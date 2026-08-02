#!/usr/bin/env julia
# =============================================================================
# run_comparison.jl
#
# Compares 4 mission strategies on the SAME synthetic environment/data:
#   1. :transect          - fixed lawnmower/TSP waypoint sweep
#   2. :ergo_nonadaptive   - ergodic control, target clarity map fixed at t0
#   3. :ergo_adaptive      - ergodic control, target clarity map recomputed
#                            every fusion step (this is what STGP_Sim.ipynb did)
#   4. :bb_ipp             - branch-and-bound informative path planning driven
#                            by the same "target_spatial_dist" reward field
#                            used by the adaptive ergodic controller
#
# All four are initialized from the identical synthetic_data / STGPKF problem
# / NGPKF grid / SOC target profile, and each strategy's full result struct is
# JLD2-dumped to its own file inside a single timestamped run directory, e.g.
#
#   results/20260801_142233/
#       env.jld2               <- shared environment (synthetic_data, grids, etc.)
#       transect.jld2
#       ergo_nonadaptive.jld2
#       ergo_adaptive.jld2
#       bb_ipp.jld2
#       manifest.jld2          <- run params, for provenance
#
# Designed to run on UMich Great Lakes HPC via the accompanying
# submit_great_lakes.sbatch. Strategies run as separate Distributed.jl worker
# processes (one strategy per worker) so each gets its own thread/core and
# there's no risk of the (mutating, global-heavy) simulator/module code
# stepping on itself across strategies.
# =============================================================================

using Distributed
using Dates

# ---- how many workers to launch -------------------------------------------
# Default: one worker per strategy (4). Override with:
#   julia run_comparison.jl --nworkers 4 --strategies transect,ergo_adaptive
const N_STRATEGIES_DEFAULT = 4

function parse_args(args)
    opts = Dict{String,String}(
        "nworkers"   => string(N_STRATEGIES_DEFAULT),
        "strategies" => "transect,ergo_nonadaptive,ergo_adaptive,bb_ipp",
        "outdir"     => "results",
        "srcdir"     => joinpath(@__DIR__, "../", "src"),
    )
    i = 1
    while i <= length(args)
        key = replace(args[i], "--" => "")
        if haskey(opts, key) && i < length(args)
            opts[key] = args[i+1]
            i += 2
        else
            i += 1
        end
    end
    return opts
end

opts = parse_args(ARGS)
strategies = Symbol.(split(opts["strategies"], ","))
nworkers_requested = parse(Int, opts["nworkers"])
SCRIPT_SRC_DIR = abspath(opts["srcdir"])

# Add workers (guard against re-adding if run interactively)
if nprocs() == 1
    addprocs(min(nworkers_requested, length(strategies)))
end

@everywhere SRC_DIR = $SCRIPT_SRC_DIR

# =============================================================================
# Everything below runs on ALL workers (main + spawned), so each worker can
# independently build the environment and simulate its assigned strategy.
# =============================================================================
@everywhere begin
    using LinearAlgebra, StaticArrays, Interpolations, Statistics, Random
    using SpatiotemporalGPs
    using JLD2
    using ForwardDiff

    include(joinpath(SRC_DIR, "jordan_lake_domain.jl"))
    include(joinpath(SRC_DIR, "kf.jl"))
    include(joinpath(SRC_DIR, "ngpkf.jl"))
    include(joinpath(SRC_DIR, "SyntheticData.jl"))
    include(joinpath(SRC_DIR, "ergodic.jl"))
    include(joinpath(SRC_DIR, "variograms.jl"))
    include(joinpath(SRC_DIR, "SOC_Controller.jl"))
    include(joinpath(SRC_DIR, "simulator_spatial.jl"))
    include(joinpath(SRC_DIR, "simulator_ST.jl"))
    include(joinpath(SRC_DIR, "Convex_bound_avoidance.jl"))
    include(joinpath(SRC_DIR, "transects.jl"))
end

# =============================================================================
# Shared mission/environment setup -- IDENTICAL for every strategy.
# =============================================================================
@everywhere function build_environment(; seed=1234)
    Random.seed!(seed)  # same synthetic wind field draw for every strategy

    # --- time scale ---
    Δt      = 2.5              # seconds
    dt_min  = Δt / 60          # minutes
    dt_hrs  = Δt / 3600        # hours
    T_begin = 9.0               # hours
    T_end   = 12.0              # hours
    ts_hrs  = T_begin:dt_hrs:T_end
    ts_min  = T_begin*60:dt_min:T_end*60

    # --- spatiotemporal kernels ---
    σt, σs = 1.0, 1.0
    lt = 0.75 * 60.0     # minutes
    ls = 0.75            # km
    kt = Matern(1/2, σt, lt)
    ks = Matern(1/2, σs, ls)

    # --- spatial domain / grid ---
    dx = 0.05  # km
    xs = 0:dx:1.6
    ys = 0:dx:1.9
    grid_pts = vec([@SVector[x, y] for x in xs, y in ys])

    # --- synthetic wind field (same for every strategy since seed is fixed) ---
    synthetic_data = STGPKF.generate_spatiotemporal_process(xs, ys, dt_min, (T_end - T_begin) * 60, ks, kt)

    # --- STGPKF problem + NGPKF grid ---
    problem = STGPKFProblem(grid_pts, ks, kt, dt_min)
    kern = ks
    ngpkf_grid = NGPKF.NGPKFGrid(synthetic_data.xs, synthetic_data.ys, kern)

    # --- ASV start location(s) ---
    x0s = [@SVector[0.75, 0.75] for _ in 1:1]

    # --- rated-value / target-q setup (kept for parity with the notebooks) ---
    target_q = 0.95
    Nx, Ny = length(xs), length(ys)
    target_q_mat = zeros(Nx, Ny)
    for i in 1:Nx, j in 1:Ny
        p = [xs[i], ys[j]]
        if p ∈ JordanLakeDomain.convex_polygon.polygon
            target_q_mat[i, j] = target_q
        end
    end

    # --- SOC target profile ---
    soc_begin, soc_end = 6000, 5500
    lcbf = SoCController.compute_lcbf(ts_hrs, dt_hrs)
    ucbf = SoCController.compute_ucbf(ts_hrs, dt_hrs)
    soc_target, v_opt = SoCController.generate_SOC_target(lcbf, ucbf, soc_begin, soc_end, ts_hrs, dt_hrs)

    # --- transect waypoints (used by strategy 1, harmless to build for all) ---
    transect_xs = 0.1:0.3:2
    transect_ys = 0.1:0.3:2
    pts = vec([[x, y] for x in transect_xs, y in transect_ys])
    transect_pts = Transects.create_points(pts)

    fuse_measurements_every_ΔT     = 5.0 / 60          # hours
    recompute_controller_every_ΔT  = 5.0 / (120.0*60)  # hours
    w_rated_val = 0.75
    σ_meas = 0.5
    σ_t = zeros(length(xs), length(ys))

    return (; Δt, dt_min, dt_hrs, T_begin, T_end, ts_hrs, ts_min,
            ks, kt, xs, ys, grid_pts, synthetic_data, problem, ngpkf_grid,
            x0s, target_q_mat, soc_begin, soc_end, soc_target,
            transect_pts, fuse_measurements_every_ΔT, recompute_controller_every_ΔT,
            w_rated_val, σ_meas, σ_t, convex_polygon = JordanLakeDomain.convex_polygon)
end

# =============================================================================
# Shared helper: builds the "information-gain target field" used by both the
# adaptive ergodic controller and the BB-IPP controller. Pulled out so both
# strategies compute the target the exact same way.
# =============================================================================
@everywhere function compute_target_spatial_dist(Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
    target_q = 0.95
    Nx, Ny = length(env.synthetic_data.xs), length(env.synthetic_data.ys)
    w_rated = ones(Nx, Ny) .* w_rated_val

    lambda_param = 0.25
    delta = -lambda_param .* ((Mean .- w_rated) .^ 2)
    q_target_temp = target_q .* exp.(delta)

    x_domain, y_domain = env.synthetic_data.xs, env.synthetic_data.ys
    for i in 1:length(x_domain), j in 1:length(y_domain)
        p = [x_domain[i], y_domain[j]]
        if !(p ∈ convex_polygon.polygon)
            q_target_temp[i, j] = 0.0
        end
    end

    q_target_itp = linear_interpolation((x_domain, y_domain), q_target_temp, extrapolation_bc=Interpolations.Line())
    q_target_weighted = q_target_itp(ErgodicController.xs(ergo_grid), ErgodicController.ys(ergo_grid))

    target_spatial_dist = zeros(size(ergo_q_map))
    for i in CartesianIndices(target_spatial_dist)
        target_spatial_dist[i] = q_target_weighted[i] > ergo_q_map[i] ?
            clarity_delta_new(ergo_q_map[i], q_target_weighted[i]) : 0.0
    end

    return target_spatial_dist, q_target_temp
end

@everywhere begin
    const C_CLARITY = 1.0
    const R_CLARITY = 0.5
    const K_CLARITY = (C_CLARITY^2 / R_CLARITY)

    function clarity_delta_new(current_clarity, target_clarity)
        den = -target_clarity*K_CLARITY + K_CLARITY*current_clarity*target_clarity + K_CLARITY - K_CLARITY*current_clarity
        return (target_clarity - current_clarity) / den
    end
end

# =============================================================================
# Strategy 1: Transect (fixed waypoint sweep)
# =============================================================================
@everywhere function heading_calculator(speed, position, waypoint)
    dx, dy = waypoint[1] - position[1], waypoint[2] - position[2]
    heading = atan(dy, dx)
    return [speed * cos(heading), speed * sin(heading)]
end

@everywhere function transect_controller(t, xs, Mean, w_rated_val, convex_polygon;
        ergo_grid, ergo_q_map, traj, transect_pts, waypoint_idx, umax=0.15, ΔT, kwargs...)

    current_waypoint = transect_pts[waypoint_idx]
    u = [heading_calculator(umax, x, current_waypoint) for x in xs]
    u = [@SVector[ux, uy] for (ux, uy) in u]

    # dummy q_target map so JLD2 output has the same field layout as the ergo runs
    q_target_temp = zeros(size(ergo_q_map))

    if norm(xs[1] - current_waypoint) < 0.1
        waypoint_idx += 1
        if waypoint_idx > length(transect_pts)
            waypoint_idx = 1
        end
    end
    return u, q_target_temp, waypoint_idx
end

@everywhere function run_transect(env, outdir)
    controller = transect_controller
    res = SimulatorST.simulate_known_transect(
        env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
        env.w_rated_val, env.convex_polygon, env.problem;
        ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data,
        transect_pts=env.transect_pts, σ_meas=env.σ_meas,
        Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
        fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
        recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
    jldsave(joinpath(outdir, "transect.jld2"); res, strategy="transect")
    return res
end

# =============================================================================
# Strategy 2: Non-adaptive ergodic (target clarity map fixed at t0)
# =============================================================================
@everywhere function make_nonadaptive_ergo_controller(env)
    cache = Ref{Union{Nothing,Matrix{Float64}}}(nothing)  # frozen q_target_weighted-derived target
    cached_q_target_temp = Ref{Union{Nothing,Matrix{Float64}}}(nothing)

    return function (t, xs, Mean, w_rated_val, convex_polygon;
            ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)

        if cache[] === nothing
            # compute ONCE, from the initial Mean estimate, and freeze it
            target_spatial_dist, q_target_temp = compute_target_spatial_dist(
                Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
            cache[] = target_spatial_dist
            cached_q_target_temp[] = q_target_temp
        else
            # target is frozen, but it's still compared against the *current*
            # clarity estimate so the controller keeps chasing unmet cells
            frozen_target_q_weighted = nothing  # not needed explicitly; recompute deficit vs frozen map
        end

        # Recompute the deficit each step against the frozen target field's
        # implied "target minus achieved" gap, using the frozen field itself
        # as the desired increment map (this is what makes it non-adaptive:
        # the desired clarity distribution never updates after t0).
        target_spatial_dist = cache[]

        u = [ErgodicController.controller_single_integrator_cvx_bound(
                ergo_grid, x, traj, target_spatial_dist, convex_polygon;
                umax=umax, do_boundary_correction=true) for x in xs]

        return u, cached_q_target_temp[]
    end
end

@everywhere function run_ergo_nonadaptive(env, outdir)
    controller = make_nonadaptive_ergo_controller(env)
    res = SimulatorST.simulate_known_param(
        env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
        env.w_rated_val, env.convex_polygon, env.problem;
        ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
        Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
        fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
        recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
    jldsave(joinpath(outdir, "ergo_nonadaptive.jld2"); res, strategy="ergo_nonadaptive")
    return res
end

# =============================================================================
# Strategy 3: Adaptive ergodic (target clarity map recomputed every call)
# =============================================================================
@everywhere function make_adaptive_ergo_controller(env)
    return function (t, xs, Mean, w_rated_val, convex_polygon;
            ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)
        target_spatial_dist, q_target_temp = compute_target_spatial_dist(
            Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
        u = [ErgodicController.controller_single_integrator_cvx_bound(
                ergo_grid, x, traj, target_spatial_dist, convex_polygon;
                umax=umax, do_boundary_correction=true) for x in xs]
        return u, q_target_temp
    end
end

@everywhere function run_ergo_adaptive(env, outdir)
    controller = make_adaptive_ergo_controller(env)
    res = SimulatorST.simulate_known_param(
        env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
        env.w_rated_val, env.convex_polygon, env.problem;
        ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
        Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
        fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
        recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
    jldsave(joinpath(outdir, "ergo_adaptive.jld2"); res, strategy="ergo_adaptive")
    return res
end

# =============================================================================
# Strategy 4: Branch-and-Bound IPP, using the SAME adaptive target field as
# strategy 3, but choosing motion via receding-horizon branch-and-bound over
# a fan of motion primitives instead of the ergodic control law.
# =============================================================================
@everywhere begin
    struct MotionPrimitive
        dtheta::Float64
        dist::Float64
    end

    function get_motion_primitives(speed, dt, M=5)
        dthetas = range(-pi/3, pi/3, length=M)
        return [MotionPrimitive(dth, speed * dt) for dth in dthetas]
    end

    mutable struct BBState
        gamma_star::Float64
        z_star::Vector{Int}
    end

    @inline function bb_reward_fast(x::Float64, y::Float64, target_grid::Matrix{Float64}, 
                                  xs::AbstractVector, ys::AbstractVector, convex_polygon)
        p = @SVector[x, y]
        if !(p ∈ convex_polygon.polygon)
            return -10.0
        end
        ix = clamp(round(Int, (x - xs[1]) / (xs[2] - xs[1])) + 1, 1, size(target_grid, 1))
        iy = clamp(round(Int, (y - ys[1]) / (ys[2] - ys[1])) + 1, 1, size(target_grid, 2))
        return target_grid[ix, iy]
    end

    function bb_recursion_fast!(x::Float64, y::Float64, z_path::Vector{Int}, gamma_parent::Float64,
            j::Int, H::Int, L_UB::Float64, primitives::Vector{MotionPrimitive},
            state::BBState, heading::Float64, target_grid::Matrix{Float64}, 
            xs::AbstractVector, ys::AbstractVector, convex_polygon)

        gamma_max = gamma_parent + (H - j) * L_UB
        if gamma_max <= state.gamma_star
            return
        end

        for (i, prim) in enumerate(primitives)
            new_heading = heading + prim.dtheta
            new_x = x + prim.dist * cos(new_heading)
            new_y = y + prim.dist * sin(new_heading)
            
            reward_i = bb_reward_fast(new_x, new_y, target_grid, xs, ys, convex_polygon)
            gamma_child = gamma_parent + reward_i
            
            z_path[j + 1] = i

            if j + 1 == H
                if gamma_child > state.gamma_star
                    state.gamma_star = gamma_child
                    state.z_star .= z_path
                end
            else
                bb_recursion_fast!(new_x, new_y, z_path, gamma_child, j + 1, H, L_UB,
                    primitives, state, new_heading, target_grid, xs, ys, convex_polygon)
            end
        end
    end

    function path_planning_bb_fast(x_start::Vector{Float64}, heading::Float64, H::Int,
            primitives::Vector{MotionPrimitive}, L_UB::Float64, target_grid::Matrix{Float64},
            xs::AbstractVector, ys::AbstractVector, convex_polygon)
        
        state = BBState(-Inf, zeros(Int, H))
        scratch_path = zeros(Int, H)
        
        bb_recursion_fast!(x_start[1], x_start[2], scratch_path, 0.0, 0, H, L_UB, 
                           primitives, state, heading, target_grid, xs, ys, convex_polygon)
                           
        return state.z_star, state.gamma_star
    end
end

@everywhere function make_bb_ipp_controller(env; H=7, M_primitives=5, primitive_stride=130)
    heading_state = Ref(0.0)

    # umax is in m/s (matches usage elsewhere: transect_controller, ergodic
    # controllers). Grid coordinates (xs, ys, convex_polygon) are in km.
    # primitive_stride = number of physical sim ticks (env.Δt seconds each)
    # a single B&B primitive represents, so the search actually resolves
    # distinct grid cells instead of everything landing in the same cell.
    dt_sec_per_primitive = primitive_stride * env.Δt          # seconds
    dist_km_per_primitive = (umax_placeholder = nothing)      # set inside closure below

    return function (t, xs, Mean, w_rated_val, convex_polygon;
            ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)

        target_spatial_dist, q_target_temp = compute_target_spatial_dist(
            Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)

        L_UB = max(maximum(target_spatial_dist), 1e-6)

        # umax [m/s] * dt_sec [s] = meters -> /1000 -> km, to match grid units
        dist_km = umax * dt_sec_per_primitive / 1000.0
        primitives = get_motion_primitives(1.0, dist_km, M_primitives)  # speed=1, "dt"=dist_km directly

        grid_xs = ErgodicController.xs(ergo_grid)
        grid_ys = ErgodicController.ys(ergo_grid)

        u_out = Vector{SVector{2,Float64}}(undef, length(xs))
        for (k, x) in enumerate(xs)
            x_start = [x[1], x[2]]
            z_star, gamma_star = path_planning_bb_fast(
                x_start, heading_state[], H, primitives, L_UB,
                target_spatial_dist, grid_xs, grid_ys, convex_polygon
            )

            if isempty(z_star) || z_star[1] == 0 || gamma_star <= -5.0
                max_idx = argmax(target_spatial_dist)
                target_x = grid_xs[max_idx[1]]
                target_y = grid_ys[max_idx[2]]
                heading_state[] = atan(target_y - x[2], target_x - x[1])
            else
                chosen_prim = primitives[z_star[1]]
                heading_state[] += chosen_prim.dtheta
            end

            u_out[k] = @SVector[umax * cos(heading_state[]), umax * sin(heading_state[])]
        end

        return u_out, q_target_temp
    end
end

@everywhere function run_bb_ipp(env, outdir)
    controller = make_bb_ipp_controller(env)
    res = SimulatorST.simulate_known_param(
        env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
        env.w_rated_val, env.convex_polygon, env.problem;
        ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
        Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
        fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
        recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
    jldsave(joinpath(outdir, "bb_ipp.jld2"); res, strategy="bb_ipp")
    return res
end

# =============================================================================
# Dispatch table
# =============================================================================
@everywhere const STRATEGY_FNS = Dict(
    :transect          => run_transect,
    :ergo_nonadaptive  => run_ergo_nonadaptive,
    :ergo_adaptive     => run_ergo_adaptive,
    :bb_ipp            => run_bb_ipp,
)

@everywhere function run_one_strategy(strategy_name, outdir, seed)
    env = build_environment(; seed=seed)  # rebuilt identically on each worker (same seed)
    fn = STRATEGY_FNS[strategy_name]
    println("[$(strategy_name)] starting on worker $(myid()) / pid $(getpid())")
    t0 = time()
    res = fn(env, outdir)
    println("[$(strategy_name)] finished in $(round(time()-t0, digits=1))s -> $(outdir)/$(strategy_name).jld2")
    return strategy_name
end

# =============================================================================
# Main driver
# =============================================================================
function main()
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    outdir = joinpath(opts["outdir"], timestamp)
    mkpath(outdir)
    println("Run directory: $(abspath(outdir))")
    println("Strategies:    $(strategies)")
    println("Workers:       $(workers())")

    # Save one shared copy of the environment for reference / plotting later
    env_main = build_environment(; seed=1234)
    jldsave(joinpath(outdir, "env.jld2");
        xs=env_main.xs, ys=env_main.ys, ts_hrs=env_main.ts_hrs, ts_min=env_main.ts_min,
        synthetic_data=env_main.synthetic_data, soc_target=env_main.soc_target,
        transect_pts=env_main.transect_pts, w_rated_val=env_main.w_rated_val)

    jldsave(joinpath(outdir, "manifest.jld2");
        strategies=String.(strategies), seed=1234, timestamp=timestamp,
        src_dir=SCRIPT_SRC_DIR)

    seed = 1234
    results = pmap(s -> run_one_strategy(s, outdir, seed), strategies)

    println("\nAll strategies complete: $(results)")
    println("Results saved under: $(abspath(outdir))")
end

main()