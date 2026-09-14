#!/usr/bin/env julia
# =============================================================================
# run_monte_carlo_comparison.jl (Unified Mission Comparisons & Visualizations)
# =============================================================================

using Distributed, Dates, Printf, CairoMakie

const SCRIPT_START_TIME = Dates.now()
const T_START_WALL = time()

# ---- Arg Parsing -----------------------------------------------------------
const N_STRATEGIES_DEFAULT = 5

function parse_args(args)
    opts = Dict{String,String}(
        "nworkers"   => string(N_STRATEGIES_DEFAULT),
        "num_mc"     => "5",
        "base_seed"  => "1234",
        "w_rated"    => "-3.5",
        "ls"         => "0.75",
        "lt"         => "45.0",
        "strategies" => "transect,bb_ipp_nonadaptive,bb_ipp_adaptive,ergo_nonadaptive,ergo_adaptive",
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

# Parse sweep arrays
w_rated_cmd = parse.(Float64, split(opts["w_rated"], ","))
ls_cmd = parse.(Float64, split(opts["ls"], ","))
lt_cmd = parse.(Float64, split(opts["lt"], ","))

nworkers_requested = parse(Int, opts["nworkers"])
num_mc = parse(Int, opts["num_mc"])
base_seed = parse(Int, opts["base_seed"])
seeds = base_seed:(base_seed + num_mc - 1)
SCRIPT_SRC_DIR = abspath(opts["srcdir"])

if nprocs() == 1
    total_tasks = length(seeds) * length(strategies) * length(w_rated_cmd) * length(ls_cmd) * length(lt_cmd)
    addprocs(min(nworkers_requested, total_tasks); exename=joinpath(Sys.BINDIR, "julia"))
end

@everywhere SRC_DIR = $SCRIPT_SRC_DIR

# =============================================================================
# Setup Modules & Environment Code Across ALL Workers
# =============================================================================
@everywhere begin
    using LinearAlgebra, StaticArrays, Interpolations, Statistics, Random
    using SpatiotemporalGPs, JLD2, ForwardDiff, Printf

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

    SimulatorST.clamp_to_domain(pts::AbstractVector{<:AbstractVector}, xs::AbstractVector, ys::AbstractVector; kwargs...) =
        SimulatorST.clamp_to_domain([SVector{2, Float64}(p[1], p[2]) for p in pts], xs, ys; kwargs...)
end

# =============================================================================
# Environment & Controller Definitions (With Worker-Level Caching)
# =============================================================================
@everywhere const ENV_CACHE = Dict{Tuple{Int, Float64, Float64}, Any}()

@everywhere function get_base_environment(; seed=1234, ls_val=0.75, lt_val=45.0)
    key = (seed, ls_val, lt_val)
    if haskey(ENV_CACHE, key)
        return ENV_CACHE[key]
    end

    Random.seed!(seed)

    Δt      = 2.5
    dt_min  = Δt / 60
    dt_hrs  = Δt / 3600
    T_begin = 9.0
    T_end   = 12.0
    ts_hrs  = T_begin:dt_hrs:T_end
    ts_min  = T_begin*60:dt_min:T_end*60

    σt, σs = 1.0, 1.0
    kt = Matern(1/2, σt, lt_val)
    ks = Matern(1/2, σs, ls_val)

    dx = 0.05
    xs = 0:dx:1.6
    ys = 0:dx:1.9
    grid_pts = vec([@SVector[x, y] for x in xs, y in ys])

    synthetic_data = STGPKF.generate_spatiotemporal_process(xs, ys, dt_min, (T_end - T_begin) * 60, ks, kt)

    problem = STGPKFProblem(grid_pts, ks, kt, dt_min)
    ngpkf_grid = NGPKF.NGPKFGrid(synthetic_data.xs, synthetic_data.ys, ks)

    x0s = [@SVector[0.75, 0.75] for _ in 1:1]

    target_q = 0.95
    Nx, Ny = length(xs), length(ys)
    target_q_mat = zeros(Nx, Ny)
    for i in 1:Nx, j in 1:Ny
        p = [xs[i], ys[j]]
        if p ∈ JordanLakeDomain.convex_polygon.polygon
            target_q_mat[i, j] = target_q
        end
    end

    soc_begin, soc_end = 6000, 5500
    lcbf = SoCController.compute_lcbf(ts_hrs, dt_hrs)
    ucbf = SoCController.compute_ucbf(ts_hrs, dt_hrs)
    soc_target, v_opt = SoCController.generate_SOC_target(lcbf, ucbf, soc_begin, soc_end, ts_hrs, dt_hrs)

    transect_xs = 0.1:0.3:2
    transect_ys = 0.1:0.3:2
    pts = vec([[x, y] for x in transect_xs, y in transect_ys])
    transect_pts = Transects.create_points(pts)

    fuse_measurements_every_ΔT     = 5.0 / 60
    recompute_controller_every_ΔT  = 5.0 / (120.0*60)
    σ_meas = 0.5
    σ_t = zeros(length(xs), length(ys))

    base_env = (; Δt, dt_min, dt_hrs, T_begin, T_end, ts_hrs, ts_min,
            ks, kt, xs, ys, grid_pts, synthetic_data, problem, ngpkf_grid,
            x0s, target_q_mat, soc_begin, soc_end, soc_target,
            transect_pts, fuse_measurements_every_ΔT, recompute_controller_every_ΔT,
            σ_meas, σ_t, convex_polygon = JordanLakeDomain.convex_polygon)

    ENV_CACHE[key] = base_env
    return base_env
end

@everywhere function build_environment(; seed=1234, w_rated_val=-3.5, ls_val=0.75, lt_val=45.0)
    base_env = get_base_environment(; seed=seed, ls_val=ls_val, lt_val=lt_val)
    return merge(base_env, (; w_rated_val=w_rated_val))
end

# =============================================================================
# Mission Controllers & Execution Definitions
# =============================================================================
@everywhere begin
    const C_CLARITY = 1.0
    const R_CLARITY = 0.5
    const K_CLARITY = (C_CLARITY^2 / R_CLARITY)

    function clarity_delta_new(current_clarity, target_clarity)
        den = -target_clarity*K_CLARITY + K_CLARITY*current_clarity*target_clarity + K_CLARITY - K_CLARITY*current_clarity
        return (target_clarity - current_clarity) / den
    end

    function compute_target_spatial_dist(Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
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

    function heading_calculator(speed, position, waypoint)
        dx, dy = waypoint[1] - position[1], waypoint[2] - position[2]
        heading = atan(dy, dx)
        return [speed * cos(heading), speed * sin(heading)]
    end

    # 1. Transect Mission
    function make_transect_controller(env)
        return function (t, xs, Mean, w_rated_val, convex_polygon;
                ergo_grid, ergo_q_map, traj, transect_pts, waypoint_idx, umax=0.15, ΔT, kwargs...)

            _, current_q_target_temp = compute_target_spatial_dist(
                Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)

            current_waypoint = transect_pts[waypoint_idx]
            u_out = Vector{SVector{2,Float64}}(undef, length(xs))
            safe_margin_km = 0.015

            for (k, x) in enumerate(xs)
                u_raw = heading_calculator(umax, x, current_waypoint)
                u_raw_sv = @SVector[u_raw[1], u_raw[2]]
                u_out[k] = ErgodicController.convex_bounary_correction(
                    convex_polygon, x, u_raw_sv; speed_max=umax, min_safe_d=safe_margin_km
                )
            end

            if norm(xs[1] - current_waypoint) < 0.1
                waypoint_idx += 1
                if waypoint_idx > length(transect_pts)
                    waypoint_idx = 1
                end
            end
            
            return u_out, current_q_target_temp, waypoint_idx
        end
    end

    function run_transect(env)
        controller = make_transect_controller(env)
        res = SimulatorST.simulate_known_transect(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data,
            transect_pts=env.transect_pts, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        return res
    end

    # 2. Non-Adaptive Ergodic Mission
    function make_nonadaptive_ergo_controller(env)
        cache = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
        return function (t, xs, Mean, w_rated_val, convex_polygon;
                ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)

            current_target_spatial_dist, current_q_target_temp = compute_target_spatial_dist(
                Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)

            if cache[] === nothing
                cache[] = current_target_spatial_dist
            end

            target_spatial_dist = cache[]
            u = [ErgodicController.controller_single_integrator_cvx_bound(
                    ergo_grid, x, traj, target_spatial_dist, convex_polygon;
                    umax=umax, do_boundary_correction=true) for x in xs]

            return u, current_q_target_temp
        end
    end

    function run_ergo_nonadaptive(env)
        controller = make_nonadaptive_ergo_controller(env)
        res = SimulatorST.simulate_known_param(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        return res
    end

    # 3. Adaptive Ergodic Mission
    function make_adaptive_ergo_controller(env)
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

    function run_ergo_adaptive(env)
        controller = make_adaptive_ergo_controller(env)
        res = SimulatorST.simulate_known_param(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        return res
    end

    # Branch & Bound Primitive Dynamics
    struct MotionPrimitive
        dtheta::Float64
        dist::Float64
    end

    function get_motion_primitives(speed, dt, M=7)
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
            return -Inf
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
            
            base_reward = bb_reward_fast(new_x, new_y, target_grid, xs, ys, convex_polygon)
            if isinf(base_reward) && base_reward < 0
                continue
            end

            turn_penalty = 1e-6 * abs(prim.dtheta)
            reward_i = base_reward - turn_penalty

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

    # 4. Non-Adaptive Branch & Bound IPP
    function make_nonadaptive_bb_ipp_controller(env; H=5, M_primitives=7, primitive_stride=20)
        heading_state = Ref(0.0)
        dt_sec_per_primitive = primitive_stride * env.Δt
        cache = Ref{Union{Nothing,Matrix{Float64}}}(nothing)

        return function (t, xs, Mean, w_rated_val, convex_polygon;
                ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)

            current_target_spatial_dist, current_q_target_temp = compute_target_spatial_dist(
                Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)

            if cache[] === nothing
                cache[] = current_target_spatial_dist
            end

            target_spatial_dist = cache[]
            L_UB = max(maximum(target_spatial_dist), 1e-6)
            dist_km = umax * dt_sec_per_primitive / 1000.0
            primitives = get_motion_primitives(1.0, dist_km, M_primitives)

            grid_xs = ErgodicController.xs(ergo_grid)
            grid_ys = ErgodicController.ys(ergo_grid)

            u_out = Vector{SVector{2,Float64}}(undef, length(xs))
            for (k, x) in enumerate(xs)
                x_start = [x[1], x[2]]
                z_star, gamma_star = path_planning_bb_fast(
                    x_start, heading_state[], H, primitives, L_UB,
                    target_spatial_dist, grid_xs, grid_ys, convex_polygon
                )

                if isempty(z_star) || z_star[1] == 0 || isinf(gamma_star) || gamma_star <= -5.0
                    centroid = ConvexBoundAvoidance.calculate_centroid(convex_polygon)
                    step_heading = atan(centroid[2] - x[2], centroid[1] - x[1])
                else
                    chosen_prim = primitives[z_star[1]]
                    dtheta_step = chosen_prim.dtheta / primitive_stride
                    step_heading = heading_state[] + dtheta_step
                end

                u_raw = @SVector[umax * cos(step_heading), umax * sin(step_heading)]
                safe_margin_km = 0.015
                u_safe = ErgodicController.convex_bounary_correction(
                    convex_polygon, x, u_raw; speed_max=umax, min_safe_d=safe_margin_km
                )

                if norm(u_safe) > 1e-4
                    heading_state[] = atan(u_safe[2], u_safe[1])
                else
                    heading_state[] = step_heading
                end
                u_out[k] = u_safe
            end
            return u_out, current_q_target_temp
        end
    end

    function run_bb_ipp_nonadaptive(env)
        controller = make_nonadaptive_bb_ipp_controller(env)
        res = SimulatorST.simulate_known_param(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        return res
    end

    # 5. Adaptive Branch & Bound IPP
    function make_adaptive_bb_ipp_controller(env; H=5, M_primitives=7, primitive_stride=20)
        heading_state = Ref(0.0)
        dt_sec_per_primitive = primitive_stride * env.Δt

        return function (t, xs, Mean, w_rated_val, convex_polygon;
                ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)

            target_spatial_dist, q_target_temp = compute_target_spatial_dist(
                Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)

            L_UB = max(maximum(target_spatial_dist), 1e-6)
            dist_km = umax * dt_sec_per_primitive / 1000.0
            primitives = get_motion_primitives(1.0, dist_km, M_primitives)

            grid_xs = ErgodicController.xs(ergo_grid)
            grid_ys = ErgodicController.ys(ergo_grid)

            u_out = Vector{SVector{2,Float64}}(undef, length(xs))
            for (k, x) in enumerate(xs)
                x_start = [x[1], x[2]]
                z_star, gamma_star = path_planning_bb_fast(
                    x_start, heading_state[], H, primitives, L_UB,
                    target_spatial_dist, grid_xs, grid_ys, convex_polygon
                )

                if isempty(z_star) || z_star[1] == 0 || isinf(gamma_star) || gamma_star <= -5.0
                    centroid = ConvexBoundAvoidance.calculate_centroid(convex_polygon)
                    step_heading = atan(centroid[2] - x[2], centroid[1] - x[1])
                else
                    chosen_prim = primitives[z_star[1]]
                    dtheta_step = chosen_prim.dtheta / primitive_stride
                    step_heading = heading_state[] + dtheta_step
                end

                u_raw = @SVector[umax * cos(step_heading), umax * sin(step_heading)]
                safe_margin_km = 0.015
                u_safe = ErgodicController.convex_bounary_correction(
                    convex_polygon, x, u_raw; speed_max=umax, min_safe_d=safe_margin_km
                )

                if norm(u_safe) > 1e-4
                    heading_state[] = atan(u_safe[2], u_safe[1])
                else
                    heading_state[] = step_heading
                end
                u_out[k] = u_safe
            end
            return u_out, q_target_temp
        end
    end

    function run_bb_ipp_adaptive(env)
        controller = make_adaptive_bb_ipp_controller(env)
        res = SimulatorST.simulate_known_param(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        return res
    end
end

@everywhere const STRATEGY_FNS = Dict(
    :transect           => run_transect,
    :bb_ipp_nonadaptive => run_bb_ipp_nonadaptive,
    :bb_ipp_adaptive    => run_bb_ipp_adaptive,
    :ergo_nonadaptive   => run_ergo_nonadaptive,
    :ergo_adaptive      => run_ergo_adaptive,
)

# =============================================================================
# Helper Function to Compute Spatial RMSE & Clarity Deficits
# =============================================================================
@everywhere function compute_run_metrics(res, env)
    synthetic_data = env.synthetic_data
    w_rated_val = env.w_rated_val
    convex_polygon = env.convex_polygon
    xs, ys = synthetic_data.xs, synthetic_data.ys

    w_hats = res.w_hats
    N_steps = length(w_hats)
    N_truth = size(synthetic_data.data, 3)
    step_ratio = N_steps > 1 ? (N_truth - 1) / (N_steps - 1) : 0.0
    
    rmse_global_series = zeros(N_steps)
    for i in 1:N_steps
        truth_idx = N_steps > 1 ? min(floor(Int, (i - 1) * step_ratio) + 1, N_truth) : 1
        truth_map = synthetic_data.data[:, :, truth_idx]
        diff_sq   = (w_hats[i] .- truth_map) .^ 2
        rmse_global_series[i] = sqrt(mean(diff_sq))
    end

    ergo_q_maps = res.ergo_q_maps
    q_target_maps = res.q_target_maps
    N_q = length(ergo_q_maps)
    
    clarity_deficit_series = zeros(N_q)
    gt_clarity_deficit_series = zeros(N_q)
    
    # New RMSE Series
    est_clarity_rmse_series = zeros(N_q)    # Estimated Target vs. Achieved Ergodic
    gt_clarity_rmse_series = zeros(N_q)     # Ground Truth Target vs. Achieved Ergodic
    target_clarity_rmse_series = zeros(N_q) # Ground Truth Target vs. Estimated Target
    
    q_step_ratio = N_q > 1 ? (N_truth - 1) / (N_q - 1) : 0.0

    for i in 1:N_q
        # Estimated clarity deficit
        deficit_map = max.(0.0, q_target_maps[i] .- ergo_q_maps[i])
        clarity_deficit_series[i] = mean(deficit_map)

        # Ground truth clarity target setup
        truth_idx = N_q > 1 ? min(floor(Int, (i - 1) * q_step_ratio) + 1, N_truth) : 1
        truth_map = synthetic_data.data[:, :, truth_idx]
        
        target_q = 0.95
        lambda_param = 0.25
        delta = -lambda_param .* ((truth_map .- w_rated_val) .^ 2)
        gt_q_target_map = target_q .* exp.(delta)

        for xi in 1:length(xs), yj in 1:length(ys)
            p = [xs[xi], ys[yj]]
            if !(p ∈ convex_polygon.polygon)
                gt_q_target_map[xi, yj] = 0.0
            end
        end

        # Ground truth clarity deficit
        gt_deficit_map = max.(0.0, gt_q_target_map .- ergo_q_maps[i])
        gt_clarity_deficit_series[i] = mean(gt_deficit_map)

        # 1. Estimated Target Clarity vs Achieved Clarity
        est_clarity_rmse_series[i] = sqrt(mean((q_target_maps[i] .- ergo_q_maps[i]) .^ 2))
        
        # 2. Ground Truth Target Clarity vs Achieved Clarity
        gt_clarity_rmse_series[i] = sqrt(mean((gt_q_target_map .- ergo_q_maps[i]) .^ 2))
        
        # 3. Ground Truth Target Clarity vs Estimated Target Clarity
        target_clarity_rmse_series[i] = sqrt(mean((gt_q_target_map .- q_target_maps[i]) .^ 2))
    end

    return rmse_global_series, clarity_deficit_series, gt_clarity_deficit_series, est_clarity_rmse_series, gt_clarity_rmse_series, target_clarity_rmse_series
end

@everywhere function run_task(task_tuple)
    seed, strategy_name, outdir, w_rated_val, ls_val, lt_val = task_tuple
    
    filename = @sprintf("trial_seed%d_%s_w%.2f_ls%.2f_lt%.2f.jld2", seed, strategy_name, w_rated_val, ls_val, lt_val)
    outpath = joinpath(outdir, filename)

    if isfile(outpath)
        data = load(outpath)
        meas = data["measurements"]
        rmse_global = data["rmse_global"]
        clarity_deficit = data["clarity_deficit"]
        gt_clarity_deficit = get(data, "gt_clarity_deficit", clarity_deficit)
        
        # Load new RMSEs with fallbacks for backwards compatibility
        est_clarity_rmse = get(data, "est_clarity_rmse", clarity_deficit)
        gt_clarity_rmse = get(data, "gt_clarity_rmse", clarity_deficit)
        target_clarity_rmse = get(data, "target_clarity_rmse", clarity_deficit)
        
        solve_time = get(data, "solve_time", 0.0)
    else
        env = build_environment(; seed=seed, w_rated_val=w_rated_val, ls_val=ls_val, lt_val=lt_val)
        fn = STRATEGY_FNS[strategy_name]
        
        t0 = time()
        res = fn(env)
        solve_time = time() - t0

        rmse_global, clarity_deficit, gt_clarity_deficit, est_clarity_rmse, gt_clarity_rmse, target_clarity_rmse = compute_run_metrics(res, env)
        meas = vec(res.measurements)

        jldsave(outpath;
            measurements = meas,
            rmse_global = rmse_global,
            clarity_deficit = clarity_deficit,
            gt_clarity_deficit = gt_clarity_deficit,
            est_clarity_rmse = est_clarity_rmse,
            gt_clarity_rmse = gt_clarity_rmse,
            target_clarity_rmse = target_clarity_rmse,
            solve_time = solve_time,
            xs = res.xs,       
            us = res.us,       
            seed = seed,       
            strategy = string(strategy_name)
        )
    end

    return (seed, strategy_name, meas, rmse_global, clarity_deficit, gt_clarity_deficit, est_clarity_rmse, gt_clarity_rmse, target_clarity_rmse, solve_time, w_rated_val, ls_val, lt_val)
end

# =============================================================================
# Main Driver & Post-Processing
# =============================================================================
function main()
    datetime_str = Dates.format(SCRIPT_START_TIME, "yyyymmdd_HHMMSS")
    data_dir = joinpath(opts["outdir"], datetime_str)
    mkpath(data_dir)

    println("="^80)
    println("Monte Carlo Parameter Sweep Routine")
    println("Start Timestamp:  $(Dates.format(SCRIPT_START_TIME, "yyyy-mm-dd HH:MM:SS"))")
    println("Output Directory: $(abspath(data_dir))")
    println("MC Seeds:         $(seeds)")
    println("Wind Speeds (W):  $(w_rated_cmd)")
    println("Spatial (Ls):     $(ls_cmd)")
    println("Temporal (Lt):    $(lt_cmd)")
    println("Strategies:       $(strategies)")
    println("Active Workers:   $(workers())")
    println("="^80)

    tasks = [(seed, strat, data_dir, w, ls, lt) 
             for seed in seeds 
             for strat in strategies 
             for w in w_rated_cmd 
             for ls in ls_cmd 
             for lt in lt_cmd]
             
    results = pmap(run_task, tasks)

    # Dict Initialization Inside main()
    measurements_dict       = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Float64}}()
    rmse_global_dict        = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Vector{Float64}}}()
    clarity_deficit_dict    = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Vector{Float64}}}()
    gt_clarity_deficit_dict = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Vector{Float64}}}()
    est_c_rmse_dict         = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Vector{Float64}}}()
    gt_c_rmse_dict          = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Vector{Float64}}}()
    tgt_c_rmse_dict         = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Vector{Float64}}}()
    solve_time_dict         = Dict{Tuple{Symbol, Float64, Float64, Float64}, Vector{Float64}}()

    for s in strategies, w in w_rated_cmd, ls in ls_cmd, lt in lt_cmd
        measurements_dict[(s, w, ls, lt)] = Float64[]
        rmse_global_dict[(s, w, ls, lt)] = Vector{Float64}[]
        clarity_deficit_dict[(s, w, ls, lt)] = Vector{Float64}[]
        gt_clarity_deficit_dict[(s, w, ls, lt)] = Vector{Float64}[]
        est_c_rmse_dict[(s, w, ls, lt)] = Vector{Float64}[]
        gt_c_rmse_dict[(s, w, ls, lt)] = Vector{Float64}[]
        tgt_c_rmse_dict[(s, w, ls, lt)] = Vector{Float64}[]
        solve_time_dict[(s, w, ls, lt)] = Float64[]
    end

    for (seed, strat, meas, rmse_g, deficit, gt_deficit, e_c_rmse, gt_c_rmse, t_c_rmse, stime, w_val, ls_val, lt_val) in results
        append!(measurements_dict[(strat, w_val, ls_val, lt_val)], meas)
        push!(rmse_global_dict[(strat, w_val, ls_val, lt_val)], rmse_g)
        push!(clarity_deficit_dict[(strat, w_val, ls_val, lt_val)], deficit)
        push!(gt_clarity_deficit_dict[(strat, w_val, ls_val, lt_val)], gt_deficit)
        push!(est_c_rmse_dict[(strat, w_val, ls_val, lt_val)], e_c_rmse)
        push!(gt_c_rmse_dict[(strat, w_val, ls_val, lt_val)], gt_c_rmse)
        push!(tgt_c_rmse_dict[(strat, w_val, ls_val, lt_val)], t_c_rmse)
        push!(solve_time_dict[(strat, w_val, ls_val, lt_val)], stime)
    end

    allowable_buffer = 1.0
    N_mc = length(seeds)
    strategy_names_str = string.(strategies)

    # =========================================================================
    # Report Generation (TXT & CSV) - Insert this after your Makie plot block
    # =========================================================================
    txt_report_path = joinpath(data_dir, "summary_table.txt")
    open(txt_report_path, "w") do f
        println(f, "="^190)
        println(f, "MONTE CARLO SIMULATION SUMMARY TABLE")
        println(f, "Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(f, "="^190)
        
        @printf(f, "%-7s | %-5s | %-5s | %-16s | %-17s | %-18s | %-18s | %-18s | %-17s | %-17s | %-11s | %-10s\n",
            "W_rated", "Ls", "Lt", "Strategy", "RMSE (Mean±SEM)", "Est C-RMSE (±SEM)", "GT C-RMSE (±SEM)", "Tgt C-RMSE (±SEM)", "Est Def (Mean±IQR)", "GT Def (Mean±IQR)", "Time (s)", "In-Target")
        println(f, "-"^190)

        for w in w_rated_cmd, ls in ls_cmd, lt in lt_cmd
            for (k, strat) in enumerate(strategies)
                rmse_series = rmse_global_dict[(strat, w, ls, lt)]
                if isempty(rmse_series) continue end

                # Standard Error of Mean logic for RMSEs
                sem_calc(x) = N_mc > 1 ? std(x) / sqrt(N_mc) : 0.0
                
                m_rmse_g, sem_rmse_g = mean([mean(s) for s in rmse_series]), sem_calc([mean(s) for s in rmse_series])
                m_ec, sem_ec = mean([mean(s) for s in est_c_rmse_dict[(strat, w, ls, lt)]]), sem_calc([mean(s) for s in est_c_rmse_dict[(strat, w, ls, lt)]])
                m_gtc, sem_gtc = mean([mean(s) for s in gt_c_rmse_dict[(strat, w, ls, lt)]]), sem_calc([mean(s) for s in gt_c_rmse_dict[(strat, w, ls, lt)]])
                m_tc, sem_tc = mean([mean(s) for s in tgt_c_rmse_dict[(strat, w, ls, lt)]]), sem_calc([mean(s) for s in tgt_c_rmse_dict[(strat, w, ls, lt)]])

                # IQR logic for Deficits
                iqr_calc(x) = N_mc > 1 ? quantile(x, 0.75) - quantile(x, 0.25) : 0.0
                run_def = [mean(s) for s in clarity_deficit_dict[(strat, w, ls, lt)]]
                run_gt_def = [mean(s) for s in gt_clarity_deficit_dict[(strat, w, ls, lt)]]

                def_str = @sprintf("%.4f±%.4f", mean(run_def), iqr_calc(run_def))
                gt_str = @sprintf("%.4f±%.4f", mean(run_gt_def), iqr_calc(run_gt_def))
                
                time_str = @sprintf("%.2f±%.2f", mean(solve_time_dict[(strat, w, ls, lt)]), std(solve_time_dict[(strat, w, ls, lt)]))
                
                errs = measurements_dict[(strat, w, ls, lt)] .- w
                in_range = count(abs.(errs) .<= allowable_buffer) / max(1, length(errs))

                @printf(f, "%-7.2f | %-5.2f | %-5.2f | %-16s | %-17s | %-18s | %-18s | %-18s | %-17s | %-17s | %-11s | %-9.1f%%\n",
                    w, ls, lt, strategy_names_str[k],
                    @sprintf("%.4f±%.4f", m_rmse_g, sem_rmse_g),
                    @sprintf("%.4f±%.4f", m_ec, sem_ec),
                    @sprintf("%.4f±%.4f", m_gtc, sem_gtc),
                    @sprintf("%.4f±%.4f", m_tc, sem_tc),
                    def_str, gt_str, time_str, in_range * 100.0
                )
            end
            println(f, "-"^190)
        end
    end

    csv_report_path = joinpath(data_dir, "summary_metrics.csv")
    open(csv_report_path, "w") do f
        println(f, "W_Rated,Ls,Lt,Strategy,RMSE_Mean,RMSE_SEM,Est_Clarity_RMSE_Mean,Est_Clarity_RMSE_SEM,GT_Clarity_RMSE_Mean,GT_Clarity_RMSE_SEM,Target_Clarity_RMSE_Mean,Target_Clarity_RMSE_SEM,Est_Deficit_Mean,Est_Deficit_IQR,GT_Deficit_Mean,GT_Deficit_IQR,Solve_Time_Mean,Solve_Time_Std,Error_Mean,Error_Std,Proportion_In_Target")
        for w in w_rated_cmd, ls in ls_cmd, lt in lt_cmd, (k, strat) in enumerate(strategies)
            rmse_series = rmse_global_dict[(strat, w, ls, lt)]
            if isempty(rmse_series) continue end

            sem_calc(x) = N_mc > 1 ? std(x) / sqrt(N_mc) : 0.0
            iqr_calc(x) = N_mc > 1 ? quantile(x, 0.75) - quantile(x, 0.25) : 0.0

            m_rmse, sem_rmse = mean([mean(s) for s in rmse_series]), sem_calc([mean(s) for s in rmse_series])
            m_ec, sem_ec = mean([mean(s) for s in est_c_rmse_dict[(strat, w, ls, lt)]]), sem_calc([mean(s) for s in est_c_rmse_dict[(strat, w, ls, lt)]])
            m_gtc, sem_gtc = mean([mean(s) for s in gt_c_rmse_dict[(strat, w, ls, lt)]]), sem_calc([mean(s) for s in gt_c_rmse_dict[(strat, w, ls, lt)]])
            m_tc, sem_tc = mean([mean(s) for s in tgt_c_rmse_dict[(strat, w, ls, lt)]]), sem_calc([mean(s) for s in tgt_c_rmse_dict[(strat, w, ls, lt)]])

            run_def = [mean(s) for s in clarity_deficit_dict[(strat, w, ls, lt)]]
            run_gt_def = [mean(s) for s in gt_clarity_deficit_dict[(strat, w, ls, lt)]]
            
            stimes = solve_time_dict[(strat, w, ls, lt)]
            errs = measurements_dict[(strat, w, ls, lt)] .- w
            in_range = count(abs.(errs) .<= allowable_buffer) / max(1, length(errs))
            
            @printf(f, "%.2f,%.2f,%.2f,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                w, ls, lt, strategy_names_str[k],
                m_rmse, sem_rmse, m_ec, sem_ec, m_gtc, sem_gtc, m_tc, sem_tc,
                mean(run_def), iqr_calc(run_def), mean(run_gt_def), iqr_calc(run_gt_def),
                mean(stimes), std(stimes), mean(errs), std(errs), in_range
            )
        end
    end
    
    wall_runtime_sec = time() - T_START_WALL
    println("\nSweep Complete! Total Wall Runtime: ", round(wall_runtime_sec, digits=2), " seconds")
    println("Summary metrics CSV saved to: ", csv_report_path)
    println("Summary ASCII Table saved to: ", txt_report_path)
    println("="^80)
end

main()