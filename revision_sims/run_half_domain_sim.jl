#!/usr/bin/env julia
# =============================================================================
# run_half_domain_sim.jl (Single Static Half-Domain Comparison)
# =============================================================================

using Distributed, Dates, Printf, CairoMakie
include(joinpath(@__DIR__, "half_domain_diagnostics.jl"))

const SCRIPT_START_TIME = Dates.now()
const T_START_WALL = time()

# ---- Arg Parsing -----------------------------------------------------------
# Example: julia --project=. revision_sims/run_half_domain_sim.jl --w_rated -3.5 --wind_offset 6.0
# West: x < 0.8 km, rated wind. East (including the midpoint): rated + offset.
# ls/lt are fixed estimator hyperparameters, not truth-field correlation scales.
const N_STRATEGIES_DEFAULT = 7

function parse_args(args)
    opts = Dict{String,String}(
        "nworkers"   => string(N_STRATEGIES_DEFAULT),
        "seed"  => "1234",
        "w_rated"    => "-3.5",
        "wind_offset" => "6.0",
        "animation_seconds" => "30",
        "animation_fps" => "10",
        "ls"         => "0.75",
        "lt"         => "45.0",
        "strategies" => "transect,bb_ipp_nonadaptive,bb_ipp_adaptive,bb_ipp_ground_truth,ergo_nonadaptive,ergo_adaptive,ergo_ground_truth",
        "outdir"     => "results_half_domain",
        "srcdir"     => joinpath(@__DIR__, "../", "src"),
    )
    i = 1
    while i <= length(args)
        key = replace(args[i], "--" => "")
        if haskey(opts, key) && i < length(args)
            opts[key] = args[i+1]
            i += 2
        else
            error("Unknown option or missing value: $(args[i])")
        end
    end
    return opts
end

opts = parse_args(ARGS)
strategies = Symbol.(split(opts["strategies"], ","))

# Scalar parameters deliberately reject comma-separated sweeps.
w_rated_val = parse(Float64, opts["w_rated"])
wind_offset = parse(Float64, opts["wind_offset"])
ls_val = parse(Float64, opts["ls"])
lt_val = parse(Float64, opts["lt"])
animation_seconds = parse(Float64, opts["animation_seconds"])
animation_fps = parse(Int, opts["animation_fps"])
0 < animation_seconds <= 30 || error("animation_seconds must be in (0, 30]")
animation_fps > 0 || error("animation_fps must be positive")
floor(Int, animation_seconds * animation_fps) >= 2 || error("Animation needs at least two frames")
seed = parse(Int, opts["seed"])
nworkers_requested = parse(Int, opts["nworkers"])
all(isfinite, (w_rated_val, wind_offset, ls_val, lt_val)) || error("Parameters must be finite")
abs(wind_offset) > 1.0 || error("wind_offset must lie outside the ±1 normalized wind speed target band")
ls_val > 0 && lt_val > 0 || error("ls and lt must be positive")
nworkers_requested >= 0 || error("nworkers must be nonnegative")
supported = Set([:transect, :bb_ipp_nonadaptive, :bb_ipp_adaptive, :bb_ipp_ground_truth,
    :ergo_nonadaptive, :ergo_adaptive, :ergo_ground_truth])
!isempty(strategies) && all(s -> s in supported, strategies) || error("Unknown strategy")
length(unique(strategies)) == length(strategies) || error("Duplicate strategies")
SCRIPT_SRC_DIR = abspath(opts["srcdir"])

if nprocs() == 1 && nworkers_requested > 0
    addprocs(min(nworkers_requested, length(strategies));
        exename=joinpath(Sys.BINDIR, "julia"),
        exeflags=`--project=$(dirname(Base.active_project()))`)
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
# Deterministic ground truth; the callable sampler preserves the sharp boundary.
# =============================================================================
@everywhere struct HalfDomainWind
    split_x::Float64
    west::Float64
    east::Float64
end
@everywhere (wind::HalfDomainWind)(x, y, t) = x < wind.split_x ? wind.west : wind.east

@everywhere function generate_half_domain_data(xs, ys, ts, w_rated_val, wind_offset)
    split_x = (first(xs) + last(xs)) / 2
    wind = HalfDomainWind(split_x, w_rated_val, w_rated_val + wind_offset)
    # All slices are identical; measurement noise is added only by the simulator.
    data = [wind(x, y, t) for x in xs, y in ys, t in ts]
    return (; xs, ys, ts, data, itp=wind)
end

@everywhere function build_environment(; w_rated_val=-3.5, wind_offset=6.0,
        ls_val=0.75, lt_val=45.0)
    Δt      = 2.5
    dt_min  = Δt / 60
    dt_hrs  = Δt / 3600
    T_begin = 9.0
    T_end   = 15.0
    ts_hrs  = T_begin:dt_hrs:T_end
    ts_min  = T_begin*60:dt_min:T_end*60

    σt, σs = 1.0, 1.0
    kt = Matern(1/2, σt, lt_val)
    ks = Matern(1/2, σs, ls_val)

    dx = 0.05
    xs = 0:dx:1.6
    ys = 0:dx:1.9
    grid_pts = vec([@SVector[x, y] for x in xs, y in ys])

    synthetic_data = generate_half_domain_data(xs, ys, ts_min, w_rated_val, wind_offset)

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

    soc_begin, soc_end = 6000, 5250
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

    return merge(base_env, (; w_rated_val, wind_offset, ls_val, lt_val))
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


    function target_clarity_reward_grid(q_target_map, ergo_grid, env)
        q_target_itp = linear_interpolation(
            (env.synthetic_data.xs, env.synthetic_data.ys), q_target_map,
            extrapolation_bc=Interpolations.Line())
        return q_target_itp(
            ErgodicController.xs(ergo_grid), ErgodicController.ys(ergo_grid))
    end

    function adaptive_bb_reward_grid(target_clarity, grid_xs, grid_ys, position;
            attraction_weight=0.35, high_clarity_fraction=0.9)
        max_clarity = maximum(target_clarity)
        max_clarity > 0 || return zeros(size(target_clarity))

        high_clarity = findall(q -> q >= high_clarity_fraction * max_clarity, target_clarity)
        goal_idx = high_clarity[argmin([
            hypot(grid_xs[idx[1]] - position[1], grid_ys[idx[2]] - position[2])
            for idx in high_clarity])]
        goal_x, goal_y = grid_xs[goal_idx[1]], grid_ys[goal_idx[2]]
        domain_diagonal = max(hypot(last(grid_xs) - first(grid_xs),
            last(grid_ys) - first(grid_ys)), eps(Float64))

        reward = similar(target_clarity, Float64)
        for idx in CartesianIndices(reward)
            target_score = target_clarity[idx] / max_clarity
            attraction_score = 1.0 - hypot(
                grid_xs[idx[1]] - goal_x, grid_ys[idx[2]] - goal_y) / domain_diagonal
            reward[idx] = (1.0 - attraction_weight) * target_score +
                attraction_weight * attraction_score
        end
        return reward
    end



function uniform_target_clarity_map(env, convex_polygon; target_q=0.95)
    q_target = zeros(length(env.synthetic_data.xs), length(env.synthetic_data.ys))
    for i in eachindex(env.synthetic_data.xs), j in eachindex(env.synthetic_data.ys)
        p = [env.synthetic_data.xs[i], env.synthetic_data.ys[j]]
        if p in convex_polygon.polygon
            q_target[i, j] = target_q
        end
    end
    return q_target
end

function clarity_deficit_from_target(q_target_map, ergo_q_map, ergo_grid, env)
    q_target_itp = linear_interpolation(
        (env.synthetic_data.xs, env.synthetic_data.ys), q_target_map,
        extrapolation_bc=Interpolations.Line())
    q_target_weighted = q_target_itp(
        ErgodicController.xs(ergo_grid), ErgodicController.ys(ergo_grid))

    deficit = zeros(size(ergo_q_map))
    for idx in CartesianIndices(deficit)
        deficit[idx] = q_target_weighted[idx] > ergo_q_map[idx] ?
            clarity_delta_new(ergo_q_map[idx], q_target_weighted[idx]) : 0.0
    end
    return deficit
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
    uniform_q_target = uniform_target_clarity_map(env, env.convex_polygon)
    return function (t, xs, Mean, w_rated_val, convex_polygon;
            ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)

        # Preserve the current STGPKF target for logging, but do not use it
        # to control the environment-agnostic baseline.
        _, current_q_target_temp = compute_target_spatial_dist(
            Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
        target_spatial_dist = clarity_deficit_from_target(
            uniform_q_target, ergo_q_map, ergo_grid, env)

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

    # 4. Ground Truth Adaptive Ergodic Mission
    function ground_truth_wind_map(env, t)
        truth_idx = clamp(
            round(Int, (t - env.ts_min[1]) / env.dt_min) + 1,
            1,
            size(env.synthetic_data.data, 3)
        )
        return env.synthetic_data.data[:, :, truth_idx]
    end

    function make_ground_truth_ergo_controller(env)
        return function (t, xs, Mean, w_rated_val, convex_polygon;
                ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)
            truth_wind = ground_truth_wind_map(env, t)
            target_spatial_dist, q_target_temp = compute_target_spatial_dist(
                truth_wind, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
            u = [ErgodicController.controller_single_integrator_cvx_bound(
                    ergo_grid, x, traj, target_spatial_dist, convex_polygon;
                    umax=umax, do_boundary_correction=true) for x in xs]
            return u, q_target_temp
        end
    end

    function run_ergo_ground_truth(env)
        controller = make_ground_truth_ergo_controller(env)
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
    uniform_q_target = uniform_target_clarity_map(env, env.convex_polygon)

    return function (t, xs, Mean, w_rated_val, convex_polygon;
            ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)

        # Preserve the live STGPKF target map for logging. Planning uses the
        # evolving deficit against a fixed uniform target over the domain.
        _, current_q_target_temp = compute_target_spatial_dist(
            Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
        target_spatial_dist = clarity_deficit_from_target(
            uniform_q_target, ergo_q_map, ergo_grid, env)

        dist_km = umax * dt_sec_per_primitive / 1000.0
        primitives = get_motion_primitives(1.0, dist_km, M_primitives)
        grid_xs = ErgodicController.xs(ergo_grid)
        grid_ys = ErgodicController.ys(ergo_grid)

        u_out = Vector{SVector{2,Float64}}(undef, length(xs))
        for (k, x) in enumerate(xs)
            x_start = [x[1], x[2]]
            planning_reward = adaptive_bb_reward_grid(
                target_spatial_dist, grid_xs, grid_ys, x)
            planning_upper_bound = max(maximum(planning_reward), 1e-6)
            z_star, gamma_star = path_planning_bb_fast(
                x_start, heading_state[], H, primitives, planning_upper_bound,
                planning_reward, grid_xs, grid_ys, convex_polygon)

            if isempty(z_star) || z_star[1] == 0 || isinf(gamma_star) || gamma_star <= -5.0
                centroid = ConvexBoundAvoidance.calculate_centroid(convex_polygon)
                step_heading = atan(centroid[2] - x[2], centroid[1] - x[1])
            else
                chosen_prim = primitives[z_star[1]]
                dtheta_step = chosen_prim.dtheta / primitive_stride
                step_heading = heading_state[] + dtheta_step
            end

            u_raw = @SVector[umax * cos(step_heading), umax * sin(step_heading)]
            u_safe = ErgodicController.convex_bounary_correction(
                convex_polygon, x, u_raw; speed_max=umax, min_safe_d=0.015)

            heading_state[] = norm(u_safe) > 1e-4 ?
                atan(u_safe[2], u_safe[1]) : step_heading
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

            _, q_target_temp = compute_target_spatial_dist(
                Mean, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)

            # Plan from the live target-clarity map derived from the latest STGPKF mean.
            target_clarity_grid = target_clarity_reward_grid(q_target_temp, ergo_grid, env)
            dist_km = umax * dt_sec_per_primitive / 1000.0
            primitives = get_motion_primitives(1.0, dist_km, M_primitives)

            grid_xs = ErgodicController.xs(ergo_grid)
            grid_ys = ErgodicController.ys(ergo_grid)

            u_out = Vector{SVector{2,Float64}}(undef, length(xs))
            for (k, x) in enumerate(xs)
                x_start = [x[1], x[2]]
                planning_reward = adaptive_bb_reward_grid(
                    target_clarity_grid, grid_xs, grid_ys, x)
                planning_upper_bound = max(maximum(planning_reward), 1e-6)
                z_star, gamma_star = path_planning_bb_fast(
                    x_start, heading_state[], H, primitives, planning_upper_bound,
                    planning_reward, grid_xs, grid_ys, convex_polygon
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

# Ground-truth Branch & Bound IPP (oracle target-clarity benchmark)
function make_ground_truth_bb_ipp_controller(env; H=5, M_primitives=7, primitive_stride=20)
    heading_state = Ref(0.0)
    dt_sec_per_primitive = primitive_stride * env.Δt

    return function (t, xs, Mean, w_rated_val, convex_polygon;
            ergo_grid, ergo_q_map, traj, umax=0.15, ΔT, kwargs...)
        truth_wind = ground_truth_wind_map(env, t)
        _, q_target_temp = compute_target_spatial_dist(
            truth_wind, ergo_q_map, w_rated_val, convex_polygon, ergo_grid, env)
        target_clarity_grid = target_clarity_reward_grid(q_target_temp, ergo_grid, env)

        dist_km = umax * dt_sec_per_primitive / 1000.0
        primitives = get_motion_primitives(1.0, dist_km, M_primitives)
        grid_xs = ErgodicController.xs(ergo_grid)
        grid_ys = ErgodicController.ys(ergo_grid)

        u_out = Vector{SVector{2,Float64}}(undef, length(xs))
        for (k, x) in enumerate(xs)
            x_start = [x[1], x[2]]
            planning_reward = adaptive_bb_reward_grid(
                target_clarity_grid, grid_xs, grid_ys, x)
            planning_upper_bound = max(maximum(planning_reward), 1e-6)
            z_star, gamma_star = path_planning_bb_fast(
                x_start, heading_state[], H, primitives, planning_upper_bound,
                planning_reward, grid_xs, grid_ys, convex_polygon)

            if isempty(z_star) || z_star[1] == 0 || isinf(gamma_star) || gamma_star <= -5.0
                centroid = ConvexBoundAvoidance.calculate_centroid(convex_polygon)
                step_heading = atan(centroid[2] - x[2], centroid[1] - x[1])
            else
                chosen_prim = primitives[z_star[1]]
                dtheta_step = chosen_prim.dtheta / primitive_stride
                step_heading = heading_state[] + dtheta_step
            end

            u_raw = @SVector[umax * cos(step_heading), umax * sin(step_heading)]
            u_safe = ErgodicController.convex_bounary_correction(
                convex_polygon, x, u_raw; speed_max=umax, min_safe_d=0.015)
            heading_state[] = norm(u_safe) > 1e-4 ?
                atan(u_safe[2], u_safe[1]) : step_heading
            u_out[k] = u_safe
        end
        return u_out, q_target_temp
    end
end

function run_bb_ipp_ground_truth(env)
    controller = make_ground_truth_bb_ipp_controller(env)
    return SimulatorST.simulate_known_param(
        env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
        env.w_rated_val, env.convex_polygon, env.problem;
        ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
        Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
        fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
        recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
end

end

@everywhere const STRATEGY_FNS = Dict(
    :transect           => run_transect,
    :bb_ipp_nonadaptive => run_bb_ipp_nonadaptive,
    :bb_ipp_adaptive    => run_bb_ipp_adaptive,
    :bb_ipp_ground_truth  => run_bb_ipp_ground_truth,
    :ergo_nonadaptive   => run_ergo_nonadaptive,
    :ergo_adaptive      => run_ergo_adaptive,
    :ergo_ground_truth  => run_ergo_ground_truth,
)

# =============================================================================

@everywhere function ground_truth_target_clarity_map(env, t)
    truth_map = ground_truth_wind_map(env, t)
    target = 0.95 .* exp.(-0.25 .* ((truth_map .- env.w_rated_val) .^ 2))
    for i in eachindex(env.synthetic_data.xs), j in eachindex(env.synthetic_data.ys)
        if !([env.synthetic_data.xs[i], env.synthetic_data.ys[j]] in env.convex_polygon.polygon)
            target[i, j] = 0.0
        end
    end
    return target
end

# Helper Function to Compute Spatial RMSE & Clarity Deficits
# =============================================================================
@everywhere function compute_run_metrics(res, env)
    w_hats = res.w_hats
    N_steps = length(w_hats)

    rmse_global_series = zeros(N_steps)
    for i in 1:N_steps
        truth_map = ground_truth_wind_map(env, res.w_hat_ts[i])
        rmse_global_series[i] = sqrt(mean((w_hats[i] .- truth_map) .^ 2))
    end

    ergo_q_maps = res.ergo_q_maps
    q_target_maps = res.q_target_maps
    N_q = length(ergo_q_maps)

    clarity_deficit_series = zeros(N_q)
    gt_clarity_deficit_series = zeros(N_q)
    est_clarity_rmse_series = zeros(N_q)
    gt_clarity_rmse_series = zeros(N_q)
    target_clarity_rmse_series = zeros(N_q)

    # Targets are emitted on controller timestamps; clarity maps are emitted on
    # filter timestamps. Compare estimated and true targets at each clarity time.
    target_ts = [first(res.ts); collect(res.ts[1:end-1])]
    for i in 1:N_q
        metric_time = res.w_hat_ts[i]
        target_idx = clamp(
            searchsortedlast(target_ts, metric_time), 1, length(q_target_maps))
        estimated_target = q_target_maps[target_idx]
        target_time = target_ts[target_idx]
        ground_truth_target = ground_truth_target_clarity_map(env, target_time)
        achieved_clarity = ergo_q_maps[i]

        clarity_deficit_series[i] =
            mean(max.(0.0, estimated_target .- achieved_clarity))
        gt_clarity_deficit_series[i] =
            mean(max.(0.0, ground_truth_target .- achieved_clarity))
        est_clarity_rmse_series[i] =
            sqrt(mean((estimated_target .- achieved_clarity) .^ 2))
        gt_clarity_rmse_series[i] =
            sqrt(mean((ground_truth_target .- achieved_clarity) .^ 2))
        target_clarity_rmse_series[i] =
            sqrt(mean((ground_truth_target .- estimated_target) .^ 2))
    end

    return rmse_global_series, clarity_deficit_series,
        gt_clarity_deficit_series, est_clarity_rmse_series,
        gt_clarity_rmse_series, target_clarity_rmse_series
end

# Sample maps on their actual update timestamps, holding each until the next
# update. Keep every path point so frame skipping never straightens the route.
@everywhere function animation_samples(res, env; seconds=30.0, fps=10)
    nframes = min(length(res.xs), floor(Int, seconds * fps))
    indices = unique(round.(Int, range(1, length(res.xs); length=nframes)))
    times = collect(res.ts)[indices]
    target_ts = [first(res.ts); collect(res.ts[1:end-1])]
    clarity_indices = [clamp(searchsortedlast(res.w_hat_ts, t),
        1, length(res.ergo_q_maps)) for t in times]
    target_indices = [clamp(searchsortedlast(target_ts, t),
        1, length(res.q_target_maps)) for t in times]
    truth_target_maps = [ground_truth_target_clarity_map(env, target_ts[i])
        for i in target_indices]
    return (; indices, times,
        clarity_maps=res.ergo_q_maps[clarity_indices],
        target_maps=res.q_target_maps[target_indices],
        truth_target_maps, fps)
end

function animate_strategy(strategy, trial, env, outdir)
    animation = trial["animation"]
    history = trial["xs"]
    frame = Observable(1)
    clarity = lift(i -> animation.clarity_maps[i], frame)
    target = lift(i -> animation.target_maps[i], frame)
    title = lift(i -> @sprintf("%s | Mission time: %.1f min", strategy,
        animation.times[i] - first(env.ts_min)), frame)
    fig = Figure(size=(1500, 560))
    Label(fig[0, 1:6], title; fontsize=20)
    polygon = hcat(env.convex_polygon.vertices, env.convex_polygon.vertices[:, 1])
    fields = (env.synthetic_data.data[:, :, 1], clarity, target)
    titles = ("Ground-truth wind and path", "Achieved clarity", "Target clarity")
    for panel in 1:3
        col = 2panel - 1
        ax = Axis(fig[1, col]; title=titles[panel], xlabel="West → East (km)",
            ylabel="South → North (km)", aspect=DataAspect())
        limits!(ax, first(env.xs), last(env.xs), first(env.ys), last(env.ys))
        colorrange = panel == 1 ? extrema(env.synthetic_data.data[:, :, 1]) : (0.0, 1.0)
        hm = heatmap!(ax, env.xs, env.ys, fields[panel]; colorrange, colormap=:viridis)
        Colorbar(fig[1, col+1], hm; label=panel == 1 ? "Normalized wind speed" : "Clarity")
        lines!(ax, polygon[1, :], polygon[2, :]; color=:white, linewidth=2)
        vlines!(ax, [env.synthetic_data.itp.split_x]; color=:white, linestyle=:dash)
        for robot in eachindex(first(history))
            path = lift(frame) do i
                [Point2f(state[robot][1], state[robot][2])
                    for state in history[1:animation.indices[i]]]
            end
            current = lift(points -> [last(points)], path)
            lines!(ax, path; color=:orange, linewidth=2)
            scatter!(ax, current; color=:red, strokecolor=:white, strokewidth=1, markersize=12)
        end
    end
    mkpath(outdir)
    path = joinpath(outdir, "$(strategy).mp4")
    record(fig, path, eachindex(animation.indices); framerate=animation.fps) do i
        frame[] = i
    end
    println("Animation saved: $path ($(length(animation.indices) / animation.fps) s)")
    return path
end

@everywhere function run_task(task)
    strategy, seed, outdir, animation_seconds, animation_fps = task
    # Fresh estimator state and the same noise seed for every strategy, regardless
    # of which worker runs it. The truth is deterministic and identical everywhere.
    env = build_environment(; HALF_DOMAIN_CONFIG...)
    Random.seed!(seed)
    t0 = time()
    res = STRATEGY_FNS[strategy](env)
    solve_time = time() - t0
    rmse, deficit, gt_deficit, est_c_rmse, gt_c_rmse, tgt_c_rmse = compute_run_metrics(res, env)
    measurements = vec(res.measurements)
    jldsave(joinpath(outdir, "trial_$(strategy).jld2");
        strategy=string(strategy), seed, measurements, solve_time,
        metric_times=collect(res.w_hat_ts),
        rmse_global=rmse, clarity_deficit=deficit, gt_clarity_deficit=gt_deficit,
        est_clarity_rmse=est_c_rmse, gt_clarity_rmse=gt_c_rmse,
        target_clarity_rmse=tgt_c_rmse, xs=res.xs, us=res.us,
        animation=animation_samples(res, env; seconds=animation_seconds, fps=animation_fps),
        w_rated=env.w_rated_val, wind_offset=env.wind_offset,
        ls=env.ls_val, lt=env.lt_val)
    errors = measurements .- env.w_rated_val
    return (; strategy=string(strategy), rmse=mean(rmse),
        est_clarity_rmse=mean(est_c_rmse), gt_clarity_rmse=mean(gt_c_rmse),
        target_clarity_rmse=mean(tgt_c_rmse), est_deficit=mean(deficit),
        gt_deficit=mean(gt_deficit), solve_time, error_mean=mean(errors),
        error_std=length(errors) > 1 ? std(errors) : 0.0,
        in_target=100 * count(abs.(errors) .<= 1.0) / max(1, length(errors)))
end

function main()
    data_dir = joinpath(opts["outdir"], Dates.format(SCRIPT_START_TIME, "yyyymmdd_HHMMSS"))
    mkpath(data_dir)
    config = (; w_rated_val, wind_offset, ls_val, lt_val)
    @everywhere HALF_DOMAIN_CONFIG = $config
    env = build_environment(; config...)
    wind = env.synthetic_data.itp
    println("Single static half-domain comparison")
    println("West (x < $(wind.split_x) km): $(wind.west) normalized wind speed; east: $(wind.east) normalized wind speed")
    println("Strategies: $(strategies); measurement-noise seed: $seed")
    println("Output directory: $(abspath(data_dir))")
    jldsave(joinpath(data_dir, "environment.jld2");
        xs=env.xs, ys=env.ys, ts_min=env.ts_min,
        wind_map=env.synthetic_data.data[:, :, 1], split_x=wind.split_x,
        west_wind=wind.west, east_wind=wind.east, time_invariant=true,
        w_rated=w_rated_val, wind_offset, ls=ls_val, lt=lt_val, seed)

    tasks = [(strategy, seed, data_dir, animation_seconds, animation_fps) for strategy in strategies]
    rows = nprocs() == 1 ? map(run_task, tasks) : pmap(run_task, tasks)
    open(joinpath(data_dir, "summary_metrics.csv"), "w") do io
        println(io, join(string.(propertynames(first(rows))), ","))
        for row in rows
            println(io, join(values(row), ","))
        end
    end
    open(joinpath(data_dir, "summary_table.txt"), "w") do io
        println(io, "SINGLE STATIC HALF-DOMAIN COMPARISON")
        println(io, "West=$(wind.west), east=$(wind.east) normalized wind speed; split=$(wind.split_x) km")
        println(io, "One run per strategy; metrics are mission averages, without Monte Carlo uncertainty.")
        @printf(io, "%-24s %12s %12s %12s %12s\n", "Strategy", "RMSE", "GT deficit", "In-target %", "Time (s)")
        for row in rows
            @printf(io, "%-24s %12.4f %12.4f %12.2f %12.2f\n",
                row.strategy, row.rmse, row.gt_deficit, row.in_target, row.solve_time)
        end
    end

    figures_dir = joinpath(data_dir, "figures")
    mkpath(figures_dir)
    fig = Figure(size=(1500, 550))
    for (i, (metric, label)) in enumerate([
            (:rmse, "Global RMSE"), (:gt_deficit, "Ground-truth clarity deficit"),
            (:in_target, "Measurements within ±1 normalized wind speed of rated (%)")])
        ax = Axis(fig[1, i]; title=label,
            xticks=(collect(eachindex(rows)), [row.strategy for row in rows]),
            xticklabelrotation=pi/4)
        barplot!(ax, collect(eachindex(rows)), [getproperty(row, metric) for row in rows])
    end
    save(joinpath(figures_dir, "strategy_comparison.png"), fig)
    save(joinpath(figures_dir, "strategy_comparison.pdf"), fig)
    truth_fig = Figure(size=(650, 550))
    ax = Axis(truth_fig[1, 1]; title="Static ground-truth wind", xlabel="West → East (km)",
        ylabel="South → North (km)", aspect=DataAspect())
    hm = heatmap!(ax, env.xs, env.ys, env.synthetic_data.data[:, :, 1])
    vertices = env.convex_polygon.vertices
    lines!(ax, [vertices[1, :]; vertices[1, 1]], [vertices[2, :]; vertices[2, 1]]; color=:black)
    vlines!(ax, [wind.split_x]; color=:black, linestyle=:dash)
    Colorbar(truth_fig[1, 2], hm; label="Normalized wind speed")
    save(joinpath(figures_dir, "ground_truth.png"), truth_fig)
    # Save the synchronized estimated-target / ground-truth deficit comparison (PNG and PDF).
    HalfDomainDiagnostics.plot_deficits(data_dir)
    HalfDomainDiagnostics.plot_rmse(data_dir)
    for strategy in strategies
        trial = load(joinpath(data_dir, "trial_$(strategy).jld2"))
        animate_strategy(strategy, trial, env, joinpath(data_dir, "animations"))
    end
    println("Comparison complete in $(round(time() - T_START_WALL; digits=2)) s: $(abspath(data_dir))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
