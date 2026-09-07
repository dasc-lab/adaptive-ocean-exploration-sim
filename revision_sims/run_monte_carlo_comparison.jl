#!/usr/bin/env julia
# =============================================================================
# run_monte_carlo_comparison.jl
# =============================================================================

using Distributed, Dates, Printf, CairoMakie

const SCRIPT_START_TIME = Dates.now()
const T_START_WALL = time()

# ---- Arg Parsing -----------------------------------------------------------
const N_STRATEGIES_DEFAULT = 4

function parse_args(args)
    opts = Dict{String,String}(
        "nworkers"   => string(N_STRATEGIES_DEFAULT),
        "num_mc"     => "5",
        "base_seed"  => "1234",
        "w_rated"    => "-3.5",
        "ls"         => "0.75",
        "lt"         => "45.0",
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
num_mc = parse(Int, opts["num_mc"])
base_seed = parse(Int, opts["base_seed"])
seeds = base_seed:(base_seed + num_mc - 1)
SCRIPT_SRC_DIR = abspath(opts["srcdir"])

if nprocs() == 1
    total_tasks = length(seeds) * length(strategies)
    addprocs(min(nworkers_requested, total_tasks); exename=joinpath(Sys.BINDIR, "julia"))
end

@everywhere SRC_DIR = $SCRIPT_SRC_DIR

# =============================================================================
# Setup Modules & Environment Code Across ALL Workers
# =============================================================================
@everywhere begin
    using LinearAlgebra, StaticArrays, Interpolations, Statistics, Random
    using SpatiotemporalGPs, JLD2, ForwardDiff

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
# Environment & Controller Definitions
# =============================================================================
@everywhere function build_environment(;seed=1234, w_rated_val=-3.5, ls_val=0.75, lt_val=45.0)
    Random.seed!(seed)

    Δt      = 2.5
    dt_min  = Δt / 60
    dt_hrs  = Δt / 3600
    T_begin = 9.0
    T_end   = 12.0
    ts_hrs  = T_begin:dt_hrs:T_end
    ts_min  = T_begin*60:dt_min:T_end*60

    σt, σs = 1.0, 1.0
    lt = lt_val
    ls = ls_val
    kt = Matern(1/2, σt, lt)
    ks = Matern(1/2, σs, ls)

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

    return (; Δt, dt_min, dt_hrs, T_begin, T_end, ts_hrs, ts_min,
            ks, kt, xs, ys, grid_pts, synthetic_data, problem, ngpkf_grid,
            x0s, target_q_mat, soc_begin, soc_end, soc_target,
            transect_pts, fuse_measurements_every_ΔT, recompute_controller_every_ΔT,
            w_rated_val, σ_meas, σ_t, convex_polygon = JordanLakeDomain.convex_polygon)
end

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

    function run_transect(env, outpath)
        controller = make_transect_controller(env)
        res = SimulatorST.simulate_known_transect(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data,
            transect_pts=env.transect_pts, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        jldsave(outpath; res, strategy="transect")
        return res
    end

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

    function run_ergo_nonadaptive(env, outpath)
        controller = make_nonadaptive_ergo_controller(env)
        res = SimulatorST.simulate_known_param(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        jldsave(outpath; res, strategy="ergo_nonadaptive")
        return res
    end

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

    function run_ergo_adaptive(env, outpath)
        controller = make_adaptive_ergo_controller(env)
        res = SimulatorST.simulate_known_param(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        jldsave(outpath; res, strategy="ergo_adaptive")
        return res
    end

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

    function make_bb_ipp_controller(env; H=5, M_primitives=7, primitive_stride=20)
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

    function run_bb_ipp(env, outpath)
        controller = make_bb_ipp_controller(env)
        res = SimulatorST.simulate_known_param(
            env.ts_min, env.x0s, env.soc_begin, controller, env.soc_target,
            env.w_rated_val, env.convex_polygon, env.problem;
            ngpkf_grid=env.ngpkf_grid, EnvData=env.synthetic_data, σ_meas=env.σ_meas,
            Q_process=diagm(vec(env.σ_t .^ 2 .* env.fuse_measurements_every_ΔT)),
            fuse_measurements_every_ΔT=env.fuse_measurements_every_ΔT,
            recompute_controller_every_ΔT=env.recompute_controller_every_ΔT)
        jldsave(outpath; res, strategy="bb_ipp")
        return res
    end
end

@everywhere const STRATEGY_FNS = Dict(
    :transect          => run_transect,
    :ergo_nonadaptive  => run_ergo_nonadaptive,
    :ergo_adaptive     => run_ergo_adaptive,
    :bb_ipp            => run_bb_ipp,
)

w_rated_cmd = parse(Float64, opts["w_rated"])
ls_cmd = parse(Float64, opts["ls"])
lt_cmd = parse(Float64, opts["lt"])

# =============================================================================
# Helper Function to Compute Spatial RMSE & Clarity Deficit Over Time
# =============================================================================
@everywhere function compute_run_metrics(res, synthetic_data, w_rated_val)
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
    for i in 1:N_q
        deficit_map = max.(0.0, q_target_maps[i] .- ergo_q_maps[i])
        clarity_deficit_series[i] = mean(deficit_map)
    end

    return rmse_global_series, clarity_deficit_series
end

# UNPACK ls_val and lt_val explicitly to prevent scope errors on worker nodes
@everywhere function run_task(task_tuple)
    seed, strategy_name, outdir, w_rated_val, ls_val, lt_val = task_tuple
    outpath = joinpath(outdir, "trial_seed$(seed)_$(strategy_name).jld2")

    # Use locally unpacked ls_val and lt_val instead of global ls_cmd / lt_cmd
    env = build_environment(; seed=seed, w_rated_val=w_rated_val, ls_val=ls_val, lt_val=lt_val)

    if isfile(outpath)
        res = load(outpath, "res")
    else
        fn = STRATEGY_FNS[strategy_name]
        res = fn(env, outpath)
    end

    rmse_global, clarity_deficit = compute_run_metrics(
        res, env.synthetic_data, w_rated_val
    )

    return (seed, strategy_name, vec(res.measurements), rmse_global, clarity_deficit)
end

# =============================================================================
# Main Driver & Post-Processing
# =============================================================================
function main()

    datetime_str = Dates.format(SCRIPT_START_TIME, "yyyymmdd_HHMMSS")
    data_dir = joinpath(opts["outdir"], datetime_str)
    mkpath(data_dir)

    println("="^80)
    println("Monte Carlo Comparison Routine")
    println("Start Timestamp:  $(Dates.format(SCRIPT_START_TIME, "yyyy-mm-dd HH:MM:SS"))")
    println("Output Directory: $(abspath(data_dir))")
    println("MC Seeds:         $(seeds)")
    println("Rated Wind Speed: $(w_rated_cmd) m/s")
    println("Spatial Ls:       $(ls_cmd)")
    println("Temporal Lt:      $(lt_cmd)")
    println("Strategies:       $(strategies)")
    println("Active Workers:   $(workers())")
    println("="^80)

    # Pass ls_cmd and lt_cmd explicitly into the worker tasks tuple
    tasks = [(seed, strat, data_dir, w_rated_cmd, ls_cmd, lt_cmd) for seed in seeds for strat in strategies]
    results = pmap(run_task, tasks)

    measurements_dict    = Dict(s => Float64[] for s in strategies)
    rmse_global_dict     = Dict(s => Vector{Vector{Float64}}() for s in strategies)
    clarity_deficit_dict = Dict(s => Vector{Vector{Float64}}() for s in strategies)

    for (seed, strat, meas, rmse_g, deficit) in results
        append!(measurements_dict[strat], meas)
        push!(rmse_global_dict[strat], rmse_g)
        push!(clarity_deficit_dict[strat], deficit)
    end

    sample_env = build_environment(; seed=base_seed, w_rated_val=w_rated_cmd, ls_val=ls_cmd, lt_val=lt_cmd)
    w_rated_val = sample_env.w_rated_val

    allowable_buffer = 1.0
    N_mc = length(seeds)

    strategy_names_str = String[]
    
    m_rmse_g_vals   = Float64[]
    sem_rmse_g_vals = Float64[]
    m_deficit_vals  = Float64[]
    f_deficit_vals  = Float64[]
    m_error_vals    = Float64[]
    std_error_vals  = Float64[]
    in_buffer_props = Float64[]

    med_rmse_g_vals = Float64[]
    q25_rmse_g_vals = Float64[]
    q75_rmse_g_vals = Float64[]
    
    med_fdef_vals   = Float64[]
    q25_fdef_vals   = Float64[]
    q75_fdef_vals   = Float64[]

    println("\n" * "="^80)
    println("SUMMARY METRICS Across Monte Carlo Trials (N = $N_mc Seeds)")
    println("="^80)

    for strat in strategies
        seed_means_global = [mean(s) for s in rmse_global_dict[strat]]
        m_rmse_g = mean(seed_means_global)
        sem_rmse_g = N_mc > 1 ? std(seed_means_global) / sqrt(N_mc) : 0.0

        med_rmse_g = median(seed_means_global)
        q25_rmse_g = quantile(seed_means_global, 0.25)
        q75_rmse_g = quantile(seed_means_global, 0.75)

        deficit_matrix = hcat(clarity_deficit_dict[strat]...)
        mean_deficit_series = vec(mean(deficit_matrix, dims=2))
        m_deficit = mean(mean_deficit_series)
        
        seed_final_deficits = [s[end] for s in clarity_deficit_dict[strat]]
        f_deficit = mean(seed_final_deficits)
        med_fdef  = median(seed_final_deficits)
        q25_fdef  = quantile(seed_final_deficits, 0.25)
        q75_fdef  = quantile(seed_final_deficits, 0.75)

        meas = measurements_dict[strat]
        errs = meas .- w_rated_val
        m_err = mean(errs)
        s_err = std(errs)
        prop_in_range = count(abs.(errs) .<= allowable_buffer) / length(errs)

        push!(strategy_names_str, string(strat))
        push!(m_rmse_g_vals, m_rmse_g)
        push!(sem_rmse_g_vals, sem_rmse_g)
        push!(med_rmse_g_vals, med_rmse_g)
        push!(q25_rmse_g_vals, q25_rmse_g)
        push!(q75_rmse_g_vals, q75_rmse_g)

        push!(m_deficit_vals, m_deficit)
        push!(f_deficit_vals, f_deficit)
        push!(med_fdef_vals, med_fdef)
        push!(q25_fdef_vals, q25_fdef)
        push!(q75_fdef_vals, q75_fdef)

        push!(m_error_vals, m_err)
        push!(std_error_vals, s_err)
        push!(in_buffer_props, prop_in_range)

        println("Strategy: $(strat)")
        @printf("  - Spatial RMSE (Mean ± SEM):       %.4f ± %.4f\n", m_rmse_g, sem_rmse_g)
        @printf("  - Spatial RMSE (Median [Q1, Q3]):  %.4f [%.4f, %.4f]\n", med_rmse_g, q25_rmse_g, q75_rmse_g)
        @printf("  - Mean Clarity Deficit:            %.4f\n", m_deficit)
        @printf("  - Final Deficit (Median [Q1, Q3]): %.4f [%.4f, %.4f]\n", med_fdef, q25_fdef, q75_fdef)
        @printf("  - Measurement Error Mean:          %.4f\n", m_err)
        @printf("  - Measurement Error Std:           %.4f\n", s_err)
        @printf("  - Target Dwell (±%.1f):             %.2f%%\n\n", allowable_buffer, prop_in_range * 100)
    end

    # =========================================================================
    # VISUALIZATION 1: Per-Seed Metric Distributions (Violin + Jitter Points)
    # =========================================================================
    fig_dist = Figure(size = (1100, 500))
    
    ax_rmse_v = Axis(fig_dist[1, 1],
        title = "Spatial RMSE Distribution Across Seeds",
        xticks = (1:length(strategies), strategy_names_str),
        ylabel = "Mean Spatial RMSE"
    )
    ax_def_v = Axis(fig_dist[1, 2],
        title = "Final Clarity Deficit Distribution Across Seeds",
        xticks = (1:length(strategies), strategy_names_str),
        ylabel = "Final Clarity Deficit"
    )

    colors = [:dodgerblue, :orange, :crimson, :forestgreen]

    for (k, strat) in enumerate(strategies)
        rmse_vals = [mean(s) for s in rmse_global_dict[strat]]
        cats = fill(k, length(rmse_vals))
        violin!(ax_rmse_v, cats, rmse_vals; color = (colors[k], 0.35), strokecolor = colors[k], strokewidth = 1.5)
        
        jitter = (rand(length(rmse_vals)) .- 0.5) .* 0.18
        scatter!(ax_rmse_v, cats .+ jitter, rmse_vals; color = colors[k], markersize = 8, strokewidth = 0.5, strokecolor = :black)

        def_vals = [s[end] for s in clarity_deficit_dict[strat]]
        violin!(ax_def_v, cats, def_vals; color = (colors[k], 0.35), strokecolor = colors[k], strokewidth = 1.5)
        scatter!(ax_def_v, cats .+ jitter, def_vals; color = colors[k], markersize = 8, strokewidth = 0.5, strokecolor = :black)
    end

    Label(fig_dist[0, 1:2], "Monte Carlo Seed Variability Analysis", fontsize = 18, font = :bold)
    
    output_dist_fig = joinpath(data_dir, "mc_metric_distributions.png")
    save(output_dist_fig, fig_dist)
    println("Seed distribution plot successfully saved to: $output_dist_fig")

    # =========================================================================
    # VISUALIZATION 2: Time-Series Ribbons (Median ± 25th-75th Percentiles)
    # =========================================================================
    fig_ribbon = Figure(size = (1100, 500))

    ax_rmse_t = Axis(fig_ribbon[1, 1], title = "Global Spatial RMSE Trajectory (Median ± IQR)", xlabel = "Time Step", ylabel = "RMSE")
    ax_def_t  = Axis(fig_ribbon[1, 2], title = "Clarity Deficit Trajectory (Median ± IQR)", xlabel = "Time Step", ylabel = "Clarity Deficit")

    for (k, strat) in enumerate(strategies)
        rmse_mat = hcat(rmse_global_dict[strat]...)
        N_steps = size(rmse_mat, 1)
        steps = 1:N_steps

        med_rmse_t = [median(rmse_mat[i, :]) for i in 1:N_steps]
        q25_rmse_t = [quantile(rmse_mat[i, :], 0.25) for i in 1:N_steps]
        q75_rmse_t = [quantile(rmse_mat[i, :], 0.75) for i in 1:N_steps]

        band!(ax_rmse_t, steps, q25_rmse_t, q75_rmse_t; color = (colors[k], 0.25))
        lines!(ax_rmse_t, steps, med_rmse_t; color = colors[k], linewidth = 2, label = strategy_names_str[k])

        def_mat = hcat(clarity_deficit_dict[strat]...)
        N_q = size(def_mat, 1)
        steps_q = 1:N_q

        med_def_t = [median(def_mat[i, :]) for i in 1:N_q]
        q25_def_t = [quantile(def_mat[i, :], 0.25) for i in 1:N_q]
        q75_def_t = [quantile(def_mat[i, :], 0.75) for i in 1:N_q]

        band!(ax_def_t, steps_q, q25_def_t, q75_def_t; color = (colors[k], 0.25))
        lines!(ax_def_t, steps_q, med_def_t; color = colors[k], linewidth = 2, label = strategy_names_str[k])
    end

    axislegend(ax_rmse_t, position = :rt, framevisible = true)
    axislegend(ax_def_t, position = :rt, framevisible = true)
    Label(fig_ribbon[0, 1:2], "Time-Series Convergence & Dispersion across Seeds", fontsize = 18, font = :bold)

    output_ribbon_fig = joinpath(data_dir, "mc_time_series_ribbons.png")
    save(output_ribbon_fig, fig_ribbon)
    println("Time-series ribbon plot successfully saved to: $output_ribbon_fig")

    # =========================================================================
    # VISUALIZATION 3: Measurement Histograms Across MC Runs
    # =========================================================================
    fig_hist = Figure(size = (1000, 900))

    axs = [
        Axis(fig_hist[1, 1], title = "Transect",              xlabel = "Error", ylabel = "Frequency", limits = ((-5, 5), nothing)),
        Axis(fig_hist[1, 2], title = "Non-adaptive Ergodic", xlabel = "Error", ylabel = "Frequency", limits = ((-5, 5), nothing)),
        Axis(fig_hist[2, 1], title = "BB-IPP",               xlabel = "Error", ylabel = "Frequency", limits = ((-5, 5), nothing)),
        Axis(fig_hist[2, 2], title = "Adaptive Ergodic",     xlabel = "Error", ylabel = "Frequency", limits = ((-5, 5), nothing))
    ]

    hist_data = [
        (measurements_dict[:transect] .- w_rated_val,      "Transect"),
        (measurements_dict[:ergo_nonadaptive] .- w_rated_val, "Non-adaptive Ergodic"),
        (measurements_dict[:bb_ipp] .- w_rated_val,            "BB-IPP"),
        (measurements_dict[:ergo_adaptive] .- w_rated_val,    "Adaptive Ergodic")
    ]

    for (ax, (errs, name)) in zip(axs, hist_data)
        m_e, s_e = mean(errs), std(errs)
        h = hist!(ax, errs, 
                  bins = 50, 
                  color = (:skyblue, 0.8), 
                  strokewidth = 0.5, 
                  strokecolor = :white)
        
        lbl = "$(name)\nMean: $(round(m_e, digits=2)), Std: $(round(s_e, digits=2))\nMedian: $(round(median(errs), digits=2))"
        axislegend(ax, [h], [lbl], position = :rt, framevisible = true, backgroundcolor = (:white, 0.8))
    end

    Label(fig_hist[0, 1:2], "Measurement Histograms Across Monte Carlo Runs", fontsize = 18, font = :bold)

    output_fig_path = joinpath(data_dir, "measurement_histograms.png")
    save(output_fig_path, fig_hist)
    println("Histogram plot successfully saved to:  $output_fig_path")

    # =========================================================================
    # REPORTS: Text & CSV Summaries
    # =========================================================================
    wall_runtime_sec = time() - T_START_WALL
    txt_report_path = joinpath(data_dir, "summary_metrics.txt")
    open(txt_report_path, "w") do f
        println(f, "="^80)
        println(f, "MONTE CARLO SIMULATION SUMMARY REPORT")
        println(f, "="^80)
        println(f, "Execution Timestamp: ", Dates.format(SCRIPT_START_TIME, "yyyy-mm-dd HH:MM:SS"))
        println(f, "Total Wall Runtime:  ", round(wall_runtime_sec, digits=2), " seconds")
        println(f, "Number of MC Seeds:  ", num_mc, " (Base Seed: ", base_seed, ")")
        println(f, "Target Reference Value: ", w_rated_val)
        println(f, "="^80)
        println(f, "")
        for i in 1:length(strategies)
            println(f, "Strategy: ", strategy_names_str[i])
            @printf(f, "  - Global Spatial RMSE (Mean ± SEM): %.6f ± %.6f\n", m_rmse_g_vals[i], sem_rmse_g_vals[i])
            @printf(f, "  - Global Spatial RMSE (Median [IQR]): %.6f [%.6f, %.6f]\n", med_rmse_g_vals[i], q25_rmse_g_vals[i], q75_rmse_g_vals[i])
            @printf(f, "  - Mean Clarity Deficit:            %.6f\n", m_deficit_vals[i])
            @printf(f, "  - Final Clarity Deficit (Mean):    %.6f\n", f_deficit_vals[i])
            @printf(f, "  - Final Clarity Deficit (Median):  %.6f [%.6f, %.6f]\n", med_fdef_vals[i], q25_fdef_vals[i], q75_fdef_vals[i])
            @printf(f, "  - Error Mean:                      %.6f\n", m_error_vals[i])
            @printf(f, "  - Error Std Dev:                   %.6f\n", std_error_vals[i])
            @printf(f, "  - Proportion In Range (±%.1f):      %.4f (%.2f%%)\n\n", allowable_buffer, in_buffer_props[i], in_buffer_props[i] * 100)
        end
    end
    println("Summary report text saved to:         $txt_report_path")

    csv_report_path = joinpath(data_dir, "summary_metrics.csv")
    open(csv_report_path, "w") do f
        println(f, "Strategy,Global_RMSE_Mean,Global_RMSE_SEM,Global_RMSE_Median,Global_RMSE_Q25,Global_RMSE_Q75,Mean_Clarity_Deficit,Final_Deficit_Mean,Final_Deficit_Median,Final_Deficit_Q25,Final_Deficit_Q75,Error_Mean,Error_Std,Proportion_In_Target_Range")
        for i in 1:length(strategies)
            @printf(f, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                strategy_names_str[i],
                m_rmse_g_vals[i], sem_rmse_g_vals[i],
                med_rmse_g_vals[i], q25_rmse_g_vals[i], q75_rmse_g_vals[i],
                m_deficit_vals[i],
                f_deficit_vals[i],
                med_fdef_vals[i], q25_fdef_vals[i], q75_fdef_vals[i],
                m_error_vals[i],
                std_error_vals[i],
                in_buffer_props[i]
            )
        end
    end
    println("Summary metrics CSV saved to:         $csv_report_path")
    println("="^80)
end

main()