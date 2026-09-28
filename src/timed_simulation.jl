# Included inside SimulatorST. This engine is shared by the revision-simulation
# adaptive and transect runners; legacy experiment entry points remain separate.
const MOTION_MODEL_VERSION = "shared-hard-boundary-instant-heading-v1"
const FILTER_TIMING_VERSION = "timestamped-sequential-v1"

"""Convert a cadence to integer simulation steps, avoiding floating-point drift."""
function cadence_steps(interval, dt; allow_substep=false)
    isfinite(interval) && interval > 0 || throw(ArgumentError("cadence must be finite and positive"))
    allow_substep && interval <= dt && return 1
    ratio = interval / dt
    n = round(Int, ratio)
    n >= 1 && isapprox(ratio, n; atol=1e-9, rtol=1e-9) ||
        throw(ArgumentError("cadence must be an integer multiple of the sampling timestep"))
    return n
end

"""Limit a complete integration step to the convex mission polygon and grid.
Velocity is scaled down, never rotated or renormalized after the safety limit.
"""
function feasible_velocity(x, u, dt_hours, polygon, grid_xs, grid_ys)
    all(isfinite, u) || throw(ArgumentError("controller returned non-finite velocity"))
    vertices = polygon.vertices
    center = vec(sum(vertices; dims=2)) / size(vertices, 2)
    displacement = u * (3.6 * dt_hours) # m/s -> km/h
    α = 1.0
    for i in axes(vertices, 2)
        a, b = vertices[:, i], vertices[:, mod1(i+1, size(vertices, 2))]
        edge = b - a
        normal = SVector(-edge[2], edge[1])
        dot(normal, center - a) < 0 && (normal = -normal)
        slack = dot(normal, x - a)
        slack >= -1e-10 || throw(ArgumentError("vehicle lies outside motion polygon"))
        travel = dot(normal, displacement)
        travel < 0 && (α = min(α, max(0.0, slack) / -travel))
    end
    for k in 1:2
        lo, hi = k == 1 ? extrema(grid_xs) : extrema(grid_ys)
        lo - 1e-10 <= x[k] <= hi + 1e-10 || throw(ArgumentError("vehicle lies outside estimator grid"))
        if displacement[k] > 0
            α = min(α, max(0.0, hi - x[k]) / displacement[k])
        elseif displacement[k] < 0
            α = min(α, max(0.0, x[k] - lo) / -displacement[k])
        end
    end
    return u * clamp(α, 0.0, 1.0)
end

"""Run a mission with measurements at every ts, including both endpoints.

A fusion event processes pending samples sequentially at their acquisition times:
PREDICT one model step, then CORRECT with all vehicles sampled at that time.
The first sample is corrected at t0 without prediction. The terminal event always
flushes the buffer. Between fusion events the controller holds the last published
posterior. Target maps have explicit controller timestamps; commands and energy
updates have exactly one entry per integration interval.
"""
function simulate_timed_mission(ts, x0, b0, controller, soc_profile,
        w_rated_val, convex_polygon, stgp_problem;
        ngpkf_grid, EnvData, σ_meas=0, σ_process=0,
        Q_process=σ_process^2 * I,
        fuse_measurements_every_ΔT=5.0/60,
        recompute_controller_every_ΔT=fuse_measurements_every_ΔT,
        motion_polygon=convex_polygon, transect_pts=nothing,
        solar_day=SoCController.dayOfYear, solar_latitude=SoCController.lat)
    times = collect(ts)
    length(times) >= 2 || throw(ArgumentError("at least two timestamps required"))
    dt = times[2] - times[1]
    dt > 0 && all(d -> isapprox(d, dt; atol=1e-10, rtol=1e-9), diff(times)) ||
        throw(ArgumentError("timestamps must increase uniformly"))
    isapprox(stgp_problem.ss_model.dt, dt; atol=1e-10, rtol=1e-9) ||
        throw(ArgumentError("STGPKF prediction timestep does not match simulation timestep"))
    isfinite(σ_meas) && σ_meas >= 0 || throw(ArgumentError("invalid measurement noise"))
    # STGPKF already obtains process covariance from its temporal kernel.
    # Historically this extra argument was silently ignored; reject nonzero input.
    (Q_process isa UniformScaling ? iszero(Q_process.λ) : all(iszero, Q_process)) ||
        throw(ArgumentError("extra Q_process is unsupported; configure the STGPKF temporal kernel"))
    length(soc_profile) == length(times) || throw(DimensionMismatch("SOC profile and time grid differ"))
    length(x0) == 1 || throw(ArgumentError("shared-SOC mission currently supports one vehicle"))
    fuse_steps = cadence_steps(fuse_measurements_every_ΔT, dt)
    control_steps = cadence_steps(recompute_controller_every_ΔT, dt; allow_substep=true)
    ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))
    state = stgpkf_initialize(stgp_problem)
    xs = [copy(x0)]
    bs = [Float64(b0)]
    us = Vector{typeof(copy(x0))}()
    speeds = Float64[]
    measurements = Float64[]
    measurement_ts = Float64[]
    measurement_positions = typeof(first(x0))[]
    w_hat_ts = Float64[]
    w_hats = Matrix{Float64}[]
    ergo_q_maps = Matrix{Float64}[]
    q_target_ts = Float64[]
    q_target_maps = Matrix{Float64}[]
    # Single vehicle is enforced above; every measurement has one matching position.
    last_fused = 0
    prediction_steps = 0
    waypoint_idx = 1
    error_sum, pid_error = 0.0, 0.0
    boat = SoCController.ASV_Params()
    u = [zero(first(x0))]
    for x in x0
        feasible_velocity(x, zero(x), dt/60, motion_polygon, EnvData.xs, EnvData.ys)
    end
    @progress for (it, t) in enumerate(times)
        x = first(xs[end])
        y = measure(EnvData, x[1], x[2], t, σ_meas)
        isfinite(y) || throw(ArgumentError("non-finite measurement at t=$t"))
        push!(measurements, y)
        push!(measurement_ts, t)
        push!(measurement_positions, x)
        if it == 1 || mod(it-1, fuse_steps) == 0 || it == length(times)
            for j in (last_fused+1):it
                if j > 1
                    state = stgpkf_predict(stgp_problem, state)
                    prediction_steps += 1
                end
                state = stgpkf_correct(stgp_problem, state,
                    [measurement_positions[j]], [measurements[j]], (σ_meas^2) * I(1))
            end
            last_fused = it
            μ = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
            q = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
            all(isfinite, μ) && all(v -> isfinite(v) && 0 <= v <= 1, q) ||
                error("invalid filter output at t=$t")
            push!(w_hat_ts, t)
            push!(w_hats, reshape(μ, length(EnvData.xs), length(EnvData.ys)))
            push!(ergo_q_maps, ngpkf_to_ergo(ngpkf_grid, ergo_grid,
                reshape(q, length(EnvData.xs), length(EnvData.ys))))
        end
        # The terminal controller evaluation records a contemporaneous target,
        # but does not integrate motion or consume an extra battery step.
        if it < length(times)
            speed, error_sum, pid_error = SoCController.speed_controller(
                bs[end], soc_profile[it], error_sum, pid_error)
        else
            speed = last(speeds)
        end
        if it == 1 || mod(it-1, control_steps) == 0 || it == length(times)
            kwargs = (; ngpkf_grid, ergo_grid, ergo_q_map=ergo_q_maps[end],
                traj=vcat(xs...), umax=speed, ΔT=dt)
            if transect_pts === nothing
                u, target = controller(t, xs[end], w_hats[end], w_rated_val, convex_polygon; kwargs...)
            else
                u, target, waypoint_idx = controller(t, xs[end], w_hats[end], w_rated_val,
                    convex_polygon; kwargs..., transect_pts, waypoint_idx)
            end
            size(target) == size(w_hats[end]) || throw(DimensionMismatch("target grid differs"))
            all(v -> isfinite(v) && 0 <= v <= 1, target) || error("invalid target clarity")
            push!(q_target_ts, t)
            push!(q_target_maps, copy(target))
        end
        it == length(times) && break
        length(u) == length(x0) || throw(DimensionMismatch("controller output count differs"))
        # Respect speed limits without undoing boundary/waypoint speed reductions.
        raw = SVector{2,Float64}(first(u))
        norm(raw) > speed && (raw *= speed / norm(raw))
        safe = feasible_velocity(x, raw, dt/60, motion_polygon, EnvData.xs, EnvData.ys)
        applied = [safe]
        push!(us, applied)
        push!(speeds, norm(safe))
        push!(xs, step(t, xs[end], applied, dt * 60 / 1000))
        push!(bs, SoCController.batterymodel!(boat, solar_day, t/60, solar_latitude,
            norm(safe), bs[end], dt/60))
    end
    @assert prediction_steps == length(times)-1
    @assert last_fused == length(measurements) == length(measurement_positions)
    return (; ts=times, xs, us, speeds, bs, measurements, measurement_ts,
        measurement_positions, w_hat_ts, w_hats, ergo_q_maps, q_target_maps,
        q_target_ts, prediction_steps, filter_timing_version=FILTER_TIMING_VERSION,
        motion_model_version=MOTION_MODEL_VERSION, filter_state=state, solar_day, solar_latitude)
end
