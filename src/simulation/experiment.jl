"""Compose one experiment. Configuration and component definitions are reusable.

run_experiment creates independent estimator/planner state for every invocation.
"""
struct Experiment{D<:AbstractDomain,F<:AbstractField,S<:AbstractSensor,V<:AbstractVehicle,E<:AbstractEstimator,P<:AbstractPlanner,C<:AbstractController}
    domain::D
    field::F
    sensor::S
    vehicle::V
    estimator::E
    planner::P
    controller::C
    initial_state::VehicleState
end
function Experiment(; domain, field, sensor, vehicle, estimator, planner, controller, initial_state)
    in_domain(domain, initial_state.position) || throw(ArgumentError("initial state must lie inside domain"))
    Experiment(domain, field, sensor, vehicle, estimator, planner, controller, initial_state)
end

function steps_for(interval, dt)
    isfinite(interval) && interval > 0 || throw(ArgumentError("interval must be finite and positive"))
    ratio = interval / dt
    isfinite(ratio) && ratio <= typemax(Int) / 2 || throw(ArgumentError("interval is too large"))
    n = round(Int, ratio)
    n >= 1 && isapprox(ratio, n; atol=1e-9, rtol=1e-10) ||
        throw(ArgumentError("interval must be an integer multiple of dt"))
    n
end

"""Timing in seconds. duration, sensor_every and plan_every must be integer multiples of dt.

Observations occur at t0 and then on sensor ticks, including the terminal tick
only if scheduled. State estimates are saved at each tick at query_points.
"""
struct SimulationConfig
    t0::Float64
    duration::Float64
    dt::Float64
    sensor_every::Float64
    plan_every::Float64
    seed::Int
    query_points::Vector{SVector{2,Float64}}
    function SimulationConfig(; t0=0.0, duration=60.0, dt=1.0,
            sensor_every=dt, plan_every=dt, seed=1, query_points=[])
        isfinite(t0) && isfinite(dt) && dt > 0 || throw(ArgumentError("time must be finite and dt positive"))
        steps_for(duration, dt)
        steps_for(sensor_every, dt)
        steps_for(plan_every, dt)
        new(t0, duration, dt, sensor_every, plan_every, seed, point2.(collect(query_points)))
    end
end

struct SimulationResult{B}
    config::SimulationConfig
    times::Vector{Float64}
    states::Vector{VehicleState}
    commands::Vector{SVector{2,Float64}}
    observations::Vector{Observation}
    estimates::Vector{Vector{FieldEstimate}}
    belief::B
end
