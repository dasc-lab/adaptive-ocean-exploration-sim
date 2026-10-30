"""Advance belief → observe/assimilate → plan → control → move on integer ticks.

Truth is supplied only to the sensor. Planners receive a PlanningContext whose
belief_at query exposes estimated means/variances. No global random seed is set.
"""
function run_experiment(experiment::Experiment; config=SimulationConfig())
    rng = MersenneTwister(config.seed)
    estimator = deepcopy(experiment.estimator)
    planner = deepcopy(experiment.planner)
    controller = deepcopy(experiment.controller)
    vehicle = deepcopy(experiment.vehicle)
    sensor = deepcopy(experiment.sensor)
    belief = initialize(estimator, config.t0)
    progress = initialize(planner, config.t0)
    n = steps_for(config.duration, config.dt)
    sample_steps = steps_for(config.sensor_every, config.dt)
    plan_steps = steps_for(config.plan_every, config.dt)
    times = [config.t0 + i*config.dt for i in 0:n]
    all(diff(times) .> 0) || throw(ArgumentError("time step is below floating-point resolution at t0"))
    states = VehicleState[experiment.initial_state]
    commands = SVector{2,Float64}[]
    observations = Observation[]
    estimates = Vector{FieldEstimate}[]
    target = nothing
    for tick in 0:n
        time = times[tick+1]
        state = states[end]
        predict!(belief, estimator, time)
        if tick % sample_steps == 0
            reading = observe(sensor, experiment.field, state, time, rng)
            push!(observations, reading)
            assimilate!(belief, estimator, reading)
        end
        push!(estimates, [estimate(estimator, belief, p, time) for p in config.query_points])
        tick == n && break
        if tick % plan_steps == 0
            context = PlanningContext(time, state, experiment.domain,
                p -> estimate(estimator, belief, p, time))
            target = plan!(progress, planner, context)
        end
        command = point2(control(controller, state, target, time, config.dt))
        push!(commands, command)
        push!(states, advance(vehicle, state, command, experiment.domain, time, config.dt))
    end
    SimulationResult(deepcopy(config), times, states, commands, observations, estimates, belief)
end
