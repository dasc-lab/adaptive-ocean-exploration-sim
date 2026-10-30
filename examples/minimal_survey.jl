using Simulator
include(joinpath(@__DIR__, "..", "projects", "modular_demo", "configs", "default.jl"))

function minimal_survey()
    experiment = demo_experiment()
    result = run_experiment(experiment; config=demo_config())
    println("Collected ", length(result.observations), " observations in ",
        result.times[end] - result.times[1], " seconds")
    println("Final field RMSE: ", last(field_rmse(result, experiment.field)))
    # Replay the same observations with a fresh estimator, without a truth field.
    replay = replay_observations(experiment.estimator, result.observations)
    println("Replay final belief time: ", replay.time)
    if !isempty(ARGS)
        println("Trajectory written to ", write_trajectory_csv(ARGS[1], result))
    end
    result
end
if abspath(PROGRAM_FILE) == @__FILE__
    minimal_survey()
end
