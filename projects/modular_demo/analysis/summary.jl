using Simulator

# Analysis consumes completed results; it does not participate in control.
function summarize(result::SimulationResult; truth=nothing)
    (; observations=length(result.observations),
        duration_s=last(result.times)-first(result.times),
        remaining_battery_wh=last(result.states).battery_wh,
        final_rmse=isnothing(truth) ? missing : last(field_rmse(result, truth)))
end
