"""RMSE over saved query points using truth at each matching timestamp.

Evaluation is separate from planning; callers must supply an appropriate truth
field explicitly. Replay data alone do not provide full-field ground truth.
"""
function field_rmse(result::SimulationResult, truth::AbstractField)
    points = result.config.query_points
    isempty(points) && throw(ArgumentError("configure query_points to compute field RMSE"))
    [sqrt(sum((estimate.mean - field_value(truth, p, time))^2
        for (estimate,p) in zip(estimates, points)) / length(points))
        for (time,estimates) in zip(result.times, result.estimates)]
end
