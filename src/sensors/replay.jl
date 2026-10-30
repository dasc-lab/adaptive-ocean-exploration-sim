"""Assimilate recorded observations without a simulated field or moving vehicle.

Input must already be chronological. Acquisition times/positions are retained;
coordinates and values must already use the estimator's units.
"""
function replay_observations(estimator::AbstractEstimator, observations; initial_time=nothing)
    readings = collect(observations)
    isempty(readings) && throw(ArgumentError("replay requires at least one observation"))
    times = getproperty.(readings, :time)
    issorted(times) || throw(ArgumentError("observations must be chronological"))
    t0 = isnothing(initial_time) ? first(times) : Float64(initial_time)
    isfinite(t0) && t0 <= first(times) || throw(ArgumentError("initial time must precede observations"))
    local_estimator = deepcopy(estimator)
    belief = initialize(local_estimator, t0)
    for reading in readings
        predict!(belief, local_estimator, reading.time)
        assimilate!(belief, local_estimator, reading)
    end
    belief
end
