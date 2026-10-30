"""Adapter over the existing numerical GP / square-root Kalman filter.

A scalar spatial GP on a fixed grid. process_variance_rate adds independent grid
variance per second (random-walk dynamics). It is not a temporal-kernel STGP.
Every run owns a fresh GPBelief. Kernel parameters use metres and field units.
"""
struct GridGP{G} <: AbstractEstimator
    grid::G
    process_variance_rate::Float64
end
function GridGP(xs, ys; sigma=1.0, length_scale=100.0, process_variance_rate=0.0)
    all(isfinite, (sigma, length_scale, process_variance_rate)) &&
        sigma > 0 && length_scale > 0 && process_variance_rate >= 0 ||
        throw(ArgumentError("GP scales must be positive and process variance rate nonnegative"))
    axes = map(x -> Float64.(collect(x)), (xs, ys))
    for axis in axes
        !isempty(axis) && all(isfinite, axis) && all(diff(axis) .> 0) ||
            throw(ArgumentError("GP grid axes must be finite and strictly increasing"))
    end
    GridGP(NGPKF.NGPKFGrid(axes..., NGPKF.MaternKernel(Float64(sigma), Float64(length_scale))), Float64(process_variance_rate))
end
mutable struct GPBelief{S}
    state::S
    time::Float64
end
initialize(estimator::GridGP, time) = GPBelief(NGPKF.initialize(estimator.grid), Float64(time))
function predict!(belief::GPBelief, estimator::GridGP, time)
    isfinite(time) && time >= belief.time || throw(ArgumentError("prediction time must be finite and nondecreasing"))
    dt = time - belief.time
    if dt > 0 && estimator.process_variance_rate > 0
        belief.state = KF.predict(belief.state, I, estimator.process_variance_rate * dt * I)
    end
    belief.time = time
    belief
end
function assimilate!(belief::GPBelief, estimator::GridGP, reading::Observation)
    reading.time == belief.time || throw(ArgumentError("predict belief to acquisition time before assimilation"))
    reading.variance > 0 || throw(ArgumentError("GridGP requires positive observation variance"))
    belief.state = NGPKF.correct(estimator.grid, belief.state,
        [reading.position], [reading.value]; R_measure=reshape([reading.variance], 1, 1))
    belief
end
function estimate(estimator::GridGP, belief::GPBelief, position, time=belief.time)
    time == belief.time || throw(ArgumentError("query time must match belief time; call predict! first"))
    grid = estimator.grid
    p = point2(position)
    k = [grid.kernel(p, q) for q in grid.grid_pts]
    weights = grid.invK * k
    mean = dot(weights, KF.μ(belief.state))
    # Conditional GP interpolation plus uncertainty retained in the grid posterior.
    variance = grid.kernel(p, p) - dot(k, weights) + sum(abs2, belief.state.F * weights)
    FieldEstimate(mean, max(0.0, variance))
end
