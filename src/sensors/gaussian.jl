"""Point sensor with independent Gaussian noise, in the field's value units."""
struct GaussianSensor <: AbstractSensor
    sigma::Float64
    function GaussianSensor(sigma)
        isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sensor sigma must be finite and nonnegative"))
        new(sigma)
    end
end
function observe(sensor::GaussianSensor, field::AbstractField, state::VehicleState, time, rng)
    truth = field_value(field, state.position, time)
    Observation(time, state.position, truth + sensor.sigma * randn(rng); variance=sensor.sigma^2)
end
