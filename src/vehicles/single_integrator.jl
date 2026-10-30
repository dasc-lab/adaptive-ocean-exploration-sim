"""Constant electrical demand in watts, a minimal replaceable energy model."""
struct ConstantPower <: AbstractPowerModel
    watts::Float64
    function ConstantPower(watts)
        isfinite(watts) && watts >= 0 || throw(ArgumentError("power must be finite and nonnegative"))
        new(watts)
    end
end
power_watts(model::ConstantPower, state, velocity, time) = model.watts

"""Planar velocity-driven vehicle, speed in m/s. Uses an explicit power model."""
struct SingleIntegrator{P<:AbstractPowerModel} <: AbstractVehicle
    max_speed::Float64
    power_model::P
    function SingleIntegrator(max_speed; power_model=ConstantPower(0.0))
        isfinite(max_speed) && max_speed > 0 || throw(ArgumentError("maximum speed must be finite and positive"))
        new{typeof(power_model)}(max_speed, power_model)
    end
end
function advance(vehicle::SingleIntegrator, state::VehicleState, command, domain, time, dt)
    dt > 0 && isfinite(dt) || throw(ArgumentError("dt must be finite and positive"))
    u = point2(command)
    speed = norm(u)
    speed > vehicle.max_speed && (u *= vehicle.max_speed / speed)
    position = constrain_motion(domain, state.position, dt * u)
    actual_velocity = (position - state.position) / dt
    power = power_watts(vehicle.power_model, state, actual_velocity, time)
    isfinite(power) || throw(ArgumentError("power model returned nonfinite demand"))
    energy = state.battery_wh - power * dt / 3600
    energy >= 0 || throw(ArgumentError("battery exhausted before completing integration step"))
    VehicleState(position; battery_wh=energy)
end
