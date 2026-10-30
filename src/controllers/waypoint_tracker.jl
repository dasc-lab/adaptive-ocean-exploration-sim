"""Track a waypoint with a speed limit and exact one-step arrival (m/s)."""
struct WaypointTracker <: AbstractController
    speed::Float64
    function WaypointTracker(speed)
        isfinite(speed) && speed > 0 || throw(ArgumentError("tracking speed must be finite and positive"))
        new(speed)
    end
end
function control(controller::WaypointTracker, state::VehicleState, target, time, dt)
    delta = point2(target) - state.position
    distance = norm(delta)
    distance == 0 && return zero(delta)
    min(controller.speed, distance / dt) * delta / distance
end
