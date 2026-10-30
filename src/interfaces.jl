"""Extension points use local Cartesian metres, seconds, m/s, and battery Wh.

Fields represent truth; estimators and planners receive observations/beliefs only.
See docs/architecture.md for each operation's contract.
"""
abstract type AbstractDomain end
abstract type AbstractField end
abstract type AbstractSensor end
abstract type AbstractVehicle end
abstract type AbstractEstimator end
abstract type AbstractPlanner end
abstract type AbstractController end
abstract type AbstractPowerModel end

function field_value end
function in_domain end
function constrain_motion end
function observe end
function initialize end
function predict! end
function assimilate! end
function estimate end
function plan! end
function control end
function advance end
function power_watts end

point2(p) = begin
    length(p) == 2 || throw(ArgumentError("expected a two-dimensional position"))
    q = SVector{2,Float64}(p)
    all(isfinite, q) || throw(ArgumentError("position must be finite"))
    q
end

"""Scalar observation at acquisition time, with known measurement variance."""
struct Observation
    time::Float64
    position::SVector{2,Float64}
    value::Float64
    variance::Float64
    function Observation(time, position, value; variance)
        all(isfinite, (time, value, variance)) && variance >= 0 ||
            throw(ArgumentError("observation values must be finite and variance nonnegative"))
        new(time, point2(position), value, variance)
    end
end

struct FieldEstimate
    mean::Float64
    variance::Float64
end

"""Position in metres and remaining battery energy in Wh (Inf disables capacity)."""
struct VehicleState
    position::SVector{2,Float64}
    battery_wh::Float64
    function VehicleState(position; battery_wh=Inf)
        battery_wh >= 0 || throw(ArgumentError("battery_wh must be nonnegative"))
        new(point2(position), battery_wh)
    end
end

"""Information provided to a planner. belief_at(position) returns a FieldEstimate.
No environmental field or sensor object is provided.
"""
struct PlanningContext{D,F}
    time::Float64
    state::VehicleState
    domain::D
    belief_at::F
end
