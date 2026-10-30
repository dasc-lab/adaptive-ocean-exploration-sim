"""Closed axis-aligned navigable region in local Cartesian metres."""
struct BoxDomain <: AbstractDomain
    lower::SVector{2,Float64}
    upper::SVector{2,Float64}
    function BoxDomain(lower, upper)
        lo, hi = point2(lower), point2(upper)
        all(lo .< hi) || throw(ArgumentError("domain lower bounds must be below upper bounds"))
        new(lo, hi)
    end
end
in_domain(d::BoxDomain, p) = all(d.lower .<= p .<= d.upper)

# Stop along the commanded segment at the first boundary; do not slide or teleport.
function constrain_motion(d::BoxDomain, position, displacement)
    in_domain(d, position) || throw(ArgumentError("initial position is outside domain"))
    α = 1.0
    for i in 1:2
        if displacement[i] > 0
            α = min(α, (d.upper[i] - position[i]) / displacement[i])
        elseif displacement[i] < 0
            α = min(α, (d.lower[i] - position[i]) / displacement[i])
        end
    end
    clamp.(position + α * displacement, d.lower, d.upper)
end
