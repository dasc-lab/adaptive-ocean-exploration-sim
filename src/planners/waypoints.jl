"""Visit explicit waypoints, optionally cycling. Distances/tolerance are metres."""
struct WaypointPlanner <: AbstractPlanner
    waypoints::Vector{SVector{2,Float64}}
    loop::Bool
    tolerance::Float64
    function WaypointPlanner(waypoints; loop=false, tolerance=0.01)
        points = point2.(collect(waypoints))
        isempty(points) && throw(ArgumentError("at least one waypoint is required"))
        isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("tolerance must be finite and nonnegative"))
        new(points, loop, tolerance)
    end
end
mutable struct WaypointProgress
    index::Int
end
initialize(::WaypointPlanner, time) = WaypointProgress(1)
function plan!(progress::WaypointProgress, planner::WaypointPlanner, context::PlanningContext)
    all(p -> in_domain(context.domain, p), planner.waypoints) || throw(ArgumentError("waypoints must lie in domain"))
    # Bound the loop so coincident waypoints cannot hang a run.
    for _ in eachindex(planner.waypoints)
        target = planner.waypoints[progress.index]
        norm(target - context.state.position) > planner.tolerance && return target
        if progress.index < length(planner.waypoints)
            progress.index += 1
        elseif planner.loop
            progress.index = 1
        else
            return target
        end
    end
    planner.waypoints[progress.index]
end

"""Generate alternating horizontal survey lines in a rectangular domain."""
function lawnmower_waypoints(domain::BoxDomain; spacing, margin=0.0)
    isfinite(spacing) && spacing > 0 || throw(ArgumentError("spacing must be finite and positive"))
    isfinite(margin) && margin >= 0 || throw(ArgumentError("margin must be finite and nonnegative"))
    lo, hi = domain.lower .+ margin, domain.upper .- margin
    all(lo .< hi) || throw(ArgumentError("margin leaves no survey area"))
    ys = collect(lo[2]:spacing:hi[2])
    last(ys) < hi[2] && push!(ys, hi[2])
    points = SVector{2,Float64}[]
    for (i,y) in enumerate(ys)
        ends = isodd(i) ? (lo[1],hi[1]) : (hi[1],lo[1])
        append!(points, [SVector(x,y) for x in ends])
    end
    points
end
