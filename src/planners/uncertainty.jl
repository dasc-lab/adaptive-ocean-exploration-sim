"""Choose the candidate with greatest posterior variance; no truth access."""
struct UncertaintyPlanner <: AbstractPlanner
    candidates::Vector{SVector{2,Float64}}
    function UncertaintyPlanner(candidates)
        points = point2.(collect(candidates))
        isempty(points) && throw(ArgumentError("at least one candidate is required"))
        new(points)
    end
end
initialize(::UncertaintyPlanner, time) = nothing
function plan!(::Nothing, planner::UncertaintyPlanner, context::PlanningContext)
    candidates = filter(p -> in_domain(context.domain, p), planner.candidates)
    isempty(candidates) && throw(ArgumentError("no candidates lie inside domain"))
    candidates[argmax([context.belief_at(p).variance for p in candidates])]
end
