using Simulator
include(joinpath(@__DIR__, "..", "configs", "default.jl"))

"""Swap analytic / dataset-backed fields and fixed / adaptive planners."""
function compare_sources()
    synthetic = AnalyticField((p,t) -> p[1]/100 + p[2]/80 + t/300)
    xs, ys, ts = [0.0,100.0], [0.0,80.0], [0.0,60.0]
    sampled = GriddedField(xs, ys, ts,
        [field_value(synthetic, (x,y), t) for x in xs, y in ys, t in ts])
    [(; source, adaptive, result=run_experiment(demo_experiment(; field, adaptive); config=demo_config()))
        for (source,field) in ((:analytic,synthetic),(:gridded,sampled)) for adaptive in (false,true)]
end
if abspath(PROGRAM_FILE) == @__FILE__
    for run in compare_sources()
        println(run.source, " adaptive=", run.adaptive,
            " observations=", length(run.result.observations))
    end
end
