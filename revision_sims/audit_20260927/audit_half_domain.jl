using JLD2, StaticArrays, Statistics, Printf
base=joinpath(@__DIR__,"..","results_half_domain","20260925_130353")
e=load(joinpath(base,"environment.jld2"))
xs,ys=e["xs"],e["ys"]
vertices=[(0.01,0.01),(0.9723,-0.061),(1.4687,1.7472),(0.078,1.5507)]
function inside(x,y)
 c=[(vertices[mod1(i+1,4)][1]-vertices[i][1])*(y-vertices[i][2])-(vertices[mod1(i+1,4)][2]-vertices[i][2])*(x-vertices[i][1]) for i=1:4]
 return all(c .>= -1e-10)||all(c .<= 1e-10)
end
mask=[inside(x,y) for x in xs,y in ys]
west=mask .& [x<e["split_x"] for x in xs,y in ys]
qtrue=[inside(x,y) ? .95*exp(-.25*((x<e["split_x"] ? e["west_wind"] : e["east_wind"])-e["w_rated"])^2) : 0. for x in xs,y in ys]
println("grid cells=$(length(mask)), in polygon=$(sum(mask)), west=$(sum(west)), west/full=$(mean(west))")
println("q west=.95; q east=",.95*exp(-9))
println("strategy,scalar_deficit,snapshot_deficit,max_snapshot_scalar_error,west_q,west_qtarget,hidden_west_shortfall_fraction,west_within_100m_of_any_path_fraction,west_position_pct")
for name in ["transect","bb_ipp_nonadaptive","bb_ipp_adaptive","bb_ipp_ground_truth","ergo_nonadaptive","ergo_adaptive","ergo_ground_truth"]
 d=load(joinpath(base,"trial_"*name*".jld2"));a=d["animation"]
 deficit=[mean(max.(qtrue-q,0)) for q in a.clarity_maps]
 inds=[searchsortedlast(d["metric_times"],t) for t in a.times]
 err=maximum(abs.(deficit-d["gt_clarity_deficit"][inds]))
 westq=mean(mean(q[west]) for q in a.clarity_maps)
 westtarget=mean(mean(q[west]) for q in a.target_maps)
 hidden=mean(mean((q[west].<qtrue[west]).&(qt[west].<=q[west])) for (q,qt) in zip(a.clarity_maps,a.target_maps))
 path=[first(s) for s in d["xs"]]
 covered=mean(minimum((p[1]-xs[i])^2+(p[2]-ys[j])^2 for p in path)<=.1^2 for i in eachindex(xs),j in eachindex(ys) if west[i,j])
 @printf("%s,%.10f,%.10f,%.3g,%.6f,%.6f,%.6f,%.6f,%.4f\n",name,mean(d["gt_clarity_deficit"]),mean(deficit),err,westq,westtarget,hidden,covered,100mean(p[1]<e["split_x"] for p in path))
 if name=="ergo_adaptive"
  dt=diff(d["metric_times"])
  println("FILTER_TIMING: min=",minimum(dt)*60," s; median=",median(dt)*60," s; max=",maximum(dt)*60," s; n=",length(dt),"; integrated_prediction_seconds=",length(dt)*2.5,"; elapsed_seconds=",(last(d["metric_times"])-first(d["metric_times"]))*60)
 end
end
