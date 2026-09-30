#!/usr/bin/env julia

using CairoMakie, JLD2, Printf, Statistics

length(ARGS) in (1, 2) || error(
    "Usage: julia --project=. revision_sims/aggregate_lambda_cd_sweep.jl SWEEP_ROOT [OUTPUT_DIR]")
root = abspath(ARGS[1])
outdir = length(ARGS) == 2 ? abspath(ARGS[2]) : joinpath(root, "aggregate")
isdir(root) || error("Sweep root does not exist: $root")
mkpath(outdir)

summary_files = String[]
for (dir, _, files) in walkdir(root)
    "summary_metrics.csv" in files && push!(summary_files, joinpath(dir, "summary_metrics.csv"))
end
isempty(summary_files) && error("No summary_metrics.csv files found below $root")

rows = NamedTuple[]
for path in sort(summary_files)
    lines = readlines(path)
    length(lines) >= 2 || continue
    header = split(first(lines), ',')
    for line in Iterators.drop(lines, 1)
        values = split(line, ',')
        length(values) == length(header) || error("Malformed CSV row in $path")
        raw = Dict(header .=> values)
        push!(rows, (;
            environment=raw["environment"],
            lambda_cd=parse(Float64, raw["lambda_cd"]),
            strategy=raw["strategy"],
            rmse=parse(Float64, raw["rmse"]),
            gt_deficit=parse(Float64, raw["gt_deficit"]),
            in_target=parse(Float64, raw["in_target"]),
            source=path))
    end
end
sort!(rows; by=r -> (r.environment, r.lambda_cd, r.strategy))

open(joinpath(outdir, "lambda_cd_sweep.csv"), "w") do io
    println(io, "environment,lambda_cd,strategy,rmse,gt_deficit,in_target,source")
    for r in rows
        println(io, join((r.environment, r.lambda_cd, r.strategy, r.rmse,
            r.gt_deficit, r.in_target, r.source), ','))
    end
end

environments = ["half_domain", "moving_pocket"]
metrics = [(:rmse, "Global wind RMSE"),
           (:gt_deficit, "Ground-truth clarity deficit"),
           (:in_target, "Measurements within ±1 of rated (%)")]
fig = Figure(size=(1500, 850))
for (row_index, environment) in enumerate(environments)
    adaptive = filter(r -> r.environment == environment && r.strategy == "ergo_adaptive", rows)
    isempty(adaptive) && continue
    for (column_index, (metric, label)) in enumerate(metrics)
        ax = Axis(fig[row_index, column_index];
            title="$(replace(environment, '_' => ' ')): $label",
            xlabel="Target-clarity decay rate λ_cd", ylabel=label)
        x = [r.lambda_cd for r in adaptive]
        y = [getproperty(r, metric) for r in adaptive]
        lines!(ax, x, y; color=:navy, linewidth=2)
        scatter!(ax, x, y; color=:navy, markersize=12)
        baseline = filter(r -> r.environment == environment &&
            r.strategy == "ergo_nonadaptive", rows)
        if length(baseline) == 1
            hlines!(ax, [getproperty(only(baseline), metric)];
                color=:darkorange, linestyle=:dash, linewidth=2,
                label="Explicit non-adaptive")
            axislegend(ax; position=:rt)
        end
    end
end
save(joinpath(outdir, "lambda_cd_sweep.pdf"), fig)
save(joinpath(outdir, "lambda_cd_sweep.png"), fig)

# Plot the empirical distribution of the measurements actually collected by
# each sweep condition. Use common bin edges within each environment so curve
# heights are directly comparable. The rated value is shown as a reference.
hist_fig = Figure(size=(1800, 850))
for (column_index, environment) in enumerate(environments)
    environment_rows = filter(r -> r.environment == environment, rows)
    distributions = NamedTuple[]
    for r in environment_rows
        trial_path = joinpath(dirname(r.source), "trial_$(r.strategy).jld2")
        isfile(trial_path) || error("Missing trial file for histogram: $trial_path")
        trial = load(trial_path)
        measurements = Float64.(vec(trial["measurements"]))
        isempty(measurements) && error("No measurements in $trial_path")
        nonadaptive = r.strategy == "ergo_nonadaptive"
        label = nonadaptive ? "non-adaptive" :
            @sprintf("λ=%.3g", r.lambda_cd)
        push!(distributions, (; measurements, nonadaptive,
            rated=Float64(trial["w_rated"]),
            label=@sprintf("%s (μ=%.2f, σ=%.2f)", label,
                mean(measurements), std(measurements))))
    end
    isempty(distributions) && continue
    all_measurements = reduce(vcat, [d.measurements for d in distributions])
    lo, hi = extrema(all_measurements)
    lo == hi && (lo -= 0.5; hi += 0.5)
    edges = range(lo, hi; length=36)
    ax = Axis(hist_fig[1, column_index];
        title="$(replace(environment, '_' => ' ')): collected measurements",
        xlabel="Measured normalized wind speed", ylabel="Probability")
    colors = cgrad(:viridis, max(length(distributions), 2); categorical=true)
    for (i, d) in enumerate(distributions)
        color = d.nonadaptive ? :darkorange : colors[i]
        hist!(ax, d.measurements; bins=edges, normalization=:probability,
            color=(color, 0.10), strokecolor=color, strokewidth=2,
            label=d.label)
    end
    vlines!(ax, [first(distributions).rated]; color=:red, linestyle=:dash,
        linewidth=2, label="rated wind")
    Legend(hist_fig[2, column_index], ax; orientation=:vertical,
        tellwidth=false, framevisible=false)
end
save(joinpath(outdir, "measurement_histograms.pdf"), hist_fig)
save(joinpath(outdir, "measurement_histograms.png"), hist_fig)

for environment in environments
    adaptive = filter(r -> r.environment == environment &&
        r.strategy == "ergo_adaptive" && iszero(r.lambda_cd), rows)
    fixed = filter(r -> r.environment == environment &&
        r.strategy == "ergo_nonadaptive" && iszero(r.lambda_cd), rows)
    if length(adaptive) == 1 && length(fixed) == 1
        a, b = only(adaptive), only(fixed)
        @printf("%s lambda=0 equivalence: ΔRMSE=%.3e, ΔGT-deficit=%.3e, Δin-target=%.3e\n",
            environment, a.rmse-b.rmse, a.gt_deficit-b.gt_deficit,
            a.in_target-b.in_target)
    end
end
println("Wrote aggregate results to $outdir")
