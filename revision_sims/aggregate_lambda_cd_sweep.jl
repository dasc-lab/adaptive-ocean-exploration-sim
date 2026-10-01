#!/usr/bin/env julia

using CairoMakie, JLD2, Printf, Statistics
include(joinpath(@__DIR__, "..", "src", "jordan_lake_domain.jl"))

function fixed_reference_deficit(trial, env; lambda_ref=0.25)
    xs, ys = env["xs"], env["ys"]
    inside = [[x, y] in JordanLakeDomain.convex_polygon.polygon for x in xs, y in ys]
    clarity_maps = trial["clarity_maps"]
    metric_times = trial["metric_times"]
    length(clarity_maps) == length(metric_times) || error("Clarity timestamp mismatch")
    deficits = Float64[]
    for (clarity, t) in zip(clarity_maps, metric_times)
        truth = if get(env, "time_invariant", false)
            env["wind_map"]
        else
            time_index = clamp(searchsortedlast(env["ts_min"], t), 1, size(env["wind_data"], 3))
            env["wind_data"][:, :, time_index]
        end
        target = 0.95 .* exp.(-lambda_ref .* (truth .- trial["w_rated"]).^2) .* inside
        push!(deficits, mean(max.(0.0, target .- clarity)))
    end
    return mean(deficits)
end

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
        trial_path = joinpath(dirname(path), "trial_$(raw["strategy"]).jld2")
        isfile(trial_path) || error("Missing trial file: $trial_path")
        trial = load(trial_path)
        env = load(joinpath(dirname(path), "environment.jld2"))
        push!(rows, (;
            environment=raw["environment"],
            lambda_cd=parse(Float64, raw["lambda_cd"]),
            strategy=raw["strategy"],
            rmse=parse(Float64, raw["rmse"]),
            gt_deficit=parse(Float64, raw["gt_deficit"]),
            reference_deficit=fixed_reference_deficit(trial, env),
            in_target=parse(Float64, raw["in_target"]),
            source=path))
    end
end
sort!(rows; by=r -> (r.environment, r.lambda_cd, r.strategy))

open(joinpath(outdir, "lambda_cd_sweep.csv"), "w") do io
    println(io, "environment,lambda_cd,strategy,rmse,gt_deficit,reference_deficit,in_target,source")
    for r in rows
        println(io, join((r.environment, r.lambda_cd, r.strategy, r.rmse,
            r.gt_deficit, r.reference_deficit, r.in_target, r.source), ','))
    end
end

environments = ["half_domain", "moving_pocket"]
strategy_order = ["transect", "transect_half",
    "bb_ipp_nonadaptive", "bb_ipp_adaptive", "bb_ipp_ground_truth",
    "ergo_nonadaptive", "ergo_adaptive", "ergo_ground_truth"]
strategy_labels = Dict(
    "transect" => "Transect",
    "transect_half" => "Half-domain transect",
    "bb_ipp_nonadaptive" => "BB-IPP non-adaptive",
    "bb_ipp_adaptive" => "BB-IPP adaptive",
    "bb_ipp_ground_truth" => "BB-IPP ground truth",
    "ergo_nonadaptive" => "Ergodic non-adaptive",
    "ergo_adaptive" => "Ergodic adaptive",
    "ergo_ground_truth" => "Ergodic ground truth")
strategy_palette = cgrad(:tab10, length(strategy_order); categorical=true)
strategy_colors = Dict(strategy => strategy_palette[i]
    for (i, strategy) in enumerate(strategy_order))
metrics = [(:rmse, "Global wind RMSE"),
           (:gt_deficit, "Deficit against each run's own λ target"),
           (:reference_deficit, "Deficit against fixed λ=0.25 target"),
           (:in_target, "Measurements within ±1 of rated (%)")]
for environment in environments
    environment_rows = filter(r -> r.environment == environment, rows)
    isempty(environment_rows) && continue
    present_strategies = filter(s -> any(r -> r.strategy == s, environment_rows), strategy_order)
    environment_label = replace(environment, '_' => ' ')
    fig = Figure(size=(1500, 1050))
    for (column_index, (metric, label)) in enumerate(metrics)
        plot_row, plot_column = fldmod1(column_index, 2)
        ax = Axis(fig[plot_row, plot_column];
            title=label,
            xlabel="Target-clarity decay rate λ_cd", ylabel=label)
        for strategy in present_strategies
            series = sort(filter(r -> r.strategy == strategy, environment_rows);
                by=r -> r.lambda_cd)
            lines!(ax, [r.lambda_cd for r in series], [getproperty(r, metric) for r in series];
                color=strategy_colors[strategy], linewidth=2,
                label=strategy_labels[strategy])
            scatter!(ax, [r.lambda_cd for r in series], [getproperty(r, metric) for r in series];
                color=strategy_colors[strategy], markersize=9)
        end
        vlines!(ax, [0.25]; color=:gray45, linestyle=:dot, linewidth=1.5)
    end
    Label(fig[0, 1:2], "$environment_label: target-clarity decay sweep", fontsize=20)
    Legend(fig[3, 1:2], [LineElement(color=strategy_colors[s], linewidth=3)
        for s in present_strategies], [strategy_labels[s] for s in present_strategies];
        orientation=:horizontal, nbanks=2, framevisible=false, tellwidth=false)
    save(joinpath(outdir, "$(environment)_lambda_cd_sweep.pdf"), fig)
    save(joinpath(outdir, "$(environment)_lambda_cd_sweep.png"), fig)
end

# Mission-focus versus global-reconstruction tradeoff. Points toward the upper
# left are preferable; labels make the non-monotonic controller response clear.
for environment in environments
    environment_rows = filter(r -> r.environment == environment, rows)
    isempty(environment_rows) && continue
    present_strategies = filter(s -> any(r -> r.strategy == s, environment_rows), strategy_order)
    environment_label = replace(environment, '_' => ' ')
    ncolumns = min(4, length(present_strategies))
    nrows = cld(length(present_strategies), ncolumns)
    tradeoff_fig = Figure(size=(450ncolumns, 400nrows + 100))
    for (index, strategy) in enumerate(present_strategies)
        plot_row, plot_column = fldmod1(index, ncolumns)
        series = sort(filter(r -> r.strategy == strategy, environment_rows); by=r -> r.lambda_cd)
        ax = Axis(tradeoff_fig[plot_row, plot_column];
            title=strategy_labels[strategy],
            xlabel=plot_row == nrows ? "Global wind RMSE" : "",
            ylabel=plot_column == 1 ? "In-target measurements (%)" : "")
        lines!(ax, [r.rmse for r in series], [r.in_target for r in series];
            color=strategy_colors[strategy], linewidth=1.5)
        scatter!(ax, [r.rmse for r in series], [r.in_target for r in series];
            color=[r.lambda_cd == 0.25 ? :crimson : strategy_colors[strategy] for r in series],
            markersize=12)
        for r in series
            text!(ax, r.rmse, r.in_target; text=@sprintf("  %.3g", r.lambda_cd),
                align=(:left, :center), fontsize=10)
        end
    end
    Label(tradeoff_fig[0, 1:ncolumns],
        "$environment_label tradeoffs; point labels are λ_cd values and λ=0.25 is highlighted",
        fontsize=17)
    save(joinpath(outdir, "$(environment)_sampling_reconstruction_tradeoff.pdf"), tradeoff_fig)
    save(joinpath(outdir, "$(environment)_sampling_reconstruction_tradeoff.png"), tradeoff_fig)
end

# Plot each empirical distribution separately; overlaid histograms obscure the
# strongly non-monotonic changes in this sweep. Each figure contains only one
# environment, uses common bin edges, and retains every run (including the
# explicit non-adaptive lambda-zero reference).
for environment in environments
    environment_rows = filter(r -> r.environment == environment, rows)
    distributions = NamedTuple[]
    for r in environment_rows
        trial_path = joinpath(dirname(r.source), "trial_$(r.strategy).jld2")
        isfile(trial_path) || error("Missing trial file for histogram: $trial_path")
        trial = load(trial_path)
        measurements = Float64.(vec(trial["measurements"]))
        isempty(measurements) && error("No measurements in $trial_path")
        push!(distributions, (; measurements, lambda_cd=r.lambda_cd,
            strategy=r.strategy,
            rated=Float64(trial["w_rated"]), mean=mean(measurements),
            std=std(measurements)))
    end
    isempty(distributions) && continue
    all_measurements = reduce(vcat, [d.measurements for d in distributions])
    lo, hi = extrema(all_measurements)
    lo == hi && (lo -= 0.5; hi += 0.5)
    edges = range(lo, hi; length=36)
    lambdas = sort(unique(d.lambda_cd for d in distributions))
    present_strategies = filter(s -> any(d -> d.strategy == s, distributions), strategy_order)
    ncolumns = length(lambdas)
    nrows = length(present_strategies)
    hist_fig = Figure(size=(330ncolumns + 180, 260nrows + 100), fontsize=11)
    for d in distributions
        plot_row = findfirst(==(d.strategy), present_strategies)
        plot_column = findfirst(==(d.lambda_cd), lambdas)
        ax = Axis(hist_fig[plot_row, plot_column];
            title=@sprintf("λ=%.3g\nμ=%.2f, σ=%.2f", d.lambda_cd, d.mean, d.std),
            xlabel=plot_row == nrows ? "Measured normalized wind speed" : "",
            ylabel=plot_column == 1 ? "Probability" : "")
        color = d.lambda_cd == 0.25 ? :crimson : strategy_colors[d.strategy]
        hist!(ax, d.measurements; bins=edges, normalization=:probability,
            color=(color, 0.20), strokecolor=color, strokewidth=1.5)
        vlines!(ax, [d.rated]; color=:red, linestyle=:dash, linewidth=1.5)
    end
    for (plot_row, strategy) in enumerate(present_strategies)
        Label(hist_fig[plot_row, 0], strategy_labels[strategy];
            rotation=pi/2, fontsize=13, tellheight=false)
    end
    environment_label = replace(environment, '_' => ' ')
    Label(hist_fig[0, 1:ncolumns],
        "$environment_label: collected measurements across target-clarity decay rates",
        fontsize=20)
    Label(hist_fig[nrows + 1, 1:ncolumns],
        "Dashed line: rated wind. Panels report empirical measurement mean and standard deviation.",
        fontsize=14)
    save(joinpath(outdir, "$(environment)_measurement_histograms.pdf"), hist_fig)
    save(joinpath(outdir, "$(environment)_measurement_histograms.png"), hist_fig)
end

for environment in environments, family in ("ergo", "bb_ipp")
    adaptive = filter(r -> r.environment == environment &&
        r.strategy == "$(family)_adaptive" && iszero(r.lambda_cd), rows)
    fixed = filter(r -> r.environment == environment &&
        r.strategy == "$(family)_nonadaptive" && iszero(r.lambda_cd), rows)
    if length(adaptive) == 1 && length(fixed) == 1
        a, b = only(adaptive), only(fixed)
        @printf("%s %s lambda=0 equivalence: ΔRMSE=%.3e, ΔGT-deficit=%.3e, Δin-target=%.3e\n",
            environment, family, a.rmse-b.rmse, a.gt_deficit-b.gt_deficit,
            a.in_target-b.in_target)
    end
end
println("Wrote aggregate results to $outdir")
