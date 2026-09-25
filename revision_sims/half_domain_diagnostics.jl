module HalfDomainDiagnostics
using JLD2, CairoMakie, Statistics, Printf
include(joinpath(@__DIR__, "..", "src", "jordan_lake_domain.jl"))

const STRATEGY_ORDER = ["transect", "bb_ipp_nonadaptive", "bb_ipp_adaptive",
    "bb_ipp_ground_truth", "ergo_nonadaptive", "ergo_adaptive", "ergo_ground_truth"]
const STRATEGY_LABELS = ["Transect", "BB-IPP uniform", "BB-IPP adaptive",
    "BB-IPP (Ground Truth)", "Ergodic uniform", "Ergodic adaptive",
    "Ergodic (Ground Truth)"]
const STRATEGY_COLORS = [:gray, :peru, :orangered, :purple, :steelblue, :navy, :seagreen]

"""Replot synchronized saved maps without rerunning the missions.
Ground-truth BB-IPP and ergodic use ground-truth targets rather than STGPKF targets.
Ergodic (Ground Truth) uses a ground-truth target, not an STGPKF-derived target.
"""
function plot_deficits(data_dir)
    env = load(joinpath(data_dir, "environment.jld2"))
    xs, ys = env["xs"], env["ys"]
    inside = [[x, y] ∈ JordanLakeDomain.convex_polygon.polygon for x in xs, y in ys]
    split_x = get(env, "split_x", (first(xs) + last(xs)) / 2)
    west = inside .& [x < split_x for x in xs, y in ys]
    truth_target = 0.95 .* exp.(-0.25 .* (env["wind_map"] .- env["w_rated"]).^2) .* inside
    order, labels, palette = STRATEGY_ORDER, STRATEGY_LABELS, STRATEGY_COLORS
    rows = NamedTuple[]
    series = NamedTuple[]
    for (index, strategy) in enumerate(order)
        path = joinpath(data_dir, "trial_$(strategy).jld2")
        isfile(path) || continue
        trial = load(path)
        a = trial["animation"]
        estimated = trial["clarity_deficit"]
        gt = trial["gt_clarity_deficit"]
        metric_times = trial["metric_times"]
        length(metric_times) == length(estimated) == length(gt) ||
            error("Clarity-deficit timestamp mismatch: $strategy")
        west_clarity = mean([mean(q[west]) for q in a.clarity_maps])
        west_target = mean([mean(q[west]) for q in a.target_maps])
        # Fraction of truly valuable west cells assigned less than half their true target.
        west_underweighted = mean([mean(q[west] .< 0.475) for q in a.target_maps])
        target_source = if strategy in ("ergo_ground_truth", "bb_ipp_ground_truth")
            "ground_truth"
        elseif strategy in ("ergo_nonadaptive", "bb_ipp_nonadaptive")
            "uniform_target"
        else
            "STGPKF_mean"
        end
        push!(rows, (; strategy, target_source,
            frames=length(estimated), estimated_deficit=mean(estimated), gt_deficit=mean(gt),
            west_clarity, west_target, west_underweighted))
        push!(series, (; times=metric_times .- first(env["ts_min"]), estimated, gt,
            label=labels[index], color=palette[index]))
    end
    isempty(rows) && error("No saved strategy maps found in $data_dir")
    fig = Figure(size=(1500, 850), fontsize=15)
    ax_est = Axis(fig[1, 1]; title="Estimated-target clarity deficit", xlabel="Mission time (min)",
        ylabel="Mean positive deficit")
    ax_gt = Axis(fig[1, 2]; title="Ground-truth-target clarity deficit", xlabel="Mission time (min)",
        ylabel="Mean positive deficit")
    for s in series
        lines!(ax_est, s.times, s.estimated; color=s.color, label=s.label)
        lines!(ax_gt, s.times, s.gt; color=s.color, label=s.label)
    end
    upper = 1.05 * maximum(max(maximum(s.estimated), maximum(s.gt)) for s in series)
    ylims!(ax_est, 0, upper)
    ylims!(ax_gt, 0, upper)
    ax_mean = Axis(fig[2, 1:2]; title="Average deficit over all filter updates",
        ylabel="Mean positive deficit", xticks=(1:length(rows), [s.label for s in series]))
    barplot!(ax_mean, repeat(collect(1:length(rows)); inner=2),
        vec(permutedims(hcat([r.estimated_deficit for r in rows], [r.gt_deficit for r in rows])));
        dodge=repeat([1, 2], length(rows)), color=repeat([:steelblue, :orange], length(rows)))
    Legend(fig[3, 1:2], [PolyElement(color=:steelblue), PolyElement(color=:orange)],
        ["Estimated target (Ground Truth strategy uses GT)", "Ground-truth target"]; orientation=:horizontal)
    Legend(fig[0, 1:2], ax_est; orientation=:horizontal, nbanks=2)
    Label(fig[4, 1:2], "Deficit = spatial mean of max(target − achieved clarity, 0). Full grid normalization; all filter updates.", fontsize=13)
    outdir = joinpath(data_dir, "figures")
    mkpath(outdir)
    for extension in ("png", "pdf")
        save(joinpath(outdir, "clarity_deficit_comparison.$extension"), fig)
    end
    open(joinpath(data_dir, "clarity_deficit_diagnostics.csv"), "w") do io
        println(io, join(string.(propertynames(first(rows))), ","))
        for row in rows
            println(io, join(values(row), ","))
        end
    end
    open(joinpath(data_dir, "clarity_deficit_timeseries.csv"), "w") do io
        println(io, "strategy,mission_minutes,estimated_target_deficit,ground_truth_target_deficit")
        for (row, s) in zip(rows, series), i in eachindex(s.times)
            println(io, join((row.strategy, s.times[i], s.estimated[i], s.gt[i]), ","))
        end
    end
    return rows
end

"""RMSE time histories, means and temporal distributions from saved results.
Clarity RMSE is recomputed from synchronized snapshots, not legacy scalar metrics.
Wind RMSE uses every saved filter update; older trials lack update timestamps.
"""
function plot_rmse(data_dir)
    env = load(joinpath(data_dir, "environment.jld2"))
    inside = [[x, y] ∈ JordanLakeDomain.convex_polygon.polygon for x in env["xs"], y in env["ys"]]
    truth_target = 0.95 .* exp.(-0.25 .* (env["wind_map"] .- env["w_rated"]).^2) .* inside
    series = NamedTuple[]
    rms(a, b) = sqrt(mean(abs2, a .- b))
    for (i, strategy) in enumerate(STRATEGY_ORDER)
        path = joinpath(data_dir, "trial_$(strategy).jld2")
        isfile(path) || continue
        trial = load(path)
        estimated = trial["est_clarity_rmse"]
        gt = trial["gt_clarity_rmse"]
        target_error = trial["target_clarity_rmse"]
        wind = trial["rmse_global"]
        metric_times = get(trial, "metric_times", nothing)
        metric_times === nothing && error("Missing metric timestamps: $strategy")
        length(metric_times) == length(wind) == length(estimated) == length(gt) == length(target_error) ||
            error("RMSE timestamp mismatch: $strategy")
        push!(series, (; strategy, label=STRATEGY_LABELS[i], color=STRATEGY_COLORS[i],
            wind, metric_times, times=metric_times .- first(env["ts_min"]), estimated, gt, target_error))
    end
    isempty(series) && error("No saved RMSE results found in $data_dir")
    outdir = joinpath(data_dir, "figures")
    mkpath(outdir)
    save_figure(fig, name) = foreach(ext -> save(joinpath(outdir, "$name.$ext"), fig), ("png", "pdf"))
    have_times = all(s.metric_times !== nothing for s in series)
    labels = [s.label for s in series]
    colors = [s.color for s in series]
    positions = collect(eachindex(series))

    specs = ((:estimated, "Estimated target vs achieved clarity"),
        (:gt, "Ground-truth target vs achieved clarity"),
        (:target_error, "Estimated target vs ground-truth target"))
    clarity_fig = Figure(size=(1500, 850), fontsize=15)
    clarity_specs = specs[1:2]
    ax_est = Axis(clarity_fig[1, 1]; title=clarity_specs[1][2], xlabel="Mission time (min)",
        ylabel="Clarity RMSE")
    ax_gt = Axis(clarity_fig[1, 2]; title=clarity_specs[2][2], xlabel="Mission time (min)",
        ylabel="Clarity RMSE")
    for (ax, (metric, _)) in zip((ax_est, ax_gt), clarity_specs)
        for s in series
            lines!(ax, s.times, getproperty(s, metric); color=s.color, label=s.label)
        end
    end
    upper = 1.05 * max(maximum(maximum(s.estimated) for s in series),
        maximum(maximum(s.gt) for s in series), eps())
    ylims!(ax_est, 0, upper)
    ylims!(ax_gt, 0, upper)
    ax_mean = Axis(clarity_fig[2, 1:2]; title="Average RMSE over all filter updates",
        ylabel="Clarity RMSE", xticks=(positions, labels))
    barplot!(ax_mean, repeat(collect(1:length(series)); inner=2),
        vec(permutedims(hcat([mean(s.estimated) for s in series], [mean(s.gt) for s in series])));
        dodge=repeat([1, 2], length(series)), color=repeat([:steelblue, :orange], length(series)))
    Legend(clarity_fig[3, 1:2], [PolyElement(color=:steelblue), PolyElement(color=:orange)],
        ["Estimated target (Ground Truth strategy uses GT)", "Ground-truth target"]; orientation=:horizontal)
    Legend(clarity_fig[0, 1:2], ax_est; orientation=:horizontal, nbanks=2)
    Label(clarity_fig[4, 1:2], "RMSE = sqrt(mean((target − reference)^2)) over the full grid. All filter updates.", fontsize=13)
    save_figure(clarity_fig, "clarity_rmse_comparison")

    # --- NEW: Grid of subplots for direct Ground Truth vs STGPKF comparison ---
    grid_fig = Figure(size=(1500, 800), fontsize=14)
    for (i, s) in enumerate(series)
        r, c = cld(i, 3), mod1(i, 3)
        ax = Axis(grid_fig[r, c], title=s.label, xlabel="Mission time (min)", ylabel="Clarity RMSE")
        
        lines!(ax, s.times, s.gt, label="Ground Truth RMSE", color=:dodgerblue, linewidth=2.5)
        lines!(ax, s.times, s.estimated, label="STGPKF Estimated RMSE", color=:darkorange, linewidth=2.5, linestyle=:dash)
        
        ylims!(ax, 0, 1) # Uniform y-axis for reliable visual comparison across strategies
        
        if i == 1
            axislegend(ax, position=:rt, framevisible=false)
        end
    end
    Label(grid_fig[0, 1:3], "Clarity RMSE over time per strategy: Ground Truth vs STGPKF Estimates", fontsize=20)
    save_figure(grid_fig, "clarity_rmse_subplots")
    # --------------------------------------------------------------------------

    hist_clarity = Figure(size=(1800, 600), fontsize=14)
    for (col, (metric, title)) in enumerate(specs)
        ax = Axis(hist_clarity[1, col]; title, xlabel="Clarity RMSE", ylabel="Fraction of saved snapshots")
        for s in series
            hist!(ax, getproperty(s, metric); bins=range(0, 1; length=26), normalization=:probability,
                color=(s.color, 0.12), strokecolor=s.color, strokewidth=1.5)
        end
        xlims!(ax, 0, 1)
        ylims!(ax, 0, 1)
    end
    Legend(hist_clarity[0, 1:3], [PolyElement(color=(s.color, 0.3), strokecolor=s.color) for s in series],
        labels; orientation=:horizontal, nbanks=2)
    Label(hist_clarity[2, 1:3], "Distributions over time in one mission, not independent trials. Ergodic (Ground Truth) uses the true target.", fontsize=13)
    save_figure(hist_clarity, "clarity_rmse_histograms")

    # Field-estimation RMSE outputs follow all clarity diagnostics.
    wind_fig = Figure(size=(1500, 600), fontsize=15)
    ax = Axis(wind_fig[1, 1]; title="Global wind-estimation RMSE", ylabel="RMSE (normalized wind speed)",
        xlabel=have_times ? "Mission time (min)" : "Filter update index (timestamps not saved)")
    for s in series
        x = have_times ? s.metric_times .- first(env["ts_min"]) : collect(eachindex(s.wind))
        lines!(ax, x, s.wind; color=s.color, label=s.label)
    end
    means_ax = Axis(wind_fig[1, 2]; title="Mean RMSE over all saved filter updates",
        ylabel="RMSE (normalized wind speed)", xticks=(positions, labels), xticklabelrotation=pi/4)
    barplot!(means_ax, positions, [mean(s.wind) for s in series]; color=colors)
    Legend(wind_fig[0, 1:2], ax; orientation=:horizontal, nbanks=2)
    Label(wind_fig[2, 1:2], "Spatial RMSE compares the estimated wind field with ground truth over the full grid.", fontsize=13)
    save_figure(wind_fig, "wind_rmse_comparison")

    wind_edges = range(0, max(maximum(maximum(s.wind) for s in series), eps()); length=26)
    hist_fig = Figure(size=(1500, 1100), fontsize=14)
    for (i, s) in enumerate(series)
        row, col = cld(i, 2), mod1(i, 2)
        ax = Axis(hist_fig[row, col]; title=s.label,
            xlabel=row == 3 ? "Wind RMSE (normalized wind speed)" : "",
            ylabel=col == 1 ? "Fraction of filter updates" : "")
        hist!(ax, s.wind; bins=wind_edges, normalization=:probability, color=s.color)
        xlims!(ax, first(wind_edges), last(wind_edges))
        ylims!(ax, 0, 1)
    end
    Label(hist_fig[0, 1:2], "Wind RMSE distributions within each mission", fontsize=20)
    save_figure(hist_fig, "wind_rmse_histograms")

    open(joinpath(data_dir, "rmse_diagnostics.csv"), "w") do io
        println(io, "strategy,wind_updates,clarity_snapshots,wind_rmse_mean,estimated_clarity_rmse_mean,gt_clarity_rmse_mean,target_rmse_mean")
        for s in series
            println(io, join((s.strategy, length(s.wind), length(s.times), mean(s.wind),
                mean(s.estimated), mean(s.gt), mean(s.target_error)), ","))
        end
    end
    open(joinpath(data_dir, "rmse_timeseries.csv"), "w") do io
        println(io, "strategy,metric,sample_index,mission_minutes,rmse")
        for s in series
            for i in eachindex(s.wind)
                minutes = s.metric_times === nothing ? "" : s.metric_times[i] - first(env["ts_min"])
                println(io, join((s.strategy, "wind", i, minutes, s.wind[i]), ","))
            end
            for (metric, _) in specs, i in eachindex(s.times)
                println(io, join((s.strategy, string(metric), i, s.times[i], getproperty(s, metric)[i]), ","))
            end
        end
    end
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: julia --project=. revision_sims/half_domain_diagnostics.jl RESULTS_DIRECTORY")
    HalfDomainDiagnostics.plot_rmse(ARGS[1])
    for row in HalfDomainDiagnostics.plot_deficits(ARGS[1])
        println(row)
    end
end
