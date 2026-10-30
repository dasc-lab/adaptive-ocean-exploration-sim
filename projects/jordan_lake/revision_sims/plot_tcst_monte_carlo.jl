#!/usr/bin/env julia
# Standalone: julia --project=. projects/jordan_lake/revision_sims/plot_tcst_monte_carlo.jl [OUTPUT_DIR]
# Set TCST_MC_ROOT to a replacement parameter-grid directory when corrected
# results arrive. Never pool batches. The current default is the legacy batch.
if !isdefined(Main, :Fieldline)
    include(joinpath(@__DIR__, "plot_tcst_simulation_figures.jl"))
end

function read_mc_grid()
    root = get(ENV, "TCST_MC_ROOT", joinpath(RESULTS, "data_results", "results_mc"))
    rows = NamedTuple[]
    for ls in [.2, .75, 1.0], lt in [30.0, 75.0, 120.0]
        condition = joinpath(root, "ls_$(ls)_lt_$(lt)")
        isdir(condition) || error("Missing Monte Carlo condition: $condition")
        batches = sort(filter(d -> isfile(joinpath(condition, d, "plot_summary_data.csv")), readdir(condition)))
        isempty(batches) && error("No completed summary in $condition")
        # Select the most recent summarized batch, then fail on incompleteness
        # rather than silently falling back to an older implementation.
        batch = joinpath(condition, last(batches))
        files = filter(f -> startswith(f, "trial_seed") && endswith(f, ".jld2"), readdir(batch))
        summary = collect(CSV.File(joinpath(batch, "plot_summary_data.csv")))
        for w in [0.0, .5, 1.0, 1.5], strategy in setdiff(ORDER, ["transect_half", "bb_ipp_ground_truth"])
            matches = filter(r -> String(r.Strategy) == strategy && Float64(r.W_Rated) == w, summary)
            length(matches) == 1 || error("Missing/duplicate condition: $batch, $strategy, $w")
            r = only(matches)
            suffix = @sprintf("_%s_w%.2f_ls%.2f_lt%.2f.jld2", strategy, w, ls, lt)
            trial_files = filter(f -> endswith(f, suffix), files)
            seeds = sort([parse(Int, match(r"^trial_seed(\d+)_", f).captures[1]) for f in trial_files])
            seeds == collect(1234:1263) || error("Expected seeds 1234:1263: $batch, $strategy, $w")
            push!(rows, (; ls, lt, w, strategy, n=length(seeds),
                rmse=Float64(r.RMSE_Mean), rmse_sem=Float64(r.RMSE_SEM),
                deficit=Float64(r.GT_Deficit_Mean), deficit_sem=Float64(r.GT_Deficit_SEM),
                in_target=Float64(r.In_Target_Percent), source=batch))
        end
    end
    return rows
end

function plot_tcst_monte_carlo()
    effects_path = get(ENV, "TCST_MC_EFFECTS", "")
    if isempty(effects_path)
        rows = read_mc_grid()
        gains = NamedTuple[]
        for planner in ["ergo", "bb_ipp"], a in rows
            a.strategy == planner * "_adaptive" || continue
            b = only(filter(r -> r.strategy == planner * "_nonadaptive" &&
                (r.ls, r.lt, r.w) == (a.ls, a.lt, a.w), rows))
            push!(gains, (; planner, ls=a.ls, lt=a.lt, w=a.w,
                rmse=100 * (a.rmse / b.rmse - 1),
                deficit=100 * (a.deficit / b.deficit - 1),
                in_target=a.in_target - b.in_target))
        end
        CSV.write(joinpath(OUTPUT, "monte_carlo_condition_summary.csv"), rows)
        CSV.write(joinpath(OUTPUT, "monte_carlo_adaptation_effects.csv"), gains)
    else
        gains = [(; planner=String(r.Planner), ls=Float64(r.Ls), lt=Float64(r.Lt),
                    w=Float64(r.W_Rated), rmse=Float64(r.RMSE_Percent),
                    deficit=Float64(r.GT_Deficit_Percent),
                    in_target=Float64(r.In_Target_PP))
                 for r in CSV.File(effects_path)]
    end
    # Source group SEMs describe seed variability. The plotted envelope instead
    # describes environmental/model variation and is NOT a confidence interval.
    # Match the other full-width paper figures: 800 pt width, Computer Modern
    # typography from the shared publication theme, and 14 pt axis text. Color
    # identifies spatial scale and line style identifies temporal scale so every
    # one of the nine environmental settings can be traced across rated speeds.
    fig = Figure(size=(800, 650), fontsize=14)
    metrics = [:rmse, :deficit, :in_target]
    labels = ["Change in global RMSE (%)", "Change in GT deficit (%)", "Change in-target sampling (pp)"]
    ls_values = [.2, .75, 1.0]
    lt_values = [30.0, 75.0, 120.0]
    ls_colors = Dict(ls => Fieldline.categorical(i) for (i, ls) in enumerate(ls_values))
    lt_styles = Dict(30.0 => :solid, 75.0 => :dash, 120.0 => :dot)

    Legend(fig[0, 1],
        [LineElement(color=ls_colors[ls], linewidth=2) for ls in ls_values],
        ["Ls = $(ls) km" for ls in ls_values];
        orientation=:horizontal, tellwidth=false, framevisible=false)
    Legend(fig[0, 2],
        [LineElement(color=Fieldline.categorical(:ink), linewidth=2,
                     linestyle=lt_styles[lt]) for lt in lt_values],
        [@sprintf("Lt = %.0f min", lt) for lt in lt_values];
        orientation=:horizontal, tellwidth=false, framevisible=false)

    for (j, planner) in enumerate(["ergo", "bb_ipp"]), i in 1:3
        metric = metrics[i]
        subset = filter(r -> r.planner == planner, gains)
        wvals = [0.0, .5, 1.0, 1.5]
        groups = [getproperty.(filter(r -> r.w == w, subset), metric) for w in wvals]
        title = i == 1 ? (planner == "ergo" ? "Ergodic" : "BB-IPP") : ""
        ax = Axis(fig[i, j]; title, titlesize=18, ylabel=labels[i],
            xlabel=i == 3 ? "Rated normalized wind speed" : "", xticks=wvals)
        band!(ax, wvals, minimum.(groups), maximum.(groups);
            color=(Fieldline.neutral(:light, :grid), .42))
        for ls in ls_values, lt in lt_values
            ys = [getproperty(only(filter(r -> (r.ls, r.lt, r.w) == (ls, lt, w), subset)), metric) for w in wvals]
            lines!(ax, wvals, ys; color=ls_colors[ls], linestyle=lt_styles[lt], linewidth=1.35)
        end
        scatterlines!(ax, wvals, mean.(groups); color=Fieldline.categorical(:ink),
            linewidth=2.5, markersize=7)
        hlines!(ax, [0]; color=Fieldline.categorical(:ink), linestyle=:dash, linewidth=2)
        # Identical vertical limits for both planners, for each metric.
        all_values = getproperty.(gains, metric)
        lo, hi = extrema(vcat(all_values, [0.0]))
        margin = max((hi-lo) * .1, .2)
        ylims!(ax, lo-margin, hi+margin)
    end
    colgap!(fig.layout, 24)
    rowgap!(fig.layout, 8)
    save_figure(fig, "monte_carlo_environment_sensitivity")
    for planner in ["ergo", "bb_ipp"]
        selected = filter(r -> r.planner == planner, gains)
        println(planner, ": ", [(m, mean(getproperty.(selected,m)), extrema(getproperty.(selected,m))) for m in metrics])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_tcst_monte_carlo()
end
