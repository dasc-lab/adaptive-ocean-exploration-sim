#!/usr/bin/env julia
# Reproduce every figure included by Sections/simulation.tex (PDF + PNG).
# Usage, from the simulation repository:
# julia --project=. revision_sims/plot_tcst_simulation_figures.jl [OUTPUT_DIR] [RESULTS_ROOT]
# Existing aggregated metrics and saved fields are used; simulations are not rerun.
using CairoMakie, CSV, JLD2, StaticArrays, Statistics, Printf, LaTeXStrings
include(joinpath(@__DIR__, "..", "color-scheme", "julia", "Fieldline.jl"))
using .Fieldline
include(joinpath(@__DIR__, "..", "src", "jordan_lake_domain.jl"))

const DEFAULT_OUTPUT = "/home/kmgovind/OneDrive/Research/CPS/lake-testing-tcst/Figures"
length(ARGS) <= 2 || error("Usage: plot_tcst_simulation_figures.jl [OUTPUT_DIR] [RESULTS_ROOT]")
const OUTPUT = isempty(ARGS) ? DEFAULT_OUTPUT : abspath(ARGS[1])
const RESULTS = length(ARGS) < 2 ? (@__DIR__) : abspath(ARGS[2])
mkpath(OUTPUT)

# Use the shared colors, scientific colormap, and stroke weights. White paper
# overrides are local to this script; the shared slide/screen theme is unchanged.
base = Fieldline.makie_theme()
set_theme!(; merge(base, (
    backgroundcolor=:white, fontsize=14,
    fonts=theme_latexfonts().fonts,
    Lines=(linewidth=2,), ScatterLines=(linewidth=2,),
    Axis=merge(base.Axis, (backgroundcolor=:white, xgridvisible=false, ygridvisible=false,
                          titlesize=18, xlabelsize=14, ylabelsize=14, xticklabelsize=14, yticklabelsize=14)),
    Legend=merge(base.Legend, (backgroundcolor=:white, framevisible=false, labelsize=10)),
))...)
const STROKES = Fieldline.PALETTE["strokes"]
const ORDER = ["transect", "transect_half", "bb_ipp_nonadaptive", "bb_ipp_adaptive",
               "bb_ipp_ground_truth", "ergo_nonadaptive", "ergo_adaptive", "ergo_ground_truth"]
const LABELS = ["Full-domain transect", "Half-domain transect (GT)", "BB-IPP, non-adaptive",
                "BB-IPP, adaptive", "BB-IPP, ground truth", "Ergodic, non-adaptive",
                "Ergodic, adaptive", "Ergodic, ground truth"]
const STRATEGY_COLORS = Dict(s => Fieldline.categorical(i) for (i, s) in enumerate(ORDER))
const ENVIRONMENTS = ["half_domain", "moving_pocket"]

function save_figure(fig, name)
    save(joinpath(OUTPUT, "$name.pdf"), fig; pt_per_unit=1)
    save(joinpath(OUTPUT, "$name.png"), fig; px_per_unit=3)
    println("Saved $name.pdf and $name.png")
end

function read_results()
    rows = NamedTuple[]
    for study in ("lambda_results", "lambda_fine")
        path = joinpath(RESULTS, study, "aggregate", "lambda_cd_sweep.csv")
        isfile(path) || error("Missing aggregate: $path")
        for r in CSV.File(path)
            push!(rows, (; environment=String(r.environment), lambda_cd=Float64(r.lambda_cd),
                strategy=String(r.strategy), rmse=Float64(r.rmse), gt_deficit=Float64(r.gt_deficit),
                reference_deficit=Float64(r.reference_deficit), in_target=Float64(r.in_target),
                source=String(r.source)))
        end
    end
    return rows
end

# Linear near zero, logarithmic above .01; explicit transformed ticks also work
# in Makie versions without a symmetric logarithmic scale.
sweep_x(x) = x <= .01 ? x : .01 + .01125 * log10(x / .01)
function plot_sweep(rows)
    selected = Dict{Tuple{String,Float64},NamedTuple}()
    for r in rows
        r.strategy == "ergo_adaptive" || continue
        key = (r.environment, r.lambda_cd)
        if haskey(selected, key)
            all(isapprox(getproperty(r, k), getproperty(selected[key], k); rtol=1e-7, atol=1e-9)
                for k in (:rmse, :reference_deficit, :in_target)) || error("Conflicting repeated run: $key")
        end
        selected[key] = r
    end
    fig = Figure(size=(800, 340))
    metrics = [:rmse, :reference_deficit, :in_target]
    titles = ["(a) Field estimation", "(b) Target clarity", "(c) Targeted sampling"]
    labels = ["RMSE relative to\nnon-adaptive strategy", "Fixed-target deficit relative to\nnon-adaptive strategy", "In-target measurements (%)"]
    ticks = ([sweep_x(x) for x in [0, .01, .05, .25, 1]], ["0", ".01", ".05", ".25", "1"])
    for i in 1:3
        ax = Axis(fig[2, i]; title=titles[i], xlabel=L"\lambda_{\mathrm{cd}}", ylabel=labels[i], xticks=ticks)
        for (j, env) in enumerate(ENVIRONMENTS)
            xs = sort([x for (e, x) in keys(selected) if e == env])
            baseline = getproperty(selected[(env, 0.0)], metrics[i])
            ys = [getproperty(selected[(env, x)], metrics[i]) / (i == 3 ? 1 : baseline) for x in xs]
            scatterlines!(ax, sweep_x.(xs), ys; color=Fieldline.categorical(j),
                marker=j == 1 ? :circle : :rect, markersize=6,
                linewidth=2)
        end
        i < 3 && hlines!(ax, [1]; color=Fieldline.categorical(:ink), linestyle=:dash, linewidth=2)
        i == 3 && ylims!(ax, 0, 100)
        xlims!(ax, -.001, sweep_x(1) + .001)
    end
    Legend(fig[1, 1:3], [LineElement(color=Fieldline.categorical(i), linewidth=2) for i in 1:2],
        ["Half-domain", "Moving pockets"]; orientation=:horizontal, tellwidth=false)
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)
    save_figure(fig, "lambda_cd_sweep")
end

function nominal_rows(rows, env)
    return Dict(r.strategy => r for r in rows if r.environment == env && r.lambda_cd == .25)
end

function plot_comparison(rows, env, filename)
    selected = nominal_rows(rows, env)
    strategies = filter(s -> haskey(selected, s), ORDER)
    n = length(strategies)
    fig = Figure(size=(800, 420))
    titles = ["(a) Global RMSE", "(b) Clarity deficit", "(c) In-target\nmeasurements"]
    labels = ["Normalized wind speed", "Fixed-target deficit", "Measurements (%)"]
    for (i, metric) in enumerate([:rmse, :reference_deficit, :in_target])
        ys = collect(n:-1:1)
        ax = Axis(fig[1, i]; title=titles[i], xlabel=labels[i],
            yticks=(ys, [LABELS[findfirst(==(s), ORDER)] for s in strategies]),
            yticklabelsvisible=i == 1, yticksvisible=false)
        values = [getproperty(selected[s], metric) for s in strategies]
        upper = i == 3 ? 120.0 : maximum(values) * 1.5
        for (y, s, value) in zip(ys, strategies, values)
            color = STRATEGY_COLORS[s]
            truth = occursin("ground_truth", s) || s == "transect_half"
            fill = truth ? CairoMakie.Makie.LinePattern(linecolor=Fieldline.neutral(:light, :text_muted),
                backgroundcolor=color, width=.7, tilesize=(8, 8)) : color
            barplot!(ax, [y], [value]; direction=:x, width=.65, color=fill)
            text!(ax, value + upper * .018, y; text=@sprintf("%.*f", i == 3 ? 1 : (i == 2 ? 4 : 3), value),
                align=(:left, :center), fontsize=14)
        end
        xlims!(ax, 0, upper)
        ylims!(ax, .4, n + .6)
    end
    colgap!(fig.layout, 36)
    save_figure(fig, filename)
end

function plot_environments(rows)
    fields = NamedTuple[]
    for env in ENVIRONMENTS
        row = nominal_rows(rows, env)["ergo_adaptive"]
        data = load(joinpath(dirname(row.source), "environment.jld2"))
        if env == "half_domain"
            push!(fields, (; xs=collect(data["xs"]), ys=collect(data["ys"]), field=data["wind_map"],
                title="(a) Half-domain\n(static)", split=Float64(data["split_x"]), rated=Float64(data["w_rated"])))
        else
            times = collect(data["ts_min"])
            for (elapsed, letter) in zip([0, 180, 360], ['b', 'c', 'd'])
                index = argmin(abs.(times .- first(times) .- elapsed))
                push!(fields, (; xs=collect(data["xs"]), ys=collect(data["ys"]), field=data["wind_data"][:, :, index],
                    title="($letter) Pockets, $(elapsed ÷ 60) h", split=nothing, rated=Float64(data["w_rated"])))
            end
        end
    end
    fig = Figure(size=(800, 350))
    vertices = JordanLakeDomain.convex_polygon.vertices
    boundary = [vertices vertices[:, 1]]
    polygon = JordanLakeDomain.convex_polygon.polygon
    for (i, f) in enumerate(fields)
        ax = Axis(fig[1, i]; title=f.title, xlabel="East (km)", ylabel=i == 1 ? "North (km)" : "",
            yticklabelsvisible=i == 1, aspect=DataAspect(), xticks=[0, .8, 1.6], yticks=[0, .6, 1.2, 1.8])
        mask = [[x, y] in polygon for x in f.xs, y in f.ys]
        wind = copy(f.field)
        wind[.!mask] .= NaN
        heatmap!(ax, f.xs, f.ys, wind; colormap=Fieldline.sequential(scientific=true),
            colorrange=(-3.5, 2.5), nan_color=:white, rasterize=3)
        lines!(ax, boundary[1, :], boundary[2, :]; color=Fieldline.categorical(:ink))
        if isnothing(f.split)
            contour!(ax, f.xs, f.ys, wind; levels=[f.rated + 1], color=Fieldline.categorical(:vermilion), linewidth=2)
        else
            intersections = Float64[]
            for j in 1:4
                x1, y1 = vertices[:, j]
                x2, y2 = vertices[:, mod1(j + 1, 4)]
                if min(x1, x2) <= f.split <= max(x1, x2) && x1 != x2
                    push!(intersections, y1 + (f.split - x1) / (x2 - x1) * (y2 - y1))
                end
            end
            lines!(ax, [f.split, f.split], collect(extrema(intersections)); color=Fieldline.categorical(:vermilion), linestyle=:dash, linewidth=2)
        end
        limits!(ax, -.05, 1.6, -.1, 1.9)
    end
    Colorbar(fig[2, 1:4]; colormap=Fieldline.sequential(scientific=true), colorrange=(-3.5, 2.5),
        vertical=false, label="Normalized wind speed", ticks=[-3.5, -1.5, .5, 2.5],
        width=Relative(.65), height=11, labelsize=14, ticklabelsize=14)
    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 5)
    save_figure(fig, "simulation_environments")
end

function concentration_diagnostics(rows)
    # Descriptive occupancy of fixed 100 m cells, using actual sensing locations.
    # Smaller effective cell count and larger top-ten share mean more concentration.
    open(joinpath(OUTPUT, "planner_sampling_concentration.csv"), "w") do io
        println(io, "environment,strategy,cell_width_km,occupied_cells,effective_cells,top10_measurement_pct")
        for env in ENVIRONMENTS, strategy in ["bb_ipp_adaptive", "ergo_adaptive"]
            row = nominal_rows(rows, env)[strategy]
            positions = load(joinpath(dirname(row.source), "trial_$strategy.jld2"), "measurement_positions")
            counts = Dict{Tuple{Int,Int},Int}()
            for p in positions
                key = (floor(Int, p[1] / .1), floor(Int, p[2] / .1))
                counts[key] = get(counts, key, 0) + 1
            end
            probabilities = sort(collect(values(counts)) ./ length(positions); rev=true)
            effective = exp(-sum(p * log(p) for p in probabilities))
            top10 = 100sum(probabilities[1:min(10, end)])
            println(io, join((env, strategy, .1, length(counts), effective, top10), ','))
            @printf("%s / %s: %.1f effective cells; %.1f%% of measurements in top 10 cells\n", env, strategy, effective, top10)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = read_results()
    concentration_diagnostics(rows)
    plot_sweep(rows)
    plot_comparison(rows, "half_domain", "revision_half_domain_comparison")
    plot_comparison(rows, "moving_pocket", "revision_moving_pocket_comparison")
    plot_environments(rows)
    include(joinpath(@__DIR__, "plot_tcst_monte_carlo.jl"))
    plot_tcst_monte_carlo()
end
