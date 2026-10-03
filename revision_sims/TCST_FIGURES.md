# Reproducing the TCST simulation figures

From the simulation repository, run:

```sh
julia --project=. revision_sims/plot_tcst_simulation_figures.jl
```

The script reads the existing `lambda_results/aggregate/lambda_cd_sweep.csv`
and `lambda_fine/aggregate/lambda_cd_sweep.csv`, checks overlapping settings,
and loads saved `environment.jld2` files for the environment snapshots.
It does not rerun any simulations. All five plots included in
`Sections/simulation.tex` are regenerated as vector PDFs and high-resolution PNGs.

| Manuscript figure | Output |
| --- | --- |
| Target-clarity decay sensitivity | `lambda_cd_sweep.pdf` |
| Half-domain strategy comparison | `revision_half_domain_comparison.pdf` |
| Moving-pocket strategy comparison | `revision_moving_pocket_comparison.pdf` |
| Static field and moving-pocket snapshots | `simulation_environments.pdf` |
| Monte Carlo environmental sensitivity | `monte_carlo_environment_sensitivity.pdf` |

The default output directory is the manuscript's `Figures` directory. Alternate
output and results directories can be supplied as positional arguments:

```sh
julia --project=. revision_sims/plot_tcst_simulation_figures.jl OUTPUT_DIR RESULTS_ROOT
```

`RESULTS_ROOT` must contain the two aggregate directories above. Source paths in
the aggregate files must point to the saved runs on the current machine.

Styling comes directly from `color-scheme/julia/Fieldline.jl` and
`color-scheme/palette.yaml`: categorical colors, cividis field maps, and stroke
weights. The publication script overrides figure/panel/legend backgrounds to
white and removes grids. Typography matches the other manuscript plots:
Computer Modern (Makie's native TeX fonts), line width 2, guide and tick sizes
14, legend size 10, and title size 18. Canvas sizes accommodate these settings;
the manuscript scales the PDFs to its full text width. Ground-truth methods
use hatched bars; colors and method order are consistent between environments.

The script also writes `planner_sampling_concentration.csv`, a descriptive
check based on the actual measurement positions in 100 m cells. Effective cell
count is the exponential of occupancy entropy; the top-ten share is the fraction
of measurements in the ten most-sampled cells. These diagnostics support the
qualitative planner discussion and are not treated as multi-seed statistics.

The Julia script is the canonical generator. Earlier Python plotting scripts
in the manuscript's Figures directory were development helpers.

## Monte Carlo data and replacement

`plot_tcst_monte_carlo.jl` can also be run independently. It uses the same theme
and accepts the same output/results arguments. The default input is
`revision_sims/data_results/results_mc`, with nine `ls_*_lt_*` directories.
It selects the latest summarized batch per condition, requires exactly one
summary for each of the six retained strategies and four rated wind speeds,
and verifies the presence of all 30 trial seeds (1234--1263) for every group.
The current batch is September 21, 2026: 216 groups and 6,480 trial files.
It is a legacy exploratory result, not numerically comparable with the corrected
six-hour experiments. The draft and its figure caption explicitly flag this.

The plot shows adaptive versus uniform-target changes within each planner.
Ergodic uses the same cobalt blue and BB-IPP the same vermilion orange as the
two-series sensitivity plots. These colors are selected by planner, not by
the eight-strategy bar-chart ordering, which previously assigned low-contrast
yellow to adaptive ergodic control. The white background, Computer Modern
fonts, and axis sizing are unchanged. Faint condition curves and lighter
shading distinguish the environmental range from the marked mean curve.
The envelope is the range across nine environmental/model settings, not a
confidence interval. The condition summary retains the source RMSE/deficit SEMs
over seeds. Aggregate gains give equal weight to settings. No measurement-level
significance test or unsupported paired-noise assumption is used.

To use replacement results, first arrange the completed summaries/trials in
the same condition-directory layout, then set `TCST_MC_ROOT` to that directory.
The current raw Slurm tree is not automatically mixed with these batches.
Run the standalone Monte Carlo script, verify the version tags and summaries,
and update the numerical prose/counts and remove the provisional notice only
after confirming the corrected batch. Additional BB-IPP ground-truth runs can
be retained on disk; this adaptation figure uses only the two adaptive/uniform
pairs. The standalone script does not regenerate the other four figures.
