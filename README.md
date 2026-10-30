# Adaptive ocean exploration simulation

A Julia package for composing environmental exploration experiments from data
sources, sensors, vehicles, estimators, planners and controllers.

The package name remains `Simulator`. New interfaces use metres, seconds, m/s,
watts, and battery Wh. Historical modules retain their original units.

## Quick start

From the repository root, using the existing Julia environment:

```sh
julia --project=. examples/minimal_survey.jl
julia --project=. projects/modular_demo/experiments/compare_sources.jl
julia --project=. -e 'include("test/runtests.jl")'
```

The examples need no external datasets and write no files by default. See
`projects/modular_demo/configs/default.jl` to select components. See
`docs/architecture.md` for extension methods, timing, units and migration limits.

## Layout

- `src/interfaces.jl`: component contracts and shared data types.
- `src/domains/`, `fields/`, `sensors/`, `vehicles/`: environment and sensing.
- `src/estimators/`, `planners/`, `controllers/`: estimation and decisions.
- `src/simulation/`: experiment composition and scheduling.
- `src/metrics/`, `visualization/`: independent evaluation and result export.
- `src/legacy/`: historical APIs and Jordan Lake-specific implementations.
- `projects/`: study configurations, experiments and analysis.
- `examples/`: small runnable demonstrations.
- `test/fixtures/`: small tracked reference datasets.
- `data/`, `outputs/`: ignored local datasets and generated artifacts.
- `color-scheme/`: shared plotting palette submodule.

`projects/jordan_lake/` retains the archived research workspace. Compatibility
includes at old src paths keep its runners and notebooks working. The new runner
currently includes a numerical GP adapter, waypoint/uncertainty planners and a
velocity-driven vehicle; other historical algorithms remain available through
their existing modules while adapters are migrated incrementally.

## Data and outputs

Keep large datasets, cached results, plots and videos outside tracked source.
The data/ and outputs/ directories ignore everything except their README files;
small test fixtures remain tracked. These rules do not remove the existing
Jordan Lake archive or rewrite repository history. No Git LFS is introduced.
Record data sources/checksums in project docs and use explicit export commands
when saving a result. Existing Julia dependencies remain in Project.toml for
legacy compatibility.
