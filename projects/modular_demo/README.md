# Modular demonstration

This study demonstrates the component API with a scalar field on a 100 m × 80 m
rectangle. `configs/default.jl` selects the field, sensor, vehicle, GP, planner,
controller, and timing. `experiments/compare_sources.jl` swaps analytic and
sampled-array fields and fixed/adaptive planners. Its sampled field is synthetic
data used to exercise the dataset adapter, not a real observation dataset.

From the repository root:

```sh
julia --project=. examples/minimal_survey.jl
julia --project=. projects/modular_demo/experiments/compare_sources.jl
julia --project=. examples/minimal_survey.jl outputs/modular_demo/trajectory.csv
```

The first two commands write no results. The third explicitly exports a local
trajectory CSV. `analysis/summary.jl` illustrates independent result analysis.
The minimal example also replays its observations through a new estimator.

No large dataset download is required. The tracked fixture under test/fixtures/
shows a small recorded-observation format. Future real datasets should have
provenance, value units, coordinate frame, time origin, download instructions,
and checksums recorded in their study README.
