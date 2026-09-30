# Target-clarity decay sweep

The target map is

```text
q_target(p,t) = 0.95 exp(-lambda_cd (w_hat(p,t) - w_rated)^2).
```

The sweep uses `lambda_cd = 0, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0`
for both the static half-domain and moving-pocket environments. The existing
paper value is 0.25. All other parameters and the measurement-noise seed are
held fixed. At zero, the array job runs both `ergo_adaptive` and
`ergo_nonadaptive`; their trajectories and metrics should agree up to
floating-point noise because the adaptive target becomes spatially constant.

Submit from the repository root or `revision_sims`:

```bash
sbatch revision_sims/run_lambda_cd_sweep.sbatch
```

The 16 array tasks are the Cartesian product of two environments and eight
decay rates. Results are written under
`results_lambda_cd_sweep/slurm_<jobid>/task_<array-index>/attempt_0/` in the
configured Turbo root. For a short pipeline check, set `DURATION_MINUTES`:

```bash
DURATION_MINUTES=0.25 sbatch revision_sims/run_lambda_cd_sweep.sbatch
```

After retrieving the complete job directory, aggregate and plot it with:

```bash
julia --project=. revision_sims/aggregate_lambda_cd_sweep.jl PATH_TO_SLURM_JOB
```

The aggregator writes a combined CSV, metric-sweep plots, and collected-
measurement histograms as PDF and PNG files. Histogram legend labels report
the empirical measurement mean and standard deviation for every decay rate and
for the explicit non-adaptive reference. It also prints the lambda-zero
differences between the adaptive and explicit non-adaptive runs.
