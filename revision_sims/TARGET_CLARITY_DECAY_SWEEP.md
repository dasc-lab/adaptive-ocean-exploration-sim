# Target-clarity decay sweep

The target map is

```text
q_target(p,t) = 0.95 exp(-lambda_cd (w_hat(p,t) - w_rated)^2).
```

The sweep uses `lambda_cd = 0, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0`
for both the static half-domain and moving-pocket environments. The existing
paper value is 0.25. All other parameters and the measurement-noise seed are
held fixed. Every decay-rate task runs the complete original strategy set:
eight strategies for the half-domain environment and seven for the moving-
pocket environment. At zero, `ergo_adaptive` and `ergo_nonadaptive` should
agree up to floating-point noise because the adaptive target becomes spatially
constant; the same check applies to the corresponding BB-IPP pair.

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

The aggregator writes one combined CSV plus a separate set of figures for each
environment. For example, the half-domain outputs are
`half_domain_lambda_cd_sweep.*`,
`half_domain_sampling_reconstruction_tradeoff.*`, and
`half_domain_measurement_histograms.*`; the moving-pocket files use the
`moving_pocket_` prefix. Metric figures compare all strategies as functions of
the decay rate and show both the deficit against each run's own target and the
deficit against a common `lambda_cd = 0.25` reference target. Tradeoff figures
use one panel per strategy, and measurement
histograms use strategies as rows and decay rates as columns. Each histogram
panel reports the empirical measurement mean and standard deviation for that
run. The aggregator also prints the lambda-zero differences between the
adaptive and explicit non-adaptive ergodic runs.
