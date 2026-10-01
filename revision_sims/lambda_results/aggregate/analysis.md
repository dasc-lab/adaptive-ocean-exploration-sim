# Target-clarity decay sweep: analysis

## Data integrity

- All 16 array tasks completed: eight decay rates in each of the static
  half-domain and moving-pocket environments.
- No failures were found in the saved job logs.
- At `lambda_cd = 0`, the adaptive and explicit non-adaptive ergodic runs are
  numerically identical in both environments for RMSE, clarity deficit, and
  in-target measurement percentage. This confirms the limiting-case argument.
- These are matched deterministic runs with one measurement-noise seed, not
  Monte Carlo estimates. Small differences and apparent optima should not yet
  be interpreted as statistically robust.

## Metric interpretation

The target-clarity map changes with `lambda_cd`; consequently, the native
ground-truth clarity deficit has a different target at every sweep point and is
not directly comparable across decay rates. The aggregate CSV and sensitivity
plot therefore include `reference_deficit`, which evaluates every saved
trajectory against the same ground-truth target generated with
`lambda_cd = 0.25`.

The most interpretable cross-sweep metrics are therefore:

1. percentage of measurements within one normalized wind-speed unit of rated;
2. global wind-field RMSE;
3. deficit against the fixed `lambda_cd = 0.25` target; and
4. the complete empirical measurement distributions.

## Static half-domain environment

The response has a broad useful range from approximately `0.025` to `0.25`:

| lambda_cd | In target (%) | Global RMSE | Fixed-reference deficit |
|---:|---:|---:|---:|
| 0 | 46.66 | 1.671 | 0.0862 |
| 0.025 | 91.96 | 2.101 | 0.0867 |
| 0.05 | 92.84 | 2.535 | 0.0932 |
| 0.1 | 95.21 | 2.686 | 0.0959 |
| 0.25 | 95.57 | 2.839 | 0.0943 |

`lambda_cd = 0.025` is the clearest knee point. Relative to the non-adaptive
limit, it raises the in-target fraction by 45.30 percentage points while
increasing global RMSE by about 25.7%. Increasing the decay rate from `0.025`
to `0.25` gains only another 3.61 percentage points in target-focused sampling,
but increases RMSE by a further 35.2% relative to the `0.025` run.

The histogram panels explain the abrupt loss above `0.25`. At `0.1` and `0.25`,
the collected measurements form a narrow mode around rated wind. At `0.5` and
above, a second mode near the uninteresting eastern value reappears. Thus an
excessively sharp target does not simply make the controller more selective;
it changes the closed-loop trajectory enough to lose sustained concentration.

## Moving-pocket environment

The moving-pocket response is more sharply tuned and strongly non-monotonic:

| lambda_cd | In target (%) | Global RMSE | Fixed-reference deficit |
|---:|---:|---:|---:|
| 0 | 10.97 | 1.547 | 0.00825 |
| 0.025 | 38.79 | 1.902 | 0.00888 |
| 0.05 | 46.64 | 2.119 | 0.01007 |
| 0.1 | 79.53 | 2.397 | 0.01048 |
| 0.25 | 32.36 | 1.958 | 0.01031 |

`lambda_cd = 0.1` maximizes mission-focused sampling, improving the in-target
fraction by 68.56 percentage points over the non-adaptive limit, at the cost of
about 54.9% higher global RMSE. Its histogram is tightly concentrated near the
rated value. This indicates successful tracking rather than a small change in
the thresholded metric.

The current paper value `0.25` is strictly dominated by `0.025` in this run:
`0.025` has higher in-target sampling (38.79% versus 32.36%), lower global RMSE
(1.902 versus 1.958), and lower fixed-reference deficit (0.00888 versus
0.01031). Values above `0.25` progressively return toward broad, predominantly
off-target sampling.

## Recommended interpretation

There is no environment-independent optimum in these data. The decay rate acts
as a closed-loop selectivity parameter and creates a sampling-versus-global-
reconstruction tradeoff. Moderate positive values consistently improve
mission-focused sampling over the uniform-target limit, but very sharp targets
can degrade tracking.

For the current deterministic evidence:

- `0.025` is a defensible balanced setting and the half-domain knee point.
- `0.1` is the strongest mission-focused setting and is clearly preferred in
  the moving-pocket case when near-rated sampling is the primary objective.
- `0.25` is defensible for the static half-domain result but should not be
  presented as generally robust or optimal.

Before selecting a final paper value, repeat at least the most informative
settings (`0`, `0.025`, `0.05`, `0.1`, and `0.25`) over multiple measurement-
noise seeds. Report median and interquartile range or mean and confidence
interval for in-target percentage, global RMSE, and fixed-reference deficit.
