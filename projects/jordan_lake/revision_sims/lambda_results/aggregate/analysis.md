# Target-clarity decay sweep: consolidated analysis

## Scope and integrity

- This analysis uses only `slurm_62968554`, the new complete sweep. The older
  `slurm_62796139` job contains 18 ergodic-focused rows and is excluded to avoid
  double counting.
- The new job contains all 120 expected runs: eight decay rates for eight
  half-domain strategies (64 runs) and seven moving-pocket strategies (56).
- Each condition is one matched deterministic run with seed 1234. The sweep
  measures sensitivity, not between-seed uncertainty.
- Strategies whose planning rule does not use `lambda_cd` reproduce identical
  trajectories, RMSE, and measurement distributions across the sweep. Their
  repeated points are controls, not independent replications.

## Comparable metrics

The native ground-truth deficit is evaluated against a target that changes with
`lambda_cd`, so its downward trend is partly definitional. The consolidated CSV
therefore recomputes `reference_deficit` against one fixed ground-truth target
with `lambda_cd = 0.25`. Both quantities are retained and plotted:

- `gt_deficit` measures how well each strategy satisfies the target generated
  by that run's own decay-rate parameter;
- `reference_deficit` holds the target fixed and supports direct comparisons
  across decay rates.

Cross-rate interpretation should emphasize:

1. measurements within one normalized wind-speed unit of rated;
2. global wind-field RMSE;
3. fixed-reference clarity deficit; and
4. the full empirical measurement distributions.

## Half-domain environment

The fixed baselines expose the intended tradeoff. The full transect gives the
lowest global RMSE (1.552), while the oracle half-domain transect gives the
highest in-target fraction (95.68%) and a low reference deficit (0.0768), at the
cost of the worst global RMSE (3.423).

For adaptive ergodic control, `lambda_cd = 0.025` is the clearest balanced knee:

| lambda_cd | Global RMSE | Own-target deficit | Fixed-reference deficit | In target (%) |
|---:|---:|---:|---:|---:|
| 0 | 1.671 | 0.16525 | 0.08619 | 46.66 |
| 0.025 | 2.101 | 0.08669 | 0.08669 | 91.96 |
| 0.05 | 2.535 | 0.09317 | 0.09317 | 92.84 |
| 0.1 | 2.686 | 0.09590 | 0.09590 | 95.21 |
| 0.25 | 2.839 | 0.09425 | 0.09425 | 95.57 |

Moving from zero to 0.025 gains 45.30 percentage points of targeted sampling
for a 25.7% RMSE increase. Moving from 0.025 to 0.25 gains only another 3.61
points while increasing RMSE by a further 35.2%. Above 0.25 the response is
non-monotonic and targeted sampling falls sharply.

Adaptive BB-IPP reacts more abruptly: every positive rate produces roughly
94--95% in-target sampling. Its best fixed-reference deficit is 0.08044 at
`lambda_cd = 0.75`, but 0.025 is almost as good (0.08101) with lower RMSE
(2.672 versus 2.722). Thus 0.025 is also the most economical BB-IPP setting.

The oracle strategies are useful upper-information benchmarks, but they do not
dominate every metric. Oracle ergodic control has the best reference deficit
(0.07163 for every positive rate) while the half-domain transect has the highest
in-target fraction. This reinforces that global reconstruction, target-focused
sampling, and target-region clarity are distinct objectives.

## Moving-pocket environment

The moving case is harder and more strongly non-monotonic. Fixed broad-coverage
baselines have good global RMSE but collect few near-rated samples: transect
RMSE is 1.366 with 7.79% in target; non-adaptive ergodic RMSE is 1.547 with
10.97% in target.

Adaptive ergodic control produces the largest focused-sampling response:

| lambda_cd | Global RMSE | Own-target deficit | Fixed-reference deficit | In target (%) |
|---:|---:|---:|---:|---:|
| 0 | 1.547 | 0.16562 | 0.00825 | 10.97 |
| 0.025 | 1.902 | 0.03772 | 0.00888 | 38.79 |
| 0.05 | 2.119 | 0.02645 | 0.01007 | 46.64 |
| 0.1 | 2.397 | 0.01760 | 0.01048 | 79.53 |
| 0.25 | 1.958 | 0.01031 | 0.01031 | 32.36 |
| 0.5 | 1.860 | 0.00708 | 0.01047 | 20.32 |
| 0.75 | 1.699 | 0.00552 | 0.01004 | 15.23 |
| 1.0 | 1.721 | 0.00445 | 0.00944 | 19.70 |

`lambda_cd = 0.1` is the mission-focus optimum in this run, improving the
in-target fraction by 68.56 percentage points over zero, but with 54.9% higher
global RMSE. The current paper value 0.25 is dominated by 0.025 here: 0.025 has
higher in-target sampling, lower RMSE, and lower reference deficit.

Adaptive BB-IPP is less sensitive once the rate is positive. Its in-target
fraction rises from 29.00% at 0.025 to a maximum of 34.74% at 0.25, while RMSE
stays near 2.3 and reference deficit stays near 0.0097. This is more stable but
does not achieve the 79.53% peak of adaptive ergodic control.

Oracle ergodic control gives the smallest moving-pocket reference deficit
(0.00657 at 0.025) and maintains about 41--48% in-target sampling for positive
rates. Oracle BB-IPP, surprisingly, collects only about 1.6--2.3% in-target
measurements. That result should be explained from its reward and trajectory,
not interpreted as a general failure of ground-truth information.

## Limiting-case discrepancy

Ergodic adaptive and non-adaptive runs are exactly identical at
`lambda_cd = 0` in both environments, supporting the intended limiting-case
argument.

BB-IPP does **not** satisfy the same test. At zero, adaptive BB-IPP differs from
non-adaptive BB-IPP by:

- half-domain: RMSE -0.133, deficit -0.0123, in-target +8.99 points;
- moving pocket: RMSE +0.274, deficit +0.0516, in-target -5.21 points.

The implementation explains this discrepancy. Non-adaptive BB-IPP plans over
the current clarity deficit relative to a uniform target, whereas adaptive
BB-IPP plans over the target-clarity map itself. Setting the target map uniform
does not make these rewards equal. Consequently, BB-IPP should not be used as
evidence for the zero-decay equivalence unless the reward definitions are
aligned and the affected sweep is rerun.

## Recommended paper interpretation

- Treat `lambda_cd` as a selectivity parameter with an explicit tradeoff, not a
  parameter having one universal optimum.
- Use 0.025 as the balanced setting supported across both environments.
- Use 0.1 when maximizing moving-target sample acquisition is the primary aim.
- Do not claim that 0.25 is robustly optimal; it is competitive in the static
  environment but dominated in the moving-pocket adaptive-ergodic run.
- Base cross-rate clarity claims on the fixed-reference deficit, not the native
  deficit whose target changes with the swept parameter.
- Repeat the key rates (0, 0.025, 0.05, 0.1, 0.25) over multiple seeds before
  making inferential or statistical claims.
