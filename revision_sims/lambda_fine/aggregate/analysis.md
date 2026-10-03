# Fine target-clarity decay sweep

## Scope

The complete fine sweep contains 22 matched deterministic simulations: one
adaptive-ergodic run at each of 11 target-clarity decay rates in the static
half-domain and moving-pocket environments. All runs use the same seed, mission
duration, initial condition, and estimator configuration.

The tested values are `0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
0.045, 0.05`.

`own-target deficit` evaluates each run against the target generated with that
run's decay rate. `fixed-target deficit` evaluates all trajectories against the
same ground-truth target generated with `lambda_cd = 0.25`.

## Half-domain results

| lambda_cd | Wind RMSE | Own-target deficit | Fixed-target deficit | In target (%) |
|---:|---:|---:|---:|---:|
| 0 | 1.67070 | 0.16525 | 0.08619 | 46.66 |
| 0.005 | 1.63124 | 0.12447 | 0.07848 | 74.86 |
| 0.01 | 1.62013 | 0.09463 | 0.07743 | 85.14 |
| 0.015 | 1.91812 | 0.08518 | 0.08372 | 85.49 |
| 0.02 | 1.97558 | 0.08286 | 0.08286 | 87.33 |
| 0.025 | 2.10065 | 0.08669 | 0.08669 | 91.96 |
| 0.03 | 2.21933 | 0.09014 | 0.09014 | 92.19 |
| 0.035 | 2.28730 | 0.08939 | 0.08939 | 92.65 |
| 0.04 | 2.41800 | 0.09026 | 0.09026 | 92.78 |
| 0.045 | 2.46858 | 0.09158 | 0.09158 | 92.89 |
| 0.05 | 2.53546 | 0.09317 | 0.09317 | 92.84 |

The fine grid reveals a narrow, beneficial low-selectivity regime that the
coarse sweep missed. Relative to the non-adaptive limit (`lambda_cd = 0`), the
`lambda_cd = 0.01` run:

- reduces global RMSE by 3.0%;
- reduces fixed-target deficit by 10.2%; and
- increases in-target sampling by 38.48 percentage points.

Thus mild adaptation is a Pareto improvement in this static environment. It
rejects the conclusion that broad search is always best for complete coverage.
Instead, a weak target bias can steer effort away from the irrelevant half
without collapsing exploration within the interesting half.

The useful regime is narrow. At `lambda_cd >= 0.015`, global RMSE worsens. At
`lambda_cd = 0.025`, fixed-target deficit is already slightly worse than the
non-adaptive limit. Beyond approximately 0.03, in-target sampling saturates near
92--93% while RMSE and fixed-target deficit continue to degrade. This is the
onset of over-concentration.

The oracle half-domain transect from the full comparison has fixed-target
deficit 0.07680. The adaptive `lambda_cd = 0.01` run reaches 0.07743, only about
0.8% higher, while retaining much lower global RMSE (1.620 versus 3.423). This
is strong evidence that mild adaptation approaches the target-region coverage
of the oracle spatial baseline without abandoning the full field.

## Moving-pocket results

| lambda_cd | Wind RMSE | Own-target deficit | Fixed-target deficit | In target (%) |
|---:|---:|---:|---:|---:|
| 0 | 1.54739 | 0.16562 | 0.00825 | 10.97 |
| 0.005 | 1.61867 | 0.12149 | 0.00855 | 16.36 |
| 0.01 | 1.71895 | 0.08422 | 0.00868 | 22.16 |
| 0.015 | 1.76459 | 0.05942 | 0.00932 | 29.21 |
| 0.02 | 1.78550 | 0.04236 | 0.00858 | 31.27 |
| 0.025 | 1.90155 | 0.03767 | 0.00888 | 38.79 |
| 0.03 | 1.85202 | 0.03362 | 0.00926 | 42.58 |
| 0.035 | 2.00141 | 0.03223 | 0.00965 | 44.16 |
| 0.04 | 1.99438 | 0.02947 | 0.00968 | 41.35 |
| 0.045 | 2.10385 | 0.02700 | 0.00958 | 48.92 |
| 0.05 | 2.11855 | 0.02637 | 0.01007 | 46.64 |

Unlike the half-domain case, no positive decay rate improves the fixed-target
deficit or global RMSE relative to zero. Every increase in targeted sampling is
therefore purchased with some loss of complete dynamic-field coverage.

The response is gradual below 0.03 rather than having one dominant optimum.
For example, `lambda_cd = 0.02` raises in-target sampling from 10.97% to 31.27%
while increasing RMSE by 15.4% and fixed-target deficit by only 3.9%. This is a
reasonable compromise if target acquisition is valuable, but it is not a
Pareto improvement. At 0.045, in-target sampling reaches 48.92%, with a 36.0%
RMSE penalty and a 16.1% fixed-deficit penalty.

The steadily decreasing own-target deficit should not be interpreted as better
complete coverage: the target itself becomes narrower as the decay rate rises.
The fixed-target metric correctly exposes the coverage cost.

## Interpretation

The effect of target selectivity is environment dependent:

- In a static environment with one extended interesting region, mild adaptation
  can improve relevant sampling, global reconstruction, and complete
  interesting-region clarity simultaneously.
- In a dynamic multi-pocket environment, adaptation primarily reallocates time
  toward already identified pockets. Target acquisition improves, but complete
  coverage and reconstruction degrade.
- In both cases, excessive selectivity eventually produces over-concentration;
  the transition occurs at much smaller decay rates than the original paper
  setting of 0.25.

The strongest defensible conclusion is not that adaptation universally
outperforms broad search. It is that target-clarity decay provides an
interpretable control over sensing allocation, with a low-selectivity regime
that can be beneficial and a high-selectivity regime that exposes the
structural discovery limitation of mean-driven target generation.

## Statistical limitation

These are matched single-seed simulations. The narrow half-domain optimum near
0.01 and the local fluctuations in the moving-pocket curves should be treated
as mechanistic results until repeated over multiple measurement-noise seeds.
