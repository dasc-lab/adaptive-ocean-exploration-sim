# Aggregated revision simulations (2026-10-03)

This directory separates and aggregates the two completed result sets found in `revision_sims/slurm_files`.

## Source batches

### Corrected Monte Carlo study

- Source: `slurm_62968580`
- Code commit recorded by the runs: `5602fe1d733bafe4a32e8b6f6181f78e0e890e8d`
- Design: 3 spatial length scales x 3 temporal length scales x 4 rated wind speeds x 7 strategies x 30 seeds
- Total: 7,560 three-hour trials
- Strategies: transect; uniform-target, adaptive, and ground-truth-target variants of BB-IPP and ergodic control
- Pairing: each environment seed and measurement-noise seed is shared across strategies within a condition
- Timing metadata checked in the saved trials: timestamped sequential filtering, 2.5 s sampling, and 5 s filter fusion

The `mc` directory contains:

- `mc_summary_all.csv`: all 252 condition-by-strategy summary rows
- `mc_adaptation_effects.csv`: adaptive-minus-uniform comparisons for both planners in all 36 environmental settings
- `mc_oracle_target_effects.csv`: ground-truth-target-minus-adaptive comparisons
- `mc_strategy_overall.csv`: equally weighted summaries across all 36 environmental settings
- `analysis.md`: concise numerical findings used in the manuscript
- `monte_carlo_environment_sensitivity.{pdf,png}`: paper-ready sensitivity figure

### Fine target-clarity-weight sweep

- Source: `slurm_62996784`
- Code commit recorded by the runs: `128eee83cbd81b4733c69e6ad2c952c288a10fca`
- Design: 11 values of `lambda_cd` from 0 to 0.05 in the half-domain and moving-pocket environments
- Total: 22 adaptive-ergodic diagnostic runs

The `lambda` directory contains the combined CSV plus figures for each environment. These runs are deterministic sensitivity diagnostics, so each marker is one run rather than a Monte Carlo confidence estimate.

## Interpretation guardrails

- A higher in-target fraction means the strategy collects more measurements near the rated wind speed. It does not imply lower spatial clarity deficit, because repeated samples can remain concentrated in one portion of the interesting region.
- Ground-truth clarity deficit evaluates spatial coverage of all interesting regions. It is the appropriate common-target comparison between adaptive and non-adaptive strategies.
- Global RMSE evaluates reconstruction over the full domain, including regions that the adaptive objective may intentionally deprioritize.
- The min-max bands in the Monte Carlo sensitivity figure represent variation among the nine length-scale settings. They are not confidence intervals.
- The Monte Carlo generator and estimator use matched spatial and temporal length scales. The study tests environmental sensitivity, not estimator-kernel misspecification.

