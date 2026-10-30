# How the latest results fit the TCST paper

## Defensible central claim

Online target adaptation consistently changes *where* the vehicle samples. Across all 36 Monte Carlo settings, both adaptive planners increased the fraction of measurements near the rated wind speed. The mean increases were 15.50 percentage points for ergodic control and 14.85 percentage points for BB-IPP.

The study does not support a general claim that adaptation improves every performance metric. Adaptive ergodic control increased global RMSE in 34 of 36 settings and ground-truth clarity deficit in 35 of 36. Adaptive BB-IPP had a mixed clarity-deficit response, improving in 19 settings and worsening in 17. The paper should therefore present adaptation as a tunable shift toward scientifically relevant conditions, with reconstruction and spatial-coverage costs that depend on the environment and planner.

## Recommended simulation narrative

1. **Validate the metric in the half-domain diagnostic.** The restricted half-domain transect lowers ground-truth clarity deficit relative to the full-domain transect. This answers the sanity-check question: the metric responds when coverage is genuinely confined to the entire interesting half.
2. **Show that target-map selectivity matters.** The fine `lambda_cd` sweep recovers uniform coverage at zero. In the static half-domain environment, a mild value near 0.01 improves RMSE, fixed-target deficit, and near-target sampling together. Stronger selectivity overconcentrates the path. In moving pockets, increased near-target sampling comes with reconstruction and clarity costs.
3. **Use the controlled planners to explain mechanism.** Near-target sampling and spatial clarity are different. A vehicle may repeatedly sample one favorable location while leaving other favorable regions uncovered. Ground-truth-target runs separate target-estimation error from the path planner's allocation behavior.
4. **Use the Monte Carlo study for robustness.** Its strongest result is the 36-of-36 increase in targeted sampling for each adaptive planner. Present the RMSE and ground-truth-deficit results as an exploration-exploitation tradeoff rather than secondary evidence that adaptation is universally better.
5. **Use the oracle comparison as a limitation and future-work result.** Ground-truth target maps reduce clarity deficit by 8.61% for ergodic control and 7.81% for BB-IPP relative to estimated adaptive targets, but increase RMSE by about 5% for both. Better target estimation helps satisfy the spatial requirement, but cannot remove the coverage tradeoff.

## Claims to remove or soften

- Do not claim that adaptive strategies generally lower ground-truth clarity deficit.
- Do not claim that adaptive ergodic control generally preserves or improves global RMSE; the corrected sweep shows a small but systematic average reconstruction cost.
- Do not claim that one planner is globally superior. BB-IPP's clarity-deficit response is more variable; ergodic control is more consistent but, at the nominal target selectivity, usually incurs a clarity-deficit increase relative to its uniform target.
- Avoid treating in-target percentage as a proxy for clarity deficit. Use it as a complementary measurement-concentration metric.

## Manuscript changes made

- Updated the Monte Carlo design to seven strategies and 7,560 trials.
- Documented matched field and measurement-noise seeds across strategies.
- Replaced provisional Monte Carlo values with the corrected 36-setting aggregation.
- Added the temporal-scale trend and ground-truth-target diagnostic.
- Replaced the Monte Carlo figure with an environmental-sensitivity plot that shows all nine length-scale combinations and their equally weighted mean.
- Preserved the half-domain and moving-pocket studies as mechanistic examples rather than treating their single runs as statistical evidence.

## Suggested placement

- Put the fine `lambda_cd` sweep before the planner comparison, since it establishes that target-map construction changes the exploration-exploitation balance.
- Put the half-domain and moving-pocket comparisons next, since they make the behaviors visible and address the PI's diagnostic questions.
- End the simulation section with the Monte Carlo environmental-sensitivity study, which establishes which conclusions persist over seeds and correlation scales.
- Restate in the conclusion that adaptation reliably increases targeted sampling, while the target-map weight and planner govern the resulting costs in field reconstruction and spatial clarity.

