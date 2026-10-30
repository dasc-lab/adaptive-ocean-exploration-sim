# Half-domain audit, 27 September 2026

Source: results_half_domain/20260925_130353. No simulations or controllers were modified or rerun. Run `julia --project=. revision_sims/audit_20260927/audit_half_domain.jl` to reproduce the saved-data checks.

## Verified metric calculation on saved snapshots

The audit independently builds the polygon mask and static truth target (0.95 west of x=0.8 km, 0.95 exp(-9) east, zero outside polygon). For every strategy, spatial mean max(q_true - q_achieved, 0) matches saved scalar deficits at corresponding filter timestamps, maximum discrepancy 8.33e-17. This validates scoring of the 300 stored clarity snapshots, not the underlying filter covariance or all unsaved maps. Snapshot averages differ slightly from mission averages over 2881 filter entries and must not be interchanged.

## Actual target-region coverage

Fraction of the 472 in-polygon western grid points within 100 m of any saved trajectory position during the entire mission:

| Planner | Uniform target | Estimated adaptive target | Ground-truth target |
|---|---:|---:|---:|
| BB-IPP | 83.26% | 12.92% | 84.11% |
| Ergodic | 69.70% | 25.42% | 73.09% |

100 m is an illustrative spatial-coverage threshold, not a sensor radius or GP cutoff. It is cumulative coverage and does not measure revisit quality or temporal freshness. This verifies spatial concentration at this scale rather than merely inferring it from wind RMSE. Measurement concentration is a different metric from position residence: adaptive BB-IPP is physically west of the split for 99.71% of stored path points; the noisy near-rated measurement fraction is 95.42%.

At saved snapshots, about 67% of western cell-time pairs for each estimated-target adaptive strategy have estimated target <= achieved clarity while true target > achieved clarity. This makes estimated deficit vanish in those cells despite true shortfall. It directly affects the ergodic deficit demand; adaptive BB-IPP uses target clarity rather than deficit as reward input. Diagnostic target maps in uniform runs are logged estimates, not planning targets.

## The oracle comparison matters

Mission-average true deficit:

| Planner | Uniform target | Estimated adaptive target | Ground-truth target |
|---|---:|---:|---:|
| BB-IPP | 0.109034 | 0.116225 | 0.097326 |
| Ergodic | 0.104905 | 0.120641 | 0.093344 |

Ground-truth targeting reduces deficit relative to uniform targets by 10.74% (BB-IPP) and 11.02% (ergodic). Thus these data do not show that prioritizing the important half is intrinsically ineffective. The failure is specific to estimated-target closed-loop runs, subject to the filter audit below.

## Confirmed timing inconsistency; effect unresolved

run_half_domain_sim.jl constructs STGPKFProblem with dt_min=2.5/60 (2.5 s). In src/simulator_ST.jl:459-466, prediction occurs only inside the measurement-fusion conditional and only one fixed-time prediction is applied per fusion. Saved adaptive ergodic metric timestamps contain 2880 intervals, median 7.5 s (minimum approximately 5 s), spanning 21597.5 seconds. The corresponding 2880 prediction steps advance the configured model by 7200 seconds. Therefore the temporal model is not propagated over actual elapsed mission time. The correction-before-prediction order and treatment of batched measurements also need auditing for consistent timestamps. Floating-point threshold scheduling appears to contribute to the 7.5 s fusion intervals instead of the configured 5 s cadence.

This affects estimator mean/covariance and closed-loop decisions. Even a static truth field is processed with a finite temporal kernel (Lt=45 min). The current results cannot establish the effect of this inconsistency on strategy rankings. Fix and validate time handling, then rerun controlled comparisons before making final performance claims. Equal code across strategies does not guarantee equal impact.

## Objective confound

Adaptive BB-IPP feeds target clarity through a nearest-high-target attraction reward; uniform BB-IPP feeds remaining clarity demand through the same attraction transform. Thus BB-IPP comparison changes more than target information. For a clean ablation, compare uniform, estimated, and true targets using the same deficit-based reward construction. The ergodic variants already share a demand transformation, but finite-horizon/myopic control, entire-history occupancy, estimator misspecification, and feedback still prevent any general deficit guarantee.

## Recommended sequence

1. Correct filter prediction/fusion timestamp handling and validate on an analytically checkable single-point case.
2. Recompute/rerun the same half-domain comparisons with unchanged other settings and explicit run provenance.
3. Present uniform -> estimated target -> true target, with common true deficit, western achieved clarity, spatial coverage, and target underestimation.
4. Use a matched deficit-based BB-IPP ablation to separate target adaptation from objective changes.

Do not dismiss the PI concern with a generic exploration/exploitation argument. The saved-data mechanism is supported, and an actual estimator implementation issue remains to be corrected and quantified.
