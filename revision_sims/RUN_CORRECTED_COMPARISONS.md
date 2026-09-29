# Corrected revision simulations (28 September 2026)

Run from the repository root with the existing Julia environment. Existing results are preserved. **Rerun all strategies being compared:** the corrected runs are not numerically comparable with the September 25 trajectories or metrics.

## PI sanity check first

```bash
julia --project=. revision_sims/run_half_domain_sim.jl \
  --strategies transect,transect_half --nworkers 2
```

Both missions last six hours and use the same estimator, measurement noise seed, speed/energy policy, starting position, grid, and 0.3 km waypoint spacing. For the half-domain comparison, every strategy starts 0.05 km west of the computed split at y=0.75 km, keeping the common initial position inside the oracle half-transect domain. `transect_half` is an oracle spatial baseline: it knows that the west equal-area portion is interesting. The vertical split is computed from the continuous Jordan Lake polygon (currently x ≈ 0.652547 km), rather than the rectangular estimator grid. Its waypoints and integrated motion are restricted to the intersection of the original mission polygon with x <= split_x - 1e-6 km. A dedicated waypoint column lies another 1e-6 km inside that boundary, so the route covers the full width even though the regular 0.3 km lattice ends at x=0.4 km. Ground truth, estimator grid, and scoring domain remain unchanged. It must not receive an artificially smaller evaluation domain. A post-run assertion verifies every sampled position remains inside its restricted polygon.

It appears in the default comparison, summary CSV/table, plots and `animations/transect_half.mp4`. Compare **ground-truth-target clarity deficit** between the two transects. A large improvement is a hypothesis to test, not an assertion enforced by the code. Near-rated measurement percentage uses noisy measurements, so need not reach 100% even when all positions are in the interesting half.

## Full comparisons

```bash
julia --project=. revision_sims/run_half_domain_sim.jl --nworkers 4
julia --project=. revision_sims/run_single_env_comparison.jl --nworkers 4
```

Default outputs are timestamped directories under `revision_sims/results_half_domain` and `revision_sims/results_single_env`, independent of the shell's working directory. Half-domain defaults to eight strategies; moving pockets defaults to seven. Set `--nworkers 0` for serial execution. All strategies retain their original reward definitions; BB-IPP's target-clarity reward versus ergodic's clarity-demand reward remains an experimental distinction, not a corrected ablation.

For a quick pipeline check (not scientific evidence):

```bash
julia --project=. revision_sims/run_half_domain_sim.jl \
  --nworkers 0 --duration_minutes 0.25 \
  --animation_seconds 1 --animation_fps 2 --outdir /tmp/half-domain-check
```

`--duration_minutes` defaults to 360 and must be a positive multiple of 2.5 seconds. Very short missions retain the original terminal SOC goal and are intended only to check execution.

## HPC submissions and Turbo storage

Push/pull the corrected repository to your cluster checkout, including `src/timed_simulation.jl`, the new sbatch files, and `revision_sims/hpc_job_common.sh`. From the **repository root**, prepare the Julia environment once if necessary:

```bash
module load julia/1.11.4
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Submit the two full six-hour mission comparisons:

```bash
sbatch revision_sims/run_half_domain_comparison.sbatch
sbatch revision_sims/run_single_env_comparison.sbatch
```

Both request one node, eight CPUs, 96 GB total memory, and 12 hours of wall time, using the existing `cvermill0` account and `standard` partition. These are initial resource requests, not a runtime guarantee. Seven single-threaded workers leave one CPU for the coordinator; the eight half-domain strategies are queued over those workers. Julia, OpenBLAS, MKL, and OpenMP threading are capped at one thread per process. All eight half-domain strategies and all seven pocket strategies are explicitly selected.

Default output locations are:

```text
/nfs/turbo/coe-corelab/kmgovind/jordan-lake-sims/
  results_half_domain/slurm_<jobid>/attempt_0/<timestamp>/
  results_single_env/slurm_<jobid>/attempt_0/<timestamp>/
```

Each timestamped directory contains the runner's trial data, metrics, plots, videos, and source snapshot. Its parent `attempt_0` directory contains `job.log`, `git_commit.txt`, and `git_status.txt`. Slurm's bootstrap stdout/stderr also remain in the submission directory as `slurm-*.out` and `slurm-*.err`; no pre-existing `logs` directory is required. Requeues use a new attempt number. Keep the complete job directory when downloading so data and provenance stay together.

The existing Monte Carlo array is also updated:

```bash
sbatch revision_sims/run_mc_comparison.sbatch
```

It retains its 30 seeds, wind-speed sweep, nine length-scale combinations, 16 CPUs, and 96 GB. The wall-time request is 24 hours because the seventh strategy and corrected sequential filter updates make the former 12-hour request too tight. It includes ground-truth BB-IPP as well as the other six strategies, reserves one CPU for the coordinator, and writes fresh output under `results_mc/slurm_<array-jobid>/task_<index>/attempt_0/`. It does not reuse the older unversioned Turbo directories.

The Monte Carlo strategies and metrics match the current pocket comparison, including live STGPKF targets for evaluating nonadaptive strategies, speed-preserving ergodic boundary steering, incremental BB-IPP heading execution, timestamped filter updates, and contemporaneous ground-truth scoring. Each seed uses one deterministic generated truth field and a separate deterministic measurement-noise stream shared across strategies, independent of worker scheduling. The saved `measurement_seed` records that derived stream. Monte Carlo deliberately remains a three-hour randomized-GP experiment with its original SOC endpoint and rated-wind/length-scale sweep; it is not a stochastic replication of the six-hour deterministic pocket field. New trials are tagged `matched-measurement-noise-v1`, and older cached trials are rejected.

The scripts also support submission from `revision_sims` using their basenames. To submit from elsewhere, export the checkout's absolute `REPO_ROOT`. To change the storage destination, export `TURBO_ROOT`; its parent directory must already exist and be accessible. `DURATION_MINUTES` can shorten the two controlled comparisons for a pipeline check, but defaults to 360. The older `run_comparison_script.sbatch` launches the legacy experiment and is not the job for these corrected comparisons.

To retrieve results from your local machine, replace `YOUR_HPC_LOGIN_HOST` and `<jobid>` with your actual SSH login host and the ID printed by `sbatch`:

```bash
rsync -avP "kmgovind@YOUR_HPC_LOGIN_HOST:/nfs/turbo/coe-corelab/kmgovind/jordan-lake-sims/results_half_domain/slurm_<jobid>/" ./half-domain-hpc/
rsync -avP "kmgovind@YOUR_HPC_LOGIN_HOST:/nfs/turbo/coe-corelab/kmgovind/jordan-lake-sims/results_single_env/slurm_<jobid>/" ./pocket-hpc/
```

Shell syntax and mocked launches were validated locally, including repository-root/subdirectory submission, serial fallback, Turbo destination, array indexing, and strategy lists. Cluster module availability, storage permissions, scheduling, and full-run resource usage must be checked on the cluster.

## Corrections applied

- Shared engine for `simulate_known_param` and `simulate_known_transect`, used by half-domain, single-environment and Monte Carlo comparison runners.
- One measurement at each acquisition timestamp, including mission endpoints. No duplicated initial measurement, missing terminal samples, or position/sample index shift.
- At t0: correct the prior with the initial measurement, without predicting into the future. At each later acquisition timestamp: predict exactly 2.5 seconds, then correct with that timestamp's sample.
- Pending samples are fused sequentially, in acquisition order, every five seconds. Controller inputs hold the most recently published posterior between fusion events. Thus fusion cadence controls publication/availability without falsely treating samples from different times as simultaneous. The terminal fusion flushes any remaining samples.
- Integer-step cadence scheduling removes floating-point drift from nominal five-second fusion. Prediction timestep must equal the simulation timestep; incompatible cadences fail explicitly.
- Initial planning uses the actual posterior mean, not an artificial field set everywhere to the rated wind speed.
- Explicit target timestamps replace inferred array-index alignment. Ground-truth metrics use the **metric timestamp**, including in moving fields, rather than the last target-map timestamp.
- Velocity is m/s, coordinates km, simulation timestamps minutes: displacement is `u * dt_minutes * 60 / 1000`. The old `u * dt_minutes / 60` was 3.6 times too small. BB-IPP primitive lengths already used the correct conversion and now agree with vehicle motion.
- One energy update per integration interval; no extra initial battery step. Use actual applied speed after waypoint/boundary limits. Do not renormalize safety-limited velocities back to full speed.
- The simulation uses the same solar day and latitude as the SOC-profile generator (`SoCController.dayOfYear` and `.lat`), rather than a separate hard-coded day 288 versus profile day 91.
- Waypoints advance before computing the next heading; the final approach cannot overshoot. Integrated motion is constrained to the motion polygon and rectangular environment grid.
- Nonadaptive strategies plan against a fixed uniform target, but continue to compute, save, display and score live STGPKF-derived target maps. `est_deficit`, `est_clarity_rmse` and target-map RMSE use that live estimated target. `gt_deficit` continues to use the ground-truth target. The uniform planning target is saved separately as `fixed_planning_target`; the time-varying evaluation maps are `evaluation_target_maps`. Nonadaptive videos label this distinction explicitly. Planning demand still updates with achieved clarity against the fixed uniform target; the initial demand distribution is not frozen.
- Nonzero extra `Q_process` now raises an error instead of being silently ignored: STGPKF process covariance comes from its temporal kernel. The shared-SOC simulator explicitly supports one vehicle instead of silently mishandling multiple vehicles.

## Boundary-change rollback

The September 28 boundary/motion changes have been reverted to the version used by the September 28 morning runs. Ergodic and BB-IPP again use their earlier planner-specific boundary correction, and BB-IPP again executes its incremental heading change. The earlier filter-timing and live-target evaluation fixes and the HPC jobs remain. Results from the reverted boundary experiment should be kept separate from new runs.

## Ergodic speed-preserving boundary steering

Ergodic's existing inward boundary blend now determines direction only: its output is rescaled to the incoming command's magnitude. If outward and inward vectors cancel, the controller selects the inward direction at that same speed. An intentional zero command remains zero. BB-IPP's boundary correction and incremental heading execution are unchanged. The simulator's final hard feasibility limiter remains in place and can still shorten a step that would leave the admissible domain; it is not renormalized after clipping.

## Reproducibility and limitations

New controlled trials save acquisition times/positions, target timestamps, all filter means/clarity maps, all evaluation targets and fixed nonadaptive planning targets, applied speeds, battery history, prediction counts, noise level and timing version. Each controlled run also copies the runner, diagnostics, src directory, Project.toml and Manifest.toml into `source_snapshot`. Expect larger trial files and more computation: exact sequential correction performs an update for each acquisition rather than incorrectly batching different times as simultaneous.

Monte Carlo uses the corrected shared engine and scoring timestamps too. It refuses cached trials with a different/missing timing or target-metric version; use a fresh output directory. Legacy simulation functions outside the three revision comparison runners have not been rewritten and should not be used to generate matched comparisons with these results.

The sharp/static half-domain truth still deliberately differs from the smooth, finite-temporal-scale GP model. Ground-truth deficit measures unmet estimator clarity, not independently calibrated physical accuracy. This implementation correction does not assume or guarantee any strategy ranking.

## Tests

```bash
julia --project=. test/runtests.jl
# or the focused numerical suite:
julia --project=. test/test_revision_timing.jl
```

The numerical suite checks posterior means and covariances against an independent dense Kalman filter, fusion-cadence invariance for a fixed path, terminal flushing, controller timestamps, current-time truth scoring, one energy step per interval, the 1 m/s distance conversion, waypoint arrival and complete west-half route confinement.
