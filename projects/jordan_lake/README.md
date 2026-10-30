# Jordan Lake and legacy exploration experiments

Archived from the repository root as the first step toward modular experiments.
This directory groups the historical research workspace, including synthetic
studies and the standalone BB-IPP demonstration. Archiving does not imply every
experiment uses Jordan Lake field data.

## Contents

- Root `*.ipynb`: early simulation and controller studies, with saved outputs.
- `notebooks/`: later experiments, post-processing, and associated figures.
- `revision_sims/`: comparison runners, Slurm jobs, analysis, logs, results, and
  historical source snapshots.
- `lakedata_sim/`: field-data experiments, CSV data, and visualizations.
- `bb-ipp/`: standalone demonstration and rendered media.
- `results/`, `results_cache/`, `tmp.gif`: historical generated artifacts.
- `ERA5_SSR_Hatteras_2022.nc`: historical solar radiation dataset.
- `visualization_of_jordan_lake_domain.jl`: domain visualization script.
- `Project.toml.bak`: old dependency configuration.
- `README_legacy.md`: original repository objectives and interface notes.

The existing `src/`, `test/`, Julia environment, and plotting palette remain at
the repository root. No source module API was redesigned in this archive pass.
Existing local edits and untracked run files were retained. Generated results
and source snapshots were moved without changing their contents; embedded
historical absolute paths may still refer to the original machine/layout.

## Running archived work

From the repository root, use the shared Julia environment:

```sh
julia --project=. projects/jordan_lake/revision_sims/run_half_domain_sim.jl --nworkers 0
julia --project=. projects/jordan_lake/revision_sims/run_single_env_comparison.jl --nworkers 0
julia --project=. projects/jordan_lake/visualization_of_jordan_lake_domain.jl
```

Runner defaults locate shared `src/` relative to their own files. Slurm jobs
accept submission from the repository root or an archive subdirectory; an
explicit `REPO_ROOT` must refer to the repository root. See the runner-specific
documents under `revision_sims/` for parameters and cluster requirements.

Open notebooks with the working directory set to the notebook's containing
folder and the repository's Julia environment selected. Shared source includes
and existing environment activation calls were adjusted for the new location.
Saved cell outputs are preserved. Legacy notebooks may retain pre-existing
external data paths or dependency/API issues; they were not rerun or modernized.
