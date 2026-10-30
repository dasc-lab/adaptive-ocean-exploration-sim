# Adaptive ocean exploration simulation

Julia modules for exploring environmental fields with autonomous vehicles.
The repository is being separated into reusable code and project experiments.

- `src/`: existing data generation, controllers, estimators, simulation engines,
  geometry, and utilities. Module interfaces and source layout are unchanged.
- `test/`: existing checks for shared code and the Jordan Lake experiment.
- `projects/jordan_lake/`: archived notebooks, experiment runners, field data,
  results, caches, animations, and the original README.
- `Project.toml` / `Manifest.toml`: shared Julia environment.
- `color-scheme/`: shared plotting palette (Git submodule).

See `projects/jordan_lake/README.md` for the archive map and launch commands.
Jordan Lake assumptions still exist in several source modules; extracting those
and deciding module interfaces is the next organization step.
