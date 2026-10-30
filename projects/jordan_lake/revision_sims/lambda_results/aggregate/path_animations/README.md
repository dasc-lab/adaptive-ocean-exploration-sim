# Target-clarity sweep path animations

These 120 MP4s are collected from the complete `slurm_62968554` sweep and are
organized as:

```text
path_animations/
  half_domain/
    lambda_0/
    lambda_0.025/
    lambda_0.05/
    lambda_0.1/
    lambda_0.25/
    lambda_0.5/
    lambda_0.75/
    lambda_1.0/
  moving_pocket/
    <the same eight lambda directories>/
```

Each half-domain directory contains eight strategy animations. Each
moving-pocket directory contains seven. Filenames are the strategy names, so
opening one lambda directory gives a direct comparison of all strategies under
the same target-clarity decay setting.

The files are copies because the shared-drive filesystem does not permit
symbolic links. The original result animations remain under
`lambda_results/slurm_62968554/task_*/attempt_0/.../animations/`.
