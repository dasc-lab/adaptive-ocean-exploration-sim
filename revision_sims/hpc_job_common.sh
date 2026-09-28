#!/bin/bash
# Sourced by revision Slurm jobs; run from an existing cluster checkout.
prepare_revision_job() {
    local result_group="$1"
    [[ -f "$REPO_ROOT/Project.toml" && -d "$REPO_ROOT/src" ]] || {
        echo "Invalid REPO_ROOT: $REPO_ROOT" >&2; return 1;
    }
    local allocated_cpus="${SLURM_CPUS_PER_TASK:-1}"
    [[ "$allocated_cpus" =~ ^[1-9][0-9]*$ ]] || return 1
    # Reserve a CPU for the coordinator. A one-CPU allocation runs serially.
    NWORKERS=$((allocated_cpus - 1))
    export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export JULIA_PROJECT="$REPO_ROOT"

    TURBO_ROOT="${TURBO_ROOT:-/nfs/turbo/coe-corelab/kmgovind/jordan-lake-sims}"
    [[ "$TURBO_ROOT" = /* && -d "$(dirname "$TURBO_ROOT")" ]] || {
        echo "Turbo parent unavailable or path not absolute: $TURBO_ROOT" >&2; return 1;
    }
    local job_key="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Missing Slurm job ID}}"
    OUT_DIR="$TURBO_ROOT/$result_group/slurm_$job_key"
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        OUT_DIR="$OUT_DIR/task_$SLURM_ARRAY_TASK_ID"
    fi
    # Preserve previous results even if a job is requeued.
    OUT_DIR="$OUT_DIR/attempt_${SLURM_RESTART_COUNT:-0}"
    mkdir -p "$OUT_DIR"
    [[ -w "$OUT_DIR" ]] || { echo "Cannot write $OUT_DIR" >&2; return 1; }
    # Slurm bootstrap logs stay in the submission directory; full job output is
    # also written to Turbo, including Julia errors and worker diagnostics.
    exec > >(tee -a "$OUT_DIR/job.log") 2>&1
    echo "Repository: $REPO_ROOT"
    echo "Turbo output: $OUT_DIR"
    echo "Slurm job: $job_key; workers: $NWORKERS; Julia/BLAS threads: 1"
    module purge
    module load julia/1.11.4
    cd "$REPO_ROOT"
    julia --version
    # Runner snapshots preserve source; this records checkout status as well.
    if command -v git >/dev/null && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git rev-parse HEAD > "$OUT_DIR/git_commit.txt"
        git status --short > "$OUT_DIR/git_status.txt"
    fi
}
