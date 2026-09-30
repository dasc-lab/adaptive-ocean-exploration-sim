#!/bin/bash
#SBATCH --job-name=stgp-lambda-cd
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --time=3:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kmgovind@umich.edu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this file with sbatch}}"
if [[ ! -f "$REPO_ROOT/Project.toml" && -f "$REPO_ROOT/../Project.toml" ]]; then
    REPO_ROOT="$REPO_ROOT/.."
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
source "$REPO_ROOT/revision_sims/hpc_job_common.sh"

LAMBDA_VALUES=(0 0.025 0.05 0.1 0.25 0.5 0.75 1.0)
task_id="${SLURM_ARRAY_TASK_ID:?This job must run as a Slurm array}"
lambda_index=$((task_id % ${#LAMBDA_VALUES[@]}))
environment_index=$((task_id / ${#LAMBDA_VALUES[@]}))
lambda_cd="${LAMBDA_VALUES[$lambda_index]}"

case "$environment_index" in
    0)
        environment="half_domain"
        runner="$REPO_ROOT/revision_sims/run_half_domain_sim.jl"
        ;;
    1)
        environment="moving_pocket"
        runner="$REPO_ROOT/revision_sims/run_single_env_comparison.jl"
        ;;
    *)
        echo "Unexpected environment index: $environment_index" >&2
        exit 1
        ;;
esac

prepare_revision_job "results_lambda_cd_sweep"

strategies="ergo_adaptive"
if [[ "$lambda_cd" == "0" ]]; then
    strategies="ergo_nonadaptive,ergo_adaptive"
fi

echo "Environment: $environment; lambda_cd: $lambda_cd; strategies: $strategies"
julia --project="$REPO_ROOT" "$runner" \
    --nworkers "$NWORKERS" \
    --seed 1234 \
    --duration_minutes "${DURATION_MINUTES:-360}" \
    --w_rated -3.5 --wind_offset 6.0 --ls 0.75 --lt 45.0 \
    --lambda_cd "$lambda_cd" \
    --strategies "$strategies" \
    --animation_seconds "${ANIMATION_SECONDS:-2}" \
    --animation_fps "${ANIMATION_FPS:-2}" \
    --outdir "$OUT_DIR" \
    --srcdir "$REPO_ROOT/src"
