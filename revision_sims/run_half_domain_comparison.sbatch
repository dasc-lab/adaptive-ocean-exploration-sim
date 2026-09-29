#!/bin/bash
#SBATCH --job-name=stgp-half-domain
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kmgovind@umich.edu

set -euo pipefail
# Submit from the repository root or revision_sims; REPO_ROOT overrides both.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this file with sbatch}}"
if [[ ! -f "$REPO_ROOT/Project.toml" && -f "$REPO_ROOT/../Project.toml" ]]; then
    REPO_ROOT="$REPO_ROOT/.."
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
source "$REPO_ROOT/revision_sims/hpc_job_common.sh"

prepare_revision_job "results_half_domain"

julia --project="$REPO_ROOT" "$REPO_ROOT/revision_sims/run_half_domain_sim.jl" \
    --nworkers "$NWORKERS" \
    --seed 1234 \
    --duration_minutes "${DURATION_MINUTES:-360}" \
    --w_rated -3.5 --wind_offset 6.0 --ls 0.75 --lt 45.0 \
    --strategies transect,transect_half,bb_ipp_nonadaptive,bb_ipp_adaptive,bb_ipp_ground_truth,ergo_nonadaptive,ergo_adaptive,ergo_ground_truth \
    --outdir "$OUT_DIR" \
    --srcdir "$REPO_ROOT/src"
