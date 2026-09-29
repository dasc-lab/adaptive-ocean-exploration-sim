#!/bin/bash
#SBATCH --job-name=stgp-mc-array
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --array=0-8                   # 9 tasks total (3 Ls * 3 Lt)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
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

prepare_revision_job "results_mc"

# 1. Define length scale parameter spaces
LS_VALS=("0.2" "0.75" "1.0")
LT_VALS=("30.0" "75.0" "120.0")

# 2. Build length scale combinations (9 total)
LS_ARR=()
LT_ARR=()

for ls in "${LS_VALS[@]}"; do
    for lt in "${LT_VALS[@]}"; do
        LS_ARR+=("$ls")
        LT_ARR+=("$lt")
    done
done

# 3. Map Task ID (0-8) to length scale combination
LS="${LS_ARR[$SLURM_ARRAY_TASK_ID]}"
LT="${LT_ARR[$SLURM_ARRAY_TASK_ID]}"

# Sweep all wind speeds within each length-scale job
W_RATED_SWEEP="0.0,0.5,1.0,1.5"

NUM_MC=30
BASE_SEED=1234

echo "Array Task ID ${SLURM_ARRAY_TASK_ID}: Running Ls = ${LS}, Lt = ${LT} across W = [${W_RATED_SWEEP}]"

# 5. Run Julia script
julia --project="$REPO_ROOT" "$REPO_ROOT/revision_sims/run_monte_carlo_comparison.jl" \
    --nworkers "$NWORKERS" \
    --num_mc "$NUM_MC" \
    --base_seed "$BASE_SEED" \
    --w_rated "$W_RATED_SWEEP" \
    --ls "$LS" \
    --lt "$LT" \
    --strategies transect,ergo_nonadaptive,ergo_adaptive,bb_ipp_nonadaptive,bb_ipp_adaptive,ergo_ground_truth,bb_ipp_ground_truth \
    --outdir "$OUT_DIR" \
    --srcdir "$REPO_ROOT/src"
