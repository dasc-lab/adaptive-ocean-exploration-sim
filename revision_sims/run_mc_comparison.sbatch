#!/bin/bash
#SBATCH --job-name=stgp-mc-grid
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kmgovind@umich.edu

set -euo pipefail
# One array element owns one (Ls, Lt, W) condition and runs all seeds and
# strategies for that condition. Full trial histories are discarded after their
# compact per-seed and aggregate statistics are written.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this file with sbatch}}"
if [[ ! -f "$REPO_ROOT/Project.toml" && -f "$REPO_ROOT/../Project.toml" ]]; then
    REPO_ROOT="$REPO_ROOT/.."
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
source "$REPO_ROOT/revision_sims/hpc_job_common.sh"
prepare_revision_job "results_mc"

LS_VALS=("0.2" "0.75" "1.0")
LT_VALS=("30.0" "75.0" "120.0")
W_VALS=("0.0" "0.5" "1.0" "1.5")

# W changes fastest, followed by Lt and Ls: 3 * 3 * 4 = 36 conditions.
idx=$SLURM_ARRAY_TASK_ID
w_idx=$((idx % ${#W_VALS[@]})); idx=$((idx / ${#W_VALS[@]}))
lt_idx=$((idx % ${#LT_VALS[@]})); idx=$((idx / ${#LT_VALS[@]}))
ls_idx=$idx
LS="${LS_VALS[$ls_idx]}"
LT="${LT_VALS[$lt_idx]}"
W_RATED="${W_VALS[$w_idx]}"

NUM_MC=30
BASE_SEED=1234
CONDITION_OUT="$OUT_DIR/ls_${LS}_lt_${LT}_w_${W_RATED}"
mkdir -p "$CONDITION_OUT"

echo "Array task $SLURM_ARRAY_TASK_ID: Ls=$LS Lt=$LT W=$W_RATED seeds=$BASE_SEED:$((BASE_SEED + NUM_MC - 1))"
echo "This condition runs $((NUM_MC * 7)) missions over $NWORKERS workers."

julia --project="$REPO_ROOT" "$REPO_ROOT/revision_sims/run_monte_carlo_comparison.jl" \
    --nworkers "$NWORKERS" \
    --num_mc "$NUM_MC" \
    --base_seed "$BASE_SEED" \
    --w_rated "$W_RATED" \
    --ls "$LS" \
    --lt "$LT" \
    --strategies transect,ergo_nonadaptive,ergo_adaptive,bb_ipp_nonadaptive,bb_ipp_adaptive,ergo_ground_truth,bb_ipp_ground_truth \
    --save_trials false \
    --skip_plots true \
    --outdir "$CONDITION_OUT" \
    --srcdir "$REPO_ROOT/src"
