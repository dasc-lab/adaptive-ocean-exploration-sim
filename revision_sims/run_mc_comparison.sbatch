#!/bin/bash
#SBATCH --job-name=stgp-mc-array
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --array=0-5                   # 6 tasks total (3 Ls * 2 Lt)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kmgovind@umich.edu

set -euo pipefail
mkdir -p logs

module purge
module load julia/1.11.4

cd "$SLURM_SUBMIT_DIR"

# 1. Define length scale parameter spaces
LS_VALS=("0.2" "0.75" "1.0")
LT_VALS=("30.0" "120.0")

# 2. Build length scale combinations (6 total)
LS_ARR=()
LT_ARR=()

for ls in "${LS_VALS[@]}"; do
    for lt in "${LT_VALS[@]}"; do
        LS_ARR+=("$ls")
        LT_ARR+=("$lt")
    done
done

# 3. Map Task ID (0-5) to length scale combination
LS="${LS_ARR[$SLURM_ARRAY_TASK_ID]}"
LT="${LT_ARR[$SLURM_ARRAY_TASK_ID]}"

# Sweep all wind speeds within each length-scale job
W_RATED_SWEEP="0.0,0.5,1.0,1.5"

NUM_MC=30
BASE_SEED=1234

# 4. Direct Turbo NFS storage output directory
TURBO_BASE="/nfs/turbo/coe-corelab/kmgovind/jordan-lake-sims/results_mc"
OUT_DIR="${TURBO_BASE}/ls_${LS}_lt_${LT}"

mkdir -p "$OUT_DIR"

echo "Array Task ID ${SLURM_ARRAY_TASK_ID}: Running Ls = ${LS}, Lt = ${LT} across W = [${W_RATED_SWEEP}]"

# 5. Run Julia script
julia --project run_monte_carlo_comparison.jl \
    --nworkers "$SLURM_CPUS_PER_TASK" \
    --num_mc "$NUM_MC" \
    --base_seed "$BASE_SEED" \
    --w_rated "$W_RATED_SWEEP" \
    --ls "$LS" \
    --lt "$LT" \
    --strategies transect,ergo_nonadaptive,ergo_adaptive,bb_ipp \
    --outdir "$OUT_DIR" \
    --srcdir ../src
