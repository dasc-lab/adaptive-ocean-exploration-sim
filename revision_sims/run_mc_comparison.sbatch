#!/bin/bash
#SBATCH --job-name=stgp-mc-array
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --array=0-15                  # Index range for 16 environments (4 winds * 2 Ls * 2 Lt)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kmgovind@umich.edu

set -euo pipefail
mkdir -p logs

module purge
module load julia/1.11.4

cd "$SLURM_SUBMIT_DIR"

# 1. Define the parameter spaces
W_VALS=("0.0" "0.5" "1.0" "1.5")
LS_VALS=("0.2" "1.0")
LT_VALS=("30.0" "120.0")

# 2. Dynamically build the combinations into flat arrays
W_ARR=()
LS_ARR=()
LT_ARR=()

for w in "${W_VALS[@]}"; do
    for ls in "${LS_VALS[@]}"; do
        for lt in "${LT_VALS[@]}"; do
            W_ARR+=("$w")
            LS_ARR+=("$ls")
            LT_ARR+=("$lt")
        done
    done
done

# 3. Map Slurm Array Task ID to the specific combination
W_RATED="${W_ARR[$SLURM_ARRAY_TASK_ID]}"
LS="${LS_ARR[$SLURM_ARRAY_TASK_ID]}"
LT="${LT_ARR[$SLURM_ARRAY_TASK_ID]}"

NUM_MC=30
BASE_SEED=1234
# Ensure outputs are separated to prevent plot overwriting
OUT_DIR="results_mc/w_${W_RATED}_ls_${LS}_lt_${LT}"

echo "Array Task ID ${SLURM_ARRAY_TASK_ID}: Running Wind = ${W_RATED}, Ls = ${LS}, Lt = ${LT}"

# 4. Execute the Julia script with the mapped parameters
julia --project run_monte_carlo_comparison_3.jl \
    --nworkers "$SLURM_CPUS_PER_TASK" \
    --num_mc "$NUM_MC" \
    --base_seed "$BASE_SEED" \
    --w_rated "$W_RATED" \
    --ls "$LS" \
    --lt "$LT" \
    --strategies transect,ergo_nonadaptive,ergo_adaptive,bb_ipp \
    --outdir "$OUT_DIR" \
    --srcdir ../src