#!/bin/bash
#SBATCH --job-name=stgp-mc-array
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --array=0-3                   # Index range for 4 environments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=01:15:00               # Only needs time for ONE environment (~35 min + buffer)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kmgovind@umich.edu

set -euo pipefail
mkdir -p logs

module purge
module load julia/1.11.4

cd "$SLURM_SUBMIT_DIR"

# Array of rated wind speed environments
W_RATED_ENVIRONMENTS=("0" "-3.0" "0.5" "2.0")

# Map Slurm Array Task ID to specific environment value
W_RATED="${W_RATED_ENVIRONMENTS[$SLURM_ARRAY_TASK_ID]}"
NUM_MC=10
BASE_SEED=1234
OUT_DIR="results_mc/w_rated_${W_RATED}"

echo "Array Task ID ${SLURM_ARRAY_TASK_ID}: Running Rated Wind Speed = ${W_RATED}"

julia --project run_monte_carlo_comparison.jl \
    --nworkers "$SLURM_CPUS_PER_TASK" \
    --num_mc "$NUM_MC" \
    --base_seed "$BASE_SEED" \
    --w_rated "$W_RATED" \
    --strategies transect,ergo_nonadaptive,ergo_adaptive,bb_ipp \
    --outdir "$OUT_DIR" \
    --srcdir ../src