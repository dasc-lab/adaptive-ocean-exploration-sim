#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# Configuration Parameters
# ==========================================
NUM_MC=30
BASE_SEED=1234

# Defined as comma-separated sweeps
W_RATED="0.0, 0.5, 1.0, 1.5, 2.0"
LS="0.2, 0.75, 1.5"
LT="30.0, 120.0"

STRATEGIES="transect,ergo_nonadaptive,ergo_adaptive,bb_ipp"
OUT_DIR="./data_results"
SRC_DIR="../src"
SCRIPT_PATH="run_monte_carlo_comparison.jl"

# Use all logical CPU cores available on your Linux system / Slurm node
NWORKERS=$(nproc)

# ==========================================
# Execution Setup
# ==========================================
mkdir -p "$OUT_DIR"

echo "==> Launching Monte Carlo simulation sweep across $NWORKERS workers..."
julia --project "$SCRIPT_PATH" \
    --nworkers "$NWORKERS" \
    --num_mc "$NUM_MC" \
    --base_seed "$BASE_SEED" \
    --w_rated "$W_RATED" \
    --ls "$LS" \
    --lt "$LT" \
    --strategies "$STRATEGIES" \
    --outdir "$OUT_DIR" \
    --srcdir "$SRC_DIR"

echo "==> Sweep finished. Results stored in: $OUT_DIR"
