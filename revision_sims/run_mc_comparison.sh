#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# Configuration Parameters
# ==========================================
NUM_MC=30
BASE_SEED=1234

# Defined as comma-separated sweeps
W_RATED="0.0,0.5,1.0,1.5"
LS="0.2,0.75,1.0"
LT="30.0,75.0,120.0"

STRATEGIES="transect,ergo_nonadaptive,ergo_adaptive,bb_ipp_nonadaptive,bb_ipp_adaptive,ergo_ground_truth,bb_ipp_ground_truth"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results_mc}"
SRC_DIR="$REPO_ROOT/src"
SCRIPT_PATH="$SCRIPT_DIR/run_monte_carlo_comparison.jl"

# Keep one logical CPU for the coordinator process.
AVAILABLE_CPUS=$(nproc)
NWORKERS=$((AVAILABLE_CPUS > 1 ? AVAILABLE_CPUS - 1 : 0))
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ==========================================
# Execution Setup
# ==========================================
mkdir -p "$OUT_DIR"

echo "==> Launching Monte Carlo simulation sweep across $NWORKERS workers..."
julia --project="$REPO_ROOT" "$SCRIPT_PATH" \
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
