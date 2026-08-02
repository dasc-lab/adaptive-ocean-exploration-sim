#!/bin/bash
#SBATCH --job-name=stgp-strategy-compare
#SBATCH --account=cvermill0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    # 1 core per strategy (4 strategies)
#SBATCH --mem-per-cpu=12G                    # bumped from 8G: longer sim + deeper BB-IPP tree hold more state
#SBATCH --time=8:00:00                      # bumped from 4h: 2x sim duration + H=20 BB-IPP search
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kmgovind@umich.edu

set -euo pipefail
mkdir -p logs

module purge
module load julia/1.11.4      # adjust to whatever `module avail julia` shows on Great Lakes

# Instantiate once before first real run:
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'

cd "$SLURM_SUBMIT_DIR"

julia --project run_comparison.jl \
    --nworkers 4 \
    --strategies transect,ergo_nonadaptive,ergo_adaptive,bb_ipp \
    --outdir results \
    --srcdir ../src
