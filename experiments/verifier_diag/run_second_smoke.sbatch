#!/usr/bin/env bash
#SBATCH --job-name=second-masks-smoke
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=.runtime/second-smoke-%j.out

set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/experiments/verifier_diag/run_second_full.sbatch" "$@" --profile smoke
