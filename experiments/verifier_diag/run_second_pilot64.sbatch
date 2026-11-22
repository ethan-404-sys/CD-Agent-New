#!/usr/bin/env bash
#SBATCH --job-name=second-masks-pilot64
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=.runtime/second-pilot64-%j.out

set -euo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/experiments/verifier_diag/run_second_full.sbatch" "$@" --profile pilot64
