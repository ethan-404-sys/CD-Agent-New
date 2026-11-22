#!/usr/bin/env bash
#SBATCH --job-name=initial-mask-check
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/core-check-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/core-check-%j.err
set -euo pipefail
ROOT=/guisongxia01/pangchao/wangyihan
REPO=$ROOT/CD-Agent
OUT=${VERIFIER_DATASET_OUT:-$REPO/outputs/verifier_init_masks/verifier_v1_600}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cpu=$ROOT/conda_envs/exp5-env/bin/python
cd "$REPO"
"$cpu" -B experiments/verifier_init_masks/initialize_verifier_masks.py regression --output "$OUT"
"$cpu" -B experiments/verifier_init_masks/initialize_verifier_masks.py check-failure --output "$OUT"
