#!/usr/bin/env bash
#SBATCH --job-name=t32-dev32-p0fix
#SBATCH --partition=gre
#SBATCH --nodelist=gpu22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/t3_visual_ref/20261123_t32_dev32_p0fix/logs/slurm-%j.log
set -euo pipefail

export BAILIAN_CSV=/guisongxia01/pangchao/wangyihan/wyh/百炼-apiKey.csv
export OUT_DIR="${SLURM_SUBMIT_DIR}/outputs/t3_visual_ref/20261123_t32_dev32_p0fix"
exec bash "${SLURM_SUBMIT_DIR}/experiments/t3_visual_ref/run_t32_dev32.sbatch" formal
