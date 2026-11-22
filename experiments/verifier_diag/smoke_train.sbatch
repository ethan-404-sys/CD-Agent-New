#!/usr/bin/env bash
#SBATCH --job-name=hv-verifier-smoke
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=.runtime/verifier-train-%j.bootstrap.log

set -euo pipefail
repo=${SLURM_SUBMIT_DIR:?Submit from the CD-Agent repository root}
cd "$repo"
account=${CD_AGENT_ACCOUNT_ROOT:-$(dirname "$repo")}
envs=${CD_AGENT_CONDA_ENVS:-$account/conda_envs}
out="${CD_AGENT_RUNTIME:-$repo/.runtime}/smoke/verifier-train-${SLURM_JOB_ID:?}"
mkdir -p "$out/logs"
printf 'Smoke log: %s\n' "$out/logs/train.log"
exec >>"$out/logs/train.log" 2>&1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
exec "$envs/omniovcd-env/bin/python" -u -B experiments/verifier_diag/train_verifier.py \
    --smoke "$@" --run-dir "$out"
