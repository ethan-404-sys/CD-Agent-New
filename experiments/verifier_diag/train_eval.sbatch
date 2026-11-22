#!/usr/bin/env bash
#SBATCH --job-name=hv-verifier-train-eval
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=.runtime/verifier-train-eval-%j.bootstrap.log

# Submit from the repo root via expctl after smoke acceptance. Pass --epochs and
# repeat --manifest for each train/val shard when not using the default manifests.
set -euo pipefail
repo=${SLURM_SUBMIT_DIR:?Submit from the CD-Agent repository root}
cd "$repo"
account=${CD_AGENT_ACCOUNT_ROOT:-$(dirname "$repo")}
envs=${CD_AGENT_CONDA_ENVS:-$account/conda_envs}
day=$(date -u +%Y%m%d)
out="${CD_AGENT_OUTPUTS:-$repo/outputs}/verifier_diag/${day}_sam3-verifier-${SLURM_JOB_ID:?}"
mkdir -p "$out/logs"
log="$out/logs/train-eval.log"
printf 'Train/eval log: %s\n' "$log"
exec >>"$log" 2>&1
started=$SECONDS
notify() {
    local code=$?
    trap - EXIT
    QQ_SUB="SAM 3 HV-Verifier train/eval $SLURM_JOB_ID" \
        QQ_BODY="exit=$code; elapsed=$((SECONDS-started))s; metrics=$out/metrics.json; evidence=$out/viz/index.html; log=$log" \
        "$account/.local/bin/wangyihan-send-qq" || printf 'QQ notification failed\n' >&2
    exit "$code"
}
trap notify EXIT
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
"$envs/omniovcd-env/bin/python" -u -B experiments/verifier_diag/train_verifier.py "$@" --run-dir "$out"
