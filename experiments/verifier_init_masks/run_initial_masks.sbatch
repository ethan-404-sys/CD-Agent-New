#!/usr/bin/env bash
#SBATCH --job-name=initial-masks600
#SBATCH --partition=gre
#SBATCH --exclude=gpu10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/core-generate-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/core-generate-%j.err
set -euo pipefail
ROOT=/guisongxia01/pangchao/wangyihan
REPO=$ROOT/CD-Agent
OUT=${VERIFIER_DATASET_OUT:-$REPO/outputs/verifier_init_masks/verifier_v1_600}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$OUT/runtime" MPLCONFIGDIR="$OUT/runtime/matplotlib"
mkdir -p "$TMPDIR" "$OUT/logs"
started=$SECONDS
finish() {
    code=$?; trap - EXIT
    QQ_SUB="Initial masks $SLURM_JOB_ID" QQ_BODY="status: exit=$code; elapsed: $((SECONDS-started))s; output: $OUT; result: inspect dataset_report.json and logs/core-generate-$SLURM_JOB_ID.out" "$ROOT/.local/bin/wangyihan-send-qq" || true
    exit "$code"
}
trap finish EXIT
cd "$REPO"
cpu=$ROOT/conda_envs/exp5-env/bin/python
sam=$ROOT/conda_envs/omniovcd-env/bin/python
dynamic=$ROOT/conda_envs/ovcd-de-env/bin/python
script=$REPO/experiments/verifier_init_masks/initialize_verifier_masks.py
test -f "$OUT/logs/frozen_regression.json"
test -f "$OUT/logs/failure_fixture_qc.json"
# Exercise the observed failure first, then every real method/dataset smoke.
"$dynamic" -B "$script" infer --output "$OUT" --method dynamic_earth_mci --source-id levir_train_69
for method in dynamic_earth_mci omniovcd rea_ovcd; do
    interpreter=$sam
    if [[ "$method" == dynamic_earth_mci ]]; then interpreter=$dynamic; fi
    "$interpreter" -B "$script" infer --output "$OUT" --method "$method" --smoke
done
for method in dynamic_earth_mci omniovcd rea_ovcd; do
    interpreter=$sam
    if [[ "$method" == dynamic_earth_mci ]]; then interpreter=$dynamic; fi
    "$interpreter" -B "$script" infer --output "$OUT" --method "$method"
done
"$cpu" -B "$script" qc --output "$OUT"
"$cpu" -B "$script" export --output "$OUT"
