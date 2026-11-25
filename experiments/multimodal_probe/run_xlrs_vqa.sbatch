#!/usr/bin/env bash
#SBATCH --job-name=xlrs-vqa
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/multimodal_probe/20261123_xlrs_vqa_probe/logs/slurm-%j.out

set -euo pipefail

repo=${SLURM_SUBMIT_DIR:?}
account=/guisongxia01/pangchao/wangyihan
python=$account/conda_envs/exp5-env/bin/python
loader=$account/Chang-Agent/tools/run_with_bailian_csv.py
csv=$account/wyh/百炼-apiKey.csv
endpoint=https://dashscope.aliyuncs.com/compatible-mode/v1
formal_out=$repo/outputs/multimodal_probe/20261123_xlrs_vqa_probe
manifest=$formal_out/inputs/cd_subset/dev_cd_vqa.jsonl
mode=${1:-${XLRS_VQA_MODE:-formal}}
args=()
case "$mode" in
    smoke)
        out=$repo/.runtime/smoke/xlrs_vqa/$SLURM_JOB_ID
        args+=(--smoke)
        ;;
    formal)
        out=$formal_out
        ;;
    *)
        printf 'Expected smoke or formal\n' >&2
        exit 2
        ;;
esac
mkdir -p "$out/logs"

finish() {
    status=$?
    if [[ "$mode" == formal ]]; then
        QQ_SUB="XLRS VQA probe · ${SLURM_JOB_ID}" \
        QQ_BODY="exit=$status; output=$out; metrics=$out/metrics.json; logs=$out/logs" \
            "$account/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || printf 'QQ notification failed\n' >&2
    fi
}
trap finish EXIT

while IFS= read -r name; do
    if [[ ${name,,} == *proxy* ]]; then unset "$name"; fi
done < <(compgen -e)
unset DASHSCOPE_API_KEY
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$repo"
printf 'XLRS startup: node=%s mode=%s job=%s\n' "${SLURMD_NODENAME:-unknown}" "$mode" "$SLURM_JOB_ID"
for required in "$python" "$loader" "$csv" "$manifest"; do
    if [[ ! -r "$required" ]]; then
        printf 'Required file is not readable on the compute node: %s\n' "$required" >&2
        exit 1
    fi
done
"$python" -B "$loader" "$csv" env DASHSCOPE_BASE_URL="$endpoint" \
    "$python" -B experiments/multimodal_probe/xlrs_vqa_run.py --manifest "$manifest" --out "$out" "${args[@]}" \
    2>&1 | tee "$out/logs/run-${SLURM_JOB_ID}.log"
