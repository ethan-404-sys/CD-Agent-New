#!/usr/bin/env bash
#SBATCH --job-name=t32-dev32
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9G
#SBATCH --time=03:00:00
set -euo pipefail

repo=${SLURM_SUBMIT_DIR:?}
account=/guisongxia01/pangchao/wangyihan
py=$account/conda_envs/exp5-env/bin/python
worker_py=$account/conda_envs/omniovcd-env/bin/python
key_loader=$account/Chang-Agent/tools/run_with_bailian_csv.py
mode=${1:?Expected smoke or formal}
out=${OUT_DIR:-$repo/outputs/t3_visual_ref/t32_dev32}
work=$out
if [[ "$mode" == smoke ]]; then
    work=$repo/.runtime/smoke/t32-dev32-${SLURM_JOB_ID}
elif [[ "$mode" != formal ]]; then
    exit 2
fi
mkdir -p "$work/logs"
: "${BAILIAN_CSV:?Supply the existing Bailian credential CSV}"
bailian_endpoint=https://dashscope.aliyuncs.com/compatible-mode/v1
while IFS= read -r name; do
    if [[ ${name,,} == *proxy* ]]; then unset "$name"; fi
done < <(compgen -e)
export AGENT_RUN="$work" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AGENT_ASK_SLOTS=3
export PYTHONPATH="$repo/src:$repo/third_party/OmniOVCD:$repo/third_party/DynamicEarth/third_party/segment_anything"
export PYTHONUNBUFFERED=1
started=$SECONDS
worker=''
phase='SAM3 startup'
finish() {
    status=$?
    sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,MaxRSS,AveRSS -P > "$work/logs/memory-${SLURM_JOB_ID}.log" 2>&1 || true
    if [[ -n "$worker" ]]; then kill "$worker" 2>/dev/null || true; fi
    if [[ "$mode" == formal ]]; then
        QQ_SUB="T3.2 Dev-32 · ${SLURM_JOB_ID}" \
        QQ_BODY="status=$status; final phase=$phase; duration=$((SECONDS-started))s; output=$out; logs=$work/logs" \
            "$account/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || printf 'QQ notification failed\n' >&2
    fi
}
trap finish EXIT
port=$("$py" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
"$worker_py" -m cd_agent.workers.sam3_worker --port "$port" > "$work/logs/sam3-${SLURM_JOB_ID}.log" 2>&1 &
worker=$!
ready=no
for ((attempt=0; attempt<300; attempt++)); do
    if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        ready=yes
        break
    fi
    if ! kill -0 "$worker" 2>/dev/null; then
        printf 'SAM3 worker exited during startup. See %s\n' "$work/logs/sam3-${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    sleep 2
done
if [[ $ready != yes ]]; then
    printf 'SAM3 worker did not become ready within ten minutes. See %s\n' "$work/logs/sam3-${SLURM_JOB_ID}.log" >&2
    exit 1
fi
printf 'SAM3 ready on %s\n' "${SLURMD_NODENAME:-unknown}"
phase='Planner and Verifier inference'
args=(--inputs "$out/inputs" --traces "$work/traces" --sam3-worker "http://127.0.0.1:$port")
if [[ "$mode" == smoke ]]; then
    args+=(--smoke --limit 1 --concurrency 1)
fi
"$py" "$key_loader" "$BAILIAN_CSV" env DASHSCOPE_BASE_URL="$bailian_endpoint" \
    "$py" -m cd_agent.t32_dev32 infer "${args[@]}"
if [[ "$mode" == formal ]]; then
    phase='Offline scoring and replay generation'
    "$py" -m cd_agent.t32_dev32 report --out "$out"
    if [[ -f "$out/demo_g3/coverage.json" ]]; then
        "$py" -m cd_agent.t32_replay --out "$out" --demo "$out/demo_g3"
    fi
    if [[ -n "${BASELINE_DIR:-}" && -f "${BASELINE_DIR}/metrics.json" ]]; then
        phase='Baseline vs current A/B comparison'
        "$py" "$repo/experiments/t3_visual_ref/compare_t32_runs.py" \
            --baseline "$BASELINE_DIR" --current "$out" --out "$out/compare"
    fi
    phase=$("$py" -c 'import json,sys; d=json.load(open(sys.argv[1]))["overall"]; print("completed", d["count"], "micro IoU", round(d["micro"]["iou"],4), "micro F1", round(d["micro"]["f1"],4))' "$out/metrics.json")
fi
