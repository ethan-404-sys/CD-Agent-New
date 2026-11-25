#!/usr/bin/env bash
#SBATCH --job-name=bright-xbd
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/multimodal_probe/20261123_bright_xbd_probe/logs/slurm-%j.log
set -euo pipefail

repo=${SLURM_SUBMIT_DIR:?}
account=/guisongxia01/pangchao/wangyihan
py=$account/conda_envs/exp5-env/bin/python
worker_py=$account/conda_envs/omniovcd-env/bin/python
key_loader=$account/Chang-Agent/tools/run_with_bailian_csv.py
bailian_csv=$account/wyh/百炼-apiKey.csv
bailian_endpoint=https://dashscope.aliyuncs.com/compatible-mode/v1
out=$repo/outputs/multimodal_probe/20261123_bright_xbd_probe
mode=${1:-formal}  # expctl invokes the script without positional arguments.
work=$out
args=()
case "$mode" in
    smoke)
        work=$repo/.runtime/smoke/bright_xbd/$SLURM_JOB_ID
        args+=(--smoke --max-cost-cny 2)
        ;;
    formal) args+=(--max-cost-cny 40) ;;
    *) printf 'Expected smoke or formal\n' >&2; exit 2 ;;
esac
cd "$repo"
mkdir -p "$work/logs"
while IFS= read -r name; do
    if [[ ${name,,} == *proxy* ]]; then unset "$name"; fi
done < <(compgen -e)
unset DASHSCOPE_API_KEY
export AGENT_RUN="$work/runtime-$SLURM_JOB_ID" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AGENT_ASK_SLOTS=3
export PYTHONPATH="$repo:$repo/src:$repo/third_party/DynamicEarth/third_party/segment_anything"
export PYTHONUNBUFFERED=1

started=$SECONDS
worker=''
phase='Sample verification'
finish() {
    status=$?
    if [[ -n "$worker" ]]; then kill "$worker" 2>/dev/null || true; fi
    sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,MaxRSS,AveRSS -P \
        > "$work/logs/memory-${SLURM_JOB_ID}.log" 2>&1 || true
    if [[ "$mode" == formal ]]; then
        QQ_SUB="BRIGHT + xBD probe · ${SLURM_JOB_ID}" \
        QQ_BODY="status=$status; final phase=$phase; duration=$((SECONDS-started))s; output=$out; logs=$work/logs" \
            "$account/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || printf 'QQ notification failed\n' >&2
    fi
}
trap finish EXIT

if [[ "$mode" == smoke ]]; then
    "$py" -B -m experiments.multimodal_probe.bright_xbd_prepare --out "$out"
else
    "$py" -B -m experiments.multimodal_probe.bright_xbd_prepare --out "$out" --verify-only
fi

phase='SAM3 startup'
port=$("$py" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
"$worker_py" -B -m cd_agent.workers.sam3_worker --port "$port" > "$work/logs/sam3-${SLURM_JOB_ID}.log" 2>&1 &
worker=$!
ready=no
for ((attempt=0; attempt<300; attempt++)); do
    if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        ready=yes
        break
    fi
    if ! kill -0 "$worker" 2>/dev/null; then
        printf 'SAM3 exited; see %s\n' "$work/logs/sam3-${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    sleep 2
done
if [[ $ready != yes ]]; then
    printf 'SAM3 startup timed out; see %s\n' "$work/logs/sam3-${SLURM_JOB_ID}.log" >&2
    exit 1
fi

phase='Planner and Verifier inference'
"$py" -B "$key_loader" "$bailian_csv" env DASHSCOPE_BASE_URL="$bailian_endpoint" \
    "$py" -B -m experiments.multimodal_probe.bright_xbd_run infer \
    --inputs "$out/inputs" --work "$work" --sam3-worker "http://127.0.0.1:$port" "${args[@]}"
kill "$worker" 2>/dev/null || true
worker=''

phase='Offline scoring and evidence rendering'
"$py" -B -m experiments.multimodal_probe.bright_xbd_run report --data "$out" --work "$work" "${args[@]}"
if [[ "$mode" == smoke ]]; then
    printf 'BRIGHT_XBD_SMOKE_PASSED\n'
fi
phase=complete
