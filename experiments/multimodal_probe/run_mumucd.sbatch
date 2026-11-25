#!/usr/bin/env bash
#SBATCH --job-name=mumucd-tirana
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9G
#SBATCH --time=01:30:00
#SBATCH --output=.runtime/smoke/mumucd/logs/slurm-%j.log

set -euo pipefail
repo=${SLURM_SUBMIT_DIR:?}
account=/guisongxia01/pangchao/wangyihan
py=$account/conda_envs/exp5-env/bin/python
prep_py=$account/conda_envs/ovcd-ape-env/bin/python
worker_py=$account/conda_envs/omniovcd-env/bin/python
key_loader=$account/Chang-Agent/tools/run_with_bailian_csv.py
bailian_csv="$account/wyh/百炼-apiKey.csv"
bailian_endpoint=https://dashscope.aliyuncs.com/compatible-mode/v1
mode=${1:-${MUMUCD_MODE:-smoke}}
out=$repo/outputs/multimodal_probe/20261123_mumucd_tirana_probe
limit=9
extra=()
case "$mode" in
    smoke)
        out=$repo/.runtime/smoke/mumucd/$SLURM_JOB_ID
        limit=1
        extra=(--smoke)
        ;;
    formal) ;;
    *) printf 'Expected mode smoke or formal\n' >&2; exit 2 ;;
esac
cd "$repo"
mkdir -p "$out/logs"
# For formal submission, set SBATCH_OUTPUT to "$out/logs/slurm-%j.log" and
# create that logs directory first. This in-run log stays with either mode.
exec > >(tee -a "$out/logs/pipeline-$SLURM_JOB_ID.log") 2>&1
while IFS= read -r name; do
    if [[ ${name,,} == *proxy* ]]; then unset "$name"; fi
done < <(compgen -e)
unset DASHSCOPE_API_KEY
export AGENT_RUN="$out" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AGENT_ASK_SLOTS=3
export PYTHONPATH="$repo:$repo/src:$repo/third_party/DynamicEarth/third_party/segment_anything"

worker=''
phase='NetCDF preparation'
started=$SECONDS
finish() {
    status=$?
    if [[ -n "$worker" ]]; then kill "$worker" 2>/dev/null || true; fi
    sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,MaxRSS,AveRSS -P > "$out/logs/memory-$SLURM_JOB_ID.log" 2>&1 || true
    if [[ "$mode" == formal ]]; then
        QQ_SUB="MUMUCD Tirana · $SLURM_JOB_ID" \
        QQ_BODY="status=$status; phase=$phase; elapsed=$((SECONDS-started))s; output=$out; logs=$out/logs" \
            "$account/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || printf 'QQ notification failed\n' >&2
    fi
}
trap finish EXIT
trap 'exit 143' TERM

# ovcd-ape-env sees a working Rasterio/GDAL wheel but lacks its pure-Python
# attrs dependency. Make exp5-env's attrs visible only while importing Rasterio.
attrs_site=$("$py" -B -c 'from pathlib import Path; import attrs; print(Path(attrs.__file__).resolve().parents[1])')
"$prep_py" -B -c '
import runpy
import sys
sys.path.append(sys.argv.pop(1))
import rasterio
sys.path.pop()
runpy.run_module("experiments.multimodal_probe.mumucd_prepare", run_name="__main__", alter_sys=True)
' "$attrs_site" --out "$out"

phase='SAM3 startup'
port=$("$py" -B -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
"$worker_py" -B -m cd_agent.workers.sam3_worker --port "$port" > "$out/logs/sam3-$SLURM_JOB_ID.log" 2>&1 &
worker=$!
ready=no
for ((attempt=0; attempt<300; attempt++)); do
    if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        ready=yes
        break
    fi
    if ! kill -0 "$worker" 2>/dev/null; then break; fi
    sleep 2
done
if [[ "$ready" != yes ]]; then
    printf 'SAM3 did not become ready; see %s\n' "$out/logs/sam3-$SLURM_JOB_ID.log" >&2
    exit 1
fi

phase='Planner and frozen Verifier'
"$py" -B "$key_loader" "$bailian_csv" env DASHSCOPE_BASE_URL="$bailian_endpoint" \
    "$py" -B -m experiments.multimodal_probe.mumucd_run infer \
    --inputs "$out/inputs" --traces "$out/traces" --sam3-worker "http://127.0.0.1:$port" \
    --limit "$limit" --max-cost-cny 20 "${extra[@]}"

phase='Offline scoring and visual evidence'
"$py" -B -m experiments.multimodal_probe.mumucd_run report --out "$out" --limit "$limit" --max-cost-cny 20
phase='complete'
