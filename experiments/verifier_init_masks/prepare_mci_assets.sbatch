#!/usr/bin/env bash
#SBATCH --job-name=mci-assets
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/mci-assets-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/mci-assets-%j.err
set -euo pipefail
root=/guisongxia01/pangchao/wangyihan
models="$root/CD-Agent/models"
reference="$root/CD-Agent/third_party"
out="${VERIFIER_DATASET_OUT:-$repo/outputs/verifier_init_masks/verifier_v1_600}"
original=/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth
local_clip=/persist_data/home/guisongxia/pangchao/chengyu/codes/DSP/pretrained/ViT-B-16.pt
started=$SECONDS
finish() {
    code=$?
    trap - EXIT
    QQ_SUB="Verifier MCI assets $SLURM_JOB_ID" \
    QQ_BODY="status=exit $code, elapsed=$((SECONDS-started))s, result=original MCI assets preparation, models=$models, logs=$out/logs/mci-assets-$SLURM_JOB_ID.out" \
    "$root/.local/bin/wangyihan-send-qq" || true
    exit "$code"
}
trap finish EXIT

fetch() {
    local target=$1 minimum=$2
    shift 2
    if [[ -f "$target" ]] && (( $(stat -c %s "$target") >= minimum )); then
        printf 'Reuse %s\n' "$target"
        return
    fi
    local temporary="$target.part.$SLURM_JOB_ID"
    local url
    for url in "$@"; do
        if curl -q --noproxy '*' --fail --location --silent --show-error --connect-timeout 12 --max-time 240 \
                --output "$temporary" "$url" \
                && (( $(stat -c %s "$temporary") >= minimum )); then
            mv -T "$temporary" "$target"
            printf 'Installed %s (%s bytes)\n' "$target" "$(stat -c %s "$target")"
            return
        fi
    done
    rm -f "$temporary"
    printf 'Unable to download %s\n' "$target" >&2
    return 1
}

mkdir -p "$models"/{dinov2-base,CLIP,SimFeatUp}
exec 9>"$models/SimFeatUp/.mci-assets.lock"
flock -n 9
if [[ ! -s "$models/CLIP/ViT-B-16.pt" ]]; then
    cp "$local_clip" "$models/CLIP/.ViT-B-16.pt.$SLURM_JOB_ID"
    mv -T "$models/CLIP/.ViT-B-16.pt.$SLURM_JOB_ID" "$models/CLIP/ViT-B-16.pt"
fi
printf 'CLIP %s bytes\n' "$(stat -c %s "$models/CLIP/ViT-B-16.pt")"

if [[ ! -d "$third_party/DynamicEarth" ]]; then
    stage="$reference/.DynamicEarth.$SLURM_JOB_ID"
    mkdir -p "$stage/third_party"
    cp -r "$original/dynamic_earth" "$stage/"
    for vendor in SegEarth_OV segment_anything SimFeatUp; do
        cp -r "$original/third_party/$vendor" "$stage/third_party/"
    done
    mv -T "$stage" "$third_party/DynamicEarth"
fi
test -r "$third_party/DynamicEarth/dynamic_earth/identifier/segearth_ov_ext.py"
test -r "$third_party/DynamicEarth/third_party/SimFeatUp/setup.py"
printf 'DynamicEarth source %s\n' "$third_party/DynamicEarth"

upstream=https://raw.githubusercontent.com/likyoo/SegEarth-OV/main/simfeatup_dev/weights
cdn=https://cdn.jsdelivr.net/gh/likyoo/SegEarth-OV@main/simfeatup_dev/weights
simfeatup=xclip_jbu_one_million_aid.ckpt
failed=0
staged="$root/CD-Agent/.runtime/mci-assets/$simfeatup"
target="$models/SimFeatUp/$simfeatup"
if [[ ! -s "$target" && -s "$staged" ]]; then
    install -m 664 "$staged" "$target.part.$SLURM_JOB_ID"
    mv -T "$target.part.$SLURM_JOB_ID" "$target"
fi
fetch "$target" 5690776 "$upstream/$simfeatup" "$cdn/$simfeatup" || failed=1
hf=https://huggingface.co/facebook/dinov2-base/resolve/main
mirror=https://hf-mirror.com/facebook/dinov2-base/resolve/main
for file in config.json preprocessor_config.json; do
    fetch "$models/dinov2-base/$file" 100 "$hf/$file" "$mirror/$file" || failed=1
done
fetch "$models/dinov2-base/model.safetensors" 340000000 "$hf/model.safetensors" "$mirror/model.safetensors" || failed=1
(( failed == 0 ))
python3 - "$models" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

models = Path(sys.argv[1])
dino = models / 'dinov2-base'
config = json.loads((dino / 'config.json').read_text())
assert config['model_type'] == 'dinov2' and config['hidden_size'] == 768
assert json.loads((dino / 'preprocessor_config.json').read_text())
with (dino / 'model.safetensors').open('rb') as stream:
    header_size = int.from_bytes(stream.read(8), 'little')
    assert 0 < header_size < 1000000
    assert len(json.loads(stream.read(header_size))) > 100
assert zipfile.is_zipfile(models / 'CLIP/ViT-B-16.pt')
assert zipfile.is_zipfile(models / 'SimFeatUp/xclip_jbu_one_million_aid.ckpt')
print('ASSETS_READY', models, flush=True)
PY
