#!/usr/bin/env bash
#SBATCH --job-name=mci-runtime
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/mci-runtime-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/CD-Agent/outputs/verifier_init_masks/verifier_v1_600/logs/mci-runtime-%j.err
set -euo pipefail
root=/guisongxia01/pangchao/wangyihan
repo="$root/CD-Agent"
out="${VERIFIER_DATASET_OUT:-$repo/outputs/verifier_init_masks/verifier_v1_600}"
source_dir="$repo/third_party/DynamicEarth/third_party/SimFeatUp"
runtime="$repo/.runtime/dynamic-earth"
environment="$root/conda_envs/ovcd-de-env"
python="$environment/bin/python"
started=$SECONDS
build=${MCI_BUILD_CACHE:-"$repo/.runtime/.dynamic-earth-build"}
finish() {
    code=$?
    trap - EXIT
    QQ_SUB="Verifier MCI runtime $SLURM_JOB_ID" \
    QQ_BODY="status=exit $code, elapsed=$((SECONDS-started))s, result=SimFeatUp CUDA runtime preparation, output=$runtime, logs=$out/logs/mci-runtime-$SLURM_JOB_ID.out" \
    "$root/.local/bin/wangyihan-send-qq" || true
    exit "$code"
}
trap finish EXIT
export CUDA_HOME=${MCI_CUDA_HOME:-/persist_data/apps/cuda-12.4}
export PATH="$environment/bin:$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MAX_JOBS="$SLURM_CPUS_PER_TASK"
export TORCH_CUDA_ARCH_LIST=${MCI_CUDA_ARCH_LIST:-"7.5;8.0;8.6;8.9;9.0+PTX"}
check_extension() {
    PYTHONPATH="$source_dir:$1" "$python" -B -c 'import torch; from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv; print("SIMFEATUP_EXTENSION_READY", flush=True)'
}

mkdir -p "$repo/.runtime"
exec 9>"$repo/.runtime/.mci-runtime.lock"
flock -n 9
if [[ -d "$runtime" ]]; then
    check_extension "$runtime"
    exit
fi
test -x "$CUDA_HOME/bin/nvcc"
test -r "$source_dir/setup.py"
case "$build" in
    "$repo/.runtime/.dynamic-earth-build"|"$repo/.runtime/.dynamic-earth-build."[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]) ;;
    *) printf 'Build cache must stay in %s/.runtime\n' "$repo" >&2; exit 1 ;;
esac
mkdir -p "$build"
cd "$source_dir"
"$python" -uB setup.py build_ext --build-temp "$build/objects" --build-lib "$build/library"
check_extension "$build/library"
mv -T "$build/library" "$runtime"
rm -rf "$build"
