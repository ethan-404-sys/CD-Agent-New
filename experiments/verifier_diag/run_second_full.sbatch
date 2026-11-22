#!/usr/bin/env bash
#SBATCH --job-name=second-masks-full
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=.runtime/second-full-%j.out

# Submit from the repo root. Each zero-based shard owns disjoint output directories.
set -euo pipefail
repo=${SLURM_SUBMIT_DIR:?Submit from the CD-Agent repository root}
job_id=${SLURM_JOB_ID:?Run in Slurm}
cd "$repo"

profile=full
split=
shard=0
num_shards=1
ckpt=()
while (( $# )); do
    case "$1" in
        --profile) profile=${2:?}; shift 2 ;;
        --split) split=${2:?}; shift 2 ;;
        --shard) shard=${2:?}; shift 2 ;;
        --num-shards) num_shards=${2:?}; shift 2 ;;
        --scannet-ckpt) ckpt=(--scannet-ckpt "${2:?}"); shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ ! $shard =~ ^(0|[1-9][0-9]*)$ || ! $num_shards =~ ^[1-9][0-9]*$ ]] ||
   (( shard >= num_shards || num_shards > 400 )); then
    printf 'Need 0 <= --shard < --num-shards <= 400\n' >&2
    exit 2
fi
if [[ -n $split && $split != train && $split != val && $split != test ]]; then
    printf 'Unknown split: %s\n' "$split" >&2
    exit 2
fi
limit=()
case "$profile" in
    smoke)
        out="$repo/.runtime/smoke/second-masks-$job_id"
        split=train
        limit=(--limit 2)
        ;;
    pilot64)
        out="$repo/outputs/verifier_diag/second_pilot64"
        split=train
        limit=(--limit 64)
        ;;
    full)
        out="$repo/outputs/verifier_diag/second_full_v1"
        if (( num_shards > 1 )); then
            printf -v shard_dir 'shard-%03d-of-%03d' "$shard" "$num_shards"
            out="$out/$shard_dir"
        fi
        ;;
    *) printf 'Unknown profile: %s\n' "$profile" >&2; exit 2 ;;
esac
if [[ $profile != full ]] && (( num_shards != 1 )); then
    printf 'Only the full profile supports sharding\n' >&2
    exit 2
fi

mkdir -p "$out/logs"
log="$out/logs/slurm-$job_id.log"
printf 'Pipeline log: %s\n' "$log"
exec >>"$log" 2>&1
account=${CD_AGENT_ACCOUNT_ROOT:-$(dirname "$repo")}
envs=${CD_AGENT_CONDA_ENVS:-$account/conda_envs}
cpu="$envs/exp5-env/bin/python"
ovcd="$envs/omniovcd-env/bin/python"
started=$SECONDS
notify() {
    code=$?
    trap - EXIT
    QQ_SUB="SECOND $profile $job_id" \
    QQ_BODY="exit=$code; elapsed=$((SECONDS-started))s; output=$out; log=$log" \
        "$account/.local/bin/wangyihan-send-qq" || printf 'QQ notification failed\n' >&2
    exit "$code"
}
if [[ $profile != smoke ]]; then
    trap notify EXIT
fi
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
runner=experiments.verifier_diag.prepare_second_masks
splits=(train val test)
if [[ -n $split ]]; then
    splits=("$split")
fi
for split in "${splits[@]}"; do
    run=$out
    if [[ $profile == full ]]; then
        run="$out/$split"
    fi
    "$cpu" -B -m "$runner" prepare --out "$run" --split "$split" \
        --shard "$shard" --num-shards "$num_shards" "${limit[@]}"
    "$cpu" -B -m "$runner" infer --out "$run" --method scannet "${ckpt[@]}"
    "$ovcd" -B -m "$runner" infer --out "$run" --method omniovcd
    "$ovcd" -B -m "$runner" infer --out "$run" --method rea_ovcd
    "$cpu" -B -m "$runner" export --out "$run"
    "$cpu" -B -m cd_agent.verify run "$run"
    test -s "$run/viz/index.html"
done
