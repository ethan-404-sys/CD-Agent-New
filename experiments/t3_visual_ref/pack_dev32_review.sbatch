#!/usr/bin/env bash
#SBATCH --job-name=t3-dev32-pack
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
set -euo pipefail

repo=${SLURM_SUBMIT_DIR:?}
account=/guisongxia01/pangchao/wangyihan
py=$account/conda_envs/exp5-env/bin/python
src_dev32=${SRC_DEV32:-$account/wyh/CD-Agent/results/t32_dev32}
pack_out=${PACK_OUT:-$repo/outputs/t3_visual_ref/dev32_review_pack}
zip_out=${ZIP_OUT:-$repo/outputs/t3_visual_ref/dev32_review_pack.zip}
prep_out=${PREP_OUT:-$repo/outputs/t3_visual_ref/20261123_t32_dev32_p0fix}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"

"$py" "$repo/experiments/t3_visual_ref/build_dev32_review_pack.py" \
    --source "$src_dev32" \
    --out "$pack_out" \
    --zip "$zip_out"

"$py" -m cd_agent.t32_dev32 prepare \
    --reuse-from "$src_dev32" \
    --out "$prep_out"

ls -lh "$zip_out" "$prep_out/sample.json"
