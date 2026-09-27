#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu,gpu-preempt,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -t 0-04:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint=a100

# Latent utilization + semantic–code Spearman (shared encoder vs none).
# See scripts/analyze_semantle_code_geometry.py.
#
# Usage (from repo root):
#   mkdir -p scripts/jobs
#   sbatch scripts/analyze_semantle_code_geometry.sh
#   sbatch --export=ALL,CONDITIONS="canonical=outputs/1784053292 noenc=outputs/1788894671" \
#     scripts/analyze_semantle_code_geometry.sh

set -eo pipefail

if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
if [ ! -d "${HF_HOME}" ]; then
    export HF_HOME="$(pwd)/models"
    mkdir -p "${HF_HOME}"
fi
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use the python already on PATH. See README.md.

BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-./data/semantle/analysis}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
CONDITIONS="${CONDITIONS:-}"

echo "[analyze_semantle_code_geometry] batch_size=${BATCH_SIZE} seed=${SEED}"
echo "[analyze_semantle_code_geometry] out_dir=${OUT_DIR} cache_dir=${CACHE_DIR} HF_HOME=${HF_HOME}"

ARGS=(
    --batch-size "${BATCH_SIZE}"
    --seed       "${SEED}"
    --out-dir    "${OUT_DIR}"
    --cache-dir  "${CACHE_DIR}"
)
if [ -n "${CONDITIONS}" ]; then
    for spec in ${CONDITIONS}; do
        ARGS+=(--condition "${spec}")
    done
fi
if [ "${NO_WANDB:-0}" = "1" ] || [ "${NO_WANDB:-0}" = "true" ] || [ "${NO_WANDB:-0}" = "yes" ]; then
    ARGS+=(--no-wandb)
else
    ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi

python -u scripts/analyze_semantle_code_geometry.py "${ARGS[@]}"
