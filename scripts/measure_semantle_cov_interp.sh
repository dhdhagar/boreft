#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu,gpu-preempt,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint=a100

# Coverage / interpolation errors for Semantle (notes/measure_cov_interp_notes.md).
#
# Usage (from repo root):
#   mkdir -p scripts/jobs
#   # Four paper ablations, full decode:
#   sbatch scripts/measure_semantle_cov_interp.sh
#   # Coverage vs training-set size (no LLM decode):
#   sbatch --export=ALL,COVERAGE_ONLY=1,FROM_LADDER=1 scripts/measure_semantle_cov_interp.sh
#   # Both in one job:
#   sbatch --export=ALL,FROM_LADDER=1 scripts/measure_semantle_cov_interp.sh
#   # Smoke test:
#   sbatch --export=ALL,N_PAIRS=4,N_MIX3=8,N_MIX5=8,N_SAMPLES=2,N_SEEDS=1,N_RANDOM=4 \
#     scripts/measure_semantle_cov_interp.sh

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

SEED="${SEED:-42}"
N_PAIRS="${N_PAIRS:-40}"
N_MIX3="${N_MIX3:-100}"
N_MIX5="${N_MIX5:-100}"
N_SAMPLES="${N_SAMPLES:-16}"
N_SEEDS="${N_SEEDS:-2}"
N_RANDOM="${N_RANDOM:-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUT_DIR="${OUT_DIR:-./data/semantle/analysis/cov_interp}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
LADDER_JSON="${LADDER_JSON:-experiments/semantle/ladder_checkpoints.json}"
TARGETS_JSON="${TARGETS_JSON:-experiments/outputs/semantle/sweep/targets.json}"
CANONICAL_DIR="${CANONICAL_DIR:-outputs/1784053292}"
CONDITIONS="${CONDITIONS:-}"

echo "[measure_semantle_cov_interp] n_pairs=${N_PAIRS} n_mix3=${N_MIX3} n_mix5=${N_MIX5}"
echo "[measure_semantle_cov_interp] n_samples=${N_SAMPLES} n_seeds=${N_SEEDS} n_random=${N_RANDOM}"
echo "[measure_semantle_cov_interp] out_dir=${OUT_DIR} cache_dir=${CACHE_DIR} HF_HOME=${HF_HOME}"

ARGS=(
    --seed              "${SEED}"
    --n-pairs           "${N_PAIRS}"
    --n-mix3            "${N_MIX3}"
    --n-mix5            "${N_MIX5}"
    --n-samples         "${N_SAMPLES}"
    --n-seeds           "${N_SEEDS}"
    --n-random          "${N_RANDOM}"
    --batch-size        "${BATCH_SIZE}"
    --out-dir           "${OUT_DIR}"
    --cache-dir         "${CACHE_DIR}"
    --ladder-json       "${LADDER_JSON}"
    --search-targets-json "${TARGETS_JSON}"
    --canonical-dir     "${CANONICAL_DIR}"
)
if [ "${COVERAGE_ONLY:-0}" = "1" ] || [ "${COVERAGE_ONLY:-0}" = "true" ]; then
    ARGS+=(--coverage-only)
fi
if [ "${FROM_LADDER:-0}" = "1" ] || [ "${FROM_LADDER:-0}" = "true" ]; then
    ARGS+=(--from-ladder)
fi
if [ "${OVERWRITE:-0}" = "1" ] || [ "${OVERWRITE:-0}" = "true" ]; then
    ARGS+=(--overwrite)
fi
if [ -n "${CONDITIONS}" ]; then
    for spec in ${CONDITIONS}; do
        ARGS+=(--condition "${spec}")
    done
fi
if [ "${NO_WANDB:-0}" = "1" ] || [ "${NO_WANDB:-0}" = "true" ]; then
    ARGS+=(--no-wandb)
else
    ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi

python -u scripts/measure_semantle_cov_interp.py "${ARGS[@]}"
