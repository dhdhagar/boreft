#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu,gpu-preempt,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint=a100

# Local eRank (1000 train targets × 8 samples, including greedy posterior)
# and Sobol-in-box pooled domain eRank (1000 × 8 T=1).
# See scripts/analyze_semantle_local_erank.py.
#
# Usage (from repo root):
#   mkdir -p scripts/jobs
#   sbatch scripts/analyze_semantle_local_erank.sh
#   sbatch --export=ALL,N_TARGETS=8,N_SAMPLES=4,N_SOBOL=16 scripts/analyze_semantle_local_erank.sh
#   sbatch --export=ALL,SKIP_LOCAL=1 scripts/analyze_semantle_local_erank.sh
#   sbatch --export=ALL,CONDITIONS="canonical=outputs/1784053292 sdpo0=outputs/1788622157 novae=outputs/1789101817 sdpo0_novae=outputs/1789021668 ce0=outputs/1789136583" \
#     scripts/analyze_semantle_local_erank.sh

set -eo pipefail

if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

# Llama is loaded via --cache-dir (read-only hub). HF_HOME stays writable so
# Qwen embeddings do not write to that hub; match search jobs' cluster cache.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
if [ ! -d "${HF_HOME}" ]; then
    export HF_HOME="$(pwd)/models"
    mkdir -p "${HF_HOME}"
fi
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use the python already on PATH. See README.md.

N_TARGETS="${N_TARGETS:-1000}"
N_SAMPLES="${N_SAMPLES:-8}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_SOBOL="${N_SOBOL:-1000}"
N_SOBOL_SAMPLES="${N_SOBOL_SAMPLES:-8}"
OUT_DIR="${OUT_DIR:-./data/semantle/analysis/breadth_n1000_s8}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
CONDITIONS="${CONDITIONS:-}"

echo "[analyze_semantle_local_erank] n_targets=${N_TARGETS} n_samples=${N_SAMPLES} seed=${SEED}"
echo "[analyze_semantle_local_erank] n_sobol=${N_SOBOL} n_sobol_samples=${N_SOBOL_SAMPLES}"
echo "[analyze_semantle_local_erank] out_dir=${OUT_DIR} cache_dir=${CACHE_DIR} HF_HOME=${HF_HOME}"

ARGS=(
    --n-targets        "${N_TARGETS}"
    --n-samples        "${N_SAMPLES}"
    --seed             "${SEED}"
    --batch-size       "${BATCH_SIZE}"
    --n-sobol          "${N_SOBOL}"
    --n-sobol-samples  "${N_SOBOL_SAMPLES}"
    --out-dir          "${OUT_DIR}"
    --cache-dir        "${CACHE_DIR}"
)
if [ "${SKIP_LOCAL:-0}" = "1" ] || [ "${SKIP_LOCAL:-0}" = "true" ] || [ "${SKIP_LOCAL:-0}" = "yes" ]; then
    ARGS+=(--skip-local)
fi
if [ "${SKIP_SOBOL:-0}" = "1" ] || [ "${SKIP_SOBOL:-0}" = "true" ] || [ "${SKIP_SOBOL:-0}" = "yes" ]; then
    ARGS+=(--skip-sobol)
fi
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

python -u scripts/analyze_semantle_local_erank.py "${ARGS[@]}"
