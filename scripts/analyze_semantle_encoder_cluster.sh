#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-08:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint="vram16"

# Qwen embedding vs Qwen3-1.7B last-token vs Llama-3.2-1B-Instruct clustering
# on Semantle word+definition text. See scripts/analyze_semantle_encoder_cluster.py.
#
# Usage:
#   sbatch scripts/analyze_semantle_encoder_cluster.sh
#   sbatch --export=ALL,N=0,SEED=0 scripts/analyze_semantle_encoder_cluster.sh
#   sbatch --export=ALL,N=1000 scripts/analyze_semantle_encoder_cluster.sh
#   sbatch --export=ALL,ENCODERS="qwen qwen_llm llama" scripts/analyze_semantle_encoder_cluster.sh
#   sbatch --export=ALL,MAX_LENGTH=512 scripts/analyze_semantle_encoder_cluster.sh
#   sbatch --export=ALL,REUSE_PREVIOUS=1 scripts/analyze_semantle_encoder_cluster.sh

set -eo pipefail

if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

# Llama is loaded via explicit --cache-dir. HF_HOME stays writable so the
# Qwen embedding model and Qwen3-1.7B do not write to the read-only llama hub.
export HF_HOME="${HF_HOME:-$(pwd)/models}"
mkdir -p "${HF_HOME}"
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use the python already on PATH. See README.md.

# ── Config ────────────────────────────────────────────────────────────────────
N="${N:-0}"  # 0 = every eligible word
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-./data/semantle/analysis}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
QWEN_LLM_MODEL="${QWEN_LLM_MODEL:-Qwen/Qwen3-1.7B}"
LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
POOLING="${POOLING:-last_instruction}"
MAX_LENGTH="${MAX_LENGTH:-128}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-64}"
QWEN_LLM_BATCH_SIZE="${QWEN_LLM_BATCH_SIZE:-16}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-16}"
LABEL_FIELD="${LABEL_FIELD:-category_normalized}"
ENCODERS="${ENCODERS:-}"
REUSE_PREVIOUS="${REUSE_PREVIOUS:-0}"

# ── Validate ──────────────────────────────────────────────────────────────────
DEFINITIONS="${DEFINITIONS:-./data/semantle/train/definitions.jsonl}"
if [ ! -f "${DEFINITIONS}" ]; then
    echo "ERROR: '${DEFINITIONS}' not found." >&2
    exit 1
fi

echo "[analyze_semantle_encoder_cluster] N=${N} SEED=${SEED} DEVICE=${DEVICE} OUT_DIR=${OUT_DIR}"
echo "[analyze_semantle_encoder_cluster] QWEN_MODEL=${QWEN_MODEL} QWEN_LLM_MODEL=${QWEN_LLM_MODEL}"
echo "[analyze_semantle_encoder_cluster] LLAMA_MODEL=${LLAMA_MODEL}"
echo "[analyze_semantle_encoder_cluster] CACHE_DIR=${CACHE_DIR} POOLING=${POOLING} MAX_LENGTH=${MAX_LENGTH}"
echo "[analyze_semantle_encoder_cluster] ENCODERS=${ENCODERS:-<all>} REUSE_PREVIOUS=${REUSE_PREVIOUS}"

# ── Run ───────────────────────────────────────────────────────────────────────
ARGS=(
    --definitions         "${DEFINITIONS}"
    --n                   "${N}"
    --seed                "${SEED}"
    --label-field         "${LABEL_FIELD}"
    --qwen-model          "${QWEN_MODEL}"
    --qwen-llm-model      "${QWEN_LLM_MODEL}"
    --llama-model         "${LLAMA_MODEL}"
    --cache-dir           "${CACHE_DIR}"
    --pooling             "${POOLING}"
    --max-length          "${MAX_LENGTH}"
    --qwen-batch-size     "${QWEN_BATCH_SIZE}"
    --qwen-llm-batch-size "${QWEN_LLM_BATCH_SIZE}"
    --llama-batch-size    "${LLAMA_BATCH_SIZE}"
    --device              "${DEVICE}"
    --out-dir             "${OUT_DIR}"
)
if [ -n "${ENCODERS}" ]; then
    ARGS+=(--encoders ${ENCODERS})
fi
if [ "${REUSE_PREVIOUS}" = "1" ] || [ "${REUSE_PREVIOUS}" = "true" ] || [ "${REUSE_PREVIOUS}" = "yes" ]; then
    ARGS+=(--reuse-previous)
fi
if [ "${NO_WANDB:-0}" = "1" ] || [ "${NO_WANDB:-0}" = "true" ] || [ "${NO_WANDB:-0}" = "yes" ]; then
    ARGS+=(--no-wandb)
else
    ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi

python -u scripts/analyze_semantle_encoder_cluster.py "${ARGS[@]}"
