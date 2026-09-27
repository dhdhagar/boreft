#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint="vram16"

# Qwen embedding vs Qwen3-1.7B vs Llama vs MiST clustering.
# Scores SMILES+definition and the SMILES-only prompt
# ("The description for molecule '{SMILES}'.").
# See scripts/analyze_molopt_encoder_cluster.py.
#
# Usage:
#   sbatch scripts/analyze_molopt_encoder_cluster.sh
#   sbatch --export=ALL,N=1000 scripts/analyze_molopt_encoder_cluster.sh
#   sbatch --export=ALL,VARIANTS="smiles_defn smiles_prompt" scripts/analyze_molopt_encoder_cluster.sh
#   sbatch --export=ALL,ENCODERS="qwen qwen_llm llama mist" scripts/analyze_molopt_encoder_cluster.sh
#   sbatch --export=ALL,REUSE_PREVIOUS=0 scripts/analyze_molopt_encoder_cluster.sh
#
# MiST is a local Qwen2.5-3B checkpoint (~6GB bf16). MIST_MODEL is a directory
# or a folder name under $HF_HOME/mist. If that name is missing, the script
# uses mist_models.json "preferred" (qwen_pretranined_v6 on this cluster).
# Default batch size is 4 for 16GB GPUs.

set -eo pipefail

if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

# Llama is loaded via explicit --cache-dir, not HF_HOME. MiST is resolved
# under $HF_HOME/mist. When HF_HOME is unset, use the default Hugging Face cache.
if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="${HOME}/.cache/huggingface"
fi
mkdir -p "${HF_HOME}"
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use the python already on PATH. See README.md.

# ── Config ────────────────────────────────────────────────────────────────────
N="${N:-0}"  # 0 = every eligible molecule
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-./data/molopt/analysis}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
QWEN_LLM_MODEL="${QWEN_LLM_MODEL:-Qwen/Qwen3-1.7B}"
LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
MIST_MODEL="${MIST_MODEL:-Qwen2.5-3B_pretrained-v4-cot}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
POOLING="${POOLING:-last_instruction}"
MAX_LENGTH="${MAX_LENGTH:-128}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-64}"
QWEN_LLM_BATCH_SIZE="${QWEN_LLM_BATCH_SIZE:-16}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-16}"
MIST_BATCH_SIZE="${MIST_BATCH_SIZE:-4}"
LABEL_FIELD="${LABEL_FIELD:-category_normalized}"
ENCODERS="${ENCODERS:-}"
VARIANTS="${VARIANTS:-}"
REUSE_PREVIOUS="${REUSE_PREVIOUS:-1}"

# ── Validate ──────────────────────────────────────────────────────────────────
DEFINITIONS="${DEFINITIONS:-./data/molopt/train/definitions.jsonl}"
RDKIT_DEFINITIONS="${RDKIT_DEFINITIONS:-./data/molopt/train/definitions_rdkit.jsonl}"
if [ ! -f "${DEFINITIONS}" ]; then
    echo "ERROR: '${DEFINITIONS}' not found." >&2
    echo "       Build it with: python scripts/prepare_molopt_chebi20.py --download" >&2
    exit 1
fi
if [ ! -f "${RDKIT_DEFINITIONS}" ]; then
    echo "ERROR: '${RDKIT_DEFINITIONS}' not found." >&2
    exit 1
fi

echo "[analyze_molopt_encoder_cluster] N=${N} SEED=${SEED} DEVICE=${DEVICE} OUT_DIR=${OUT_DIR}"
echo "[analyze_molopt_encoder_cluster] QWEN_MODEL=${QWEN_MODEL} QWEN_LLM_MODEL=${QWEN_LLM_MODEL}"
echo "[analyze_molopt_encoder_cluster] LLAMA_MODEL=${LLAMA_MODEL}"
echo "[analyze_molopt_encoder_cluster] MIST_MODEL=${MIST_MODEL} HF_HOME=${HF_HOME}"
echo "[analyze_molopt_encoder_cluster] CACHE_DIR=${CACHE_DIR} POOLING=${POOLING} MAX_LENGTH=${MAX_LENGTH}"
echo "[analyze_molopt_encoder_cluster] ENCODERS=${ENCODERS:-<all>} VARIANTS=${VARIANTS:-<both>} REUSE_PREVIOUS=${REUSE_PREVIOUS}"

# ── Run ───────────────────────────────────────────────────────────────────────
ARGS=(
    --definitions         "${DEFINITIONS}"
    --rdkit-definitions   "${RDKIT_DEFINITIONS}"
    --n                   "${N}"
    --seed                "${SEED}"
    --label-field         "${LABEL_FIELD}"
    --qwen-model          "${QWEN_MODEL}"
    --qwen-llm-model      "${QWEN_LLM_MODEL}"
    --llama-model         "${LLAMA_MODEL}"
    --mist-model          "${MIST_MODEL}"
    --cache-dir           "${CACHE_DIR}"
    --pooling             "${POOLING}"
    --max-length          "${MAX_LENGTH}"
    --qwen-batch-size     "${QWEN_BATCH_SIZE}"
    --qwen-llm-batch-size "${QWEN_LLM_BATCH_SIZE}"
    --llama-batch-size    "${LLAMA_BATCH_SIZE}"
    --mist-batch-size     "${MIST_BATCH_SIZE}"
    --device              "${DEVICE}"
    --out-dir             "${OUT_DIR}"
)
if [ -n "${ENCODERS}" ]; then
    ARGS+=(--encoders ${ENCODERS})
fi
if [ -n "${VARIANTS}" ]; then
    ARGS+=(--variants ${VARIANTS})
fi
if [ "${REUSE_PREVIOUS}" = "1" ] || [ "${REUSE_PREVIOUS}" = "true" ] || [ "${REUSE_PREVIOUS}" = "yes" ]; then
    ARGS+=(--reuse-previous)
fi
if [ "${NO_WANDB:-0}" = "1" ] || [ "${NO_WANDB:-0}" = "true" ] || [ "${NO_WANDB:-0}" = "yes" ]; then
    ARGS+=(--no-wandb)
else
    ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi

python -u scripts/analyze_molopt_encoder_cluster.py "${ARGS[@]}"
