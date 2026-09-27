#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint="vram16"

# Synthetic RDKit-definition embedding experiment: Qwen cosine vs. RBF rdkit_sim,
# with and without a held-constant ChEBI sentence in front of the suffix.
# See scripts/analyze_rdkit_definition_embeds.py.
#
# Usage:
#   sbatch scripts/analyze_rdkit_definition_embeds.sh
#   sbatch --export=ALL,N_BASES=400 scripts/analyze_rdkit_definition_embeds.sh

if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

export HF_HOME="${HF_HOME:-$(pwd)/models}"
mkdir -p "${HF_HOME}"
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use the python already on PATH. See README.md.

# ── Config ────────────────────────────────────────────────────────────────────
N_BASES="${N_BASES:-200}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-./data/molopt/analysis}"
MODEL="${MODEL:-Qwen/Qwen3-Embedding-0.6B}"
TARGET_SIMS="${TARGET_SIMS:-1.0 0.9 0.7 0.5 0.3 0.1}"

# ── Validate ──────────────────────────────────────────────────────────────────
DEFINITIONS="${DEFINITIONS:-./data/molopt/train/definitions.jsonl}"
RDKIT_MAP="${RDKIT_MAP:-./data/molopt/train/definitions_rdkit_map.json}"
if [ ! -f "${DEFINITIONS}" ]; then
    echo "ERROR: '${DEFINITIONS}' not found." >&2
    echo "       Build it with: python scripts/prepare_molopt_chebi20.py --download" >&2
    exit 1
fi
if [ ! -f "${RDKIT_MAP}" ]; then
    echo "ERROR: '${RDKIT_MAP}' not found." >&2
    exit 1
fi

echo "[analyze_rdkit_definition_embeds] N_BASES=${N_BASES} SEED=${SEED} DEVICE=${DEVICE}"
echo "[analyze_rdkit_definition_embeds] MODEL=${MODEL} OUT_DIR=${OUT_DIR}"

WANDB_ARGS=()
if [ "${NO_WANDB:-0}" = "1" ] || [ "${NO_WANDB:-0}" = "true" ] || [ "${NO_WANDB:-0}" = "yes" ]; then
    WANDB_ARGS+=(--no-wandb)
else
    WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi

# ── Run ───────────────────────────────────────────────────────────────────────
# Unquoted TARGET_SIMS on purpose: space-separated list for nargs="+".
python -u scripts/analyze_rdkit_definition_embeds.py \
    --n-bases            "${N_BASES}" \
    --target-sims        ${TARGET_SIMS} \
    --seed               "${SEED}" \
    --definitions        "${DEFINITIONS}" \
    --rdkit-map          "${RDKIT_MAP}" \
    --model              "${MODEL}" \
    --batch-size         "${BATCH_SIZE}" \
    --device             "${DEVICE}" \
    --out-dir            "${OUT_DIR}" \
    "${WANDB_ARGS[@]}"
