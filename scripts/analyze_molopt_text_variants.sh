#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-08:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint="vram16"

# Which molopt text representation clusters by ChEBI category under Qwen?
# See scripts/analyze_molopt_text_variants.py.
#
# Usage:
#   sbatch scripts/analyze_molopt_text_variants.sh
#   sbatch --export=ALL,N=0,SEED=0 scripts/analyze_molopt_text_variants.sh
#   sbatch --export=ALL,N=1000 scripts/analyze_molopt_text_variants.sh
#   sbatch --export=ALL,VARIANTS="smiles defn smiles_defn" scripts/analyze_molopt_text_variants.sh
#
# iupac / iupac_defn use RDKit InChI of the parsed SMILES (no STOUT / no API).

set -eo pipefail

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
N="${N:-0}"  # 0 = every eligible molecule
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-./data/molopt/analysis}"
MODEL="${MODEL:-Qwen/Qwen3-Embedding-0.6B}"
LABEL_FIELD="${LABEL_FIELD:-category_normalized}"
VARIANTS="${VARIANTS:-}"

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

echo "[analyze_molopt_text_variants] N=${N} SEED=${SEED} DEVICE=${DEVICE} OUT_DIR=${OUT_DIR}"
echo "[analyze_molopt_text_variants] MODEL=${MODEL} LABEL_FIELD=${LABEL_FIELD}"
echo "[analyze_molopt_text_variants] VARIANTS=${VARIANTS:-<all>}"

# ── Run ───────────────────────────────────────────────────────────────────────
ARGS=(
    --definitions         "${DEFINITIONS}"
    --rdkit-definitions   "${RDKIT_DEFINITIONS}"
    --n                   "${N}"
    --seed                "${SEED}"
    --label-field         "${LABEL_FIELD}"
    --model               "${MODEL}"
    --batch-size          "${BATCH_SIZE}"
    --device              "${DEVICE}"
    --out-dir             "${OUT_DIR}"
)
# Unquoted on purpose: VARIANTS is a space-separated list for nargs="*".
if [ -n "${VARIANTS}" ]; then
    ARGS+=(--variants ${VARIANTS})
fi
if [ "${NO_WANDB:-0}" = "1" ] || [ "${NO_WANDB:-0}" = "true" ] || [ "${NO_WANDB:-0}" = "yes" ]; then
    ARGS+=(--no-wandb)
else
    ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi

python -u scripts/analyze_molopt_text_variants.py "${ARGS[@]}"
