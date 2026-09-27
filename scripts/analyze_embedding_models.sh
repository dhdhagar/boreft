#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-04:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint="vram16"

# Compare candidate molopt embedding models on SMILES <-> description alignment.
# See notes/embedding_models.md for what the output means.
#
# Usage:
#   sbatch scripts/analyze_embedding_models.sh
#   sbatch --export=ALL,N=3000 scripts/analyze_embedding_models.sh
#   sbatch --export=ALL,MODELS="chemate qwen3" scripts/analyze_embedding_models.sh
#
# The script is CPU-runnable but slow: Qwen3-0.6B over ~5k strings dominates, so
# a GPU turns a multi-hour run into minutes. Retrieval chance level is 1/N, so
# only compare runs taken at the same N.

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
N="${N:-3000}"
SEED="${SEED:-0}"
STRUCTURE_PAIRS="${STRUCTURE_PAIRS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-./data/molopt/analysis}"
MODELS="${MODELS:-}"

# MoleculeSTM is the only model needing more than the repo's own requirements.
# ogb must stay at 1.3.5 — see the install_hint comment in the Python script.
#   pip install torch_geometric 'ogb==1.3.5'
#   pip install --no-build-isolation torch_scatter
#   pip install git+https://github.com/chao1224/MoleculeSTM.git
#   huggingface-cli download chao1224/MoleculeSTM \
#     --include 'demo/demo_checkpoints_Graph/*' --local-dir data/molopt/raw/MoleculeSTM
# Any model whose dependencies are missing is skipped with an install hint
# rather than failing the run.
MOLECULESTM_DIR="${MOLECULESTM_DIR:-./data/molopt/raw/MoleculeSTM/demo/demo_checkpoints_Graph}"

# ── Validate ──────────────────────────────────────────────────────────────────
DEFINITIONS="${DEFINITIONS:-./data/molopt/train/definitions.jsonl}"
if [ ! -f "${DEFINITIONS}" ]; then
    echo "ERROR: '${DEFINITIONS}' not found." >&2
    echo "       Build it with: python scripts/prepare_molopt_chebi20.py --download" >&2
    exit 1
fi

echo "[analyze_embedding_models] N=${N} SEED=${SEED} DEVICE=${DEVICE} OUT_DIR=${OUT_DIR}"
echo "[analyze_embedding_models] MODELS=${MODELS:-<all available>}"

# ── Run ───────────────────────────────────────────────────────────────────────
ARGS=(
    --definitions        "${DEFINITIONS}"
    --n                  "${N}"
    --seed               "${SEED}"
    --n-structure-pairs  "${STRUCTURE_PAIRS}"
    --batch-size         "${BATCH_SIZE}"
    --device             "${DEVICE}"
    --out-dir            "${OUT_DIR}"
    --moleculestm-dir    "${MOLECULESTM_DIR}"
)
# Unquoted on purpose: MODELS is a space-separated list for nargs="*".
if [ -n "${MODELS}" ]; then
    ARGS+=(--models ${MODELS})
fi
if [ "${NO_WANDB:-0}" = "1" ] || [ "${NO_WANDB:-0}" = "true" ] || [ "${NO_WANDB:-0}" = "yes" ]; then
    ARGS+=(--no-wandb)
else
    ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi

python -u scripts/analyze_embedding_models.py "${ARGS[@]}"
