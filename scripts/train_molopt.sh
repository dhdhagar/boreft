#!/bin/bash
#SBATCH -c 4                        # Number of Cores per Task
#SBATCH --mem=48G                   # Requested Memory
#SBATCH -p gpu                      # Partition
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH -t 2-00:00:00              # Job time limit
#SBATCH -o ./scripts/jobs/%j.out   # %j = job ID
#SBATCH --constraint="vram40"

# Example Run
#   sbatch scripts/train_molopt.sh
#   sbatch --export=ALL,TOP_K=500 scripts/train_molopt.sh

# Create the jobs directory if it doesn't exist
if [ ! -d "./scripts/jobs" ]; then
    echo "Creating ./scripts/jobs directory..."
    mkdir -p ./scripts/jobs
fi

# Llama is loaded via explicit --cache-dir arg, so TRANSFORMERS_CACHE is not set here.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
# Use the python already on PATH. See README.md.

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
# ChEBI-20 corpus built by `python scripts/prepare_molopt_chebi20.py --download`.
# With the default p90 oracle cap, TOP_K is the train-set size drawn from the
# capped pool; the high tail is the held-out test set. Every row has a ChEBI
# description in the matching definitions.jsonl.
MOLOPT_CSV="${MOLOPT_CSV:-./data/molopt/train/chebi20.csv}"
TOP_K="${TOP_K:-20}"
LAYER=13
LOW_RANK_DIM=64
POSITION="l1"
EPOCHS=2400
BATCH_SIZE=64
LR=1e-3
LR_SCHEDULER="linear"
WARMUP_RATIO=0.00
BETA=0.1
SIGMA_B=1.0
SEED=42
VARIANCE="learnable"

WANDB_PROJECT="molopt"
RUN_NAME="molopt-${TOP_K}mols-layer${LAYER}-rank${LOW_RANK_DIM}-beta${BETA}-epoch${EPOCHS}"
OUT_DIR="./outputs/out_${RUN_NAME}"

EVAL_STEPS=200
EVAL_N_SAMPLES=20

# ── Validate ──────────────────────────────────────────────────────────────────
if [ ! -f "${MOLOPT_CSV}" ]; then
    echo "ERROR: molopt CSV '${MOLOPT_CSV}' not found." >&2
    echo "       Build it with: python scripts/prepare_molopt_chebi20.py --download" >&2
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
python -m boreft.train \
    --task              molopt \
    --model-name        "${MODEL_NAME}" \
    --cache-dir         "${CACHE_DIR}" \
    --molopt-csv        "${MOLOPT_CSV}" \
    --train-top-k       ${TOP_K} \
    --layer             ${LAYER} \
    --low-rank-dim      ${LOW_RANK_DIM} \
    --position          "${POSITION}" \
    --epochs            ${EPOCHS} \
    --batch-size        ${BATCH_SIZE} \
    --lr                ${LR} \
    --lr-scheduler-type "${LR_SCHEDULER}" \
    --warmup-ratio      ${WARMUP_RATIO} \
    --kl-beta           ${BETA} \
    --sigma-b           ${SIGMA_B} \
    --output-dir        "${OUT_DIR}" \
    --seed              ${SEED} \
    --variance          "${VARIANCE}" \
    --wandb-project     "${WANDB_PROJECT}" \
    --wandb-run-name    "${RUN_NAME}" \
    --eval-steps        ${EVAL_STEPS} \
    --eval-n-samples    ${EVAL_N_SAMPLES} \
    --run-full-eval
