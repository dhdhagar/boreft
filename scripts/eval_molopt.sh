#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint="vram16"

# Post-training eval for a molopt checkpoint: the full RECON / RECON_TEST / DIST /
# GENZ / LIPZ suite. The embedding model, the Morgan/Tanimoto
# TFS metrics, and the held-out pool are all selected from the checkpoint's own task,
# so nothing here is molopt-specific beyond the defaults.
#
# Model, layer, rank, and sample count are read from the checkpoint's
# training_config.json unless the matching variable is exported, so an eval can
# never silently disagree with how the run was trained.
#
# Usage (OUTPUT_DIR required — same as boreft.train --output-dir):
#   sbatch --export=ALL,OUTPUT_DIR=./outputs/out_<run_name> scripts/eval_molopt.sh
#   sbatch --export=ALL,OUTPUT_DIR=./outputs/out_<run>,N_SAMPLES=10 scripts/eval_molopt.sh

if [ ! -d "./scripts/jobs" ]; then
    mkdir -p ./scripts/jobs
fi

export HF_HOME="${HF_HOME:-$(pwd)/models}"
mkdir -p "${HF_HOME}"
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
# Unbuffered Python so SLURM .out updates live (otherwise logs look "stuck").
export PYTHONUNBUFFERED=1

# Use the python already on PATH. See README.md.

# ── Config ────────────────────────────────────────────────────────────────────
# Only OUTPUT_DIR is required; every other value falls back to the checkpoint.
OUTPUT_DIR="${OUTPUT_DIR:-}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: set OUTPUT_DIR, e.g. --export=ALL,OUTPUT_DIR=./outputs/out_<run>" >&2
    exit 1
fi
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: output_dir '${OUTPUT_DIR}' does not exist." >&2
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/items.json" ]; then
    echo "ERROR: '${OUTPUT_DIR}/items.json' not found. Is this a valid checkpoint?" >&2
    exit 1
fi

echo "[eval_molopt] OUTPUT_DIR=${OUTPUT_DIR}  CACHE_DIR=${CACHE_DIR}"
echo "[eval_molopt] overrides: MODEL_NAME=${MODEL_NAME:-<checkpoint>}" \
     "LAYER=${LAYER:-<checkpoint>} LOW_RANK_DIM=${LOW_RANK_DIM:-<checkpoint>}" \
     "N_SAMPLES=${N_SAMPLES:-<checkpoint>}"

# ── Run ───────────────────────────────────────────────────────────────────────
EVAL_ARGS=(
    --output_dir    "${OUTPUT_DIR}"
    --cache_dir     "${CACHE_DIR}"
)
[ -n "${MODEL_NAME}" ]          && EVAL_ARGS+=(--model-name "${MODEL_NAME}")
[ -n "${LAYER}" ]               && EVAL_ARGS+=(--layer "${LAYER}")
[ -n "${LOW_RANK_DIM}" ]        && EVAL_ARGS+=(--low_rank_dim "${LOW_RANK_DIM}")
[ -n "${N_SAMPLES}" ]           && EVAL_ARGS+=(--n_samples "${N_SAMPLES}")
[ -n "${FULL_EVAL_N_SAMPLES}" ] && EVAL_ARGS+=(--full_eval_n_samples "${FULL_EVAL_N_SAMPLES}")
[ -n "${TORCH_DTYPE}" ]         && EVAL_ARGS+=(--torch-dtype "${TORCH_DTYPE}")
python -u -m boreft.eval.run_full_eval "${EVAL_ARGS[@]}"
