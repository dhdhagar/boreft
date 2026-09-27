#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH -o ./scripts/jobs/%j.out
#SBATCH --constraint="vram16"

# Usage (OUTPUT_DIR required — same as boreft.train --output-dir):
#   sbatch --export=ALL,OUTPUT_DIR=./outputs/out_<run_name> scripts/eval_semantle.sh
#   sbatch --export=ALL,OUTPUT_DIR=./outputs/out_<run>,N_SAMPLES=10 scripts/eval_semantle.sh
#   sbatch --export=ALL,OUTPUT_DIR=./outputs/out_<run>,TORCH_DTYPE=float16 scripts/eval_semantle.sh

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
# Honor environment when set (sbatch --export=ALL,...).
OUTPUT_DIR="${OUTPUT_DIR:-outputs/out_vae-20words-semantle-layer13-rank64-beta0.1-sigma1-epoch1000-stop1}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
LAYER="${LAYER:-13}"
LOW_RANK_DIM="${LOW_RANK_DIM:-64}"
N_SAMPLES="${N_SAMPLES:-25}"
FULL_EVAL_N_SAMPLES="${FULL_EVAL_N_SAMPLES:-}"

# ── Validate ──────────────────────────────────────────────────────────────────
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: output_dir '${OUTPUT_DIR}' does not exist." >&2
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/items.json" ]; then
    echo "ERROR: '${OUTPUT_DIR}/items.json' not found. Is this a valid checkpoint?" >&2
    exit 1
fi

echo "[eval_semantle] MODEL_NAME=${MODEL_NAME}  CACHE_DIR=${CACHE_DIR}"
echo "[eval_semantle] OUTPUT_DIR=${OUTPUT_DIR}  LAYER=${LAYER}  LOW_RANK_DIM=${LOW_RANK_DIM}"
echo "[eval_semantle] N_SAMPLES=${N_SAMPLES}  FULL_EVAL_N_SAMPLES=${FULL_EVAL_N_SAMPLES:-<all>}"

# ── Run ───────────────────────────────────────────────────────────────────────
EVAL_ARGS=(
    --output_dir    "${OUTPUT_DIR}"
    --model         "${MODEL_NAME}"
    --cache_dir     "${CACHE_DIR}"
    --layer         "${LAYER}"
    --low_rank_dim  "${LOW_RANK_DIM}"
    --n_samples     "${N_SAMPLES}"
)
[ -n "${FULL_EVAL_N_SAMPLES}" ] && EVAL_ARGS+=(--full_eval_n_samples "${FULL_EVAL_N_SAMPLES}")
[ -n "${TORCH_DTYPE}" ]         && EVAL_ARGS+=(--torch-dtype "${TORCH_DTYPE}")
python -u -m boreft.eval.semantle "${EVAL_ARGS[@]}"