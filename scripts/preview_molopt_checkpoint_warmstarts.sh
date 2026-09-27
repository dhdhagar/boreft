#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu,gpu-preempt,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -t 0-04:00:00
#SBATCH -o jobs/%j.out
#SBATCH --constraint=a100

# Sample reconstructed checkpoint-μ warmstarts and score DRD2 / GSK3B / JNK3.
#
#   mkdir -p jobs
#   sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789921464 \
#     scripts/preview_molopt_checkpoint_warmstarts.sh
#   sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789933836 \
#     scripts/preview_molopt_checkpoint_warmstarts.sh
#   sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789933841 \
#     scripts/preview_molopt_checkpoint_warmstarts.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
mkdir -p jobs
if [ ! -d src/boreft ]; then
    echo "ERROR: submit sbatch from the repo root (missing src/boreft)." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
if [ ! -d "${HF_HOME}" ]; then
    export HF_HOME="$(pwd)/models"
    mkdir -p "${HF_HOME}"
fi
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use the python already on PATH. See README.md.

REFT_OUTPUT_DIR="${REFT_OUTPUT_DIR:-}"
if [ -z "${REFT_OUTPUT_DIR}" ]; then
    echo "ERROR: REFT_OUTPUT_DIR is required." >&2
    exit 1
fi

ARGS=(
    --reft-output-dir "${REFT_OUTPUT_DIR}"
    --seeds ${SEEDS:-1 2 3 4 5}
    --warmstart-count "${WARMSTART_COUNT:-10}"
    --oracles ${ORACLES:-DRD2 GSK3B JNK3}
    --cache-dir "${CACHE_DIR:-${HF_HOME}}"
)
if [ -n "${OUT_DIR:-}" ]; then
    ARGS+=(--output-dir "${OUT_DIR}")
fi
if [ -n "${WARMSTART_FILE:-}" ]; then
    ARGS+=(--warmstart-file "${WARMSTART_FILE}")
fi
if [ "${LOAD_LATEST:-0}" = "1" ]; then
    ARGS+=(--load-latest)
fi
if [ "${SKIP_ORACLE_CHECK:-0}" = "1" ]; then
    ARGS+=(--skip-oracle-check)
fi

echo "[checkpoint_warmstarts] reft=${REFT_OUTPUT_DIR}"
python -u scripts/preview_molopt_checkpoint_warmstarts.py "${ARGS[@]}"
