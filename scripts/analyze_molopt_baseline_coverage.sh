#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu,gpu-preempt,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00:00
#SBATCH -o jobs/%j.out
#SBATCH --constraint=a100

# Coverage and μ-box distance for baseline molecules that beat BOReFT.
#
#   mkdir -p jobs
#   sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789921464 \
#     scripts/analyze_molopt_baseline_coverage.sh

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

REFT_OUTPUT_DIR="${REFT_OUTPUT_DIR:-outputs/1789921464}"
ARGS=(
    --reft-output-dir "${REFT_OUTPUT_DIR}"
    --search-root "${SEARCH_ROOT:-experiments/outputs/molopt/search}"
    --boreft-dir "${BOREFT_DIR:-boreft_mu}"
    --baseline-dirs ${BASELINE_DIRS:-random_sampling_mu bopro_mu opro_mu}
    --oracles ${ORACLES:-DRD2 GSK3B JNK3}
    --seeds ${SEEDS:-1 2 3 4 5}
    --cache-dir "${CACHE_DIR:-${HF_HOME}}"
)
if [ -n "${OUT:-}" ]; then
    ARGS+=(--output "${OUT}")
fi
if [ "${LOAD_LATEST:-0}" = "1" ]; then
    ARGS+=(--load-latest)
fi

echo "[coverage] reft=${REFT_OUTPUT_DIR} baselines=${BASELINE_DIRS:-random_sampling_mu bopro_mu opro_mu}"
python -u scripts/analyze_molopt_baseline_coverage.py "${ARGS[@]}"
