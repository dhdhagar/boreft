#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -p cpu
#SBATCH -t 0-01:00:00
#SBATCH -o ./scripts/jobs/%j.out

# Log-volume comparison of AABB k=0/0.1/0.5 vs the covering ellipsoid.
# CPU-only: reads bias_tables.pt from the canonical Semantle checkpoint.
#
# Submit from the repo root:
#   mkdir -p scripts/jobs
#   sbatch scripts/compare_search_domain_volumes.sh
#   sbatch --export=ALL,OUTPUT_DIR=outputs/1784053292 \
#     scripts/compare_search_domain_volumes.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
mkdir -p scripts/jobs
if [ ! -d src/boreft ]; then
    echo "ERROR: submit sbatch from the repo root (missing src/boreft)." >&2
    echo "       cwd=$(pwd)" >&2
    exit 1
fi

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Use the python already on PATH. See README.md.

OUTPUT_DIR="${OUTPUT_DIR:-outputs/1784053292}"
JSON_OUT="${JSON_OUT:-data/semantle/analysis/search_domain_volumes.json}"
AABB_STD_K="${AABB_STD_K:-0 0.1 0.5}"

echo "[domain-vol] output_dir=${OUTPUT_DIR} json_out=${JSON_OUT} k=${AABB_STD_K}"

python -u scripts/compare_search_domain_volumes.py \
    --output-dir "${OUTPUT_DIR}" \
    --json-out "${JSON_OUT}" \
    --aabb-std-k ${AABB_STD_K}
