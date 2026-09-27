#!/bin/bash
# BOReFT projected search on sampled Semantle targets.
#
# Default cell is observation-samples=5. Theory overlays vs canonical use the
# sweep winner (1 sample, T=0, ARD, d=64) written into a separate tree:
#
#   ./experiments/semantle/boreft.sh --boreft_dir outputs/1784053292
#   ./experiments/semantle/boreft.sh --boreft_dir outputs/1788622157 \
#     --method-dir boreft_sdpo0 --protocol-cell s1_t0_ard_d64 \
#     --targets-json experiments/outputs/semantle/sweep/targets.json
#   ./experiments/semantle/search_spaces.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" boreft "$@"
