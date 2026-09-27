#!/bin/bash
# Protocol search in the SDPO-off, no-VAE, joint, and reconstruction-off
# ablation spaces, matching the published canonical BOReFT cell (s1_t0_ard_d64).
# Writes under experiments/outputs/semantle/search/boreft_sdpo0,
# .../boreft_novae, .../boreft_sdpo0_novae, .../boreft_ce0, and
# .../boreft_ce0_t1 (CE-off at T=1) so search/boreft/ is left alone.
#   python experiments/semantle/compare_spaces.py
#
# Default is --resume (same as the hyperparameter sweep). Re-running does not
# wipe finished seeds. Pass --overwrite to start the 20 jobs from scratch.
# Replacing the no-VAE checkpoint requires --overwrite on that tree only:
#   ./experiments/semantle/boreft.sh --boreft_dir outputs/1789101817 \
#     --method-dir boreft_novae --protocol-cell s1_t0_ard_d64 \
#     --targets-json experiments/outputs/semantle/sweep/targets.json --overwrite
#
#   ./experiments/semantle/search_spaces.sh
#   ./experiments/semantle/search_spaces.sh --dry-run
#   ./experiments/semantle/search_spaces.sh --overwrite
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${DIR}/boreft.sh" --boreft_dir outputs/1788622157 \
  --method-dir boreft_sdpo0 --protocol-cell s1_t0_ard_d64 \
  --targets-json experiments/outputs/semantle/sweep/targets.json \
  "$@"

"${DIR}/boreft.sh" --boreft_dir outputs/1789101817 \
  --method-dir boreft_novae --protocol-cell s1_t0_ard_d64 \
  --targets-json experiments/outputs/semantle/sweep/targets.json \
  "$@"

"${DIR}/boreft.sh" --boreft_dir outputs/1789021668 \
  --method-dir boreft_sdpo0_novae --protocol-cell s1_t0_ard_d64 \
  --targets-json experiments/outputs/semantle/sweep/targets.json \
  "$@"

"${DIR}/boreft.sh" --boreft_dir outputs/1789136583 \
  --method-dir boreft_ce0 --protocol-cell s1_t0_ard_d64 \
  --targets-json experiments/outputs/semantle/sweep/targets.json \
  "$@"

"${DIR}/boreft.sh" --boreft_dir outputs/1789136583 \
  --method-dir boreft_ce0_t1 --protocol-cell s1_t1_ard_d64 \
  --targets-json experiments/outputs/semantle/sweep/targets.json \
  "$@"
