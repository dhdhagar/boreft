#!/bin/bash
# Discrete BO over the checkpoint train vocabulary (all unique items.json words).
#
#   ./experiments/semantle/discrete_bo.sh --boreft_dir outputs/1784053292
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" discrete_bo "$@"
