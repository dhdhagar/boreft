#!/bin/bash
# Discrete BO over the checkpoint train SMILES catalog.
#
#   ./experiments/molopt/discrete_bo.sh --boreft_dir outputs/1789222254
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" discrete_bo "$@"
