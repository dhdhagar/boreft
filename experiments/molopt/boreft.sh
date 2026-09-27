#!/bin/bash
# BOReFT projected search on TDC property oracles (DRD2, GSK3B, JNK3).
#
#   ./experiments/molopt/boreft.sh --boreft_dir outputs/1789222254
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" boreft "$@"
