#!/bin/bash
# OPRO baseline on TDC property oracles.
#
#   ./experiments/molopt/opro.sh --boreft_dir outputs/1789222254
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" opro "$@"
