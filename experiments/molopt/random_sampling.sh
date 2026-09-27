#!/bin/bash
# Random-sampling baseline on TDC property oracles.
#
#   ./experiments/molopt/random_sampling.sh --boreft_dir outputs/1789222254
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" random_sampling "$@"
