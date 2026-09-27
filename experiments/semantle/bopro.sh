#!/bin/bash
# BOPRO baseline search on sampled Semantle targets.
#
#   ./experiments/semantle/bopro.sh --boreft_dir outputs/1784053292
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" bopro "$@"
