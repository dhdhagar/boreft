#!/bin/bash
# OPRO baseline search on sampled Semantle targets.
#
#   ./experiments/semantle/opro.sh --boreft_dir outputs/1784053292
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" opro "$@"
