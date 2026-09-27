#!/bin/bash
# AutoDiscovery baseline search on sampled Semantle targets.
#
# Semantle knobs (via _common.sh): parent_context=20, exploration_constant=0.5,
# OPRO-style scored-history prompt. Writes under
# experiments/outputs/semantle/search/autodiscovery/ and overwrites any prior
# protocol run. After all 10 target jobs finish:
#   python experiments/semantle/analyze_results.py --wandb-force
#
#   ./experiments/semantle/autodiscovery.sh --boreft_dir outputs/1784053292
#   ./experiments/semantle/autodiscovery.sh --boreft_dir outputs/1784053292 --dry-run
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" autodiscovery "$@"
