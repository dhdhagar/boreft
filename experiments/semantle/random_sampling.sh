#!/bin/bash
# Random-sampling baseline search on sampled Semantle targets.
#
#   ./experiments/semantle/random_sampling.sh --boreft_dir outputs/1784053292
# LoRA-SFT proposal (last-k 0): ./experiments/semantle/random_sampling_lora.sh --epochs 1,9
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/_common.sh" random_sampling "$@"
