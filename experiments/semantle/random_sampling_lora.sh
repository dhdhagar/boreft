#!/bin/bash
# Random search with a LoRA-SFT proposal adapter from the e10 train run.
# Defaults to a task-only prompt (last-k 0). Published Random still uses last-k 20.
#
#   ./experiments/semantle/random_sampling_lora.sh
#   ./experiments/semantle/random_sampling_lora.sh --epochs 1,9
#   ./experiments/semantle/random_sampling_lora.sh --epochs 1,9 --dry-run
#   ./experiments/semantle/random_sampling_lora.sh --epochs 9 --overwrite
#   ./experiments/semantle/random_sampling_lora.sh -- --random-sampling.last-k-incontext 20
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
BOREFT_DIR="${BOREFT_DIR:-outputs/1784053292}"
ADAPTER_ROOT="${ADAPTER_ROOT:-experiments/outputs/semantle/lora_sft_random_e10}"
TARGETS_JSON="${TARGETS_JSON:-experiments/outputs/semantle/sweep/targets.json}"
EPOCHS="${EPOCHS:-1,9}"
DRY_RUN=0
OVERWRITE=0
EXTRA=()

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ $# -lt 2 || -z "${value}" || "${value}" == --* ]]; then
    echo "error: ${flag} requires a value" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--dry_run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --epochs|--epoch)
      require_value "$1" "${2:-}"
      EPOCHS="$2"
      shift 2
      ;;
    --epochs=*|--epoch=*)
      EPOCHS="${1#*=}"
      shift
      ;;
    --boreft_dir|--boreft-dir)
      require_value "$1" "${2:-}"
      BOREFT_DIR="$2"
      shift 2
      ;;
    --adapter-root)
      require_value "$1" "${2:-}"
      ADAPTER_ROOT="$2"
      shift 2
      ;;
    --targets-json|--targets_json)
      require_value "$1" "${2:-}"
      TARGETS_JSON="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "expected: [--epochs 1,9] [--dry-run] [--overwrite] [--boreft_dir DIR] [--adapter-root DIR]" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

IFS=',' read -r -a EPOCH_LIST <<< "${EPOCHS}"
if [[ ${#EPOCH_LIST[@]} -eq 0 ]]; then
  echo "no epochs to launch" >&2
  exit 1
fi

for raw in "${EPOCH_LIST[@]}"; do
  epoch="$(echo "${raw}" | tr -d '[:space:]')"
  if [[ ! "${epoch}" =~ ^[0-9]+$ ]]; then
    echo "invalid epoch: ${raw}" >&2
    exit 1
  fi
  padded="$(printf '%03d' "${epoch}")"
  adapter="${ADAPTER_ROOT}/adapters/epoch${padded}.pt"
  if [[ ! -f "${adapter}" ]]; then
    echo "missing LoRA adapter: ${adapter}" >&2
    echo "train first: sbatch scripts/train_lora_sft_random.sh" >&2
    exit 1
  fi
  method_dir="random_sampling_lora_e${epoch}"
  echo "[lora-search] epoch=${epoch}  adapter=${adapter}  method_dir=${method_dir}"
  FLAGS=(
    --boreft_dir "${BOREFT_DIR}"
    --method-dir "${method_dir}"
    --targets-json "${TARGETS_JSON}"
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    FLAGS+=(--dry-run)
  fi
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    FLAGS+=(--overwrite)
  fi
  "${DIR}/random_sampling.sh" \
    "${FLAGS[@]}" \
    -- \
    --lora-sft-adapter "${adapter}" \
    --random-sampling.last-k-incontext 0 \
    --wandb-group semantle-search \
    "${EXTRA[@]}"
done
