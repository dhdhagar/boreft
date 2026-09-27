#!/bin/bash
# Protocol search for prompt baselines on the LoRA-SFT proposal model.
# Same adapter as Random (post-SFT): epoch 9 of lora_sft_random_e10.
# Writes under search/<method>_lora_e9/ so the pretrained trees stay intact.
#
#   ./experiments/semantle/baselines_lora.sh --dry-run
#   ./experiments/semantle/baselines_lora.sh
#   ./experiments/semantle/baselines_lora.sh --methods bopro,opro --overwrite
#   ./experiments/semantle/baselines_lora.sh --epochs 9 --adapter-root experiments/outputs/semantle/lora_sft_random_e10
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
BOREFT_DIR="${BOREFT_DIR:-outputs/1784053292}"
ADAPTER_ROOT="${ADAPTER_ROOT:-experiments/outputs/semantle/lora_sft_random_e10}"
TARGETS_JSON="${TARGETS_JSON:-experiments/outputs/semantle/sweep/targets.json}"
EPOCHS="${EPOCHS:-9}"
METHODS="${METHODS:-sdpo_ttt,autodiscovery,migrate,bopro,opro}"
DRY_RUN=0
OVERWRITE=0
NO_WANDB=0
EXTRA=()

KNOWN_METHODS="sdpo_ttt autodiscovery migrate bopro opro"

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
    --no-wandb|--no_wandb) NO_WANDB=1; shift ;;
    --epochs|--epoch)
      require_value "$1" "${2:-}"
      EPOCHS="$2"
      shift 2
      ;;
    --epochs=*|--epoch=*)
      EPOCHS="${1#*=}"
      shift
      ;;
    --methods|--method)
      require_value "$1" "${2:-}"
      METHODS="$2"
      shift 2
      ;;
    --methods=*|--method=*)
      METHODS="${1#*=}"
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
      echo "expected: [--methods sdpo_ttt,autodiscovery,migrate,bopro,opro] [--epochs 9] [--dry-run] [--overwrite] [--boreft_dir DIR] [--adapter-root DIR]" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

IFS=',' read -r -a EPOCH_LIST <<< "${EPOCHS}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
if [[ ${#EPOCH_LIST[@]} -eq 0 || ${#METHOD_LIST[@]} -eq 0 ]]; then
  echo "no methods/epochs to launch" >&2
  exit 1
fi

for raw in "${METHOD_LIST[@]}"; do
  method="$(echo "${raw}" | tr -d '[:space:]')"
  case " ${KNOWN_METHODS} " in
    *" ${method} "*) ;;
    *)
      echo "unknown method: ${method}  (choose from: ${KNOWN_METHODS})" >&2
      exit 1
      ;;
  esac
  wrapper="${DIR}/${method}.sh"
  if [[ ! -x "${wrapper}" ]]; then
    echo "missing method launcher: ${wrapper}" >&2
    exit 1
  fi
done

for raw_epoch in "${EPOCH_LIST[@]}"; do
  epoch="$(echo "${raw_epoch}" | tr -d '[:space:]')"
  if [[ ! "${epoch}" =~ ^[0-9]+$ ]]; then
    echo "invalid epoch: ${raw_epoch}" >&2
    exit 1
  fi
  padded="$(printf '%03d' "${epoch}")"
  adapter="${ADAPTER_ROOT}/adapters/epoch${padded}.pt"
  if [[ ! -f "${adapter}" ]]; then
    echo "missing LoRA adapter: ${adapter}" >&2
    echo "train first: sbatch scripts/train_lora_sft_random.sh" >&2
    exit 1
  fi
  for raw_method in "${METHOD_LIST[@]}"; do
    method="$(echo "${raw_method}" | tr -d '[:space:]')"
    method_dir="${method}_lora_e${epoch}"
    echo "[lora-baselines] method=${method}  epoch=${epoch}  adapter=${adapter}  method_dir=${method_dir}"
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
    if [[ "${NO_WANDB}" -eq 1 ]]; then
      FLAGS+=(--no-wandb)
    fi
    "${DIR}/${method}.sh" \
      "${FLAGS[@]}" \
      -- \
      --lora-sft-adapter "${adapter}" \
      --wandb-group semantle-baselines-lora \
      "${EXTRA[@]}"
  done
done
