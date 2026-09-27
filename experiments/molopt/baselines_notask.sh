#!/bin/bash
# No-header molopt baselines on the frozen decoder and the p90 LoRA-SFT adapter.
# The prompt is the completion prefix only (no objective line).
#
# Each job passes --overwrite, so a finished oracle directory is deleted and
# run again. Dirs:
#   <method>_mu_notask
#   <method>_lora_p90_e8_notask
#
#   ./experiments/molopt/baselines_notask.sh --dry-run
#   ./experiments/molopt/baselines_notask.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BOREFT_DIR="${BOREFT_DIR:-outputs/1789921464}"
ADAPTER="${ADAPTER:-experiments/outputs/molopt/lora_sft_random_1789921464/adapters/epoch008.pt}"
PREFIX="Here is a valid SMILES string for a molecule (only the SMILES string; no additional text):"
ORACLES=(DRD2 GSK3B JNK3 GSK3B_JNK3)
METHODS=(random_sampling bopro opro sdpo_ttt autodiscovery migrate)
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--dry_run) DRY_RUN=1; shift ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--dry-run]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${BOREFT_DIR}" ]]; then
  echo "checkpoint directory does not exist: ${BOREFT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER}" ]]; then
  echo "LoRA adapter does not exist: ${ADAPTER}" >&2
  exit 1
fi

assert_prompt() {
  local cfg="$1"
  local got
  if [[ ! -f "${cfg}" ]]; then
    return 0
  fi
  got="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("task_description",""))' "${cfg}")"
  if [[ "${got}" != "${PREFIX}" ]]; then
    echo "refusing to resume ${cfg}: task description is not the no-header prefix" >&2
    exit 1
  fi
}

launch_set() {
  local method="$1"
  local method_dir="$2"
  shift 2
  local oracle search_dir
  for oracle in "${ORACLES[@]}"; do
    search_dir="experiments/outputs/molopt/search/${method_dir}/${oracle}"
    assert_prompt "${search_dir}/config.json"
  done
  local flags=(
    --boreft_dir "${BOREFT_DIR}"
    --method-dir "${method_dir}"
    --oracles "${ORACLES[@]}"
    --overwrite
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    flags+=(--dry-run)
  fi
  echo "[notask] ${method}  method_dir=${method_dir}  oracles=${ORACLES[*]}  overwrite"
  "${DIR}/${method}.sh" "${flags[@]}" -- --task-description "${PREFIX}" "$@"
}

for method in "${METHODS[@]}"; do
  launch_set "${method}" "${method}_mu_notask"
  launch_set "${method}" "${method}_lora_p90_e8_notask" \
    --lora-sft-adapter "${ADAPTER}"
done
