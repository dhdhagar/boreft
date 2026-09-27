#!/bin/bash
# Launch BOReFT and the standard baselines on DRD2 / GSK3B / JNK3 / GSK3B_JNK3.
# Default warm starts are labeled training means (no Sobol dump).
# Pass ``--warmstart-source file`` to dump/share Sobol files first.
#
#   ./experiments/molopt/launch_all.sh --boreft_dir outputs/1789222254
#   ./experiments/molopt/launch_all.sh --boreft_dir outputs/1789222254 --dry-run
#   ./experiments/molopt/launch_all.sh --boreft_dir outputs/1789222254 --search-prompt task
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WARMSTART_SOURCE=checkpoint
prev=""
for arg in "$@"; do
  case "${arg}" in
    --warmstart-source=*|--warmstart_source=*)
      WARMSTART_SOURCE="${arg#*=}"
      ;;
  esac
  if [[ "${prev}" == "--warmstart-source" || "${prev}" == "--warmstart_source" ]]; then
    WARMSTART_SOURCE="${arg}"
  fi
  prev="${arg}"
done

COMMON_ARGS=("$@")
if [[ "${WARMSTART_SOURCE}" == "file" ]]; then
  DUMP_OUTPUT="$("${DIR}/dump_warmstarts.sh" "$@")"
  printf '%s\n' "${DUMP_OUTPUT}"
  DUMP_JOB_ID="$(
    printf '%s\n' "${DUMP_OUTPUT}" | awk -F= '/^DUMP_JOB_ID=/{print $2}' | tail -n 1
  )"
  if [[ -n "${DUMP_JOB_ID}" ]]; then
    COMMON_ARGS+=(--dependency "afterok:${DUMP_JOB_ID}")
    echo "[molopt] search jobs depend on dump job ${DUMP_JOB_ID}"
  fi
fi

for method in boreft random_sampling discrete_bo bopro opro sdpo_ttt migrate autodiscovery; do
  "${DIR}/${method}.sh" "${COMMON_ARGS[@]}"
done
