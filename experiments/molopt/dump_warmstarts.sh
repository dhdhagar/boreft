#!/bin/bash
# Dump five shared Sobol warmstart sets (T=1 decode) for the optional
# ``--warmstart-source file`` molopt protocol. The default launcher now uses
# labeled training means (``--warmstart-source checkpoint``) and skips this.
#
#   ./experiments/molopt/dump_warmstarts.sh --boreft_dir outputs/1789222254
#   ./experiments/molopt/dump_warmstarts.sh --boreft_dir outputs/1789222254 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BOREFT_DIR=""
DRY_RUN=0
OVERWRITE=0
WARMSTART_DIR=""
SEEDS=(1 2 3 4 5)
WARMSTART_COUNT=10
SAMPLING_TEMPERATURE=1

SBATCH_PARTITION="gpu,gpu-preempt,superpod-a100"
SBATCH_GRES="gpu:1"
SBATCH_MEM="48G"
SBATCH_CPUS=4
SBATCH_CONSTRAINT="a100"
SBATCH_TIME="04:00:00"

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
    --boreft_dir|--boreft-dir)
      require_value "$1" "${2:-}"
      BOREFT_DIR="$2"
      shift 2
      ;;
    --boreft_dir=*|--boreft-dir=*)
      BOREFT_DIR="${1#*=}"
      shift
      ;;
    --warmstart-dir|--warmstart_dir)
      require_value "$1" "${2:-}"
      WARMSTART_DIR="$2"
      shift 2
      ;;
    --warmstart-dir=*|--warmstart_dir=*)
      WARMSTART_DIR="${1#*=}"
      shift
      ;;
    --dry-run|--dry_run)
      DRY_RUN=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --oracles)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        shift
      done
      ;;
    --method-dir|--method_dir|--protocol-cell|--protocol_cell|--dependency|--observation-samples|--observation_samples|--sampling-temperature|--sampling_temperature|--projection-dim|--projection_dim|--warmstart-source|--warmstart_source|--search-prompt|--search_prompt)
      require_value "$1" "${2:-}"
      shift 2
      ;;
    --method-dir=*|--method_dir=*|--protocol-cell=*|--protocol_cell=*|--dependency=*|--warmstart-source=*|--warmstart_source=*|--search-prompt=*|--search_prompt=*)
      shift
      ;;
    --no-wandb|--no_wandb|--resume|--use-ard|--use_ard)
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${BOREFT_DIR}" ]]; then
  echo "--boreft_dir is required" >&2
  exit 1
fi

if [[ -d "${BOREFT_DIR}" ]]; then
  BOREFT_DIR="$(cd "${BOREFT_DIR}" && pwd)"
elif [[ "${BOREFT_DIR}" != /* && -d "${REPO_ROOT}/${BOREFT_DIR}" ]]; then
  BOREFT_DIR="$(cd "${REPO_ROOT}/${BOREFT_DIR}" && pwd)"
else
  echo "checkpoint directory does not exist: ${BOREFT_DIR}" >&2
  exit 1
fi
case "${BOREFT_DIR}" in
  "${REPO_ROOT}"/*) BOREFT_DIR_ARG="${BOREFT_DIR#"${REPO_ROOT}"/}" ;;
  *) BOREFT_DIR_ARG="${BOREFT_DIR}" ;;
esac

RUN_ID="$(basename "${BOREFT_DIR}")"
if [[ -z "${WARMSTART_DIR}" ]]; then
  WARMSTART_DIR="${REPO_ROOT}/experiments/outputs/molopt/warmstarts/${RUN_ID}"
fi
mkdir -p "${WARMSTART_DIR}"
WARMSTART_DIR="$(cd "${WARMSTART_DIR}" && pwd)"
case "${WARMSTART_DIR}" in
  "${REPO_ROOT}"/*) WARMSTART_DIR_ARG="${WARMSTART_DIR#"${REPO_ROOT}"/}" ;;
  *) WARMSTART_DIR_ARG="${WARMSTART_DIR}" ;;
esac

PYTHON_CMD=(
  scripts/dump_molopt_sobol_warmstarts.py
  --reft-output-dir "${BOREFT_DIR_ARG}"
  --output-dir "${WARMSTART_DIR_ARG}"
  --seeds "${SEEDS[@]}"
  --warmstart-count "${WARMSTART_COUNT}"
  --sampling-temperature "${SAMPLING_TEMPERATURE}"
)
if [[ "${OVERWRITE}" -eq 1 ]]; then
  PYTHON_CMD+=(--overwrite)
fi

missing=0
for seed in "${SEEDS[@]}"; do
  if [[ ! -f "${WARMSTART_DIR}/seed_${seed}.jsonl" ]]; then
    missing=1
    break
  fi
done
if [[ "${missing}" -eq 0 && "${OVERWRITE}" -eq 0 ]]; then
  echo "[molopt] warmstarts already in ${WARMSTART_DIR_ARG}"
  echo "DUMP_JOB_ID="
  exit 0
fi

ts="$(date +%s)"
job_name="boreft_dump_molopt_warmstarts_${RUN_ID}_${ts}"
echo "[molopt] dump warmstarts -> ${WARMSTART_DIR_ARG}"
echo "  job ${job_name}"
printf '  cmd python'
printf ' %q' "${PYTHON_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "  dry-run    not submitted"
  echo "DUMP_JOB_ID="
  exit 0
fi

cd "${REPO_ROOT}"
mkdir -p jobs
submitted="$(
  sbatch --chdir="${REPO_ROOT}" \
    -J "${job_name}" \
    -e "${REPO_ROOT}/jobs/${job_name}.err" \
    -o "${REPO_ROOT}/jobs/${job_name}.log" \
    --partition="${SBATCH_PARTITION}" \
    --gres="${SBATCH_GRES}" \
    --mem="${SBATCH_MEM}" \
    -c "${SBATCH_CPUS}" \
    --constraint="${SBATCH_CONSTRAINT}" \
    --time="${SBATCH_TIME}" \
    "${REPO_ROOT}/run_sbatch.sh" \
    "${PYTHON_CMD[@]}"
)"
echo "  ${submitted}"
job_id="${submitted##* }"
echo "DUMP_JOB_ID=${job_id}"
