#!/bin/bash
# AutoDiscovery prompt-check (OPRO-style prompt, isolated output tree).
#
# Current defaults: parent_context=20, C=0.5, budget 500, seeds 1 2 3,
# train targets carnivore / general / strawberry.
#
#   ./experiments/semantle/autodiscovery_prompt_check.sh --boreft_dir outputs/1784053292
#   ./experiments/semantle/autodiscovery_prompt_check.sh --compare \
#       --new-root experiments/outputs/semantle/search/autodiscovery_prompt_check_p20_c0.5_b500 \
#       --old-root experiments/outputs/semantle/search/autodiscovery
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
COMPARE_PY="${DIR}/compare_autodiscovery_prompt.py"

METHOD_NAME="autodiscovery"
CHECK_SLUG=""
TASK_DESCRIPTION="Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)."
WARMSTART_COUNT=10
WARMSTART_SOURCE=checkpoint
BATCH_SIZE=1
BUDGET=500
SEARCH_SEEDS=(1 2 3)
TARGETS=(carnivore general strawberry)
SPLIT="train"
PARENT_CONTEXT=20
EXPLORATION_CONSTANT=0.5

BOREFT_DIR=""
DRY_RUN=0
DO_COMPARE=0
NO_WANDB=1
WANDB_PROJECT="${WANDB_PROJECT:-boreft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="semantle-search-prompt-check"
EXTRA_ARGS=()

SBATCH_PARTITION="gpu,gpu-preempt,superpod-a100"
SBATCH_GRES="gpu:1"
SBATCH_MEM="48G"
SBATCH_CPUS=4
SBATCH_CONSTRAINT="a100"
SBATCH_TIME="12:00:00"

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ $# -lt 2 || -z "${value}" || "${value}" == --* ]]; then
    echo "error: ${flag} requires a value" >&2
    exit 1
  fi
}

require_int() {
  local flag="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^-?[0-9]+$ ]]; then
    echo "error: ${flag} must be an integer, got: ${value}" >&2
    exit 1
  fi
}

require_float() {
  local flag="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: ${flag} must be a nonnegative number, got: ${value}" >&2
    exit 1
  fi
}

usage() {
  echo "usage: $0 --boreft_dir DIR [--parent-context N] [--exploration-constant C] [--slug NAME] [--budget N] [--seeds S ...] [--targets W ...] [--wandb] [--dry-run]" >&2
  echo "       $0 --compare [--new-root DIR] [--old-root DIR]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compare)
      DO_COMPARE=1
      shift
      ;;
    --boreft_dir|--boreft-dir)
      require_value "$1" "${2:-}"
      BOREFT_DIR="$2"
      shift 2
      ;;
    --boreft_dir=*|--boreft-dir=*)
      BOREFT_DIR="${1#*=}"
      shift
      ;;
    --budget)
      require_value "$1" "${2:-}"
      require_int "$1" "$2"
      BUDGET="$2"
      shift 2
      ;;
    --budget=*)
      BUDGET="${1#*=}"
      require_int "--budget" "${BUDGET}"
      shift
      ;;
    --seeds)
      SEARCH_SEEDS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        require_int "--seeds" "$1"
        SEARCH_SEEDS+=("$1")
        shift
      done
      if [[ ${#SEARCH_SEEDS[@]} -eq 0 ]]; then
        echo "error: --seeds requires at least one integer" >&2
        exit 1
      fi
      ;;
    --targets)
      TARGETS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TARGETS+=("$1")
        shift
      done
      if [[ ${#TARGETS[@]} -eq 0 ]]; then
        echo "error: --targets requires at least one word" >&2
        exit 1
      fi
      ;;
    --parent-context|--parent_context)
      require_value "$1" "${2:-}"
      require_int "$1" "$2"
      PARENT_CONTEXT="$2"
      shift 2
      ;;
    --parent-context=*|--parent_context=*)
      PARENT_CONTEXT="${1#*=}"
      require_int "--parent-context" "${PARENT_CONTEXT}"
      shift
      ;;
    --exploration-constant|--exploration_constant)
      require_value "$1" "${2:-}"
      require_float "$1" "$2"
      EXPLORATION_CONSTANT="$2"
      shift 2
      ;;
    --exploration-constant=*|--exploration_constant=*)
      EXPLORATION_CONSTANT="${1#*=}"
      require_float "--exploration-constant" "${EXPLORATION_CONSTANT}"
      shift
      ;;
    --slug)
      require_value "$1" "${2:-}"
      CHECK_SLUG="$2"
      shift 2
      ;;
    --slug=*)
      CHECK_SLUG="${1#*=}"
      shift
      ;;
    --wandb)
      NO_WANDB=0
      shift
      ;;
    --no-wandb|--no_wandb)
      NO_WANDB=1
      shift
      ;;
    --dry-run|--dry_run)
      DRY_RUN=1
      shift
      ;;
    --new-root|--old-root)
      require_value "$1" "${2:-}"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${DO_COMPARE}" -eq 1 ]]; then
  cd "${REPO_ROOT}"
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    echo "python3 not found on PATH" >&2
    exit 1
  fi
  exec "${PYTHON}" "${COMPARE_PY}" "${EXTRA_ARGS[@]}"
fi

if [[ -z "${BOREFT_DIR}" ]]; then
  echo "--boreft_dir is required (or pass --compare)" >&2
  usage
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
if [[ ! -f "${BOREFT_DIR}/items.json" ]]; then
  echo "not a BOReFT checkpoint (missing items.json): ${BOREFT_DIR}" >&2
  exit 1
fi
case "${BOREFT_DIR}" in
  "${REPO_ROOT}"/*) BOREFT_DIR_ARG="${BOREFT_DIR#"${REPO_ROOT}"/}" ;;
  *) BOREFT_DIR_ARG="${BOREFT_DIR}" ;;
esac

dir_name_for() {
  local target="$1"
  local token
  token="$(printf '%s' "${target}" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//')"
  if [[ -z "${token}" ]]; then
    token="target"
  fi
  printf '%s-%s' "${SPLIT}" "${token}"
}

python_cmd_for_run() {
  local target="$1"
  local search_dir="$2"
  PYTHON_CMD=(
    -m boreft.baselines.search
    --baseline "${METHOD_NAME}"
    --task semantle
    --task-description "${TASK_DESCRIPTION}"
    --target "${target}"
    --reft-output-dir "${BOREFT_DIR_ARG}"
    --search-dir "${search_dir}"
    --budget "${BUDGET}"
    --warmstart-count "${WARMSTART_COUNT}"
    --warmstart-source "${WARMSTART_SOURCE}"
    --batch-size "${BATCH_SIZE}"
    --seeds "${SEARCH_SEEDS[@]}"
    --overwrite
    --autodiscovery.parent-context "${PARENT_CONTEXT}"
    --autodiscovery.exploration-constant "${EXPLORATION_CONSTANT}"
  )
  if [[ "${NO_WANDB}" -eq 0 ]]; then
    PYTHON_CMD+=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
      --wandb-group "${WANDB_GROUP}"
    )
  else
    PYTHON_CMD+=(--no-wandb)
  fi
}

cd "${REPO_ROOT}"
mkdir -p jobs

if [[ -z "${CHECK_SLUG}" ]]; then
  CHECK_SLUG="autodiscovery_prompt_check_p${PARENT_CONTEXT}_c${EXPLORATION_CONSTANT}_b${BUDGET}"
fi

echo "[prompt-check] checkpoint=${BOREFT_DIR_ARG}"
echo "[prompt-check] targets=${TARGETS[*]}  seeds=${SEARCH_SEEDS[*]}  budget=${BUDGET}"
echo "[prompt-check] parent_context=${PARENT_CONTEXT}  exploration_constant=${EXPLORATION_CONSTANT}"
echo "[prompt-check] out=experiments/outputs/semantle/search/${CHECK_SLUG}"
echo "[prompt-check] after jobs finish:"
echo "  $0 --compare --new-root experiments/outputs/semantle/search/${CHECK_SLUG} --old-root experiments/outputs/semantle/search/autodiscovery"

for target in "${TARGETS[@]}"; do
  dir_name="$(dir_name_for "${target}")"
  search_dir="experiments/outputs/semantle/search/${CHECK_SLUG}/${dir_name}"
  python_cmd_for_run "${target}" "${search_dir}"
  ts="$(date +%s)"
  job_name="boreft_search_semantle_${CHECK_SLUG}_${dir_name}_${ts}"

  echo "[prompt-check] ${SPLIT}/${target}"
  echo "  search_dir ${search_dir}"
  echo "  job        ${job_name}"
  printf '  cmd        python'
  printf ' %q' "${PYTHON_CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  dry-run    not submitted"
    continue
  fi

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
done
