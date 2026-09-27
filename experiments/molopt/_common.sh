#!/bin/bash
# Shared molopt property-search launcher. Method wrappers call:
#   exec "$(dirname "$0")/_common.sh" <method_name> "$@"
#
# Default protocol is T=1, 1 observation sample, ARD, d=64 (s1_t1_ard_d64).
# Search tasks: DRD2, GSK3B, JNK3, and GSK3B_JNK3 (product of the two kinases).
# Warm starts are labeled training molecules at their learned posterior means
# (same source as Semantle ``--warmstart-source checkpoint``). A p90 catalog
# of size 1024 / 2048 / 3072 reuses
# ``experiments/molopt/warmstarts/p90_<N>.json`` so every method looks up the
# same labels and this checkpoint's μ. Sobol dumps are still available via
# ``--warmstart-source file --warmstart-dir DIR``.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KNOWN_METHODS="autodiscovery bopro boreft discrete_bo migrate opro random_sampling sdpo_ttt"

METHOD_NAME="${1:-}"
if [[ -z "${METHOD_NAME}" || "${METHOD_NAME}" == --* ]]; then
  echo "usage: $0 <method_name> --boreft_dir DIR [--oracles NAME ...] [--warmstart-source checkpoint|file|sobol] [--warmstart-dir DIR] [--warmstart-file JSON] [--search-prompt checkpoint|task] [--method-dir NAME] [--protocol-cell SLUG] [--dependency SPEC] [--dry-run] [--resume] [--overwrite] [--no-wandb] [-- extra python args]" >&2
  exit 1
fi
shift
case " ${KNOWN_METHODS} " in
  *" ${METHOD_NAME} "*) ;;
  *)
    echo "unknown method: ${METHOD_NAME}  (choose one of: ${KNOWN_METHODS})" >&2
    exit 1
    ;;
esac

BOREFT_DIR=""
DRY_RUN=0
OVERWRITE=0
NO_WANDB=0
WANDB_PROJECT="${WANDB_PROJECT:-boreft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXTRA_ARGS=()
METHOD_DIR=""
OBSERVATION_SAMPLES=1
SAMPLING_TEMPERATURE=1
USE_ARD=1
PROJECTION_DIM=64
WARMSTART_DIR=""
DEPENDENCY=""
ORACLES=(DRD2 GSK3B JNK3 GSK3B_JNK3)

BUDGET=500
WARMSTART_COUNT=10
WARMSTART_SOURCE=checkpoint
SEARCH_PROMPT=checkpoint
METHOD_DIR_SET=0
BATCH_SIZE=1
SEARCH_SEEDS=(1 2 3 4 5)

SBATCH_PARTITION="gpu,gpu-preempt,superpod-a100"
SBATCH_GRES="gpu:1"
SBATCH_MEM="48G"
SBATCH_CPUS=4
SBATCH_CONSTRAINT="a100"
SBATCH_TIME="24:00:00"

# Completion prefix must match ``task_instruction("molopt")`` / BOReFT
# ``ckpt.prompt`` (no open tag; decode appends ``[START_SMILES]`` last).
MOLOPT_COMPLETION_PREFIX="Here is a valid SMILES string for a molecule (only the SMILES string; no additional text):"

# Match ``normalize_property_oracle_name`` so aliases share one results directory.
canonical_oracle() {
  local raw="${1//β/B}"
  raw="${raw//Β/B}"
  raw="$(printf '%s' "${raw}" | tr '[:lower:]' '[:upper:]' | tr '* -' '___')"
  while [[ "${raw}" == *__* ]]; do
    raw="${raw//__/_}"
  done
  raw="${raw#_}"
  raw="${raw%_}"
  raw="${raw//GSK3BETA/GSK3B}"
  case "${raw}" in
    DRD2|GSK3B|JNK3|GSK3B_JNK3) printf '%s' "${raw}" ;;
    JNK3_GSK3B|GSK3BJNK3|JNK3GSK3B|DUAL_KINASE|DUALKINASE) printf '%s' "GSK3B_JNK3" ;;
    *)
      echo "unknown molopt oracle: ${1}" >&2
      return 1
      ;;
  esac
}

task_description_for() {
  local objective
  case "$1" in
    DRD2) objective="The task is to optimize for DRD2 binding." ;;
    GSK3B) objective="The task is to optimize for GSK3β (GSK3B) inhibition." ;;
    JNK3) objective="The task is to optimize for JNK3 inhibition." ;;
    GSK3B_JNK3) objective="The task is to optimize the product of GSK3β (GSK3B) and JNK3 inhibition." ;;
    *)
      return 1
      ;;
  esac
  printf '%s\n%s' "${objective}" "${MOLOPT_COMPLETION_PREFIX}"
}

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

apply_protocol_cell() {
  local slug="$1"
  case "${slug}" in
    s1_t0_ard_d64)
      OBSERVATION_SAMPLES=1
      SAMPLING_TEMPERATURE=0
      USE_ARD=1
      PROJECTION_DIM=64
      ;;
    s1_t1_ard_d64)
      OBSERVATION_SAMPLES=1
      SAMPLING_TEMPERATURE=1
      USE_ARD=1
      PROJECTION_DIM=64
      ;;
    *)
      echo "unknown --protocol-cell: ${slug}  (known: s1_t0_ard_d64, s1_t1_ard_d64)" >&2
      exit 1
      ;;
  esac
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
    --oracles)
      shift
      ORACLES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ORACLES+=("$1")
        shift
      done
      if [[ ${#ORACLES[@]} -eq 0 ]]; then
        echo "error: --oracles requires at least one name" >&2
        exit 1
      fi
      ;;
    --warmstart-source|--warmstart_source)
      require_value "$1" "${2:-}"
      WARMSTART_SOURCE="$2"
      shift 2
      ;;
    --warmstart-source=*|--warmstart_source=*)
      WARMSTART_SOURCE="${1#*=}"
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
    --warmstart-file|--warmstart_file)
      require_value "$1" "${2:-}"
      WARMSTART_DIR="$2"
      shift 2
      ;;
    --warmstart-file=*|--warmstart_file=*)
      WARMSTART_DIR="${1#*=}"
      shift
      ;;
    --search-prompt|--search_prompt)
      require_value "$1" "${2:-}"
      SEARCH_PROMPT="$2"
      shift 2
      ;;
    --search-prompt=*|--search_prompt=*)
      SEARCH_PROMPT="${1#*=}"
      shift
      ;;
    --dependency)
      require_value "$1" "${2:-}"
      DEPENDENCY="$2"
      shift 2
      ;;
    --dependency=*)
      DEPENDENCY="${1#*=}"
      shift
      ;;
    --dry-run|--dry_run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      OVERWRITE=0
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-wandb|--no_wandb)
      NO_WANDB=1
      shift
      ;;
    --method-dir|--method_dir)
      require_value "$1" "${2:-}"
      METHOD_DIR="$2"
      METHOD_DIR_SET=1
      shift 2
      ;;
    --method-dir=*|--method_dir=*)
      METHOD_DIR="${1#*=}"
      METHOD_DIR_SET=1
      shift
      ;;
    --protocol-cell|--protocol_cell)
      require_value "$1" "${2:-}"
      apply_protocol_cell "$2"
      shift 2
      ;;
    --protocol-cell=*|--protocol_cell=*)
      apply_protocol_cell "${1#*=}"
      shift
      ;;
    --observation-samples|--observation_samples)
      require_value "$1" "${2:-}"
      require_int "$1" "$2"
      OBSERVATION_SAMPLES="$2"
      shift 2
      ;;
    --sampling-temperature|--sampling_temperature)
      require_value "$1" "${2:-}"
      SAMPLING_TEMPERATURE="$2"
      shift 2
      ;;
    --use-ard|--use_ard)
      USE_ARD=1
      shift
      ;;
    --projection-dim|--projection_dim)
      require_value "$1" "${2:-}"
      require_int "$1" "$2"
      PROJECTION_DIM="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
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

case "${WARMSTART_SOURCE}" in
  checkpoint|file|sobol|llm|words) ;;
  *)
    echo "unknown --warmstart-source: ${WARMSTART_SOURCE}" >&2
    exit 1
    ;;
esac
case "${SEARCH_PROMPT}" in
  checkpoint|task) ;;
  *)
    echo "unknown --search-prompt: ${SEARCH_PROMPT}  (known: checkpoint, task)" >&2
    exit 1
    ;;
esac
if [[ "${SEARCH_PROMPT}" == "task" && "${METHOD_NAME}" != "boreft" ]]; then
  echo "[molopt] ignoring --search-prompt task for ${METHOD_NAME} (BOReFT only)"
  SEARCH_PROMPT=checkpoint
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

RUN_ID="$(basename "${BOREFT_DIR}")"
PINNED_P90_DIR="${REPO_ROOT}/experiments/molopt/warmstarts"

resolve_warmstart_path() {
  local raw="$1"
  python - "$raw" "${REPO_ROOT}" <<'PY'
from pathlib import Path
import sys
raw, root = Path(sys.argv[1]), Path(sys.argv[2])
for path in (raw, root / raw):
    if path.exists():
        print(path.resolve())
        raise SystemExit(0)
print(raw)
PY
}

molopt_p90_n() {
  python - "$1" <<'PY'
import json, os, sys
root = sys.argv[1]
cfg = {}
for name in ("intervention_config.json", "training_config.json"):
    path = os.path.join(root, name)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            cfg.update(payload)
split = cfg.get("oracle_split")
path = os.path.join(root, "oracle_split.json")
if not isinstance(split, dict) and os.path.isfile(path):
    with open(path, encoding="utf-8") as handle:
        split = json.load(handle)
if isinstance(split, dict):
    if cfg.get("molopt_oracle_cap_percentile") is None:
        cfg["molopt_oracle_cap_percentile"] = split.get("percentile")
cap = cfg.get("molopt_oracle_cap_percentile")
n = cfg.get("num_training_examples")
if n is None and isinstance(split, dict):
    n = split.get("n_train")
if n is None:
    items = os.path.join(root, "items.json")
    if os.path.isfile(items):
        with open(items, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            n = len(payload)
if n is None:
    n = cfg.get("train_n_samples")
task = cfg.get("task")
try:
    size = int(n)
    ok = (
        (not task or task == "molopt")
        and cap is not None
        and abs(float(cap) - 90.0) < 1e-6
        and size in (1024, 2048, 3072)
    )
except (TypeError, ValueError):
    ok = False
    size = None
if ok:
    print(size)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

P90_N=""
if [[ "${WARMSTART_SOURCE}" == "checkpoint" ]]; then
  P90_N="$(molopt_p90_n "${BOREFT_DIR}" || true)"
fi

if [[ "${METHOD_DIR_SET}" -eq 0 || -z "${METHOD_DIR}" ]]; then
  METHOD_DIR="${METHOD_NAME}"
  if [[ "${WARMSTART_SOURCE}" == "checkpoint" ]]; then
    METHOD_DIR="${METHOD_DIR}_mu"
    if [[ "${P90_N}" == "2048" || "${P90_N}" == "3072" ]]; then
      METHOD_DIR="${METHOD_DIR}_${P90_N}"
    fi
  fi
  if [[ "${SEARCH_PROMPT}" == "task" ]]; then
    METHOD_DIR="${METHOD_DIR}_task"
  fi
fi

WARMSTART_DIR_ARG=""
if [[ "${WARMSTART_SOURCE}" == "file" || "${WARMSTART_SOURCE}" == "words" ]]; then
  if [[ -z "${WARMSTART_DIR}" ]]; then
    WARMSTART_DIR="${REPO_ROOT}/experiments/outputs/molopt/warmstarts/${RUN_ID}"
  else
    WARMSTART_DIR="$(resolve_warmstart_path "${WARMSTART_DIR}")"
  fi
  case "${WARMSTART_DIR}" in
    "${REPO_ROOT}"/*) WARMSTART_DIR_ARG="${WARMSTART_DIR#"${REPO_ROOT}"/}" ;;
    *) WARMSTART_DIR_ARG="${WARMSTART_DIR}" ;;
  esac
elif [[ "${WARMSTART_SOURCE}" == "checkpoint" ]]; then
  if [[ -n "${WARMSTART_DIR}" ]]; then
    WARMSTART_DIR="$(resolve_warmstart_path "${WARMSTART_DIR}")"
    if [[ ! -f "${WARMSTART_DIR}" ]]; then
      echo "warmstart file does not exist: ${WARMSTART_DIR}" >&2
      exit 1
    fi
  elif [[ -n "${P90_N}" ]]; then
    PINNED_P90="${PINNED_P90_DIR}/p90_${P90_N}.json"
    if [[ ! -f "${PINNED_P90}" ]]; then
      echo "${P90_N}-p90 checkpoint needs pinned labels at ${PINNED_P90}" >&2
      exit 1
    fi
    WARMSTART_DIR="${PINNED_P90}"
  fi
  if [[ -n "${WARMSTART_DIR}" ]]; then
    case "${WARMSTART_DIR}" in
      "${REPO_ROOT}"/*) WARMSTART_DIR_ARG="${WARMSTART_DIR#"${REPO_ROOT}"/}" ;;
      *) WARMSTART_DIR_ARG="${WARMSTART_DIR}" ;;
    esac
  fi
fi

python_cmd_for_run() {
  local oracle="$1"
  local search_dir="$2"
  local task_description="$3"
  local resume_flag="--resume"
  local arg
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    resume_flag="--overwrite"
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    for arg in "${EXTRA_ARGS[@]}"; do
      case "${arg}" in
        --overwrite) resume_flag="--overwrite" ;;
        --resume) resume_flag="--resume" ;;
      esac
    done
  fi
  if [[ "${METHOD_NAME}" == "boreft" ]]; then
    PYTHON_CMD=(
      -m boreft.search
      --output-dir "${BOREFT_DIR_ARG}"
      --search-dir "${search_dir}"
      --target "${oracle}"
      --oracle "${oracle}"
      --budget "${BUDGET}"
      --warmstart-count "${WARMSTART_COUNT}"
      --warmstart-source "${WARMSTART_SOURCE}"
      --surrogate projected
      --acquisition log_ei
      --batch-size "${BATCH_SIZE}"
      --seeds "${SEARCH_SEEDS[@]}"
      "${resume_flag}"
      --observation-samples "${OBSERVATION_SAMPLES}"
      --sampling-temperature "${SAMPLING_TEMPERATURE}"
    )
    if [[ "${USE_ARD}" -eq 1 ]]; then
      PYTHON_CMD+=(--use-ard)
    fi
    if [[ -n "${PROJECTION_DIM}" ]]; then
      PYTHON_CMD+=(--projection-dim "${PROJECTION_DIM}")
    fi
    if [[ "${SEARCH_PROMPT}" == "task" ]]; then
      PYTHON_CMD+=(--search-prompt task)
    fi
  else
    PYTHON_CMD=(
      -m boreft.baselines.search
      --baseline "${METHOD_NAME}"
      --task molopt
      --task-description "${task_description}"
      --target "${oracle}"
      --oracle "${oracle}"
      --reft-output-dir "${BOREFT_DIR_ARG}"
      --search-dir "${search_dir}"
      --budget "${BUDGET}"
      --warmstart-count "${WARMSTART_COUNT}"
      --warmstart-source "${WARMSTART_SOURCE}"
      --batch-size "${BATCH_SIZE}"
      --seeds "${SEARCH_SEEDS[@]}"
      --sampling-temperature "${SAMPLING_TEMPERATURE}"
      "${resume_flag}"
    )
    if [[ "${METHOD_NAME}" == "discrete_bo" ]]; then
      PYTHON_CMD+=(
        --discrete-bo.candidate-source train
        --discrete-bo.candidate-count -1
      )
    fi
  fi
  if [[ -n "${WARMSTART_DIR_ARG}" ]]; then
    PYTHON_CMD+=(--warmstart-file "${WARMSTART_DIR_ARG}")
  fi
  if [[ "${NO_WANDB}" -eq 0 ]]; then
    PYTHON_CMD+=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
      --wandb-group "molopt_property"
      --wandb-run-name "${METHOD_DIR}-${oracle}"
    )
  else
    PYTHON_CMD+=(--no-wandb)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTHON_CMD+=("${EXTRA_ARGS[@]}")
  fi
}

submit_one() {
  local oracle="$1"
  local task_description
  task_description="$(task_description_for "${oracle}")" || {
    echo "no task description for oracle ${oracle}" >&2
    exit 1
  }
  local search_dir="experiments/outputs/molopt/search/${METHOD_DIR}/${oracle}"
  python_cmd_for_run "${oracle}" "${search_dir}" "${task_description}"
  local ts job_name job_desc
  job_desc="boreft_search_molopt_${RUN_ID}_${METHOD_NAME}-${oracle}"
  ts="$(date +%s)"
  job_name="${job_desc}_${ts}"

  echo "[molopt] ${METHOD_NAME}/${oracle}"
  echo "  search_dir ${search_dir}"
  echo "  warmstarts ${WARMSTART_SOURCE}${WARMSTART_DIR_ARG:+ ${WARMSTART_DIR_ARG}}"
  echo "  job        ${job_name}"
  printf '  cmd        python'
  printf ' %q' "${PYTHON_CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  dry-run    not submitted"
    return 0
  fi

  local sbatch_args=(
    --chdir="${REPO_ROOT}"
    -J "${job_name}"
    -e "${REPO_ROOT}/jobs/${job_name}.err"
    -o "${REPO_ROOT}/jobs/${job_name}.log"
    --partition="${SBATCH_PARTITION}"
    --gres="${SBATCH_GRES}"
    --mem="${SBATCH_MEM}"
    -c "${SBATCH_CPUS}"
    --constraint="${SBATCH_CONSTRAINT}"
    --time="${SBATCH_TIME}"
  )
  if [[ -n "${DEPENDENCY}" ]]; then
    sbatch_args+=(--dependency="${DEPENDENCY}")
  fi

  local submitted
  if ! submitted="$(
    sbatch "${sbatch_args[@]}" \
      "${REPO_ROOT}/run_sbatch.sh" \
      "${PYTHON_CMD[@]}"
  )"; then
    echo "sbatch failed for ${METHOD_NAME}/${oracle}" >&2
    exit 1
  fi
  echo "  ${submitted}"
}

cd "${REPO_ROOT}"
mkdir -p jobs

canonical_oracles=()
for raw_oracle in "${ORACLES[@]}"; do
  oracle="$(canonical_oracle "${raw_oracle}")" || exit 1
  duplicate=0
  if [[ ${#canonical_oracles[@]} -gt 0 ]]; then
    for seen in "${canonical_oracles[@]}"; do
      if [[ "${seen}" == "${oracle}" ]]; then
        duplicate=1
        break
      fi
    done
  fi
  if [[ "${duplicate}" -eq 0 ]]; then
    canonical_oracles+=("${oracle}")
  fi
done
ORACLES=("${canonical_oracles[@]}")

echo "[molopt] method=${METHOD_NAME}  method_dir=${METHOD_DIR}  checkpoint=${BOREFT_DIR_ARG}"
echo "[molopt] oracles=${ORACLES[*]}  seeds=${SEARCH_SEEDS[*]}  budget=${BUDGET}"
echo "[molopt] warmstarts=${WARMSTART_SOURCE}${WARMSTART_DIR_ARG:+ ${WARMSTART_DIR_ARG}}  search_prompt=${SEARCH_PROMPT}"

for oracle in "${ORACLES[@]}"; do
  submit_one "${oracle}"
done
