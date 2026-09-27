#!/bin/bash
# Shared Semantle search launcher. Method wrappers call:
#   exec "$(dirname "$0")/_common.sh" <method_name> "$@"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SAMPLE_TARGETS="${SCRIPT_DIR}/sample_targets.py"
KNOWN_METHODS="autodiscovery bopro boreft discrete_bo migrate opro random_sampling sdpo_ttt"

METHOD_NAME="${1:-}"
if [[ -z "${METHOD_NAME}" || "${METHOD_NAME}" == --* ]]; then
  echo "usage: $0 <method_name> --boreft_dir DIR [--method-dir NAME] [--protocol-cell SLUG] [--n_train N] [--n_test N] [--seed N] [--dry-run] [--resume] [--overwrite] [--no-wandb] [-- extra python args]" >&2
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
N_TRAIN=5
N_TEST=5
SEED=42
DRY_RUN=0
OVERWRITE=0
NO_WANDB=0
WANDB_PROJECT="${WANDB_PROJECT:-boreft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
EXTRA_ARGS=()
METHOD_DIR=""
OBSERVATION_SAMPLES=5
SAMPLING_TEMPERATURE=""
USE_ARD=0
PROJECTION_DIM=""
TARGETS_JSON_IN=""

BUDGET=500
# Labeled train words excluding the search target; same words across methods
# for a given target and --seeds value. Each costs 1 verification.
WARMSTART_COUNT=10
WARMSTART_SOURCE=checkpoint
BATCH_SIZE=1
SEARCH_SEEDS=(1 2 3)
TASK_DESCRIPTION="Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)."

SBATCH_PARTITION="gpu,gpu-preempt,superpod-a100"
SBATCH_GRES="gpu:1"
SBATCH_MEM="48G"
SBATCH_CPUS=4
SBATCH_CONSTRAINT="a100"
SBATCH_TIME="24:00:00"

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
    --n_train|--n-train)
      require_value "$1" "${2:-}"
      require_int "$1" "$2"
      N_TRAIN="$2"
      shift 2
      ;;
    --n_train=*|--n-train=*)
      N_TRAIN="${1#*=}"
      require_int "--n_train" "${N_TRAIN}"
      shift
      ;;
    --n_test|--n-test)
      require_value "$1" "${2:-}"
      require_int "$1" "$2"
      N_TEST="$2"
      shift 2
      ;;
    --n_test=*|--n-test=*)
      N_TEST="${1#*=}"
      require_int "--n_test" "${N_TEST}"
      shift
      ;;
    --seed)
      require_value "$1" "${2:-}"
      require_int "$1" "$2"
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      require_int "--seed" "${SEED}"
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
      shift 2
      ;;
    --method-dir=*|--method_dir=*)
      METHOD_DIR="${1#*=}"
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
    --targets-json|--targets_json)
      require_value "$1" "${2:-}"
      TARGETS_JSON_IN="$2"
      shift 2
      ;;
    --targets-json=*|--targets_json=*)
      TARGETS_JSON_IN="${1#*=}"
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "expected: --boreft_dir DIR [--method-dir NAME] [--protocol-cell s1_t0_ard_d64|s1_t1_ard_d64] [--n_train N] [--n_test N] [--seed N] [--dry-run] [--resume] [--overwrite] [--no-wandb] [-- extra python args]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${BOREFT_DIR}" ]]; then
  echo "--boreft_dir is required" >&2
  exit 1
fi

if [[ -z "${METHOD_DIR}" ]]; then
  METHOD_DIR="${METHOD_NAME}"
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

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "python3 not found on PATH" >&2
  exit 1
fi

python_cmd_for_run() {
  local target="$1"
  local search_dir="$2"
  local resume_flag="--resume"
  local arg
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    resume_flag="--overwrite"
  fi
  # Flags after `--` go to Python; honor them so `--overwrite` is not paired
  # with the launcher's default `--resume`.
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
      --target "${target}"
      --budget "${BUDGET}"
      --warmstart-count "${WARMSTART_COUNT}"
      --warmstart-source "${WARMSTART_SOURCE}"
      --surrogate projected
      --acquisition log_ei
      --batch-size "${BATCH_SIZE}"
      --seeds "${SEARCH_SEEDS[@]}"
      "${resume_flag}"
      --observation-samples "${OBSERVATION_SAMPLES}"
    )
    if [[ -n "${SAMPLING_TEMPERATURE}" ]]; then
      PYTHON_CMD+=(--sampling-temperature "${SAMPLING_TEMPERATURE}")
    fi
    if [[ "${USE_ARD}" -eq 1 ]]; then
      PYTHON_CMD+=(--use-ard)
    fi
    if [[ -n "${PROJECTION_DIM}" ]]; then
      PYTHON_CMD+=(--projection-dim "${PROJECTION_DIM}")
    fi
  else
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
      "${resume_flag}"
    )
    if [[ "${METHOD_NAME}" == "discrete_bo" ]]; then
      # Frozen train vocabulary (all unique items.json words). LLM uniqueness
      # sampling of 1024 words routinely undershoots on 1B-scale models.
      PYTHON_CMD+=(
        --discrete-bo.candidate-source train
        --discrete-bo.candidate-count -1
      )
    fi
    if [[ "${METHOD_NAME}" == "autodiscovery" ]]; then
      # Semantle protocol: OPRO scored-history prompt (in code), deeper
      # exploitation than the paper C=2.0 / k_parents=3 defaults.
      PYTHON_CMD+=(
        --autodiscovery.parent-context 20
        --autodiscovery.exploration-constant 0.5
      )
    fi
  fi
  if [[ "${NO_WANDB}" -eq 0 ]]; then
    PYTHON_CMD+=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
    )
  else
    PYTHON_CMD+=(--no-wandb)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTHON_CMD+=("${EXTRA_ARGS[@]}")
  fi
}

submit_one() {
  local split="$1"
  local target="$2"
  local dir_name="$3"
  local search_dir job_desc ts job_name
  search_dir="experiments/outputs/semantle/search/${METHOD_DIR}/${dir_name}"
  python_cmd_for_run "${target}" "${search_dir}"
  job_desc="boreft_search_semantle_${RUN_ID}_${METHOD_NAME}-${dir_name}"
  ts="$(date +%s)"
  job_name="${job_desc}_${ts}"

  echo "[semantle] ${split}/${target}"
  echo "  search_dir ${search_dir}"
  echo "  job        ${job_name}"
  printf '  cmd        python'
  printf ' %q' "${PYTHON_CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  dry-run    not submitted"
    return 0
  fi

  local submitted
  if ! submitted="$(
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
  )"; then
    echo "sbatch failed for ${split}/${target}" >&2
    exit 1
  fi
  echo "  ${submitted}"
}

cd "${REPO_ROOT}"
mkdir -p jobs

RUN_ID="$(basename "${BOREFT_DIR}")"
METHOD_OUT="experiments/outputs/semantle/search/${METHOD_DIR}"
mkdir -p "${METHOD_OUT}"
TARGETS_JSON="${METHOD_OUT}/targets.json"

if [[ -n "${TARGETS_JSON_IN}" ]]; then
  if [[ -f "${TARGETS_JSON_IN}" ]]; then
    TARGETS_SRC="$(cd "$(dirname "${TARGETS_JSON_IN}")" && pwd)/$(basename "${TARGETS_JSON_IN}")"
  elif [[ "${TARGETS_JSON_IN}" != /* && -f "${REPO_ROOT}/${TARGETS_JSON_IN}" ]]; then
    TARGETS_SRC="${REPO_ROOT}/${TARGETS_JSON_IN}"
  else
    echo "targets json does not exist: ${TARGETS_JSON_IN}" >&2
    exit 1
  fi
  "${PYTHON}" - "${TARGETS_SRC}" "${TARGETS_JSON}" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src, encoding="utf-8"))
if not data.get("runs"):
    raise SystemExit(f"{src}: no runs")
os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
print(f"[semantle] reused targets from {src}", flush=True)
PY
else
  "${PYTHON}" "${SAMPLE_TARGETS}" \
    --boreft_dir "${BOREFT_DIR}" \
    --n_train "${N_TRAIN}" \
    --n_test "${N_TEST}" \
    --seed "${SEED}" \
    --output "${TARGETS_JSON}" >/dev/null
fi

echo "[semantle] method=${METHOD_NAME}  method_dir=${METHOD_DIR}  checkpoint=${BOREFT_DIR_ARG}"
echo "[semantle] targets -> ${TARGETS_JSON}"

"${PYTHON}" - "${TARGETS_JSON}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    f"[semantle] sampled {data['n_train']} train + {data['n_test']} test "
    f"(seed={data['seed']}, train_pool={data['train_pool_size']}, "
    f"test_pool={data['test_pool_size']}, test_source={data['test_source']})"
)
if data["train_targets"]:
    print("  train: " + ", ".join(data["train_targets"]))
if data["test_targets"]:
    print("  test:  " + ", ".join(data["test_targets"]))
PY

RUN_LIST="$(mktemp)"
trap 'rm -f "${RUN_LIST}"' EXIT
"${PYTHON}" - "${TARGETS_JSON}" <<'PY' > "${RUN_LIST}"
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for run in data["runs"]:
    print("\t".join((run["split"], run["target"], run["dir_name"])))
PY

if [[ ! -s "${RUN_LIST}" ]]; then
  echo "no runs to submit (check --n_train / --n_test)" >&2
  exit 1
fi

while IFS=$'\t' read -r split target dir_name; do
  [[ -z "${split}" ]] && continue
  if [[ -z "${target}" || -z "${dir_name}" ]]; then
    printf 'malformed run line: split=%q target=%q dir_name=%q\n' \
      "${split}" "${target}" "${dir_name}" >&2
    exit 1
  fi
  submit_one "${split}" "${target}" "${dir_name}"
done < "${RUN_LIST}"
