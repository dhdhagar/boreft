#!/bin/bash
# BOReFT hyperparameter sweep on the shared Semantle protocol.
#
# Grid: observation-samples {1,2,5} × sampling-temperature {0,0.5,1,1.5}
#       × ARD {off,on} × projection-dim {8,16,32,64}
#       = 96 configs × (n_train + n_test) targets × seeds 1,2,3
#
#   ./experiments/semantle/sweep_boreft.sh --boreft_dir outputs/1784053292
#   ./experiments/semantle/sweep_boreft.sh --boreft_dir outputs/1784053292 --dry-run
#   ./experiments/semantle/sweep_boreft.sh --boreft_dir outputs/1784053292 --overwrite
#   ./experiments/semantle/sweep_boreft.sh --boreft_dir outputs/1784053292 --no-wandb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SAMPLE_TARGETS="${SCRIPT_DIR}/sample_targets.py"

BOREFT_DIR=""
N_TRAIN=5
N_TEST=5
SEED=42
DRY_RUN=0
OVERWRITE=0
NO_WANDB=0
WANDB_PROJECT="${WANDB_PROJECT:-boreft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

BUDGET=500
WARMSTART_COUNT=10
WARMSTART_SOURCE=checkpoint
BATCH_SIZE=1
SEARCH_SEEDS=(1 2 3)

SAMPLES=(1 2 5)
TEMPS=(0 0.5 1 1.5)
PROJ_DIMS=(8 16 32 64)

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
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-wandb|--no_wandb)
      NO_WANDB=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "expected: --boreft_dir DIR [--n_train N] [--n_test N] [--seed N] [--dry-run] [--overwrite] [--no-wandb]" >&2
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

slug_for() {
  local samples="$1"
  local temp="$2"
  local ard="$3"
  local dim="$4"
  local temp_tag ard_tag
  temp_tag="${temp/./p}"
  if [[ "${ard}" -eq 1 ]]; then
    ard_tag="ard"
  else
    ard_tag="noard"
  fi
  echo "s${samples}_t${temp_tag}_${ard_tag}_d${dim}"
}

submit_one() {
  local slug="$1"
  local samples="$2"
  local temp="$3"
  local ard="$4"
  local dim="$5"
  local split="$6"
  local target="$7"
  local dir_name="$8"
  local search_dir job_desc ts job_name resume_flag
  search_dir="experiments/outputs/semantle/sweep/${slug}/${dir_name}"
  job_desc="boreft_sweep_${RUN_ID}_${slug}-${dir_name}"
  ts="$(date +%s)"
  job_name="${job_desc}_${ts}"
  resume_flag="--resume"
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    resume_flag="--overwrite"
  fi

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
    --observation-samples "${samples}"
    --sampling-temperature "${temp}"
    --projection-dim "${dim}"
  )
  if [[ "${ard}" -eq 1 ]]; then
    PYTHON_CMD+=(--use-ard)
  fi
  if [[ "${NO_WANDB}" -eq 0 ]]; then
    PYTHON_CMD+=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
    )
  else
    PYTHON_CMD+=(--no-wandb)
  fi

  echo "[sweep] ${slug}  ${split}/${target}"
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
    echo "sbatch failed for ${slug} ${split}/${target}" >&2
    exit 1
  fi
  echo "  ${submitted}"
}

cd "${REPO_ROOT}"
mkdir -p jobs

RUN_ID="$(basename "${BOREFT_DIR}")"
SWEEP_OUT="experiments/outputs/semantle/sweep"
mkdir -p "${SWEEP_OUT}"
TARGETS_JSON="${SWEEP_OUT}/targets.json"

"${PYTHON}" "${SAMPLE_TARGETS}" \
  --boreft_dir "${BOREFT_DIR}" \
  --n_train "${N_TRAIN}" \
  --n_test "${N_TEST}" \
  --seed "${SEED}" \
  --output "${TARGETS_JSON}" >/dev/null

echo "[sweep] checkpoint=${BOREFT_DIR_ARG}"
echo "[sweep] targets -> ${TARGETS_JSON}"
echo "[sweep] grid samples=${SAMPLES[*]}  temps=${TEMPS[*]}  proj_dims=${PROJ_DIMS[*]}  ard=0,1"

n_ard=2
n_configs=$(( ${#SAMPLES[@]} * ${#TEMPS[@]} * n_ard * ${#PROJ_DIMS[@]} ))
"${PYTHON}" - "${TARGETS_JSON}" "${n_configs}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
n_configs = int(sys.argv[2])
print(
    f"[sweep] sampled {data['n_train']} train + {data['n_test']} test "
    f"(seed={data['seed']})"
)
if data["train_targets"]:
    print("  train: " + ", ".join(data["train_targets"]))
if data["test_targets"]:
    print("  test:  " + ", ".join(data["test_targets"]))
n_targets = len(data["runs"])
print(f"[sweep] {n_configs} configs × {n_targets} targets = {n_configs * n_targets} jobs")
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

n_submitted=0
for samples in "${SAMPLES[@]}"; do
  for temp in "${TEMPS[@]}"; do
    for ard in 0 1; do
      for dim in "${PROJ_DIMS[@]}"; do
        slug="$(slug_for "${samples}" "${temp}" "${ard}" "${dim}")"
        while IFS=$'\t' read -r split target dir_name; do
          [[ -z "${split}" ]] && continue
          if [[ -z "${target}" || -z "${dir_name}" ]]; then
            printf 'malformed run line: split=%q target=%q dir_name=%q\n' \
              "${split}" "${target}" "${dir_name}" >&2
            exit 1
          fi
          submit_one "${slug}" "${samples}" "${temp}" "${ard}" "${dim}" \
            "${split}" "${target}" "${dir_name}"
          n_submitted=$((n_submitted + 1))
        done < "${RUN_LIST}"
      done
    done
  done
done

echo "[sweep] submitted ${n_submitted} jobs (dry-run=${DRY_RUN})"
