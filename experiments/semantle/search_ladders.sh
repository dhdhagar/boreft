#!/bin/bash
# Protocol search on the training-size and rank ladders.
# Reuses the canonical 10 targets and the canonical (target, seed) warmstart
# words, encoding those words in each checkpoint's own latent space.
#
# Canonical N=3072 / rank 64 is not re-run; compare_ladders.py loads it from
# the published s1_t0_ard_d64 cell. N=512 is skipped until
# experiments/semantle/train_n512.sh finishes (or pass --n512-dir).
#
#   ./experiments/semantle/search_ladders.sh --dry-run
#   ./experiments/semantle/search_ladders.sh --list
#   ./experiments/semantle/search_ladders.sh --n-only
#   ./experiments/semantle/search_ladders.sh --rank-only
#   ./experiments/semantle/search_ladders.sh --n512-dir outputs/TIMESTAMP --sizes 512
#   python experiments/semantle/compare_ladders.py
#
# Finished cells are skipped unless --overwrite. --n-only is the whole N
# ladder (not N=512); pin a size with --sizes 512.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT_JSON="${SCRIPT_DIR}/ladder_checkpoints.json"
DUMP_WARMSTARTS="${SCRIPT_DIR}/dump_canonical_warmstarts.py"
BOREFT_SH="${SCRIPT_DIR}/boreft.sh"
TARGETS_JSON="experiments/outputs/semantle/sweep/targets.json"
WARMSTARTS_JSON="experiments/outputs/semantle/sweep/canonical_warmstarts.json"
N512_SIDECAR="${REPO_ROOT}/jobs/n512_dir.txt"

DRY_RUN=0
OVERWRITE=0
N_ONLY=0
RANK_ONLY=0
LIST_ONLY=0
DUMP_ONLY=0
SIZES=""
RANKS=""
N512_DIR="${N512_DIR:-}"
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
    --n-only|--n_only) N_ONLY=1; shift ;;
    --rank-only|--rank_only) RANK_ONLY=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    --dump-only|--dump_only) DUMP_ONLY=1; shift ;;
    --sizes|--size|--n)
      require_value "$1" "${2:-}"
      SIZES="$2"
      shift 2
      ;;
    --sizes=*|--size=*|--n=*)
      SIZES="${1#*=}"
      shift
      ;;
    --ranks|--rank)
      require_value "$1" "${2:-}"
      RANKS="$2"
      shift 2
      ;;
    --ranks=*|--rank=*)
      RANKS="${1#*=}"
      shift
      ;;
    --n512-dir|--n512_dir)
      require_value "$1" "${2:-}"
      N512_DIR="$2"
      shift 2
      ;;
    --n512-dir=*|--n512_dir=*)
      N512_DIR="${1#*=}"
      shift
      ;;
    --)
      shift
      EXTRA=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "expected: [--dry-run] [--list] [--dump-only] [--overwrite] [--n-only] [--rank-only] [--sizes 512] [--ranks 4,8] [--n512-dir DIR]" >&2
      exit 1
      ;;
  esac
done

if [[ "${N_ONLY}" -eq 1 && "${RANK_ONLY}" -eq 1 ]]; then
  echo "choose at most one of --n-only / --rank-only" >&2
  exit 1
fi

cd "${REPO_ROOT}"
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "python3 not found on PATH" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_JSON}" ]]; then
  echo "missing ${CHECKPOINT_JSON}" >&2
  exit 1
fi

LAUNCH_FLAGS=(
  --protocol-cell s1_t0_ard_d64
  --targets-json "${TARGETS_JSON}"
)
if [[ "${OVERWRITE}" -eq 1 ]]; then
  LAUNCH_FLAGS+=(--overwrite)
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  LAUNCH_FLAGS+=(--dry-run)
fi

dump_warmstarts() {
  echo "[ladders] dumping canonical warmstart words"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    "${PYTHON}" "${DUMP_WARMSTARTS}" --dry-run || echo "  (canonical search tree not on this machine; dump on the cluster)"
    return 0
  fi
  "${PYTHON}" "${DUMP_WARMSTARTS}" --out "${WARMSTARTS_JSON}"
}

checkpoint_exists() {
  local output_dir="$1"
  [[ -d "${output_dir}" || -d "${REPO_ROOT}/${output_dir}" ]]
}

launch_cell() {
  local method_dir="$1"
  local output_dir="$2"
  local wandb_group="$3"
  echo "[ladders] ${method_dir} <- ${output_dir}"
  if ! checkpoint_exists "${output_dir}"; then
    if [[ "${DRY_RUN}" -eq 1 || "${LIST_ONLY}" -eq 1 ]]; then
      echo "  missing checkpoint (would search ${method_dir})"
      return 0
    fi
    echo "checkpoint directory does not exist: ${output_dir}" >&2
    exit 1
  fi
  "${BOREFT_SH}" \
    --boreft_dir "${output_dir}" \
    --method-dir "${method_dir}" \
    "${LAUNCH_FLAGS[@]}" \
    -- \
    --warmstart-source words \
    --warmstart-file "${WARMSTARTS_JSON}" \
    --wandb-group "${wandb_group}" \
    "${EXTRA[@]}"
}

if [[ -z "${N512_DIR}" && -f "${N512_SIDECAR}" ]]; then
  N512_DIR="$(tr -d '[:space:]' < "${N512_SIDECAR}")"
fi

PLAN="$("${PYTHON}" - "${CHECKPOINT_JSON}" "${REPO_ROOT}" "${N_ONLY}" "${RANK_ONLY}" "${OVERWRITE}" "${N512_DIR}" "${SIZES}" "${RANKS}" <<'PY'
import json, sys
from pathlib import Path

path, repo, n_only, rank_only, overwrite, n512, sizes, ranks = sys.argv[1:9]
n_only, rank_only, overwrite = int(n_only), int(rank_only), int(overwrite)
data = json.loads(Path(path).read_text(encoding="utf-8"))
search_root = Path(repo) / "experiments" / "outputs" / "semantle" / "search"
seeds = ("1", "2", "3")


def parse_filter(raw, kind):
    if not str(raw).strip():
        return None
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    bad = [item for item in values if not item.isdigit()]
    if bad:
        raise SystemExit(f"invalid --{kind} value(s): {', '.join(bad)}")
    return set(values)


def cell_complete(method_dir):
    root = search_root / method_dir
    targets_path = root / "targets.json"
    if not targets_path.is_file():
        return False
    payload = json.loads(targets_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    if not runs:
        return False
    for run in runs:
        run_dir = root / str(run.get("dir_name") or "")
        for seed in seeds:
            if not (run_dir / f"seed_{seed}" / "summary.json").is_file():
                return False
    return True


size_filter = parse_filter(sizes, "sizes")
rank_filter = parse_filter(ranks, "ranks")
launch_n = not rank_only
launch_rank = not n_only
if size_filter is not None and rank_filter is None:
    launch_rank = False
if rank_filter is not None and size_filter is None:
    launch_n = False
rows = []
if launch_n:
    for n in ("1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "3072"):
        if size_filter is not None and n not in size_filter:
            continue
        cell = data["n_ladder"][n]
        method_dir = "boreft_N" + n
        if cell.get("reuse_canonical_search"):
            print(f"# reuse canonical search for N={n}", file=sys.stderr)
            continue
        output = cell.get("output_dir")
        if n == "512":
            output = n512 or output
        if not output:
            print(f"# skip N={n}: no checkpoint (train first)", file=sys.stderr)
            continue
        if not overwrite and cell_complete(method_dir):
            print(f"# skip {method_dir}: search already complete", file=sys.stderr)
            continue
        rows.append((method_dir, output, "semantle-n-ladder"))
if launch_rank:
    for rank in ("4", "8", "16", "32", "64", "128"):
        if rank_filter is not None and rank not in rank_filter:
            continue
        cell = data["rank_ladder"][rank]
        method_dir = "boreft_rank" + rank
        if cell.get("reuse_canonical_search"):
            print(f"# reuse canonical search for rank={rank}", file=sys.stderr)
            continue
        output = cell.get("output_dir")
        if not output:
            print(f"# skip rank={rank}: no checkpoint", file=sys.stderr)
            continue
        if not overwrite and cell_complete(method_dir):
            print(f"# skip {method_dir}: search already complete", file=sys.stderr)
            continue
        rows.append((method_dir, output, "semantle-rank-ladder"))
for row in rows:
    print("\t".join(row))
PY
)"

echo "[ladders] plan:"
if [[ -z "${PLAN}" ]]; then
  echo "  (no cells to launch)" >&2
else
  while IFS=$'\t' read -r method_dir output_dir wandb_group; do
    [[ -z "${method_dir}" ]] && continue
    echo "  ${method_dir}  ${output_dir}  ${wandb_group}"
  done <<< "${PLAN}"
fi

if [[ "${LIST_ONLY}" -eq 1 ]]; then
  exit 0
fi

if [[ "${DUMP_ONLY}" -eq 1 ]]; then
  dump_warmstarts
  exit 0
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[ladders] would dump ${WARMSTARTS_JSON} if missing, then submit the cells above"
elif [[ ! -f "${WARMSTARTS_JSON}" ]]; then
  dump_warmstarts
fi

if [[ -z "${PLAN}" ]]; then
  echo "no ladder cells to launch" >&2
  exit 1
fi

while IFS=$'\t' read -r method_dir output_dir wandb_group; do
  [[ -z "${method_dir}" ]] && continue
  launch_cell "${method_dir}" "${output_dir}" "${wandb_group}"
done <<< "${PLAN}"
