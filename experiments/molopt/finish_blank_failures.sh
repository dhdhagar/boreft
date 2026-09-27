#!/bin/bash
# Resume the three molopt LLM-baseline jobs that died on blank completions.
#
# OPRO/BOPRO raise if a proposal step decodes to an empty string. The crashed
# JNK3 OPRO seed was emitting ~180-character SMILES that look truncated at the
# default 128-token cap; the next sample then came back blank. This reruns with
# a longer decode budget and --resume so finished seeds are kept.
#
#   ./experiments/molopt/finish_blank_failures.sh
#   ./experiments/molopt/finish_blank_failures.sh --dry-run
#   MAX_NEW_TOKENS=512 ./experiments/molopt/finish_blank_failures.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [[ ! "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: MAX_NEW_TOKENS must be a positive integer, got: ${MAX_NEW_TOKENS}" >&2
  exit 1
fi
for arg in "$@"; do
  case "${arg}" in
    --overwrite|--overwrite=*)
      echo "error: do not pass --overwrite; it would delete finished seeds" >&2
      exit 1
      ;;
    --)
      echo "error: extra python args are already set (--max-new-tokens)" >&2
      exit 1
      ;;
  esac
done

echo "[molopt] resume blank-completion failures with max_new_tokens=${MAX_NEW_TOKENS}"

# p90 N=1024 OPRO DRD2: seed 1 died at 68/500 (no completed seeds).
"${DIR}/opro.sh" \
  --boreft_dir outputs/1789921464 \
  --method-dir opro_mu \
  --oracles DRD2 \
  "$@" \
  --resume \
  -- \
  --max-new-tokens "${MAX_NEW_TOKENS}"

# p90 N=1024 OPRO JNK3: seeds 1-2 done; seed 3 died at 363/500.
"${DIR}/opro.sh" \
  --boreft_dir outputs/1789921464 \
  --method-dir opro_mu \
  --oracles JNK3 \
  "$@" \
  --resume \
  -- \
  --max-new-tokens "${MAX_NEW_TOKENS}"

# p90 N=3072 BOPRO DRD2: seeds 1-2 done; seed 3 died at 320/500.
"${DIR}/bopro.sh" \
  --boreft_dir outputs/1789933841 \
  --method-dir bopro_mu_3072 \
  --oracles DRD2 \
  "$@" \
  --resume \
  -- \
  --max-new-tokens "${MAX_NEW_TOKENS}"
