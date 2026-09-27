#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu,gpu-preempt,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -t 0-08:00:00
#SBATCH -o jobs/%j.out
#SBATCH --constraint=a100

# LoRA SFT on a molopt BOReFT train set (checkpoint items.json).
# Prompt and gold match MiST / BOReFT: completion prefix + [START_SMILES],
# gold is ``SMILES [END_SMILES]``. See src/boreft/baselines/lora_sft.py.
#
# Submit from the repo root:
#   mkdir -p jobs
#   sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789222254 scripts/train_lora_sft_molopt.sh
#   sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789446871 scripts/train_lora_sft_molopt.sh
#   sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789222254,DRY_RUN=1 scripts/train_lora_sft_molopt.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
mkdir -p jobs
if [ ! -d src/boreft ]; then
    echo "ERROR: submit sbatch from the repo root (missing src/boreft)." >&2
    echo "       cwd=$(pwd)" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
if [ ! -d "${HF_HOME}" ]; then
    export HF_HOME="$(pwd)/models"
    mkdir -p "${HF_HOME}"
fi
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use the python already on PATH. See README.md.

REFT_OUTPUT_DIR="${REFT_OUTPUT_DIR:-}"
if [ -z "${REFT_OUTPUT_DIR}" ]; then
    echo "ERROR: REFT_OUTPUT_DIR is required (molopt BOReFT checkpoint)." >&2
    echo "       sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789222254 scripts/train_lora_sft_molopt.sh" >&2
    exit 1
fi
RUN_ID="$(basename "${REFT_OUTPUT_DIR}")"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/outputs/molopt/lora_sft_random_${RUN_ID}}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
N_EVAL="${N_EVAL:-500}"
EVAL_EPOCHS="${EVAL_EPOCHS:-1}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"

echo "[train_lora_sft_molopt] reft=${REFT_OUTPUT_DIR} out=${OUTPUT_DIR}"
echo "[train_lora_sft_molopt] rank=${LORA_RANK} epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR}"
echo "[train_lora_sft_molopt] n_eval=${N_EVAL} eval_epochs=${EVAL_EPOCHS} save_epochs=${SAVE_EPOCHS} eval_only=${EVAL_ONLY}"

_flag_on() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

if [ ! -d "${REFT_OUTPUT_DIR}" ]; then
    echo "ERROR: REFT_OUTPUT_DIR=${REFT_OUTPUT_DIR} does not exist." >&2
    exit 1
fi
if [ ! -f "${REFT_OUTPUT_DIR}/items.json" ]; then
    echo "ERROR: not a BOReFT checkpoint (missing items.json): ${REFT_OUTPUT_DIR}" >&2
    exit 1
fi

if _flag_on "${EVAL_ONLY}"; then
    if [ ! -f "${OUTPUT_DIR}/adapter.pt" ]; then
        echo "ERROR: EVAL_ONLY=1 requires ${OUTPUT_DIR}/adapter.pt" >&2
        exit 1
    fi
    ARGS=(
        --eval-only
        --task              molopt
        --output-dir        "${OUTPUT_DIR}"
        --n-eval            "${N_EVAL}"
        --reft-output-dir   "${REFT_OUTPUT_DIR}"
    )
    if [ -n "${WANDB_RUN_ID:-}" ]; then
        ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
    fi
    if _flag_on "${ALLOW_NEW_WANDB_RUN:-0}"; then
        ARGS+=(--allow-new-wandb-run)
    fi
else
    ARGS=(
        --task                  molopt
        --prompt-source         train
        --from-checkpoint-items
        --reft-output-dir       "${REFT_OUTPUT_DIR}"
        --output-dir            "${OUTPUT_DIR}"
        --lora-rank             "${LORA_RANK}"
        --epochs                "${EPOCHS}"
        --batch-size            "${BATCH_SIZE}"
        --lr                    "${LR}"
        --overwrite
        --n-eval                "${N_EVAL}"
        --eval-epochs           "${EVAL_EPOCHS}"
        --save-epochs           "${SAVE_EPOCHS}"
    )
    if _flag_on "${DRY_RUN:-0}"; then
        ARGS+=(--dry-run)
    fi
fi
if [ -n "${CACHE_DIR:-}" ]; then
    ARGS+=(--cache-dir "${CACHE_DIR}")
fi
if _flag_on "${NO_WANDB:-0}"; then
    ARGS+=(--no-wandb)
else
    ARGS+=(
        --wandb-project "${WANDB_PROJECT:-boreft}"
        --wandb-entity "${WANDB_ENTITY:-}"
        --wandb-group "${WANDB_GROUP:-molopt-lora-sft}"
    )
fi
if [ -n "${N_PROBE:-}" ]; then
    ARGS+=(--n-probe "${N_PROBE}")
fi

python -u -m boreft.baselines.lora_sft "${ARGS[@]}"
