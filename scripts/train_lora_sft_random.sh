#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p gpu,gpu-preempt,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH -t 0-04:00:00
#SBATCH -o jobs/%j.out
#SBATCH --constraint=a100

# LoRA SFT on the canonical BOReFT Semantle train draw
# (train-top-k 4000, train-n-samples 3072, seed 42).
# See src/boreft/baselines/lora_sft.py.
#
# Submit from the repo root (Slurm's cwd is the submit directory):
#   mkdir -p jobs
#   sbatch scripts/train_lora_sft_random.sh
#   sbatch --export=ALL,DRY_RUN=1 scripts/train_lora_sft_random.sh
#   sbatch --export=ALL,EPOCHS=10,EVAL_EPOCHS=1,SAVE_EPOCHS=1,N_EVAL=500 scripts/train_lora_sft_random.sh
#   sbatch --export=ALL,EVAL_ONLY=1 scripts/train_lora_sft_random.sh

set -euo pipefail

# Stay in the submit directory. Do not resolve the repo from BASH_SOURCE:
# Slurm often copies this file into a spool path, and mkdir ./scripts there
# fails with "Permission denied".
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

REFT_OUTPUT_DIR="${REFT_OUTPUT_DIR:-outputs/1784053292}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/outputs/semantle/lora_sft_random_e10}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface}"
SEMANTLE_CSV="${SEMANTLE_CSV:-data/semantle/train/computer.csv}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
SEED="${SEED:-42}"
TRAIN_TOP_K="${TRAIN_TOP_K:-4000}"
TRAIN_N_SAMPLES="${TRAIN_N_SAMPLES:-3072}"

N_EVAL="${N_EVAL:-500}"
EVAL_EPOCHS="${EVAL_EPOCHS:-1}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"

echo "[train_lora_sft_random] reft=${REFT_OUTPUT_DIR} out=${OUTPUT_DIR}"
echo "[train_lora_sft_random] csv=${SEMANTLE_CSV}"
echo "[train_lora_sft_random] n=${TRAIN_N_SAMPLES} top_k=${TRAIN_TOP_K} seed=${SEED}"
echo "[train_lora_sft_random] rank=${LORA_RANK} epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR}"
echo "[train_lora_sft_random] n_eval=${N_EVAL} eval_epochs=${EVAL_EPOCHS} save_epochs=${SAVE_EPOCHS} eval_only=${EVAL_ONLY}"

_flag_on() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

if _flag_on "${EVAL_ONLY}"; then
    if [ ! -f "${OUTPUT_DIR}/adapter.pt" ]; then
        echo "ERROR: EVAL_ONLY=1 requires ${OUTPUT_DIR}/adapter.pt" >&2
        exit 1
    fi
    ARGS=(
        --eval-only
        --output-dir        "${OUTPUT_DIR}"
        --cache-dir         "${CACHE_DIR}"
        --n-eval            "${N_EVAL}"
    )
    if [ -n "${REFT_OUTPUT_DIR}" ] && [ -d "${REFT_OUTPUT_DIR}" ]; then
        ARGS+=(--reft-output-dir "${REFT_OUTPUT_DIR}")
    fi
    if [ -n "${WANDB_RUN_ID:-}" ]; then
        ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
    fi
    if _flag_on "${ALLOW_NEW_WANDB_RUN:-0}"; then
        ARGS+=(--allow-new-wandb-run)
    fi
else
    if [ -n "${REFT_OUTPUT_DIR}" ] && [ ! -d "${REFT_OUTPUT_DIR}" ]; then
        echo "ERROR: REFT_OUTPUT_DIR=${REFT_OUTPUT_DIR} does not exist." >&2
        echo "       On Unity this is the canonical BOReFT run (items.json check)." >&2
        echo "       Without a checkpoint: python -m boreft.baselines.lora_sft --dry-run \\" >&2
        echo "         --semantle-csv data/semantle/train/computer.csv" >&2
        exit 1
    fi
    ARGS=(
        --output-dir        "${OUTPUT_DIR}"
        --cache-dir         "${CACHE_DIR}"
        --semantle-csv      "${SEMANTLE_CSV}"
        --train-top-k       "${TRAIN_TOP_K}"
        --train-n-samples   "${TRAIN_N_SAMPLES}"
        --seed              "${SEED}"
        --lora-rank         "${LORA_RANK}"
        --epochs            "${EPOCHS}"
        --batch-size        "${BATCH_SIZE}"
        --lr                "${LR}"
        --overwrite
        --n-eval            "${N_EVAL}"
        --eval-epochs       "${EVAL_EPOCHS}"
        --save-epochs       "${SAVE_EPOCHS}"
    )
    if [ -n "${REFT_OUTPUT_DIR}" ]; then
        ARGS+=(--reft-output-dir "${REFT_OUTPUT_DIR}")
    fi
    if _flag_on "${DRY_RUN:-0}"; then
        ARGS+=(--dry-run)
    fi
    if _flag_on "${FROM_CHECKPOINT_ITEMS:-0}"; then
        ARGS+=(--from-checkpoint-items)
    fi
fi
if _flag_on "${NO_WANDB:-0}"; then
    ARGS+=(--no-wandb)
else
    ARGS+=(--wandb-project "${WANDB_PROJECT:-boreft}" --wandb-entity "${WANDB_ENTITY:-}")
fi
if [ -n "${N_PROBE:-}" ]; then
    ARGS+=(--n-probe "${N_PROBE}")
fi

python -u -m boreft.baselines.lora_sft "${ARGS[@]}"
