#!/bin/bash
# Full-recipe Semantle train at N=512 (the missing N-ladder cell).
# After the job finishes, pass the printed output_dir to search:
#   ./experiments/semantle/search_ladders.sh --n512-dir outputs/TIMESTAMP --sizes 512
#
#   ./experiments/semantle/train_n512.sh
#   ./experiments/semantle/train_n512.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIDECAR="${REPO_ROOT}/jobs/n512_dir.txt"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" || "${1:-}" == "--dry_run" ]]; then
  DRY_RUN=1
fi

cd "${REPO_ROOT}"
mkdir -p jobs
JOB_DESC=boreft_sdpo-on+gold_ce1_vae_bias-network-defns_rank64_wd1e-1_enc-llm_512
TS="$(date +%s)"
JOB_NAME="${JOB_DESC}_${TS}"
OUTPUT_DIR="outputs/${TS}"

echo "[train_n512] job=${JOB_NAME}"
echo "[train_n512] output_dir=${OUTPUT_DIR}"
echo "[train_n512] after this finishes: ./experiments/semantle/search_ladders.sh --n512-dir ${OUTPUT_DIR} --sizes 512"

CMD=(
  -m boreft.train
  --wandb-project boreft
  --wandb-group sweep
  --wandb-run-name "${JOB_NAME}"
  --task semantle
  --output-dir "${OUTPUT_DIR}"
  --semantle-csv data/semantle/train/computer.csv
  --semantle-dir data/semantle/train
  --train-top-k 4000
  --train-n-samples 512
  --test-n-samples 3488
  --lr 1e-3
  --lr-scheduler-type cosine
  --warmup-ratio 0.1
  --stop-threshold-min 0.8
  --eval-epochs 5
  --eval-n-samples 512
  --run-full-eval
  --cache-dir ${HOME}/.cache/huggingface
  --model-name meta-llama/Llama-3.2-1B-Instruct
  --use-chat-template
  --low-rank-dim 64
  --layer 0
  --position marker
  --intervention-inject prefix
  --intervention-token-init none
  --batch-size 32
  --epochs 240
  --lambda-ce 1
  --bias-type vae
  --variance learnable
  --kl-beta 1.0
  --vae-free-bits-lambda 25e-3
  --kl-prior-var 1
  --lambda-sdpo 1
  --sdpo-include-gold
  --sdpo-n-onpolicy 4
  --sdpo-n-offpolicy 0
  --sdpo-max-new-tokens 32
  --add-bias-network
  --bias-network-residual
  --use-definition-embeds
  --weight-decay-mode W_and_b
  --wd-b 1e-1
  --wd-W 1e-1
  --bias_network_encoder llm_encoder
)

printf '  cmd python'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "  dry-run    not submitted"
  exit 0
fi

echo "${OUTPUT_DIR}" > "${SIDECAR}"
sbatch --chdir="${REPO_ROOT}" \
  -J "${JOB_NAME}" \
  -e "${REPO_ROOT}/jobs/${JOB_NAME}.err" \
  -o "${REPO_ROOT}/jobs/${JOB_NAME}.log" \
  --partition="gpu,gpu-preempt,superpod-a100" \
  --gres=gpu:1 \
  --mem=48G \
  -c 4 \
  --constraint="a100-80g" \
  --time=24:00:00 \
  "${REPO_ROOT}/run_sbatch.sh" \
  "${CMD[@]}"
