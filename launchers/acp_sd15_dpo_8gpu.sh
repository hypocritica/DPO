#!/usr/bin/env bash
# ACP startup command: exec bash /mnt/afs/L202500030/xietianyu/workspace/DPO/launchers/acp_sd15_dpo_8gpu.sh
set -eo pipefail

source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate dpo
cd /mnt/afs/L202500030/xietianyu/workspace/DPO

SCD_RUN_NAME="${SCD_RUN_NAME:-sd15-dpo-8gpu-10000-v1}"
mkdir -p logs
SCD_LOG="$(pwd)/logs/${SCD_RUN_NAME}.txt"
exec > >(tee -a "$SCD_LOG") 2>&1

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source paths.sh

SCD_CONFIG="configs/local-${SCD_RUN_NAME}.json"
SCD_OUT="${RUNS_ROOT}/${SCD_RUN_NAME}"
date '+%F %T %z'
git rev-parse HEAD
echo "Console log: $SCD_LOG"
echo "Experiment: $SCD_OUT"

python scripts/check_deployment.py --family sd15 --stage prepared --gpu --world-size 8

# Freeze once; a restarted ACP container reuses the same config and experiment.
if [[ ! -f "$SCD_CONFIG" ]]; then
  if [[ -e "$SCD_OUT" ]]; then
    echo "Missing frozen config $SCD_CONFIG for an existing experiment; restore it before resuming."
    exit 1
  fi
  python scripts/make_config.py --family sd15 --method dpo --world-size 8 \
    --train_batch_size 2 --gradient_accumulation_steps 16 \
    --max_train_steps 10000 --learning_rate 1e-5 --beta 5000 \
    --rank 16 --lora_alpha 16 --seed 17 --mixed_precision bf16 \
    --checkpointing_steps 200 --report_to tensorboard --launcher accelerate \
    --out "$SCD_CONFIG"
fi

SCD_RESUME=()
if [[ -e "$SCD_OUT" ]]; then
  SCD_RESUME=(--resume_from_checkpoint latest)
fi

exec python train.py --config "$SCD_CONFIG" --out "$SCD_OUT" \
  --execution standalone --run-label "$SCD_RUN_NAME" "${SCD_RESUME[@]}"
