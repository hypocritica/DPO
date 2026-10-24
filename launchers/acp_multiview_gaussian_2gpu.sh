#!/usr/bin/env bash
# Prepared data only. Run in an allocated two-GPU ACP container.
# Select SCD_FAMILY=sd15/sdxl, SCD_VIEWS=2/4/6/8, SCD_ARM=B/C before launch.
set -eo pipefail

source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate dpo
cd /mnt/afs/L202500030/xietianyu/workspace/DPO

SCD_FAMILY="${SCD_FAMILY:-sd15}"
SCD_VIEWS="${SCD_VIEWS:-2}"
SCD_ARM="${SCD_ARM:-C}"
case "$SCD_VIEWS" in 2|4|6|8) ;; *) echo "SCD_VIEWS must be 2, 4, 6 or 8"; exit 1 ;; esac
case "$SCD_ARM" in B|C) ;; *) echo "SCD_ARM must be B or C"; exit 1 ;; esac
case "$SCD_FAMILY" in
  sd15) SCD_GLOBAL_BATCH=256; SCD_DEFAULT_BATCH=1
        if [[ "$SCD_VIEWS" == 2 ]]; then SCD_DEFAULT_BATCH=2; fi ;;
  sdxl) SCD_GLOBAL_BATCH=128; SCD_DEFAULT_BATCH=1 ;;
  *) echo "SCD_FAMILY must be sd15 or sdxl"; exit 1 ;;
esac
SCD_TRAIN_BATCH_SIZE="${SCD_TRAIN_BATCH_SIZE:-$SCD_DEFAULT_BATCH}"
SCD_RUN_NAME="${SCD_RUN_NAME:-${SCD_FAMILY}-gaussian-${SCD_ARM}-k${SCD_VIEWS}-2gpu-10000-v1}"
SCD_METHOD="multiview-${SCD_ARM}-gaussian"

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
git -c safe.directory="$PWD" rev-parse HEAD
echo "Console log: $SCD_LOG"
echo "Experiment: $SCD_OUT"
echo "Ordinary Gaussian $SCD_ARM: K=$SCD_VIEWS, pairs/GPU=$SCD_TRAIN_BATCH_SIZE, global pairs=$SCD_GLOBAL_BATCH"

# Never start another preparation against the shared dataset directory.
SCD_READY="$PICKAPIC_SD15_PREPARED/ready.json"
if [[ "$SCD_FAMILY" == sdxl ]]; then SCD_READY="$PICKAPIC_SDXL_PREPARED/ready.json"; fi
if [[ ! -f "$SCD_READY" ]]; then
  echo "Data preparation must finish first: $SCD_READY"
  exit 1
fi
python scripts/check_deployment.py --family "$SCD_FAMILY" --stage prepared --gpu --world-size 2

if [[ ! -f "$SCD_CONFIG" ]]; then
  if [[ -e "$SCD_OUT" ]]; then
    echo "Missing frozen config $SCD_CONFIG for an existing experiment; restore it before resuming."
    exit 1
  fi
  python scripts/make_config.py --family "$SCD_FAMILY" --method "$SCD_METHOD" \
    --views "$SCD_VIEWS" --world-size 2 --train_batch_size "$SCD_TRAIN_BATCH_SIZE" \
    --effective-batch "$SCD_GLOBAL_BATCH" --max_train_steps 10000 \
    --mixed_precision bf16 --checkpointing_steps 200 \
    --report_to tensorboard --launcher accelerate --out "$SCD_CONFIG"
fi

# Changing K/family/batch requires a new experiment, not reusing a frozen run.
python - "$SCD_CONFIG" "$SCD_FAMILY" "$SCD_VIEWS" "$SCD_METHOD" "$SCD_TRAIN_BATCH_SIZE" "$SCD_GLOBAL_BATCH" <<'PY'
import json
import sys
path, family, views, method, batch, global_batch = sys.argv[1:]
with open(path) as handle:
    config = json.load(handle)
t = config['training']
if (config['model']['family'] != family or config.get('method') != method
        or t['noise_views'] != int(views) or t['batch_size'] != int(batch)
        or t['batch_size'] * t['gradient_accumulation_steps'] * 2 != int(global_batch)
        or t.get('world_size') != 2 or t.get('batch_views') is not True
        or 'independent_rollout' in t or 'paired_drift' in t):
    raise SystemExit('Frozen config differs from requested Gaussian experiment; use a new SCD_RUN_NAME.')
print(f"Frozen config: B={t['batch_size']}, K={t['noise_views']}, "
      f"UNet images/GPU={2*t['batch_size']*t['noise_views']}, "
      f"accumulation={t['gradient_accumulation_steps']}, global pairs={global_batch}")
PY

SCD_RESUME=()
if [[ -e "$SCD_OUT" ]]; then SCD_RESUME=(--resume_from_checkpoint latest); fi
exec python -u train.py --config "$SCD_CONFIG" --out "$SCD_OUT" \
  --execution standalone --run-label "$SCD_RUN_NAME" "${SCD_RESUME[@]}" "$@"
