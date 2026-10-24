#!/usr/bin/env bash
# Shared ACP environment and two-GPU lifecycle; source from an entry script.
set -eo pipefail
SCD_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate dpo
cd "$SCD_REPO_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source paths.sh

scd_log() {
  mkdir -p logs
  SCD_LOG="$SCD_REPO_ROOT/logs/$1.txt"
  exec > >(tee -a "$SCD_LOG") 2>&1
  date '+%F %T %z'
  git -c safe.directory="$PWD" rev-parse HEAD
  echo "Console log: $SCD_LOG"
}

scd_train_2gpu() {
  SCD_METHOD="$1"
  shift
  SCD_FAMILY="${SCD_FAMILY:-sd15}"
  case "$SCD_METHOD" in
    dpo) SCD_VIEWS=1 ;;
    multiview-B-gaussian|multiview-C-gaussian)
      SCD_VIEWS="${SCD_VIEWS:-2}"
      case "$SCD_VIEWS" in 2|4|6|8) ;; *) echo 'SCD_VIEWS must be 2, 4, 6 or 8'; exit 1 ;; esac ;;
    *) echo "Unsupported two-GPU method: $SCD_METHOD"; exit 1 ;;
  esac
  case "$SCD_FAMILY" in
    sd15) SCD_GLOBAL_BATCH=256; SCD_DEFAULT_BATCH=1; SCD_PREPARED="$PICKAPIC_SD15_PREPARED"
          if [[ "$SCD_VIEWS" -le 2 ]]; then SCD_DEFAULT_BATCH=2; fi ;;
    sdxl) SCD_GLOBAL_BATCH=128; SCD_DEFAULT_BATCH=1; SCD_PREPARED="$PICKAPIC_SDXL_PREPARED" ;;
    *) echo 'SCD_FAMILY must be sd15 or sdxl'; exit 1 ;;
  esac
  SCD_TRAIN_BATCH_SIZE="${SCD_TRAIN_BATCH_SIZE:-$SCD_DEFAULT_BATCH}"
  SCD_PROFILE="${SCD_METHOD#multiview-}"
  SCD_PROFILE="${SCD_FAMILY}-gaussian-${SCD_PROFILE%-gaussian}-k${SCD_VIEWS}"
  if [[ "$SCD_METHOD" == dpo ]]; then SCD_PROFILE="${SCD_FAMILY}-dpo"; fi
  SCD_RUN_NAME="${SCD_RUN_NAME:-${SCD_PROFILE}-2gpu-10000-v1}"
  scd_log "$SCD_RUN_NAME"
  SCD_CONFIG="configs/local-${SCD_RUN_NAME}.json"
  SCD_OUT="${RUNS_ROOT}/${SCD_RUN_NAME}"
  echo "Experiment: $SCD_OUT"
  echo "Method=$SCD_METHOD, family=$SCD_FAMILY, K=$SCD_VIEWS, GPUs=2"

  if [[ ! -f "$SCD_PREPARED/ready.json" || ! -f "$SCD_PREPARED/pairs.jsonl" ]]; then
    echo "Data preparation must finish first: $SCD_PREPARED"
    echo "Run: bash launchers/acp_prepare_data.sh pickapic $SCD_FAMILY"
    exit 1
  fi
  python -u scripts/check_deployment.py --family "$SCD_FAMILY" --stage prepared --gpu --world-size 2

  if [[ ! -f "$SCD_CONFIG" ]]; then
    if [[ -e "$SCD_OUT" ]]; then
      echo "Missing frozen config $SCD_CONFIG for an existing experiment; restore it before resuming."
      exit 1
    fi
    python -u scripts/make_config.py --family "$SCD_FAMILY" --method "$SCD_METHOD" \
      --views "$SCD_VIEWS" --world-size 2 --train_batch_size "$SCD_TRAIN_BATCH_SIZE" \
      --effective-batch "$SCD_GLOBAL_BATCH" --max_train_steps 10000 \
      --mixed_precision bf16 --checkpointing_steps 200 \
      --report_to tensorboard --launcher accelerate --out "$SCD_CONFIG"
  fi

  python - "$SCD_CONFIG" "$SCD_FAMILY" "$SCD_VIEWS" "$SCD_METHOD" "$SCD_TRAIN_BATCH_SIZE" "$SCD_GLOBAL_BATCH" <<'PY'
import json
import sys
path, family, views, method, batch, global_batch = sys.argv[1:]
with open(path) as handle:
    config = json.load(handle)
t = config['training']
gaussian = method.endswith('-gaussian')
if (config['model']['family'] != family or config.get('method') != method
        or t['noise_views'] != int(views) or t['batch_size'] != int(batch)
        or t['batch_size'] * t['gradient_accumulation_steps'] * 2 != int(global_batch)
        or t.get('world_size') != 2 or bool(t.get('batch_views')) != gaussian
        or 'independent_rollout' in t or 'paired_drift' in t):
    raise SystemExit('Frozen config differs from requested experiment; use a new SCD_RUN_NAME.')
print(f"Frozen config: B={t['batch_size']}, K={t['noise_views']}, "
      f"UNet images/GPU={2*t['batch_size']*t['noise_views']}, "
      f"accumulation={t['gradient_accumulation_steps']}, global pairs={global_batch}")
PY
  SCD_RESUME=()
  if [[ -e "$SCD_OUT" ]]; then SCD_RESUME=(--resume_from_checkpoint latest); fi
  exec python -u train.py --config "$SCD_CONFIG" --out "$SCD_OUT" \
    --execution standalone --run-label "$SCD_RUN_NAME" "${SCD_RESUME[@]}" "$@"
}
