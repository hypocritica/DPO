#!/usr/bin/env bash
# Shared ACP environment and single-node training lifecycle.
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
  if ! git -c safe.directory="$PWD" rev-parse HEAD; then
    echo 'WARNING: Could not record Git revision; continuing to deployment checks.'
  fi
  echo "Console log: $SCD_LOG"
}

scd_train() {
  SCD_METHOD="$1"
  shift
  SCD_FAMILY="${SCD_FAMILY:-sd15}"
  SCD_WORLD_SIZE="${SCD_WORLD_SIZE:-2}"
  case "$SCD_WORLD_SIZE" in 2|4|8) ;; *) echo 'SCD_WORLD_SIZE must be 2, 4 or 8'; exit 1 ;; esac
  case "$SCD_METHOD" in
    dpo) SCD_VIEWS=1 ;;
    multiview-B|multiview-C)
      SCD_VIEWS="${SCD_VIEWS:-2}"
      if [[ ! "$SCD_VIEWS" =~ ^[1-9][0-9]*$ ]]; then echo 'SCD_VIEWS must be a positive integer'; exit 1; fi ;;
    multiview-B-gaussian|multiview-C-gaussian)
      SCD_VIEWS="${SCD_VIEWS:-2}"
      case "$SCD_VIEWS" in 2|4|6|8) ;; *) echo 'SCD_VIEWS must be 2, 4, 6 or 8'; exit 1 ;; esac ;;
    *) echo "Unsupported method: $SCD_METHOD"; exit 1 ;;
  esac
  case "$SCD_FAMILY" in
    sd15) SCD_GLOBAL_BATCH=256; SCD_DEFAULT_BATCH=1; SCD_PREPARED="$PICKAPIC_SD15_PREPARED"
          if [[ "$SCD_VIEWS" -le 2 ]]; then SCD_DEFAULT_BATCH=2; fi ;;
    sdxl) SCD_GLOBAL_BATCH=128; SCD_DEFAULT_BATCH=1; SCD_PREPARED="$PICKAPIC_SDXL_PREPARED" ;;
    *) echo 'SCD_FAMILY must be sd15 or sdxl'; exit 1 ;;
  esac
  SCD_TRAIN_BATCH_SIZE="${SCD_TRAIN_BATCH_SIZE:-$SCD_DEFAULT_BATCH}"
  if [[ ! "$SCD_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo 'SCD_TRAIN_BATCH_SIZE must be a positive integer'; exit 1
  fi
  SCD_BUDGET_ARGS=(--effective-batch "$SCD_GLOBAL_BATCH")
  SCD_BUDGET_SUFFIX=""
  if [[ -n "${SCD_COMPARISON_UNIT:-}" ]]; then
    case "$SCD_COMPARISON_UNIT" in pairs|pair_views) ;; *) echo 'Invalid comparison unit'; exit 1 ;; esac
    case "$SCD_METHOD" in dpo|*-gaussian) ;; *) echo 'Comparison budgets require DPO or ordinary Gaussian views'; exit 1 ;; esac
    if [[ "$SCD_FAMILY" == sd15 ]]; then SCD_DEFAULT_BUDGET=384; else SCD_DEFAULT_BUDGET=192; fi
    SCD_BUDGET_PER_UPDATE="${SCD_BUDGET_PER_UPDATE:-$SCD_DEFAULT_BUDGET}"
    if [[ ! "$SCD_BUDGET_PER_UPDATE" =~ ^[1-9][0-9]*$ ]]; then
      echo 'SCD_BUDGET_PER_UPDATE must be a positive integer'; exit 1
    fi
    SCD_BUDGET_FACTOR=1
    if [[ "$SCD_COMPARISON_UNIT" == pair_views ]]; then SCD_BUDGET_FACTOR="$SCD_VIEWS"; fi
    SCD_BUDGET_DIVISOR=$((SCD_TRAIN_BATCH_SIZE * SCD_WORLD_SIZE * SCD_BUDGET_FACTOR))
    if (( SCD_BUDGET_PER_UPDATE % SCD_BUDGET_DIVISOR != 0 )); then
      echo "Budget $SCD_BUDGET_PER_UPDATE must be divisible by B * GPUs * budget factor = $SCD_BUDGET_DIVISOR; no rounding is allowed."
      exit 1
    fi
    SCD_GLOBAL_BATCH=$((SCD_BUDGET_PER_UPDATE / SCD_BUDGET_FACTOR))
    SCD_BUDGET_ARGS=(--budget-unit "$SCD_COMPARISON_UNIT" --budget-per-update "$SCD_BUDGET_PER_UPDATE")
    if [[ "$SCD_COMPARISON_UNIT" == pairs ]]; then
      SCD_BUDGET_SUFFIX="-fixed-pairs-p${SCD_BUDGET_PER_UPDATE}"
    else
      SCD_BUDGET_SUFFIX="-fixed-views-v${SCD_BUDGET_PER_UPDATE}"
    fi
  fi
  SCD_PROFILE="${SCD_METHOD#multiview-}"
  case "$SCD_METHOD" in
    dpo) SCD_PROFILE="${SCD_FAMILY}-dpo" ;;
    *-gaussian) SCD_PROFILE="${SCD_FAMILY}-gaussian-${SCD_PROFILE%-gaussian}-k${SCD_VIEWS}" ;;
    *) SCD_PROFILE="${SCD_FAMILY}-rollout-${SCD_PROFILE}-k${SCD_VIEWS}" ;;
  esac
  SCD_RUN_NAME="${SCD_RUN_NAME:-${SCD_PROFILE}${SCD_BUDGET_SUFFIX}-${SCD_WORLD_SIZE}gpu-10000-v1}"
  scd_log "$SCD_RUN_NAME"
  SCD_CONFIG="configs/local-${SCD_RUN_NAME}.json"
  SCD_OUT="${RUNS_ROOT}/${SCD_RUN_NAME}"
  echo "Experiment: $SCD_OUT"
  echo "Method=$SCD_METHOD, family=$SCD_FAMILY, K=$SCD_VIEWS, GPUs=$SCD_WORLD_SIZE"

  if [[ ! -f "$SCD_PREPARED/ready.json" || ! -f "$SCD_PREPARED/pairs.jsonl" ]]; then
    echo "Data preparation must finish first: $SCD_PREPARED"
    echo "Run: bash launchers/acp_prepare_data.sh pickapic $SCD_FAMILY"
    exit 1
  fi
  python -u scripts/check_deployment.py --family "$SCD_FAMILY" --stage prepared --gpu --world-size "$SCD_WORLD_SIZE"

  if [[ ! -f "$SCD_CONFIG" ]]; then
    if [[ -e "$SCD_OUT" ]]; then
      echo "Missing frozen config $SCD_CONFIG for an existing experiment; restore it before resuming."
      exit 1
    fi
    python -u scripts/make_config.py --family "$SCD_FAMILY" --method "$SCD_METHOD" \
      --views "$SCD_VIEWS" --world-size "$SCD_WORLD_SIZE" --train_batch_size "$SCD_TRAIN_BATCH_SIZE" \
      "${SCD_BUDGET_ARGS[@]}" --max_train_steps 10000 \
      --mixed_precision bf16 --checkpointing_steps 200 \
      --report_to tensorboard --launcher accelerate --out "$SCD_CONFIG"
  fi

  python - "$SCD_CONFIG" "$SCD_FAMILY" "$SCD_VIEWS" "$SCD_METHOD" "$SCD_TRAIN_BATCH_SIZE" "$SCD_GLOBAL_BATCH" "$SCD_WORLD_SIZE" "${SCD_COMPARISON_UNIT:-}" "${SCD_BUDGET_PER_UPDATE:-0}" <<'PY'
import json
import sys
path, family, views, method, batch, global_batch, world, budget_unit, budget = sys.argv[1:]
with open(path) as handle:
    config = json.load(handle)
t = config['training']
gaussian = method.endswith('-gaussian')
rollout = method in ('multiview-B', 'multiview-C')
expected_budget = {'unit': budget_unit, 'per_update': int(budget)} if budget_unit else None
if (config['model']['family'] != family or config.get('method') != method
        or t['noise_views'] != int(views) or t['batch_size'] != int(batch)
        or t['batch_size'] * t['gradient_accumulation_steps'] * int(world) != int(global_batch)
        or t.get('world_size') != int(world) or bool(t.get('batch_views')) != gaussian
        or t.get('comparison_budget') != expected_budget
        or ('independent_rollout' in t) != rollout or 'paired_drift' in t
        or t['formal_arm'] != ('B' if method.startswith('multiview-B') else 'C')
        or t['resolution'] != (1024 if family == 'sdxl' else 512)
        or t['max_steps'] != 10000 or t['checkpoint_every'] != 200):
    raise SystemExit('Frozen config differs from requested experiment; use a new SCD_RUN_NAME.')
print(f"Frozen config: B={t['batch_size']}, K={t['noise_views']}, "
      f"UNet images/GPU={2*t['batch_size']*t['noise_views']}, "
      f"accumulation={t['gradient_accumulation_steps']}, global pairs={global_batch}, "
      f"global pair-views={int(global_batch)*t['noise_views']}, comparison={expected_budget}")
PY
  SCD_RESUME=()
  if [[ -e "$SCD_OUT" ]]; then SCD_RESUME=(--resume_from_checkpoint latest); fi
  exec python -u train.py --config "$SCD_CONFIG" --out "$SCD_OUT" \
    --execution standalone --run-label "$SCD_RUN_NAME" "${SCD_RESUME[@]}" "$@"
}

# Keep the existing SD1.5/gaussian two-GPU entry scripts and run names stable.
scd_train_2gpu() {
  SCD_WORLD_SIZE=2
  scd_train "$@"
}
