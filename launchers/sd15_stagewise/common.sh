#!/usr/bin/env bash
# Dedicated frozen stagewise experiments; source the existing ACP environment only.
set -eo pipefail
SCD_STAGE_OBJECTIVE="$1"
shift
case "$SCD_STAGE_OBJECTIVE" in dpo|phase|constant|reverse) ;; *) exit 1 ;; esac
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Stagewise SD1.5: 10000 updates, K=2, 192 pairs / 384 pair-views per update.'
  echo 'Use allocated 2/4/8 GPUs (SCD_WORLD_SIZE, default 2); SCD_RUN_NAME creates a new run.'
  echo 'Runtime options only: --stop-after-step N, --resume_from_checkpoint PATH, --resume PATH.'
  exit 0
fi
SCD_RUNTIME_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-after-step|--resume_from_checkpoint|--resume)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then echo "Missing value for $1"; exit 1; fi
      if [[ "$1" == --stop-after-step && ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo '--stop-after-step must be a positive integer'; exit 1
      fi
      shift 2 ;;
    *) echo "Unsupported stagewise option: $1; only pause/resume controls are allowed."; exit 1 ;;
  esac
done
SCD_WORLD_SIZE="${SCD_WORLD_SIZE:-2}"
case "$SCD_WORLD_SIZE" in 2|4|8) ;; *) echo 'SCD_WORLD_SIZE must be 2, 4 or 8'; exit 1 ;; esac
SCD_RUN_NAME="${SCD_RUN_NAME:-sd15-stagewise-${SCD_STAGE_OBJECTIVE}-k2-p192-${SCD_WORLD_SIZE}gpu-10000-slai-v1}"
if [[ ! "$SCD_RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo 'SCD_RUN_NAME must be a simple directory name'; exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../acp_common.sh"
scd_log "$SCD_RUN_NAME"
SCD_CONFIG="configs/local-${SCD_RUN_NAME}.json"
SCD_OUT="${RUNS_ROOT}/${SCD_RUN_NAME}"
if [[ ! -f "$PICKAPIC_SD15_PREPARED/ready.json" || ! -f "$PICKAPIC_SD15_PREPARED/pairs.jsonl" ]]; then
  echo 'Prepare Pick-a-Pic first: bash launchers/acp_prepare_data.sh pickapic sd15'; exit 1
fi
if [[ ! -f "$SCD_CONFIG" && -e "$SCD_OUT" ]]; then
  echo "Missing frozen config $SCD_CONFIG for existing output; restore it before resuming."; exit 1
fi
# Regenerate the complete expected config on each launch: checks stage parameters,
# optimizer, evaluation panel/sampling, resources and every other frozen setting.
SCD_STAGE_TMP="$(mktemp -d)"
trap 'rm -rf -- "$SCD_STAGE_TMP"' EXIT
python -u scripts/make_config.py --family sd15 --method stagewise-C-gaussian \
  --views 2 --world-size "$SCD_WORLD_SIZE" --train_batch_size 2 \
  --budget-unit pair_views --budget-per-update 384 --max_train_steps 10000 --mixed_precision bf16 \
  --checkpointing_steps 200 --report_to tensorboard --launcher accelerate \
  --set "training.stagewise.objective=$SCD_STAGE_OBJECTIVE" \
  --set training.stagewise.boundary=500 --set training.stagewise.target_margin=1.0 \
  --set training.stagewise.huber_delta=1.0 --set training.stagewise.huber_weight=0.5 \
  --out "$SCD_STAGE_TMP/expected.json"
python - "$SCD_STAGE_TMP/expected.json" "$SCD_CONFIG" <<'PY'
import json
from pathlib import Path
import sys
expected, frozen = map(Path, sys.argv[1:])
if frozen.exists():
    if json.loads(frozen.read_text()) != json.loads(expected.read_text()):
        raise SystemExit('Frozen config differs from requested stagewise experiment; use a new SCD_RUN_NAME.')
else:
    with frozen.open('x') as handle:
        handle.write(expected.read_text())
c = json.loads(frozen.read_text())
t = c['training']
assert t['batch_size'] * t['world_size'] * t['gradient_accumulation_steps'] == 192
print(f"Frozen stagewise config: {frozen}; objective={t['stagewise']['objective']}; "
      f"GPUs={t['world_size']}, B=2, accumulation={t['gradient_accumulation_steps']}, "
      'K=2, global pairs=192, pair-views=384, steps=10000')
PY
rm -rf -- "$SCD_STAGE_TMP"
trap - EXIT
SCD_RESUME=()
if [[ -e "$SCD_OUT" ]]; then SCD_RESUME=(--resume_from_checkpoint latest); fi
echo "Experiment: $SCD_OUT"
exec python -u train.py --config "$SCD_CONFIG" --out "$SCD_OUT" \
  --execution standalone --run-label "$SCD_RUN_NAME" "${SCD_RESUME[@]}" "${SCD_RUNTIME_ARGS[@]}"
