#!/usr/bin/env bash
# Five independent single-view baselines on two allocated RTX 5090 GPUs.
set -eo pipefail
SCD_BASELINE_OBJECTIVE="${1:-}"
shift
case "$SCD_BASELINE_OBJECTIVE" in kto|sft|mapo|dmpo|dspo) ;; *)
  echo 'Objective must be kto, sft, mapo, dmpo or dspo'; exit 1 ;; esac
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
SD1.5 baseline: two allocated RTX 5090 GPUs, K=1, 512, BF16, LoRA rank/alpha16.
Train: 10000 updates, default B=2 per GPU, accumulation=96, global pairs=384.
SCD_TRAIN_BATCH_SIZE changes B and accumulation; existing runs resume saved recipes.
Environment: SCD_MODE=train|smoke (default train); SCD_RUN_NAME selects a new run.
Smoke: two updates, accumulation=1, no generation evaluation, separate output.
Options: --config-only, --check-only, --stop-after-step N,
         --resume PATH or --resume_from_checkpoint PATH.
Rerunning the same experiment automatically resumes latest.
Training overrides are rejected; use the generic comparison entry for new recipes.
Use one script in each allocated two-GPU container.
HELP
  exit 0
fi
SCD_BASELINE_ARGS=()
SCD_BASELINE_CONFIG_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-only|--check-only)
      if [[ "$1" == --config-only ]]; then SCD_BASELINE_CONFIG_ONLY=true; fi
      SCD_BASELINE_ARGS+=("$1"); shift ;;
    --stop-after-step|--resume|--resume_from_checkpoint)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then echo "Missing value for $1"; exit 1; fi
      if [[ "$1" == --stop-after-step && ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo '--stop-after-step must be a positive integer'; exit 1
      fi
      if [[ "$1" == --resume_from_checkpoint ]]; then
        SCD_BASELINE_ARGS+=(--resume "$2")
      else
        SCD_BASELINE_ARGS+=("$1" "$2")
      fi
      shift 2 ;;
    *) echo "Unsupported baseline option: $1; only check/pause/resume controls are allowed."; exit 1 ;;
  esac
done
SCD_MODE="${SCD_MODE:-train}"
case "$SCD_MODE" in
  train) SCD_BASELINE_SUFFIX=p384-2gpu-10000 ;;
  smoke) SCD_BASELINE_SUFFIX=2gpu-smoke ;;
  *) echo 'SCD_MODE must be train or smoke'; exit 1 ;;
esac
SCD_RUN_NAME="${SCD_RUN_NAME:-sd15-${SCD_BASELINE_OBJECTIVE}-k1-${SCD_BASELINE_SUFFIX}-slai-v1}"
if [[ ! "$SCD_RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo 'SCD_RUN_NAME must be a simple directory name'; exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../acp_common.sh"
scd_log "$SCD_RUN_NAME"
scd_resume_if_exists "$RUNS_ROOT/$SCD_RUN_NAME" "${SCD_BASELINE_ARGS[@]}"
if [[ "$SCD_BASELINE_CONFIG_ONLY" != true &&
      ( ! -f "$PICKAPIC_SD15_PREPARED/ready.json" || ! -f "$PICKAPIC_SD15_PREPARED/pairs.jsonl" ) ]]; then
  echo 'Prepare Pick-a-Pic first: bash launchers/acp_prepare_data.sh pickapic sd15'; exit 1
fi
echo "SD1.5 objective=$SCD_BASELINE_OBJECTIVE, K=1, GPUs=2, B=${SCD_TRAIN_BATCH_SIZE:-2}, mode=$SCD_MODE"
echo "Experiment: $RUNS_ROOT/$SCD_RUN_NAME"
exec python -u scripts/compare_objectives.py --family sd15 \
  --objective "$SCD_BASELINE_OBJECTIVE" --method dpo --views 1 \
  --world-size 2 --batch-size "${SCD_TRAIN_BATCH_SIZE:-2}" --budget-unit pairs --mode "$SCD_MODE" \
  --run-name "$SCD_RUN_NAME" "${SCD_BASELINE_ARGS[@]}"
