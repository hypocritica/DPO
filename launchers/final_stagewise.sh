#!/usr/bin/env bash
# Final single-view and K=6 Stagewise-C comparisons in allocated SLAI containers.
set -eo pipefail
if [[ $# -lt 3 ]]; then
  echo 'Usage: final_stagewise.sh <sd15|sdxl> <dspo|dmpo|dpo> <1|6> [check/pause/resume options]'
  exit 1
fi
SCD_FINAL_FAMILY="$1"
SCD_FINAL_OBJECTIVE="$2"
SCD_FINAL_VIEWS="$3"
shift 3
case "$SCD_FINAL_FAMILY" in
  sd15) SCD_FINAL_WORLD=2; SCD_FINAL_BUDGET=384; SCD_FINAL_BATCH=2 ;;
  sdxl) SCD_FINAL_WORLD=4; SCD_FINAL_BUDGET=192; SCD_FINAL_BATCH=1 ;;
  *) echo 'Family must be sd15 or sdxl'; exit 1 ;;
esac
case "$SCD_FINAL_OBJECTIVE" in dspo|dmpo|dpo) ;; *)
  echo 'Objective must be dspo, dmpo or dpo'; exit 1 ;;
esac
case "$SCD_FINAL_VIEWS" in
  1) SCD_FINAL_METHOD=dpo; SCD_FINAL_PROFILE=k1 ;;
  6) SCD_FINAL_METHOD=stagewise-C-gaussian; SCD_FINAL_PROFILE=k6-stage; SCD_FINAL_BATCH=1 ;;
  *) echo 'Views must be 1 or 6'; exit 1 ;;
esac
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Final DSPO/DMPO baselines and DSPO/DMPO/DPO K=6 Stagewise-C comparisons.
SD1.5: two allocated RTX 5090 GPUs, 512, 384 pair-views/update.
       K=1: B=2/GPU, accumulation=96; K=6: B=1/GPU, accumulation=32.
SDXL:  four allocated RTX 5090 GPUs, 1024, 192 pair-views/update.
       K=1: B=1/GPU, accumulation=48; K=6: B=1/GPU, accumulation=8.
Train: 10000 updates, BF16, LoRA rank/alpha16, frozen configuration.
Stage: each pair samples one shared stage, then six independent views within it.
       Preserves the selected base loss; no Huber or low-3/high-3 variant.
Environment: SCD_MODE=train|smoke (default train); SCD_RUN_NAME selects a new run.
Smoke: two updates, accumulation=1, no generation evaluation, separate output.
       The pv384/pv192 name identifies the formal recipe; smoke uses a small budget.
Options: --config-only, --check-only, --stop-after-step N,
         --resume PATH or --resume_from_checkpoint PATH.
Rerunning the same experiment automatically resumes latest.
Training overrides are rejected; use the generic comparison entry for new recipes.
Use one script in each container with the family-specific GPU allocation.
HELP
  exit 0
fi
SCD_FINAL_ARGS=()
SCD_FINAL_CONFIG_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-only|--check-only)
      if [[ "$1" == --config-only ]]; then SCD_FINAL_CONFIG_ONLY=true; fi
      SCD_FINAL_ARGS+=("$1"); shift ;;
    --stop-after-step|--resume|--resume_from_checkpoint)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then echo "Missing value for $1"; exit 1; fi
      if [[ "$1" == --stop-after-step && ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo '--stop-after-step must be a positive integer'; exit 1
      fi
      if [[ "$1" == --resume_from_checkpoint ]]; then
        SCD_FINAL_ARGS+=(--resume "$2")
      else
        SCD_FINAL_ARGS+=("$1" "$2")
      fi
      shift 2 ;;
    *) echo "Unsupported final experiment option: $1; only check/pause/resume controls are allowed."; exit 1 ;;
  esac
done
SCD_MODE="${SCD_MODE:-train}"
case "$SCD_MODE" in
  train) SCD_FINAL_DURATION=10000 ;;
  smoke) SCD_FINAL_DURATION=smoke ;;
  *) echo 'SCD_MODE must be train or smoke'; exit 1 ;;
esac
SCD_RUN_NAME="${SCD_RUN_NAME:-${SCD_FINAL_FAMILY}-${SCD_FINAL_OBJECTIVE}-${SCD_FINAL_PROFILE}-pv${SCD_FINAL_BUDGET}-${SCD_FINAL_WORLD}gpu-${SCD_FINAL_DURATION}-final-v1}"
if [[ ! "$SCD_RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo 'SCD_RUN_NAME must be a simple directory name'; exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_log "$SCD_RUN_NAME"
if [[ "$SCD_FINAL_FAMILY" == sd15 ]]; then
  SCD_FINAL_PREPARED="$PICKAPIC_SD15_PREPARED"
else
  SCD_FINAL_PREPARED="$PICKAPIC_SDXL_PREPARED"
fi
if [[ "$SCD_FINAL_CONFIG_ONLY" != true &&
      ( ! -f "$SCD_FINAL_PREPARED/ready.json" || ! -f "$SCD_FINAL_PREPARED/pairs.jsonl" ) ]]; then
  echo "Prepare Pick-a-Pic first: bash launchers/acp_prepare_data.sh pickapic $SCD_FINAL_FAMILY"; exit 1
fi
echo "Family=$SCD_FINAL_FAMILY, objective=$SCD_FINAL_OBJECTIVE, K=$SCD_FINAL_VIEWS, GPUs=$SCD_FINAL_WORLD, B=$SCD_FINAL_BATCH, mode=$SCD_MODE"
echo "Experiment: $RUNS_ROOT/$SCD_RUN_NAME"
exec python -u scripts/compare_objectives.py --family "$SCD_FINAL_FAMILY" \
  --objective "$SCD_FINAL_OBJECTIVE" --method "$SCD_FINAL_METHOD" --views "$SCD_FINAL_VIEWS" \
  --world-size "$SCD_FINAL_WORLD" --batch-size "$SCD_FINAL_BATCH" --budget-unit pair_views --mode "$SCD_MODE" \
  --run-name "$SCD_RUN_NAME" "${SCD_FINAL_ARGS[@]}"
