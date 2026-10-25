#!/usr/bin/env bash
# One-shot evaluation of two saved adapters; no checkpoint polling or image scans.
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash launchers/{sd15,sdxl}/05_evaluate.sh [STEP [RUN_A [RUN_B]]]
Defaults: STEP=1000, the matching numbered DPO and C K=2 fixed-view runs.
Environment: SCD_EVAL_STEP, SCD_EVAL_A, SCD_EVAL_B can override the defaults.
Run after both checkpoints have been retained (normally every 1000 steps).
Uses two visible allocated GPUs, one per run; monitor prompts, PickScore, 50 sampling steps.
Repeat the same command to resume evaluation. No automatic job submission or waiting.
HELP
  exit 0
fi
if [[ $# -gt 3 ]]; then echo 'Expected STEP and at most two run names; use --help.'; exit 1; fi
SCD_EVAL_STEP="${1:-${SCD_EVAL_STEP:-1000}}"
SCD_EVAL_A="${2:-${SCD_EVAL_A:-}}"
SCD_EVAL_B="${3:-${SCD_EVAL_B:-}}"
case "${SCD_FAMILY:-}" in sd15|sdxl) ;; *)
  echo 'Use launchers/sd15/05_evaluate.sh or launchers/sdxl/05_evaluate.sh'; exit 1 ;; esac
if [[ ! "$SCD_EVAL_STEP" =~ ^(0|[1-9][0-9]*)$ || -z "$SCD_EVAL_A" || -z "$SCD_EVAL_B" || "$SCD_EVAL_A" == "$SCD_EVAL_B" ]]; then
  echo 'Set a nonnegative STEP and two distinct run names.'; exit 1
fi
IFS=, read -r -a SCD_EVAL_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ ${#SCD_EVAL_GPUS[@]} -ne 2 || "${SCD_EVAL_GPUS[0]}" == "${SCD_EVAL_GPUS[1]}" ]]; then
  echo 'Evaluation requires two distinct allocated GPUs in CUDA_VISIBLE_DEVICES.'; exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_log "${SCD_FAMILY}-slai-eval-step${SCD_EVAL_STEP}"
SCD_EVAL_CONFIG="configs/evaluate-${SCD_FAMILY}-monitor.json"

scd_evaluate_one() {
  export CUDA_VISIBLE_DEVICES="$1"
  exec python -u evaluate.py --config "$SCD_EVAL_CONFIG" \
    --run "$RUNS_ROOT/$2" --step "$SCD_EVAL_STEP" \
    --benchmark pickapic --metrics pickscore \
    --out "$RUNS_ROOT/$2/eval-monitor-50steps-step${SCD_EVAL_STEP}-v1" \
    --execution standalone --run-label "$2-eval"
}
SCD_EVAL_LOG_A="$SCD_REPO_ROOT/logs/eval-${SCD_EVAL_A}-step${SCD_EVAL_STEP}.txt"
SCD_EVAL_LOG_B="$SCD_REPO_ROOT/logs/eval-${SCD_EVAL_B}-step${SCD_EVAL_STEP}.txt"
echo "GPU ${SCD_EVAL_GPUS[0]}: $SCD_EVAL_A; log: $SCD_EVAL_LOG_A"
echo "GPU ${SCD_EVAL_GPUS[1]}: $SCD_EVAL_B; log: $SCD_EVAL_LOG_B"
scd_evaluate_one "${SCD_EVAL_GPUS[0]}" "$SCD_EVAL_A" >>"$SCD_EVAL_LOG_A" 2>&1 &
SCD_EVAL_PID_A=$!
scd_evaluate_one "${SCD_EVAL_GPUS[1]}" "$SCD_EVAL_B" >>"$SCD_EVAL_LOG_B" 2>&1 &
SCD_EVAL_PID_B=$!
trap 'kill -TERM "$SCD_EVAL_PID_A" "$SCD_EVAL_PID_B" 2>/dev/null || true' TERM INT
SCD_EVAL_STATUS=0
wait "$SCD_EVAL_PID_A" || SCD_EVAL_STATUS=1
wait "$SCD_EVAL_PID_B" || SCD_EVAL_STATUS=1
if [[ $SCD_EVAL_STATUS -eq 0 ]]; then
  echo "Both evaluations completed at step $SCD_EVAL_STEP."
else
  echo "Evaluation did not complete; see the two logs above."
fi
exit "$SCD_EVAL_STATUS"
