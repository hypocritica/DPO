#!/usr/bin/env bash
# Resume the existing winner-only SFT run; retain its optimizer and global batch.
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Resume the original SD1.5 SFT experiment with a larger per-GPU batch.
SCD_TRAIN_BATCH_SIZE=8 (default): two GPUs, accumulation=24, global batch=384.
SCD_RUN_NAME selects the existing run (default: original sd15 SFT baseline).
Options: --config-only, --check-only, --stop-after-step N (absolute step).
The existing training.json, frozen config and evaluation schedule are retained.
HELP
  exit 0
fi
SCD_RUN_NAME="${SCD_RUN_NAME:-sd15-sft-k1-p384-2gpu-10000-slai-v1}"
SCD_TRAIN_BATCH_SIZE="${SCD_TRAIN_BATCH_SIZE:-8}"
if [[ ! "$SCD_RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ||
      ! "$SCD_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Use a simple existing run name and a positive SCD_TRAIN_BATCH_SIZE.'
  exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../acp_common.sh"
scd_log "${SCD_RUN_NAME}-resume-b${SCD_TRAIN_BATCH_SIZE}"
exec python -u scripts/resume_sft_batch.py --run "$RUNS_ROOT/$SCD_RUN_NAME" \
  --batch-size "$SCD_TRAIN_BATCH_SIZE" "$@"
