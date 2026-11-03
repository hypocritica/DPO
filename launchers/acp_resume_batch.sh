#!/usr/bin/env bash
# Resume any existing experiment while changing its per-GPU batch.
set -eo pipefail
if [[ $# -lt 1 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: SCD_TRAIN_BATCH_SIZE=4 bash launchers/acp_resume_batch.sh RUN_NAME [runtime options]'
  echo 'Supports every objective and method; preserves the recorded global batch, GPU count and recipe.'
  exit 0
fi
SCD_RUN_NAME="$1"
shift
if [[ ! "$SCD_RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'Pass one existing experiment name under RUNS_ROOT.'
  exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_log "${SCD_RUN_NAME}-batch-resume"
exec python -u scripts/resume_batch.py --run "$RUNS_ROOT/$SCD_RUN_NAME" "$@"
