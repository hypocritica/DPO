#!/usr/bin/env bash
# Resume an existing run, permitting batch repartitioning on this branch.
set -eo pipefail
if [[ $# -lt 1 || "${1:-}" == --help ]]; then
  echo 'Usage: bash launchers/acp_resume_storage_fix.sh RUN_NAME [--stop-after-step N]'
  echo 'SCD_TRAIN_BATCH_SIZE optionally changes B while preserving the saved global batch.'
  exit 0
fi
SCD_RUN_NAME="$1"
shift
if [[ ! "$SCD_RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'Pass one experiment name under RUNS_ROOT, without path separators.'
  exit 1
fi
SCD_RUNTIME_ARGS=("$@")
if [[ $# -gt 0 && ( $# -ne 2 || "$1" != --stop-after-step || ! "$2" =~ ^[1-9][0-9]*$ ) ]]; then
  echo 'Only --stop-after-step N is accepted here; use SCD_TRAIN_BATCH_SIZE to change B.'
  exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_log "${SCD_RUN_NAME}-storage-resume"
SCD_OUT="$RUNS_ROOT/$SCD_RUN_NAME"
exec python -u scripts/resume_batch.py --run "$SCD_OUT" "${SCD_RUNTIME_ARGS[@]}"
