#!/usr/bin/env bash
# One experiment on two allocated H100 GPUs; restart the same command to resume.
set -eo pipefail
SCD_H100_EXPERIMENT="${1:?Pass dmpo, dmpo-k6-stage or dspo}"
shift
case "$SCD_H100_EXPERIMENT" in dmpo|dmpo-k6-stage|dspo) ;; *) exit 1 ;; esac
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
SDXL on two allocated H100 GPUs: 1024, BF16, B=1/GPU, 192 pair-views/update.
K1: accumulation=96, 192 pairs/update; K6 Stage: accumulation=16, 32 pairs/update.
10000 updates; complete state every 200 steps; LoRA every 1000 steps.
No smoke, deployment self-test, startup image sampling or base generation evaluation.
Periodic generation evaluation at 200, 500, 1000, ... remains enabled.
Rerun the same command after interruption/power loss to resume latest.
An attempt without any committed checkpoint is preserved and restarted from base.
Options: --config-only, --stop-after-step N, --run-tag TAG (default v1).
Environment: SCD_H100_RUN_TAG selects a separate repeat; keep it fixed on resume.
HELP
  exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../acp_common.sh"
scd_log "sdxl-h100-${SCD_H100_EXPERIMENT}"
exec python -u scripts/run_sdxl_h100.py "$SCD_H100_EXPERIMENT" "$@"
