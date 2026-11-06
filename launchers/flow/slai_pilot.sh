#!/usr/bin/env bash
# Run one effect experiment inside an already allocated SLAI GPU container.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash launchers/flow/slai_pilot.sh {sd35_medium|zimage} {dpo|mvpo|linear_dpo|sft} [run_flow options]' >&2
  exit 2
fi
FLOW_FAMILY="$1"
FLOW_ALGORITHM="$2"
shift 2
case "$FLOW_FAMILY" in sd35_medium|zimage) ;; *) echo 'Invalid model family' >&2; exit 2;; esac
case "$FLOW_ALGORITHM" in dpo|mvpo|linear_dpo|sft) ;; *) echo 'Invalid algorithm' >&2; exit 2;; esac
export SCD_FLOW_CONDA_SH="${SCD_FLOW_CONDA_SH:-/mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh}"
export SCD_FLOW_ENV="${SCD_FLOW_ENV:-mvpo-flow}"
FLOW_LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$FLOW_LAUNCHER_DIR/train.sh" \
  --family "$FLOW_FAMILY" --algorithm "$FLOW_ALGORITHM" \
  --recipe paper-pilot --profile h100 --world-size 2 --run-tag paper-v1 \
  --execution standalone --run-label "$FLOW_FAMILY-$FLOW_ALGORITHM-paper-v1" "$@"
