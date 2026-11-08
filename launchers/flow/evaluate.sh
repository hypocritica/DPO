#!/usr/bin/env bash
# Run on a separately allocated GPU; training keeps running independently.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/launchers/flow/slai_common.sh"
flow_slai_environment "$@"
if [[ -n "${SCD_FLOW_CONDA_SH:-}" ]]; then
  set +u
  source "$SCD_FLOW_CONDA_SH"
  conda activate "$SCD_FLOW_ENV"
  set -u
fi
cd "$ROOT"
exec "${SCD_FLOW_PYTHON:-python}" -u scripts/evaluate_flow.py "$@"
