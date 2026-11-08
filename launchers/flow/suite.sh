#!/usr/bin/env bash
# Select named experiments; all GPU resources must already be allocated.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/launchers/flow/slai_common.sh"
for option in "$@"; do
  if [[ "$option" == --list || "$option" == --plan-only || "$option" == --help ]]; then
    exec "${SCD_FLOW_PYTHON:-python}" "$ROOT/scripts/run_flow_suite.py" "$@"
  fi
done
flow_slai_environment "$@"
if [[ -n "${SCD_FLOW_CONDA_SH:-}" ]]; then
  set +u
  source "$SCD_FLOW_CONDA_SH"
  conda activate "$SCD_FLOW_ENV"
  set -u
fi
cd "$ROOT"
exec "${SCD_FLOW_PYTHON:-python}" -u scripts/run_flow_suite.py "$@"
