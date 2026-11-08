#!/usr/bin/env bash
# One allocated GPU, one model load, six real optimizer updates.
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
exec "${SCD_FLOW_PYTHON:-python}" -u scripts/probe_sd35_b2.py "$@"
