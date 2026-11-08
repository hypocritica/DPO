#!/usr/bin/env bash
# One allocated H100, one base load, 20 real ZImage B4 optimizer updates.
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
exec "${SCD_FLOW_PYTHON:-python}" -u scripts/probe_zimage_b4.py "$@"
