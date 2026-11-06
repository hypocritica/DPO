#!/usr/bin/env bash
# Run only inside the resources already allocated by Slurm / SLAI.
set -euo pipefail
FLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FLOW_ROOT"
if [[ -n "${SCD_FLOW_CONDA_SH:-}" ]]; then
  # Existing Conda hooks may refer to unset variables during activation.
  set +u
  source "$SCD_FLOW_CONDA_SH"
  conda activate "${SCD_FLOW_ENV:-dpo-flow}"
  set -u
fi
exec "${SCD_FLOW_PYTHON:-python}" -u scripts/run_flow.py "$@"
