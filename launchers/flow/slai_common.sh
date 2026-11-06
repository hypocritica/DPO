#!/usr/bin/env bash
# Defaults shared by the numbered paper-v2 launchers; no allocation or downloads.
FLOW_SLAI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export SCD_RESOURCES="${SCD_RESOURCES:-$FLOW_SLAI_ROOT/configs/resources-slai-flow.json}"
export SCD_FLOW_ENV="${SCD_FLOW_ENV:-mvpo-flow}"
export SCD_FLOW_REPORTS_ROOT="${SCD_FLOW_REPORTS_ROOT:-$FLOW_SLAI_ROOT/reports/slai}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Configuration generation works on a development CPU environment too. An
# explicit interpreter bypasses automatic activation (e.g. a custom prefix).
flow_slai_environment() {
  local option
  for option in "$@"; do
    [[ "$option" == --config-only ]] && return 0
  done
  if [[ -z "${SCD_FLOW_PYTHON:-}" && -z "${SCD_FLOW_CONDA_SH:-}" ]]; then
    local conda_sh=/mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh
    if [[ -f "$conda_sh" ]]; then
      export SCD_FLOW_CONDA_SH="$conda_sh"
    fi
  fi
}
