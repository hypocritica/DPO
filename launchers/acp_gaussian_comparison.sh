#!/usr/bin/env bash
# Shared argument handling for the two explicit comparison entry scripts.
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash launchers/acp_gaussian_fixed_{pairs,views}.sh [METHOD] [runtime options]
METHOD: dpo | multiview-C-gaussian (default) | multiview-B-gaussian
Environment: SCD_FAMILY=sd15|sdxl, SCD_VIEWS=2|4|6|8, SCD_WORLD_SIZE=2|4|8
  SCD_BUDGET_PER_UPDATE: defaults to 384 for SD1.5, 192 for SDXL.
  SCD_TRAIN_BATCH_SIZE: SD1.5 defaults to 2 for K<=2, otherwise 1; SDXL defaults to 1.
  SCD_RUN_NAME: optional new experiment name; otherwise includes mode and budget.
Runtime options: --stop-after-step N, --resume_from_checkpoint PATH (or --resume PATH).
New configurations freeze 10000 updates and the selected exact per-update budget.
Nondivisible budgets and training CLI overrides are rejected. Use allocated GPUs.
HELP
  exit 0
fi
case "${SCD_COMPARISON_UNIT:-}" in pairs|pair_views) ;; *)
  echo 'Use acp_gaussian_fixed_pairs.sh or acp_gaussian_fixed_views.sh'; exit 1 ;; esac
SCD_METHOD=multiview-C-gaussian
if [[ $# -gt 0 && "$1" != --* ]]; then SCD_METHOD="$1"; shift; fi
case "$SCD_METHOD" in dpo|multiview-B-gaussian|multiview-C-gaussian) ;; *)
  echo 'Comparison methods: dpo, multiview-B-gaussian, multiview-C-gaussian'; exit 1 ;; esac
# Only runtime controls can follow a frozen comparison configuration.
SCD_RUNTIME_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-after-step|--resume_from_checkpoint|--resume)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then echo "Missing value for $1"; exit 1; fi
      if [[ "$1" == --stop-after-step && ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo '--stop-after-step must be a positive integer'; exit 1
      fi
      shift 2 ;;
    *) echo "Unsupported comparison option: $1. Frozen training settings use the documented environment variables and a new experiment name."; exit 1 ;;
  esac
done
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_train "$SCD_METHOD" "${SCD_RUNTIME_ARGS[@]}"
