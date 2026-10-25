#!/usr/bin/env bash
# SLAI: two allocated RTX 5090 GPUs, sdxl, fixed pair-view budget.
set -eo pipefail
export SCD_FAMILY=sdxl SCD_WORLD_SIZE=2 SCD_VIEWS=6
export SCD_TRAIN_BATCH_SIZE=1 SCD_BUDGET_PER_UPDATE=192
export SCD_RUN_NAME="${SCD_RUN_NAME:-sdxl-gaussian-C-k6-fixed-views-v192-2gpu-10000-slai-v1}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../acp_gaussian_fixed_views.sh" multiview-C-gaussian "$@"
