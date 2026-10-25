#!/usr/bin/env bash
# SLAI: two allocated RTX 5090 GPUs, sd15, fixed pair groups.
set -eo pipefail
export SCD_FAMILY=sd15 SCD_WORLD_SIZE=2 SCD_VIEWS=1
export SCD_TRAIN_BATCH_SIZE=2 SCD_BUDGET_PER_UPDATE=384
export SCD_RUN_NAME="${SCD_RUN_NAME:-sd15-dpo-fixed-pairs-p384-2gpu-10000-slai-v1}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../acp_gaussian_fixed_pairs.sh" dpo "$@"

