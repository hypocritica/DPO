#!/usr/bin/env bash
# Preserve the original experiment name while using this checkout's shared lifecycle.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
SCD_FAMILY=sd15
SCD_WORLD_SIZE=2
SCD_RUN_NAME="${SCD_RUN_NAME:-sd15-C-k2-2gpu-10000-v1}"
scd_train multiview-C "$@"
