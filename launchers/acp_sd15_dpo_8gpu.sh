#!/usr/bin/env bash
# Preserve the original experiment name while using this checkout's shared lifecycle.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
SCD_FAMILY=sd15
SCD_WORLD_SIZE=8
SCD_RUN_NAME="${SCD_RUN_NAME:-sd15-dpo-8gpu-10000-v1}"
scd_train dpo "$@"
