#!/usr/bin/env bash
# Prepared data only: two GPUs, B=2, accumulation=64, global pairs=256.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
SCD_FAMILY=sd15
scd_train_2gpu dpo "$@"
