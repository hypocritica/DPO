#!/usr/bin/env bash
# Two GPUs; select K with SCD_VIEWS=2/4/6/8, arm with SCD_ARM=B/C.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
case "${SCD_ARM:-C}" in B|C) ;; *) echo 'SCD_ARM must be B or C'; exit 1 ;; esac
scd_train_2gpu "multiview-${SCD_ARM:-C}-gaussian" "$@"
