#!/usr/bin/env bash
# CPU metadata conversion only; share existing SD1.5 image files.
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/prepare_sdxl_from_sd15.py" --help
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_log prepare-sdxl-from-sd15
exec python -u scripts/prepare_sdxl_from_sd15.py "$@"
