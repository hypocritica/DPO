#!/usr/bin/env bash
# SLAI: SDXL DMPO Stagewise-C, K=6, four allocated RTX 5090 GPUs.
set -eo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../final_stagewise.sh" sdxl dmpo 6 "$@"
