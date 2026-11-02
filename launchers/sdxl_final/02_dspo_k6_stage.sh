#!/usr/bin/env bash
# SLAI: SDXL DSPO Stagewise-C, K=6, four allocated RTX 5090 GPUs.
set -eo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../final_stagewise.sh" sdxl dspo 6 "$@"
