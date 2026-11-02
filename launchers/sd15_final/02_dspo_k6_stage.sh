#!/usr/bin/env bash
# SLAI: SD1.5 DSPO Stagewise-C, K=6, two allocated RTX 5090 GPUs.
set -eo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../final_stagewise.sh" sd15 dspo 6 "$@"
