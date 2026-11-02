#!/usr/bin/env bash
# SLAI: SD1.5 DSPO score-residual recipe, two allocated RTX 5090 GPUs.
set -eo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" dspo "$@"
