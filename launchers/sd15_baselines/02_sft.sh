#!/usr/bin/env bash
# SLAI: SD1.5 winner-only SFT, two allocated RTX 5090 GPUs.
set -eo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" sft "$@"
