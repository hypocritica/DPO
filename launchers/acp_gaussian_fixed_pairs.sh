#!/usr/bin/env bash
# Fixed global pair groups and optimizer updates; UNet work grows with K.
set -eo pipefail
export SCD_COMPARISON_UNIT=pairs
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/acp_gaussian_comparison.sh" "$@"
