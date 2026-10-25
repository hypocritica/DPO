#!/usr/bin/env bash
# Fixed global pair-view evaluations and optimizer updates; pair groups shrink by K.
set -eo pipefail
export SCD_COMPARISON_UNIT=pair_views
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/acp_gaussian_comparison.sh" "$@"
