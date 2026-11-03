#!/usr/bin/env bash
# Batch-resume branch: retain the original script name when changing checkouts.
set -eo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/02_sft_large_batch.sh" "$@"
