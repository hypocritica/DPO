#!/usr/bin/env bash
# Run inside an already allocated SLAI container; never submits a cluster job.
set -euo pipefail
SCD_EVAL_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCD_EVAL_REPO"
exec "${SCD_PYTHON:-python}" -u evaluate_batch.py "$@"
