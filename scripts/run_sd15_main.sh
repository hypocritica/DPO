#!/usr/bin/env bash
# Use the active legacy eval Python, or explicitly set SCD_SD15_PY.
# GPUs must already be allocated by SLAI. No scheduler, download or installation.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${SCD_SD15_PY:-python}"
exec "$python" -u "$repo/evaluate_sd15.py" "$@"
