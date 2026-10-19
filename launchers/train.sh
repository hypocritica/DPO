#!/usr/bin/env bash
# Execute only after CCI/ACP or Slurm has allocated the requested GPU.
set -euo pipefail
SCD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCD_CONFIG="${1:?Usage: bash launchers/train.sh CONFIG OUT [training arguments]}"
SCD_OUT="${2:?Unique output directory required}"
shift 2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SCD_PYTHON="${SCD_PYTHON:-python}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCD_EXECUTION=slurm
else
  SCD_EXECUTION=standalone
fi
exec "$SCD_PYTHON" "$SCD_ROOT/train.py" --config "$SCD_CONFIG" --out "$SCD_OUT" \
  --execution "$SCD_EXECUTION" --run-label "${SCD_RUN_LABEL:-$(basename -- "$SCD_OUT")}" "$@"
