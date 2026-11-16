#!/usr/bin/env bash
# Inside an allocated SLAI container, using the existing environments. No scheduler submission.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
family="${1:?Usage: bash qualitative/run_slai.sh sd15|sdxl|sd35_medium|zimage smoke|full|all}"
phase="${2:-smoke}"
case "$family" in
  sd15|sdxl) py="${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python}"; gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" ;;
  sd35_medium) py="${SCD_FLOW_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python}"; gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" ;;
  zimage) py="${SCD_FLOW_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python}"; gpus="${CUDA_VISIBLE_DEVICES:-0,1}" ;;
  *) echo "Unknown backbone: $family" >&2; exit 2 ;;
esac
out="${QUAL_OUT:-/mnt/aoss/xietianyu/self_correct/outputs/qualitative_30}"
test -x "$py"
df -h "$(dirname "$out")" 2>/dev/null || df -h /mnt/aoss/xietianyu
export PYTHONUNBUFFERED=1
if [[ "$phase" == all ]]; then
  "$py" qualitative/run.py launch --out "$out" --backbone "$family" --gpus "$gpus" --phase smoke
  "$py" qualitative/run.py launch --out "$out" --backbone "$family" --gpus "$gpus" --phase full
elif [[ "$phase" == smoke || "$phase" == full ]]; then
  "$py" qualitative/run.py launch --out "$out" --backbone "$family" --gpus "$gpus" --phase "$phase"
else
  echo 'Phase must be smoke, full, or all' >&2; exit 2
fi
# Generate review sheets only; selection is intentionally left to the user after download.
sheet_args=()
if [[ "$phase" == smoke ]]; then sheet_args+=(--smoke); fi
"$py" qualitative/layout.py contacts --out "$out" --backbone "$family" "${sheet_args[@]}"
