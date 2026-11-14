#!/usr/bin/env bash
# One preference panel per already allocated four-GPU resource group.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-}"; [[ $# == 0 ]] || shift
[[ "${1:-}" == --panel && $# -ge 2 ]] || { echo 'Usage: run_sdxl_formal.sh STAGE --panel partiprompts|pickv2|hpsv2 [args]' >&2; exit 2; }
panel="$2"; shift 2
[[ "$panel" == partiprompts || "$panel" == pickv2 || "$panel" == hpsv2 ]] || { echo 'Unsupported preference panel' >&2; exit 2; }
root="${SCD_SDXL_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/formal-5090-v1}"
out="$root/$panel"
py="${SCD_EVAL_PY:-python3}"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --resources ]]; then
    ((i+1<${#args[@]})) || { echo '--resources requires a file' >&2; exit 2; }
    export SCD_RESOURCES="${args[i+1]}"
  elif [[ "${args[i]}" == --resources=* ]]; then export SCD_RESOURCES="${args[i]#--resources=}"; fi
done
case "$stage" in
  plan)
    : "${SCD_SDXL_CATALOG:?Set explicitly reviewed exact-step SDXL catalog}"
    exec "$py" scripts/prepare_sdxl_formal.py --catalog "$SCD_SDXL_CATALOG" --root "$root" --panel "$panel" \
      --config "${SCD_SDXL_CONFIG:-configs/evaluation-suite.json}" --batch-size "${SCD_SDXL_BATCH:-4}" "$@" ;;
  index|summarize|gallery) exec "$py" evaluate_suite.py "$stage" --out "$out" "$@" ;;
  paper) exec "$py" export_paper.py --out "$out" "$@" ;;
  generate|score) ;;
  *) echo 'Stages: plan generate score index summarize gallery paper' >&2; exit 2 ;;
esac
: "${SCD_SD15_PY:?Set existing legacy eval Python for SDXL generation/scoring}"
gpus="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
IFS=',' read -r -a ids <<< "$gpus"
[[ ${#ids[@]} == 4 && ! "$gpus" =~ [[:space:]] && "$gpus" != *, ]] || { echo 'Exactly four distinct allocated GPU tokens required' >&2; exit 2; }
declare -A seen
for id in "${ids[@]}"; do
  [[ -n "$id" && "$id" != -1 && -z "${seen[$id]:-}" ]] || { echo 'Exactly four distinct allocated GPU tokens required' >&2; exit 2; }
  seen[$id]=1
done
mkdir -p "$out/logs"
log="$(mktemp "$out/logs/${stage}-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
exec > >(tee -a "$log") 2>&1
printf 'stage=%s panel=%s out=%s gpus=%s worker=%s\n' "$stage" "$panel" "$out" "$gpus" "$SCD_SD15_PY"
git rev-parse HEAD
if [[ "$stage" == score ]]; then
  set -- --metrics pickscore hpsv2 image_reward clip aesthetic "$@"
fi
"$py" evaluate_suite.py "$stage" --out "$out" --gpus "$gpus" --python "$SCD_SD15_PY" "$@"
if [[ "$stage" == generate && " $* " != *' --check-only '* ]]; then
  "$py" evaluate_suite.py index --out "$out"
fi
