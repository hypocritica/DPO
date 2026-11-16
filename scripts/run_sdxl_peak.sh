#!/usr/bin/env bash
# Two independent four-GPU supplement jobs, sharing one frozen checkpoint selection.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
group="${1:-help}"; [[ $# == 0 ]] || shift
stage="${1:-all}"; [[ $# == 0 ]] || shift
case "$group" in
  A|a) group=A; panels=(hpsv2) ;;
  B|b) group=B; panels=(partiprompts pickv2) ;;
  *) echo 'Usage: run_sdxl_peak.sh A|B [init|plan|all|generate|score|summarize|paper]'; exit 2 ;;
esac
case "$stage" in init|plan|all|generate|score|summarize|paper) ;; *) exit 2 ;; esac
[[ "$stage" != all || $# == 0 ]] || { echo 'all takes no stage options' >&2; exit 2; }
py="${SCD_EVAL_PY:-${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python}}"
worker="${SCD_SD15_PY:-$py}"
root="${SCD_SDXL_PEAK_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/peak-5090-v1}"
previous="${SCD_SDXL_PREVIOUS_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/paper-2000-5090-v1}"
config="${SCD_SDXL_CONFIG:-configs/evaluation-suite-sdxl-paper-v1.json}"
gpus="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export SCD_RESOURCES="${SCD_RESOURCES:-$PWD/configs/resources.json}" PYTHONUNBUFFERED=1
if [[ "$stage" == all || "$stage" == generate || "$stage" == score ]]; then
  IFS=',' read -r -a ids <<< "$gpus"
  [[ ${#ids[@]} == 4 && ! "$gpus" =~ [[:space:]] && "$gpus" != *, ]] || { echo 'Expected four GPU tokens' >&2; exit 2; }
  declare -A seen
  for id in "${ids[@]}"; do
    [[ -n "$id" && "$id" != -1 && -z "${seen[$id]:-}" ]] || exit 2
    seen[$id]=1
  done
fi
mkdir -p "$root/logs" "$root/locks"
exec 9>"$root/locks/group-$group.lock"
flock -n 9 || { echo "Group $group already running" >&2; exit 2; }
log="$(mktemp "$root/logs/$group-$stage-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
exec > >(tee -a "$log") 2>&1
git rev-parse HEAD
printf 'SDXL supplement group=%s stage=%s GPUs=%s root=%s\n' "$group" "$stage" "$gpus" "$root"
if [[ "$stage" == init || "$stage" == plan || "$stage" == all ]]; then
  "$py" scripts/prepare_sdxl_peak.py --root "$root/inputs"
  [[ "$stage" != init ]] || exit 0
fi
run_panel() {
  local action="$1" panel="$2"; shift 2
  local out="$root/$panel"
  case "$action" in
    plan) "$py" scripts/prepare_sdxl_formal.py --catalog "$root/inputs/catalog.json" --root "$root" \
      --panel "$panel" --panels partiprompts pickv2 hpsv2 --methods dspo stagewise_dspo_k6 dmpo stagewise_dmpo_k6 \
      --config "$config" --batch-size "${SCD_SDXL_BATCH:-4}" "$@" ;;
    generate)
      "$py" scripts/prepare_sdxl_peak.py --root "$root/inputs" --reuse-out "$out" --previous-root "$previous"
      "$py" evaluate_suite.py generate --out "$out" --gpus "$gpus" --python "$worker" "$@"
      if [[ " $* " != *' --check-only '* ]]; then "$py" evaluate_suite.py index --out "$out"; fi ;;
    score) "$py" evaluate_suite.py score --out "$out" --gpus "$gpus" --python "$worker" "$@" ;;
    summarize) "$py" evaluate_suite.py summarize --out "$out" "$@" ;;
    paper) "$py" export_paper.py --out "$out" --paper-name peak-v1 "$@" ;;
  esac
}
if [[ "$stage" == all ]]; then
  for action in plan generate score summarize paper; do
    for panel in "${panels[@]}"; do run_panel "$action" "$panel"; done
  done
else
  for panel in "${panels[@]}"; do run_panel "$stage" "$panel" "$@"; done
fi
