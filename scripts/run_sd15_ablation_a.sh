#!/usr/bin/env bash
# Additive entrypoint: never modifies existing plans, environments or code.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-all}"
[[ $# -le 1 ]] || { echo 'Usage: run_sd15_ablation_a.sh [all|plan|generate|score|summarize]' >&2; exit 2; }
case "$stage" in all|plan|generate|score|summarize) ;; *) exit 2 ;; esac
py="${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python}"
export SCD_RESOURCES="${SCD_RESOURCES:-$PWD/configs/resources.json}"
export PYTHONUNBUFFERED=1
root="${SCD_SD15_ABLATION_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/sd15/ablation-a-parti-5090-v1}"
out="$root/partiprompts"
gpus="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
[[ -x "$py" ]] || { echo "Missing Python: $py" >&2; exit 2; }
if [[ "$stage" == all || "$stage" == generate || "$stage" == score ]]; then
  IFS=',' read -r -a ids <<< "$gpus"
  [[ ${#ids[@]} == 4 && "$gpus" != *, && ! "$gpus" =~ [[:space:]] ]] || { echo 'Exactly four allocated GPU tokens required' >&2; exit 2; }
  declare -A seen
  for id in "${ids[@]}"; do
    [[ -n "$id" && "$id" != -1 && -z "${seen[$id]:-}" ]] || { echo 'Invalid GPU tokens' >&2; exit 2; }
    seen[$id]=1
  done
fi
mkdir -p "$root/logs"
exec 9>"$root/.launcher.lock"
flock -n 9 || { echo 'This ablation root is already active' >&2; exit 2; }
log="$(mktemp "$root/logs/$stage-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
exec > >(tee -a "$log") 2>&1
run_stage() {
  local action="$1" started=$SECONDS
  case "$action" in
    plan)
      "$py" evaluate_suite.py plan --catalog configs/evaluation-sd15-ablation-a.json \
        --config configs/evaluation-suite-sd15-ablation-a.json --families sd15 \
        --methods gaussian_b_k2 global_k2 global_k6 --benchmarks partiprompts --out "$out" ;;
    generate)
      "$py" evaluate_suite.py generate --out "$out" --gpus "$gpus" --python "$py"
      "$py" evaluate_suite.py index --out "$out"
      "$py" scripts/report_generation_sessions.py --out "$out" ;;
    score)
      "$py" evaluate_suite.py score --out "$out" --gpus "$gpus" --python "$py" \
        --metrics pickscore image_reward --benchmarks partiprompts ;;
    summarize)
      "$py" evaluate_suite.py summarize --out "$out" --metrics pickscore image_reward \
        --benchmarks partiprompts --summary-name ablation-a-v1 ;;
  esac
  printf 'stage=%s elapsed_seconds=%s\n' "$action" "$((SECONDS-started))"
}
if [[ "$stage" == all ]]; then
  for action in plan generate score summarize; do run_stage "$action"; done
else
  run_stage "$stage"
fi
