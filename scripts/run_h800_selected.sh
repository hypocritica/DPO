#!/usr/bin/env bash
# Independent formal stages on two already allocated H800 GPUs; no scheduler/install.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-}"; [[ $# == 0 ]] || shift
family="${SCD_SELECTED_FAMILY:-sd35_medium}"
[[ "$family" == sd35_medium || "$family" == zimage ]] || { echo 'SCD_SELECTED_FAMILY must be sd35_medium or zimage' >&2; exit 2; }
root="${SCD_SELECTED_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/$family/formal-h800-v1}"
selection="${SCD_SELECTED_SELECTION:-$root/selection.json}"
# Separate directories permit Parti now, other panels later with identical selection.
out="${SCD_SELECTED_OUT:-$root/parti}"
py="${SCD_EVAL_PY:-python3}"
# Preference scorers resolve resources through the environment as well.
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --resources ]]; then
    ((i+1<${#args[@]})) || { echo '--resources requires a file' >&2; exit 2; }
    export SCD_RESOURCES="${args[i+1]}"
  elif [[ "${args[i]}" == --resources=* ]]; then
    export SCD_RESOURCES="${args[i]#--resources=}"
  fi
done
case "$stage" in
  init) exec "$py" evaluate_selected.py init --family "$family" --out "$selection" "$@" ;;
  check) exec "$py" evaluate_selected.py check --selection "$selection" "$@" ;;
  plan) exec "$py" evaluate_selected.py plan --selection "$selection" --out "$out" "$@" ;;
  index|summarize|gallery) exec "$py" evaluate_suite.py "$stage" --out "$out" "$@" ;;
  paper) exec "$py" export_paper.py --out "$out" "$@" ;;
  generate|score-legacy|score-hpsv3|score-unifiedreward|score-geneval|score-dpg) ;;
  *) echo 'Usage: run_h800_selected.sh init|check|plan|generate|score-legacy|score-hpsv3|score-unifiedreward|score-geneval|score-dpg|index|summarize|gallery|paper [stage arguments]' >&2; exit 2 ;;
esac
# Never invent device IDs or assume 0,1: preserve actual allocated tokens/UUIDs.
gpus="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
IFS=',' read -r -a ids <<< "$gpus"
if [[ ${#ids[@]} != 2 || -z "${ids[0]//[[:space:]]/}" || -z "${ids[1]//[[:space:]]/}" || "${ids[0]}" == "${ids[1]}" || "$gpus" =~ [[:space:]] || ",$gpus," == *,-1,* || "$gpus" == *, ]]; then
  echo 'Exactly two distinct allocated GPU tokens required in SCD_EVAL_GPUS or CUDA_VISIBLE_DEVICES' >&2; exit 2
fi
case "$stage" in
  generate) : "${SCD_FLOW_PY:?Set existing Flow Python}"; worker="$SCD_FLOW_PY"; action=generate ;;
  score-legacy) : "${SCD_SD15_PY:?Set existing legacy eval Python}"; worker="$SCD_SD15_PY"; action=score; if [[ "$family" == sd35_medium ]]; then
    set -- --metrics pickscore image_reward clip aesthetic --benchmarks partiprompts pickv2 hpsv2 "$@"
  else
    set -- --metrics pickscore image_reward --benchmarks partiprompts "$@"
  fi ;;
  score-hpsv3) : "${SCD_HPSV3_PY:?Set HPSv3 Python}"; worker="$SCD_HPSV3_PY"; action=score; set -- --metrics hpsv3 --benchmarks partiprompts pickv2 hpsv2 "$@" ;;
  score-unifiedreward) : "${SCD_UNIFIEDREWARD_PY:?Set UnifiedReward Python}"; worker="$SCD_UNIFIEDREWARD_PY"; action=score; set -- --metrics unifiedreward --benchmarks partiprompts pickv2 hpsv2 "$@" ;;
  score-geneval) : "${SCD_GENEVAL_PY:?Set GenEval Python}"; worker="$SCD_GENEVAL_PY"; action=score; set -- --metrics geneval --benchmarks geneval "$@" ;;
  score-dpg) : "${SCD_DPG_PY:?Set DPG Python}"; worker="$SCD_DPG_PY"; action=score; set -- --metrics dpg --benchmarks dpg "$@" ;;
esac
mkdir -p "$out/logs"
log="$(mktemp "$out/logs/${stage}-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
exec > >(tee -a "$log") 2>&1
printf 'stage=%s family=%s out=%s gpus=%s worker=%s job=%s nodes=%s\n' \
  "$stage" "$family" "$out" "$gpus" "$worker" "${SLURM_JOB_ID:-platform-allocation}" "${SLURM_JOB_NODELIST:-unknown}"
git rev-parse HEAD
# --python selects the worker explicitly; unrelated scorer envs are never required.
"$py" evaluate_suite.py "$action" --out "$out" --gpus "$gpus" --python "$worker" "$@"
if [[ "$action" == generate && " $* " != *' --check-only '* ]]; then
  "$py" evaluate_suite.py index --out "$out"
fi
