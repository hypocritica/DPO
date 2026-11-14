#!/usr/bin/env bash
# Three independent four-GPU jobs; all must use the same pinned checkout and root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
group="${1:-help}"; [[ $# == 0 ]] || shift
stage="${1:-all}"; [[ $# == 0 ]] || shift
case "$group" in
  A|a) group=A; jobs=(main:hpsv2) ;;
  B|b) group=B; jobs=(main:partiprompts main:pickv2 supplement:partiprompts) ;;
  C|c) group=C; jobs=(main:geneval) ;;
  help|--help|-h)
    echo 'Usage: run_sdxl_paper.sh A|B|C [all|plan|generate|score|index|summarize|paper] [stage options]'
    echo 'A: HPSv2. B: Parti/PickV2 + four step3000 Parti supplements. C: GenEval four seeds.'
    echo 'Main: Base0 + nine methods2000. Set four allocated GPU tokens in SCD_EVAL_GPUS.'
    echo 'Restart the same command/root to resume; no resource submission, download or installation.'
    exit 0 ;;
  *) echo 'Group must be A, B or C' >&2; exit 2 ;;
esac
case "$stage" in all|plan|generate|score|index|summarize|paper) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
[[ "$stage" != all || $# == 0 ]] || { echo 'all takes no options; use individual stages' >&2; exit 2; }
export SCD_SD15_PY="${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python}"
export SCD_GENEVAL_PY="${SCD_GENEVAL_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-geneval-v1/bin/python}"
export SCD_EVAL_PY="${SCD_EVAL_PY:-$SCD_SD15_PY}"
export SCD_RESOURCES="${SCD_RESOURCES:-$PWD/configs/resources.json}"
export PYTHONUNBUFFERED=1
root="${SCD_SDXL_PAPER_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/sdxl/paper-2000-5090-v1}"
config="${SCD_SDXL_CONFIG:-configs/evaluation-suite-sdxl-paper-v1.json}"
batch="${SCD_SDXL_BATCH:-4}"
py="$SCD_EVAL_PY"
gpus="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ "$stage" == all || "$stage" == generate || "$stage" == score ]]; then
  IFS=',' read -r -a ids <<< "$gpus"
  [[ ${#ids[@]} == 4 && ! "$gpus" =~ [[:space:]] && "$gpus" != *, ]] || { echo 'Exactly four allocated GPU tokens required' >&2; exit 2; }
  declare -A seen
  for id in "${ids[@]}"; do
    [[ -n "$id" && "$id" != -1 && -z "${seen[$id]:-}" ]] || { echo 'GPU tokens must be distinct and nonempty' >&2; exit 2; }
    seen[$id]=1
  done
fi
# Reject an unavailable scoring environment before spending a full generation job.
executables=("$py")
if [[ "$stage" == all || "$stage" == generate ]]; then executables+=("$SCD_SD15_PY"); fi
if [[ "$stage" == all || "$stage" == score ]]; then
  if [[ "$group" == C ]]; then executables+=("$SCD_GENEVAL_PY"); else executables+=("$SCD_SD15_PY"); fi
fi
for executable in "${executables[@]}"; do
  [[ -x "$executable" ]] || { echo "Missing existing Python: $executable; set the corresponding SCD_*_PY" >&2; exit 2; }
done
mkdir -p "$root/logs" "$root/locks"
exec 9>"$root/locks/group-$group.lock"
flock -n 9 || { echo "Group $group is already active at $root" >&2; exit 2; }
log="$(mktemp "$root/logs/group-$group-$stage-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
exec > >(tee -a "$log") 2>&1
printf 'SDXL group=%s stage=%s GPUs=%s batch=%s output=%s\n' "$group" "$stage" "$gpus" "$batch" "$root"
git rev-parse HEAD
df -h "$(dirname "$root")"
started=$SECONDS

run_panel_stage() {
  local action="$1" role="$2" panel="$3"
  shift 3
  local target="$root/$role/$panel" catalog worker
  local -a planning
  if [[ "$role" == main ]]; then
    catalog=configs/evaluation-sdxl-main.json
    planning=(--panels partiprompts pickv2 hpsv2 geneval)
  else
    catalog=configs/evaluation-sdxl-parti-supplement.json
    planning=(--panels partiprompts --methods dmpo stagewise_dmpo_k6 dspo stagewise_dspo_k6)
  fi
  case "$action" in
    plan)
      "$py" scripts/prepare_sdxl_formal.py --catalog "$catalog" --root "$root/$role" --panel "$panel" \
        --config "$config" --batch-size "$batch" "${planning[@]}" "$@" ;;
    generate)
      "$py" evaluate_suite.py generate --out "$target" --gpus "$gpus" --python "$SCD_SD15_PY" "$@"
      if [[ " $* " != *' --check-only '* ]]; then
        "$py" evaluate_suite.py index --out "$target"
        "$py" scripts/report_generation_sessions.py --out "$target"
      fi ;;
    score)
      worker="$SCD_SD15_PY"
      if [[ "$panel" == geneval ]]; then worker="$SCD_GENEVAL_PY"; fi
      # Frozen legacy metric group for preference panels; dedicated GenEval otherwise.
      "$py" evaluate_suite.py score --out "$target" --gpus "$gpus" --python "$worker" "$@" ;;
    index|summarize) "$py" evaluate_suite.py "$action" --out "$target" "$@" ;;
    paper) "$py" export_paper.py --out "$target" --paper-name "$role-v1" "$@" ;;
  esac
}

# Publish/check every plan before starting this group's expensive work.
if [[ "$stage" == all ]]; then
  for job in "${jobs[@]}"; do run_panel_stage plan "${job%%:*}" "${job#*:}"; done
  for job in "${jobs[@]}"; do
    for action in generate score summarize paper; do
      phase_started=$SECONDS
      run_panel_stage "$action" "${job%%:*}" "${job#*:}"
      printf 'phase=%s job=%s elapsed_seconds=%s\n' "$action" "$job" "$((SECONDS-phase_started))"
    done
  done
else
  for job in "${jobs[@]}"; do run_panel_stage "$stage" "${job%%:*}" "${job#*:}" "$@"; done
fi
printf 'SDXL group=%s stage=%s finished in %s seconds; log=%s\n' "$group" "$stage" "$((SECONDS-started))" "$log"
