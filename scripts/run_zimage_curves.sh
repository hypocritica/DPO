#!/usr/bin/env bash
# Fixed 128-prompt Z-Image curve on GPUs already allocated by SLAI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-help}"; [[ $# == 0 ]] || shift
case "$stage" in
  help|--help|-h)
    cat <<'EOF'
Usage: bash scripts/run_zimage_curves.sh STAGE [stage CLI options]
Stages: prepare | plan | generate | score | score-hpsv3 | summarize |
        summarize-hpsv3 | index | all
all: prepare -> plan -> generate -> index -> PickScore -> summarize.
GPU/score failures stop all; rerun the same command to resume.
Generate/score accept --methods base dpo stagewise_k4 and --steps N ... .
Plan accepts --dry-run; generate/score accept --check-only.
SCD_Z_GPUS overrides SCD_EVAL_GPUS/CUDA_VISIBLE_DEVICES (1, 2 or 4 GPUs).
SCD_Z_CURVES/SCD_Z_PREP override AOSS outputs; SCD_Z_RUNS_ROOT,
SCD_Z_DPO_RUN/SCD_Z_K4_RUN select exact runs. SCD_Z_BATCH defaults to 1.
SCD_FLOW_PY generates; SCD_SD15_PY scores PickScore/plots; SCD_HPSV3_PY
is required only for score-hpsv3. Existing prepared inputs remain frozen.
No resource submission, environment installs or scorer downloads.
EOF
    exit 0 ;;
  prepare|plan|generate|score|score-hpsv3|summarize|summarize-hpsv3|index|all) ;;
  *) echo "Unknown stage: $stage; use help" >&2; exit 2 ;;
esac
if [[ "$stage" == all && $# != 0 ]]; then
  echo 'all takes no stage options; use individual stages for filters/check-only/dry-run' >&2; exit 2
fi
out="${SCD_Z_CURVES:-/mnt/aoss/xietianyu/self_correct/evaluations/zimage/curves-h800-b1-v1}"
prepared="${SCD_Z_PREP:-${out}-prepared}"
meta_py="${SCD_EVAL_PY:-${SCD_FLOW_PY:-python3}}"
gpus="${SCD_Z_GPUS-${SCD_EVAL_GPUS-${CUDA_VISIBLE_DEVICES-}}}"
if [[ "$stage" == generate || "$stage" == score || "$stage" == score-hpsv3 || "$stage" == all ]]; then
  IFS=',' read -r -a ids <<< "$gpus"
  if [[ ! ${#ids[@]} =~ ^(1|2|4)$ || "$gpus" =~ [[:space:]] || "$gpus" == *, ]]; then
    echo 'Set 1, 2 or 4 distinct allocated GPU tokens in SCD_Z_GPUS' >&2; exit 2
  fi
  seen=','
  for id in "${ids[@]}"; do
    if [[ -z "$id" || "$id" == -1 || "$seen" == *",$id,"* ]]; then
      echo 'Missing/disabled/duplicate GPU token; set SCD_Z_GPUS explicitly' >&2; exit 2
    fi
    seen+="$id,"
  done
fi
export PYTHONUNBUFFERED=1
run_stage() {
  local name="$1"; shift
  local command=() check_only=0 argument
  case "$name" in
    prepare)
      command=("$meta_py" evaluate_curves.py prepare --family zimage
        --runs-root "${SCD_Z_RUNS_ROOT:-/mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs}"
        --run-glob
        "${SCD_Z_DPO_RUN:-zimage-dpo-k1-pair_views128-2h100-b4-3200-C-float32-trainonly-paper-v2-flow-b4-stat-v1}"
        "${SCD_Z_K4_RUN:-zimage-mvpo-k4-pair_views128-2h100-b1-3200-C-float32-trainonly-stage500-paper-v2-flow-microviews4-v1}"
        --selection-metric pickscore --generation-batch-size "${SCD_Z_BATCH:-1}"
        --cpu-offload false --out "$prepared") ;;
    plan) command=("$meta_py" evaluate_curves.py plan --prepared "$prepared" --out "$out") ;;
    generate)
      : "${SCD_FLOW_PY:?Set the existing Flow Python}"
      command=("$meta_py" evaluate_curves.py generate --out "$out" --gpus "$gpus" --python "$SCD_FLOW_PY") ;;
    score)
      : "${SCD_SD15_PY:?Set the existing eval Python for PickScore}"
      command=("$meta_py" evaluate_curves.py score --out "$out" --gpus "$gpus" --python "$SCD_SD15_PY" --metrics pickscore) ;;
    score-hpsv3)
      : "${SCD_HPSV3_PY:?Set HPSv3 Python only for this optional stage}"
      command=("$meta_py" evaluate_curves.py score --out "$out" --gpus "$gpus" --python "$SCD_HPSV3_PY" --metrics hpsv3) ;;
    summarize|summarize-hpsv3)
      local metric=pickscore
      [[ "$name" != summarize-hpsv3 ]] || metric=hpsv3
      command=("${SCD_SD15_PY:-$meta_py}" evaluate_curves.py summarize --out "$out" --metric "$metric") ;;
    index) command=("$meta_py" evaluate_suite.py index --out "$out") ;;
  esac
  command+=("$@")
  # A plan preview does not create logs or an output directory.
  if [[ "$name" == plan && " $* " == *' --dry-run '* ]]; then
    "${command[@]}"
    return
  fi
  mkdir -p "$out/logs"
  local log
  log="$(mktemp "$out/logs/${name}-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
  printf 'stage=%s out=%s gpus=%s job=%s\n' "$name" "$out" "${gpus:-none}" "${SLURM_JOB_ID:-platform-allocation}" | tee -a "$log"
  git rev-parse HEAD | tee -a "$log"
  "${command[@]}" 2>&1 | tee -a "$log"
  for argument in "$@"; do [[ "$argument" != --check-only ]] || check_only=1; done
  if [[ "$name" == generate && "$check_only" == 0 ]]; then
    run_stage index
  fi
}
if [[ "$stage" == all ]]; then
  : "${SCD_FLOW_PY:?Set existing Flow Python}" "${SCD_SD15_PY:?Set existing eval Python}"
  for name in prepare plan generate score summarize; do run_stage "$name"; done
else
  run_stage "$stage" "$@"
fi
