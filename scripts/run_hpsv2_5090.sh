#!/usr/bin/env bash
# SD3.5M main eight variants, full HPSv2 prompt panel, seed17, four allocated 5090s.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-help}"; [[ $# == 0 ]] || shift
export SCD_SELECTED_FAMILY=sd35_medium
export SCD_SELECTED_GPU_COUNT=4
export SCD_SELECTED_ROOT="${SCD_HPS_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/hpsv2-5090-seed17-v2}"
export SCD_SELECTED_SELECTION="$SCD_SELECTED_ROOT/selection.json"
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/hpsv2"
export SCD_SD15_PY="${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python}"
export SCD_FLOW_PY="${SCD_FLOW_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python}"
export SCD_EVAL_PY="${SCD_EVAL_PY:-$SCD_FLOW_PY}"
export SCD_RESOURCES="${SCD_RESOURCES:-$PWD/configs/resources.json}"
export SCD_HPSV3_PY="${SCD_HPSV3_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python}"
runner=scripts/run_compact_evaluation.sh
case "$stage" in
  init)
    exec bash "$runner" init --generation-batch-size 1 --cpu-offload true \
      --from-prepared "${SCD_SD35_SOURCE_PREP:-/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-h100-b8-v1-prepared}" \
      --handoff configs/evaluation-sd35-main-selection.json "$@" ;;
  plan) exec bash "$runner" plan --benchmarks hpsv2 "$@" ;;
  check|generate|index|summarize|gallery|score-hpsv3|paper) exec bash "$runner" "$stage" "$@" ;;
  score-legacy) exec bash "$runner" score-legacy --benchmarks hpsv2 "$@" ;;
  score)
    [[ $# == 0 ]] || { echo 'score takes no options; use scorer stages' >&2; exit 2; }
    bash "$0" score-legacy
    bash "$0" score-hpsv3 ;;
  all)
    [[ $# == 0 ]] || { echo 'all takes no options; use separate stages' >&2; exit 2; }
    df -h /mnt/aoss/xietianyu
    if [[ ! -f "$SCD_SELECTED_SELECTION" ]]; then bash "$0" init; fi
    bash "$0" check
    bash "$0" plan
    started=$SECONDS
    bash "$0" generate
    "$SCD_EVAL_PY" scripts/report_generation_sessions.py --out "$SCD_SELECTED_OUT"
    bash "$0" score
    bash "$0" summarize --summary-name hpsv2-seed17
    # Existing complete exports are immutable and reusable on restart.
    bash "$0" paper --report main --paper-name hpsv2-seed17
    printf 'HPSv2 generation/scoring/export finished in %s seconds; output=%s\n' "$((SECONDS-started))" "$SCD_SELECTED_OUT" ;;
  help|--help|-h)
    echo 'Usage: run_hpsv2_5090.sh all|init|check|plan|generate|score|score-legacy|score-hpsv3|summarize|paper [options]'
    echo 'Set SCD_EVAL_GPUS to four allocated GPU tokens. No resource submission/install/download.'
    echo 'Eight SD3.5M main variants; 3200 prompts x seed17 = 25600 PNG. batch1 + CPU offload.'
    echo 'Core metrics: PickScore, ImageReward, HPSv3 on the HPSv2 prompt panel. GenEval is not run.' ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
