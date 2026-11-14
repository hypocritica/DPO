#!/usr/bin/env bash
# Compact formal evaluation on two already allocated H100/H800 GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:-help}"; [[ $# == 0 ]] || shift
family="${SCD_SELECTED_FAMILY:-sd35_medium}"
[[ "$family" == sd35_medium || "$family" == zimage ]] || { echo 'Use sd35_medium or zimage' >&2; exit 2; }
export SCD_SELECTED_FAMILY="$family"
export SCD_SELECTED_ROOT="${SCD_SELECTED_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/$family/compact-h100-v1}"
export SCD_SELECTED_OUT="${SCD_SELECTED_OUT:-$SCD_SELECTED_ROOT/core}"
runner=scripts/run_h800_selected.sh
case "$stage" in
  init) exec bash "$runner" init --config configs/evaluation-suite-compact.json "$@" ;;
  plan|check|generate|index|summarize|gallery|paper|score-hpsv3|score-geneval)
    exec bash "$runner" "$stage" "$@" ;;
  score-legacy)
    exec bash "$runner" score-legacy --metrics pickscore image_reward --benchmarks partiprompts pickv2 "$@" ;;
  score-unifiedreward)
    exec bash "$runner" score-unifiedreward --methods base dpo stagewise_k4 --benchmarks partiprompts "$@" ;;
  score)
    [[ $# == 0 ]] || { echo 'score takes no options; use separate scorer stages' >&2; exit 2; }
    bash "$0" score-legacy
    bash "$0" score-hpsv3
    bash "$0" score-geneval ;;
  help|--help|-h)
    echo 'Usage: run_compact_evaluation.sh init|check|plan|generate|score|score-legacy|score-hpsv3|score-geneval|score-unifiedreward|summarize|paper [options]'
    echo 'Full Parti + PickV2 + GenEval; SD3.5 preference seeds 17/29, Z-Image seed17, GenEval four seeds.'
    echo 'Initialize a fresh selection/output root. score runs the three core metrics and GenEval; UnifiedReward is optional.' ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
