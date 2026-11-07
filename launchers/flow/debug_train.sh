#!/usr/bin/env bash
# Small independent snapshot; run inside an already allocated GPU task.
set -euo pipefail
if (( $# < 1 )); then
  echo 'Usage: debug_train.sh sd35_medium|zimage [dpo|mvpo|linear_dpo|sft] [run_flow runtime options]' >&2
  exit 2
fi
family=$1; shift
algorithm=dpo
if [[ ${1:-} != -* && -n ${1:-} ]]; then algorithm=$1; shift; fi
case "$family" in
  sd35_medium) profile=5090; world=4; placement=offload ;;
  zimage) profile=h100; world=2; placement=resident ;;
  *) echo 'Choose sd35_medium or zimage' >&2; exit 2 ;;
esac
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Defaults: 50 steps, B1, accumulation1, 1024px, no generation/scoring, offline W&B; separate smoke output.'
  echo 'ZImage defaults 2xH100; SD3.5 defaults 4x5090. Use --stop-after-step 25 then omit it to verify resume.'
  echo 'Data: FLOW_DEBUG_PAIRS; length: FLOW_DEBUG_STEPS; identity: --run-tag (default predata-v1).'
  exit 0
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/launchers/flow/slai_common.sh"
flow_slai_environment "$@"
exec bash "$ROOT/launchers/flow/train.sh" --family "$family" --algorithm "$algorithm" \
  --profile "$profile" --world-size "$world" --execution standalone \
  --run-label "$family-$algorithm-$profile-predata-debug" \
  --smoke --smoke-steps "${FLOW_DEBUG_STEPS:-50}" --run-tag predata-v1 \
  --pairs "${FLOW_DEBUG_PAIRS:-/mnt/aoss/xietianyu/datasets/hpdv3/flow-debug-v1/prepared/pairs.jsonl}" \
  --model-report "$SCD_FLOW_REPORTS_ROOT/$family-model-validation.json" \
  --beta 500 --learning-rate 5e-6 --frozen-placement "$placement" \
  --report-to all --wandb-mode offline --wandb-project mvpo-flow-debug \
  --wandb-group "$family-predata-debug" "$@"
