#!/usr/bin/env bash
# Paper comparison recipes; execute only on already allocated GPU servers.
set -euo pipefail
if (( $# < 5 )); then
  echo "Internal usage: paper_experiment.sh FAMILY ALGORITHM K BUDGET_UNIT STAGE [options]" >&2
  exit 2
fi
family=$1 algorithm=$2 views=$3 budget_unit=$4 stage=$5
aggregation=C
boundary=500
shift 5
while [[ "${1:-}" == --internal-aggregation || "${1:-}" == --internal-boundary ]]; do
  case "$1" in
    --internal-aggregation) aggregation=${2:?Missing aggregation} ;;
    --internal-boundary) boundary=${2:?Missing boundary} ;;
  esac
  shift 2
done
profile=5090
[[ "$family" == zimage ]] && profile=a800
run_tag=flow-paper-v2
weight_dtype=
forward=()
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
Recipe: $family / $algorithm / K=$views / $budget_unit=128 / stagewise=$stage

  --profile 5090|h100|a800   5090: 4 GPUs; H100/A800: 2 GPUs
                            SD3.5 defaults 5090; ZImage defaults A800
  --weight-dtype DTYPE      Explicit model weight precision; ZImage defaults float32.
                            ZImage on 5090 requires --weight-dtype bfloat16.
  --run-tag TAG             Experiment version (default: flow-paper-v2).
                            Reuse the same tag to resume automatically.
  --execution MODE          standalone (default) or slurm; resources already allocated
  --resources FILE          Resource-path configuration
  --data-report FILE        Prepared dataset audit receipt (default: adjacent validation-report.json)
  --model-report FILE       Model receipt (default: reports/slai/FAMILY-model-validation.json)
  --config-only             Freeze configuration without launching training
  --check-only              Validate prepared inputs without launching training
  --runs-root DIR           Persistent experiment output root
  --config-dir DIR          Frozen configuration directory
  --stop-after-step N       Pause after an optimizer step
  --help                    Show this help

Other run_flow.py runtime options are forwarded. Recipe identity (family,
algorithm, K, budget, batch, world size and stage settings) cannot be overridden.
EOF
}
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --profile|--run-tag|--weight-dtype)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --profile) profile=$2 ;;
        --run-tag) run_tag=$2 ;;
        --weight-dtype) weight_dtype=$2 ;;
      esac
      shift 2 ;;
    --profile=*) profile=${1#*=}; shift ;;
    --run-tag=*) run_tag=${1#*=}; shift ;;
    --weight-dtype=*) weight_dtype=${1#*=}; shift ;;
    --family|--family=*|--algorithm|--algorithm=*|--views|--views=*|--budget|--budget=*|--budget-unit|--budget-unit=*|--world-size|--world-size=*|--batch-size|--batch-size=*|--recipe|--recipe=*|--stagewise|--stage-boundary|--stage-boundary=*|--smoke|--aggregator|--aggregator=*|--aggregation|--aggregation=*)
      echo "Recipe identity cannot be overridden: $1" >&2; exit 2 ;;
    *) forward+=("$1"); shift ;;
  esac
done
case "$profile" in
  5090) world=4 ;;
  h100|a800) world=2 ;;
  *) echo "Unknown profile: $profile (choose 5090, h100, or a800)" >&2; exit 2 ;;
esac
if [[ "$family" == zimage && "$profile" == 5090 && "$weight_dtype" != bfloat16 ]]; then
  echo "ZImage 4x5090 is an explicit precision variant: pass --weight-dtype bfloat16." >&2
  exit 2
fi
FLOW_LAUNCHERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$FLOW_LAUNCHERS/slai_common.sh"
flow_slai_environment "${forward[@]}"
args=(--family "$family" --algorithm "$algorithm" --recipe paper-v2
      --profile "$profile" --world-size "$world" --batch-size 1
      --budget-unit "$budget_unit" --budget 128 --run-tag "$run_tag"
      --execution standalone --run-label "$family-$algorithm-k$views-$budget_unit-$profile-$run_tag"
      --model-report "$SCD_FLOW_REPORTS_ROOT/$family-model-validation.json")
# Non-MVPO algorithms infer K1; their CLI --views default remains 2.
[[ "$algorithm" == mvpo ]] && args+=(--views "$views" --aggregation "$aggregation")
[[ "$stage" == true ]] && args+=(--stagewise --stage-boundary "$boundary")
[[ -n "$weight_dtype" ]] && args+=(--weight-dtype "$weight_dtype")
exec bash "$FLOW_LAUNCHERS/train.sh" "${args[@]}" "${forward[@]}"
