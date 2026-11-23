#!/usr/bin/env bash
# One submitted container = exactly two allocated GPUs. No changes to the old D run.
set -eo pipefail
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$launcher_dir/.."
job=${1:?Pass 1, 2, 3, plan, status, or report}
shift
export SCD_RESOURCES=${MVPO_SCAN_RESOURCES:-$PWD/configs/resources.json}
export PYTHONUNBUFFERED=1
bootstrap=${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python3.10}
if [[ ! -x "$bootstrap" ]]; then echo "Missing driver interpreter: $bootstrap" >&2; exit 1; fi
root=$(PYTHONPATH="$PWD/src" "$bootstrap" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_SCAN_ROOT"])')
args=(--output "$root" --resources "$SCD_RESOURCES")
if [[ ${MVPO_SKIP_NUMERICAL_CHECKS:-0} == 1 ]]; then args+=(--skip-numerical-checks); fi
if [[ -n ${MVPO_SD15_VIEW_BATCH:-} ]]; then args+=(--sd15-view-batch-size "$MVPO_SD15_VIEW_BATCH"); fi
if [[ -n ${MVPO_FLOW_VIEW_BATCH:-} ]]; then args+=(--flow-view-batch-size "$MVPO_FLOW_VIEW_BATCH"); fi
case "$job" in
  1|2|3) exec "$bootstrap" scripts/gradient_scan.py run --task "$job" "${args[@]}" "$@" ;;
  plan|status|report) exec "$bootstrap" scripts/gradient_scan.py "$job" "${args[@]}" "$@" ;;
  *) echo "Unknown task: $job" >&2; exit 2 ;;
esac
