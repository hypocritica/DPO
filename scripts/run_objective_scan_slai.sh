#!/usr/bin/env bash
# Native SD1.5 objectives; independent of the pinned task 1/2/3 checkout/output.
set -eo pipefail
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$launcher_dir/.."
job=${1:?Pass 4, 5, plan, status, report, status-4/5, or report-4/5}
shift
export SCD_RESOURCES=${MVPO_SCAN_RESOURCES:-$PWD/configs/resources.json}
export PYTHONUNBUFFERED=1
bootstrap=${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python3.10}
if [[ ! -x "$bootstrap" ]]; then echo "Missing driver interpreter: $bootstrap" >&2; exit 1; fi
root=$(PYTHONPATH="$PWD/src" "$bootstrap" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_OBJECTIVE_SCAN_ROOT"])')
args=(--output "$root" --resources "$SCD_RESOURCES")
if [[ -n ${MVPO_SD15_VIEW_BATCH:-} ]]; then args+=(--sd15-view-batch-size "$MVPO_SD15_VIEW_BATCH"); fi
case "$job" in
  4|5) exec "$bootstrap" scripts/objective_scan.py run --task "$job" "${args[@]}" "$@" ;;
  plan|status|report) exec "$bootstrap" scripts/objective_scan.py "$job" "${args[@]}" "$@" ;;
  status-[45]|report-[45]) exec "$bootstrap" scripts/objective_scan.py "${job%-*}" --task "${job#*-}" "${args[@]}" "$@" ;;
  *) echo "Unknown native objective task: $job" >&2; exit 2 ;;
esac
