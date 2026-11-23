#!/usr/bin/env bash
# Two separate SLAI submissions, exactly two allocated GPUs each.
set -eo pipefail
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$launcher_dir/.."
job=${1:?Pass 1, 2, plan, status, or report}
shift
export SCD_RESOURCES=${MVPO_SCAN_RESOURCES:-${SCD_RESOURCES:-$PWD/configs/resources.json}}
export PYTHONUNBUFFERED=1
bootstrap=${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python}
root=${MVPO_STAGE_ROOT:-}
if [[ -z "$root" ]]; then
  root=$(PYTHONPATH="$PWD/src" "$bootstrap" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_STAGE_ROOT"])')
fi
args=(--output "$root" --resources "$SCD_RESOURCES")
if [[ -n ${MVPO_STAGE_BATCH:-} ]]; then args+=(--stage-batch-size "$MVPO_STAGE_BATCH"); fi
case "$job" in
  1|2) exec "$bootstrap" scripts/stage_mechanism.py run --task "$job" "${args[@]}" "$@" ;;
  plan|status|report) exec "$bootstrap" scripts/stage_mechanism.py "$job" "${args[@]}" "$@" ;;
  *) echo "Unknown stage task: $job" >&2; exit 2 ;;
esac
