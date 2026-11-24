#!/usr/bin/env bash
set -eo pipefail
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$launcher_dir/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SCD_RESOURCES=${MVPO_GENERATION_RESOURCES:-${SCD_RESOURCES:-$PWD/configs/resources.json}}
python=${SCD_SD15_PY:-python}
export MVPO_FLEX_ROOT=${MVPO_PARALLEL_ROOT:-$("$python" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_PARALLEL_ROOT"])')}
export MVPO_FLEX_LEGACY_ROOT=${MVPO_FLEX_LEGACY_ROOT:-$("$python" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_FLEX_LEGACY_ROOT"])')}
action=${1:?Pass prepare, plan, status, generate TASK, score TASK, report, or gallery FAMILY ACTION}
shift
case "$action" in
  gallery)
    family=${1:?Pass sd15/sdxl/sd35_medium/zimage}; phase=${2:-run}; shift; if [[ $# -gt 0 ]]; then shift; fi
    gallery_root=${MVPO_PARALLEL_GALLERY_ROOT:-$("$python" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_PARALLEL_GALLERY_ROOT"])')}
    exec "$python" scripts/appendix_parallel_figures.py "$phase" --output "$gallery_root" --family "$family" --legacy-root "$MVPO_FLEX_LEGACY_ROOT" --source-root "$MVPO_FLEX_ROOT" "$@" ;;
  generate|score)
    task=${1:?Pass E1/E2/E3/sd35-METHOD or core}; shift
    exec "$python" scripts/appendix_flexible.py "$action" --config configs/appendix-parallel-5090.json --tasks "$task" "$@" ;;
  prepare|plan|status|report|refresh-history)
    exec "$python" scripts/appendix_flexible.py "$action" --config configs/appendix-parallel-5090.json "$@" ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
