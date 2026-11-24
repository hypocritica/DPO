#!/usr/bin/env bash
# New two-phase queue; the original run_multiseed_slai.sh stays unchanged.
set -eo pipefail
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$launcher_dir/.."
export SCD_RESOURCES=${MVPO_GENERATION_RESOURCES:-${SCD_RESOURCES:-$PWD/configs/resources.json}}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
python=${SCD_SD15_PY:-python}
action=${1:?Pass prepare, plan, status, generate, score, report, refresh-history, or one of the six task names}
shift
case "$action" in
  sd35-table|sd15-curves|sd35-curves-dpo-lossavg|sd35-curves-global-stage|sd35-curves-k2|sd35-lossavg-multiseed|A|B)
    exec "$python" scripts/appendix_flexible.py generate --tasks "$action" "$@" ;;
  prepare|plan|status|generate|score|report|refresh-history)
    exec "$python" scripts/appendix_flexible.py "$action" "$@" ;;
  *) echo "Unknown flexible task: $action" >&2; exit 2 ;;
esac
