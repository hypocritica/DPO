#!/usr/bin/env bash
set -eo pipefail
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$launcher_dir/.."
job=${1:?Pass inventory, plan, prepare-FAMILY, A1/A2/B/C/D1/D2/F1/F2/E1/E2/E3/E4, status or report}
shift
export SCD_RESOURCES=${MVPO_GENERATION_RESOURCES:-$PWD/configs/resources.json}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
bootstrap=${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python3.10}
root=$("$bootstrap" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_MULTISEED_ROOT"])')
case "$job" in
  inventory) exec "$bootstrap" scripts/appendix_multiseed.py inventory --output "${MVPO_INVENTORY_OUT:-$root-history-$(date +%Y%m%d-%H%M%S)}" "$@" ;;
  plan|status|report) exec "$bootstrap" scripts/appendix_multiseed.py "$job" --output "$root" "$@" ;;
  prepare-sd15|prepare-sdxl|prepare-sd35_medium|prepare-zimage)
    exec "$bootstrap" scripts/appendix_multiseed.py prepare --family "${job#prepare-}" --output "$root" "$@" ;;
  A1|A2|B|C|D1|D2|F1|F2|E1|E2|E3|E4)
    "$bootstrap" scripts/appendix_multiseed.py run --pack "$job" --output "$root" "$@"
    if [[ ${MVPO_GENERATION_AUTO_REPORT:-1} == 1 ]]; then
      case "$job" in
        A1|A2|F1|E1|E2|E3) family=sd15 ;;
        B|E4) family=sd35_medium ;;
        C) family=zimage ;;
        D1|D2|F2) family=sdxl ;;
      esac
      exec "$bootstrap" scripts/appendix_multiseed.py report --family "$family" --output "$root"
    fi ;;
  *) echo "Unknown multiseed task: $job" >&2;exit 2 ;;
esac
