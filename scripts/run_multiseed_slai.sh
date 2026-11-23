#!/usr/bin/env bash
set -eo pipefail
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$launcher_dir/.."
job=${1:?Pass group-h100-a/group-h100-b, figures-prepare/figures-export, inventory, plan, prepare-FAMILY, A1/A2/B/C/D1/D2/F1/F2/E1/E2/E3/E4, status or report}
shift
export SCD_RESOURCES=${MVPO_GENERATION_RESOURCES:-$PWD/configs/resources.json}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
bootstrap=${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python3.10}
root=$("$bootstrap" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["MVPO_MULTISEED_ROOT"])')
case "$job" in
  group-h100-a|group-h100-b)
    export SCD_EVAL_GPUS=${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
    IFS=',' read -r -a allocated <<< "$SCD_EVAL_GPUS"
    if [[ ${#allocated[@]} != 2 ]]; then echo "$job requires two allocated GPUs" >&2; exit 2; fi
    if [[ $job == group-h100-a ]]; then
      packs=(A1 A2 F1 D1 D2 F2 E1 E2 E3); families=(sd15 sdxl)
    else
      packs=(B L C E4 E5); families=(sd35_medium zimage)
    fi
    stage=0
    for pack in "${packs[@]}"; do
      stage=$((stage+1)); echo "[$job] quantitative stage $stage/${#packs[@]}: $pack"
      bash "$launcher_dir/run_multiseed_slai.sh" "$pack" "$@"
    done
    for family in "${families[@]}"; do
      echo "[$job] figure generation and export: $family"
      "$bootstrap" scripts/appendix_figures.py run --output "$root" --family "$family"
    done
    exit 0 ;;
  figures-prepare|figures-export)
    action=${job#figures-}
    for family in sd15 sdxl sd35_medium zimage; do
      "$bootstrap" scripts/appendix_figures.py "$action" --output "$root" --family "$family" "$@"
    done
    exit 0 ;;
  group-5090-a|group-5090-b|group-h100)
    case "$job" in
      group-5090-a) packs=(A1 A2 F1 D1 D2 F2 E1 E2 E3); expected_gpus=4 ;;
      group-5090-b) packs=(B E4); expected_gpus=4 ;;
      group-h100) packs=(C); expected_gpus=2 ;;
    esac
    export SCD_EVAL_GPUS=${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
    IFS=',' read -r -a allocated <<< "$SCD_EVAL_GPUS"
    if [[ ${#allocated[@]} != "$expected_gpus" ]]; then
      echo "$job requires $expected_gpus explicitly allocated GPUs in SCD_EVAL_GPUS/CUDA_VISIBLE_DEVICES" >&2; exit 2
    fi
    for pack in "${packs[@]}"; do
      bash "$launcher_dir/run_multiseed_slai.sh" "$pack" "$@"
    done
    exit 0 ;;
  inventory) exec "$bootstrap" scripts/appendix_multiseed.py inventory --output "${MVPO_INVENTORY_OUT:-$root-history-$(date +%Y%m%d-%H%M%S)}" "$@" ;;
  plan|status|report) exec "$bootstrap" scripts/appendix_multiseed.py "$job" --output "$root" "$@" ;;
  prepare-sd15|prepare-sdxl|prepare-sd35_medium|prepare-zimage)
    exec "$bootstrap" scripts/appendix_multiseed.py prepare --family "${job#prepare-}" --output "$root" "$@" ;;
  A1|A2|B|L|C|D1|D2|F1|F2|E1|E2|E3|E4|E5)
    "$bootstrap" scripts/appendix_multiseed.py run --pack "$job" --output "$root" "$@"
    if [[ ${MVPO_GENERATION_AUTO_REPORT:-1} == 1 ]]; then
      case "$job" in
        A1|A2|F1|E1|E2|E3) family=sd15 ;;
        B|L|E4|E5) family=sd35_medium ;;
        C) family=zimage ;;
        D1|D2|F2) family=sdxl ;;
      esac
      exec "$bootstrap" scripts/appendix_multiseed.py report --family "$family" --output "$root"
    fi ;;
  *) echo "Unknown multiseed task: $job" >&2;exit 2 ;;
esac
