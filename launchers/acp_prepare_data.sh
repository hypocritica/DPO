#!/usr/bin/env bash
# CPU-only preparation, separate from training. Completed outputs are reused.
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash launchers/acp_prepare_data.sh [pickapic|hpdv3] [sd15|sdxl|unfiltered] [options]
Defaults: pickapic sd15. unfiltered is HPDv3 data-only (not SD3 training).
Options (separate values): --raw PATH --out PATH --model PATH --revision REPO@COMMIT
  Pick-a-Pic: --images PATH --workers N --batch-size N --resume --adopt-legacy --hpdv3 PATH
  HPDv3: --pickapic PATH --min-confidence FLOAT
Pick-a-Pic also prepares missing fixed monitor prompts. HPDv3 exports pairs only;
its benchmark selection must be created separately for that training manifest.
HELP
  exit 0
fi
SCD_DATASET="${1:-pickapic}"
if [[ $# -gt 0 ]]; then shift; fi
SCD_FAMILY="${1:-sd15}"
if [[ $# -gt 0 ]]; then shift; fi
case "$SCD_DATASET" in
  pickapic) SCD_DATA_PREFIX=PICKAPIC ;;
  hpdv3) SCD_DATA_PREFIX=HPDV3 ;;
  *) echo 'Dataset must be pickapic or hpdv3'; exit 1 ;;
esac
case "$SCD_FAMILY" in
  sd15) SCD_FAMILY_KEY=SD15 ;;
  sdxl) SCD_FAMILY_KEY=SDXL ;;
  unfiltered) SCD_FAMILY_KEY=UNFILTERED
    [[ "$SCD_DATASET" == hpdv3 ]] || { echo 'unfiltered is HPDv3 only'; exit 1; } ;;
  *) echo 'Family must be sd15, sdxl or unfiltered'; exit 1 ;;
esac
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
SCD_RAW_KEY="${SCD_DATA_PREFIX}_RAW"
SCD_OUT_KEY="${SCD_DATA_PREFIX}_${SCD_FAMILY_KEY}_PREPARED"
SCD_RAW="${!SCD_RAW_KEY}"
SCD_PREPARED="${!SCD_OUT_KEY}"
SCD_PREP_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw|--out|--model|--images|--revision|--workers|--batch-size|--hpdv3|--pickapic|--min-confidence)
      if [[ $# -lt 2 || "$2" == --* ]]; then echo "Missing value for $1"; exit 1; fi
      case "$1" in --raw) SCD_RAW="$2" ;; --out) SCD_PREPARED="$2" ;; esac
      SCD_PREP_ARGS+=("$1" "$2"); shift 2 ;;
    --resume|--adopt-legacy) SCD_PREP_ARGS+=("$1"); shift ;;
    *) echo "Unknown option: $1 (see --help)"; exit 1 ;;
  esac
done
scd_log "prepare-${SCD_DATASET}-${SCD_FAMILY}"
if [[ -f "$SCD_PREPARED/ready.json" ]]; then
  if [[ ! -f "$SCD_PREPARED/pairs.jsonl" ]]; then
    echo "ready.json exists but pairs.jsonl is missing: $SCD_PREPARED"; exit 1
  fi
  echo "Reusing completed data (no image scan): $SCD_PREPARED"
  echo 'Preparation options do not alter completed data; use a new --out to change inputs/policy.'
else
  df -h "$SCD_RAW" "$DATASETS_ROOT"
  SCD_EXTRA=()
  if [[ "$SCD_DATASET" == hpdv3 ]]; then SCD_EXTRA=(--hpdv3 "$SCD_RAW"); fi
  python -u scripts/prepare_downloaded_data.py "$SCD_DATASET" --family "$SCD_FAMILY" \
    --raw "$SCD_RAW" --out "$SCD_PREPARED" "${SCD_EXTRA[@]}" "${SCD_PREP_ARGS[@]}"
fi
if [[ ! -f "$SCD_PREPARED/ready.json" || ! -f "$SCD_PREPARED/pairs.jsonl" ]]; then
  echo "Data preparation did not finish: $SCD_PREPARED"; exit 1
fi
if [[ "$SCD_DATASET" == pickapic ]]; then
  SCD_MONITOR_KEY="MONITOR_PROMPTS_${SCD_FAMILY_KEY}"
  SCD_MONITOR="${!SCD_MONITOR_KEY}"
  if [[ ! -f "$SCD_MONITOR" ]]; then
    python -u scripts/prepare_benchmarks.py --train-pairs "$SCD_PREPARED/pairs.jsonl" \
      --pickapic "$SCD_PREPARED/official-test-unique.jsonl" --out "$(dirname "$SCD_MONITOR")"
  else
    echo "Reusing fixed monitor prompts: $SCD_MONITOR"
  fi
fi
echo "Preparation complete: $SCD_PREPARED/pairs.jsonl"
