#!/usr/bin/env bash
# Run on already allocated SLAI CPU/GPU resources. No downloads or GPU execution.
set -Eeuo pipefail
if (( $# )); then
  if [[ "$1" == -h || "$1" == --help ]]; then
    echo 'Usage: [HPDV3_RAW=PATH] [FLOW_IMAGES=PATH] [FLOW_PREPARED=PATH] [FLOW_PANEL=PATH] bash launchers/flow/prepare_slai_data.sh'
    echo 'Runs source identification, extraction, preparation, audit and panel generation on allocated resources.'
    exit 0
  fi
  echo "Unknown argument: $1 (use --help)" >&2
  exit 2
fi
FLOW_STAGE=environment
trap 'rc=$?; printf "[ERROR] stage=%s line=%s exit=%s; inspect the log above\n" "$FLOW_STAGE" "$LINENO" "$rc" >&2; exit "$rc"' ERR
FLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FLOW_ROOT"
source launchers/flow/slai_common.sh
flow_slai_environment
if [[ -n "${SCD_FLOW_CONDA_SH:-}" ]]; then
  set +u
  source "$SCD_FLOW_CONDA_SH"
  conda activate "$SCD_FLOW_ENV"
  set -u
fi
FLOW_PYTHON="${SCD_FLOW_PYTHON:-python}"
FLOW_EXPORTS="$(PYTHONPATH="$FLOW_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$FLOW_PYTHON" - <<'PY'
import shlex
from self_correct_dpo.resources import load_resources
for key, value in load_resources().items():
    if not key.isidentifier(): raise ValueError(key)
    print(f'export {key}={shlex.quote(value)}')
PY
)"
eval "$FLOW_EXPORTS"
FLOW_IMAGES="${FLOW_IMAGES:-$HPDV3_RAW/flow-sub-images-v2}"
FLOW_PREPARED="${FLOW_PREPARED:-$HPDV3_SUB_PREPARED}"
FLOW_PANEL="${FLOW_PANEL:-$MONITOR_PROMPTS_FLOW_PILOT}"
FLOW_MONITOR_COUNT="${FLOW_MONITOR_COUNT:-64}"
mkdir -p logs/slai-data
exec 9>logs/slai-data/prepare.lock
flock -n 9 || { echo '[ERROR] Another preparation owns this checkout' >&2; exit 1; }
FLOW_LOG="logs/slai-data/prepare-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$FLOW_LOG") 2>&1
printf '[prepare] Python=%s\nRaw=%s\nImages=%s\nPrepared=%s\nPanel=%s\nLog=%s/%s\n' \
  "$FLOW_PYTHON" "$HPDV3_RAW" "$FLOW_IMAGES" "$FLOW_PREPARED" "$FLOW_PANEL" "$FLOW_ROOT" "$FLOW_LOG"
df -h "$HPDV3_RAW"

FLOW_STAGE=source
echo '[1/5] Identify existing source files (HF folders need no custom download receipt)'
FLOW_REVISION="$("$FLOW_PYTHON" -u scripts/inspect_hpdv3_source.py --raw "$HPDV3_RAW")"
printf '[source] Identity=%s\n' "$FLOW_REVISION"

FLOW_STAGE=extract
echo '[2/5] Extract selected images; train annotations are streamed with bounded parser memory'
"$FLOW_PYTHON" -u scripts/extract_hpdv3_sub.py --raw "$HPDV3_RAW" --out "$FLOW_IMAGES" \
  --workers "${FLOW_WORKERS:-4}" --reserve-gib "${FLOW_RESERVE_GIB:-30}"

FLOW_STAGE=prepare
echo '[3/5] Prepare verified preference pairs and splits'
if [[ -f "$FLOW_PREPARED/ready.json" && -f "$FLOW_PREPARED/audit.json" ]]; then
  echo "[prepare] Reusing completed preparation: $FLOW_PREPARED"
elif [[ -e "$FLOW_PREPARED" ]]; then
  echo "[ERROR] Incomplete prepared output exists: $FLOW_PREPARED; preserve it and choose a fresh FLOW_PREPARED path" >&2
  exit 1
else
  "$FLOW_PYTHON" -u scripts/prepare_hpdv3_sub.py --raw "$FLOW_IMAGES" --out "$FLOW_PREPARED" \
    --revision "$FLOW_REVISION" --workers "${FLOW_WORKERS:-4}"
fi

FLOW_STAGE=audit
echo '[4/5] Generate the reusable data validation report'
"$FLOW_PYTHON" -u scripts/audit_flow_data.py --pairs "$FLOW_PREPARED/pairs.jsonl" \
  --extraction "$FLOW_IMAGES/extraction.json" --raw "$HPDV3_RAW" --preparation-log "$FLOW_LOG"

FLOW_STAGE=panel
echo '[5/5] Prepare the canonical evaluation panel'
if [[ -f "$FLOW_PANEL" ]]; then
  PYTHONPATH="$FLOW_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$FLOW_PYTHON" - "$FLOW_PANEL" "$FLOW_MONITOR_COUNT" "$FLOW_PREPARED/ready.json" <<'PY'
import json, sys
from pathlib import Path
from self_correct_dpo.common import load_suite
panel=Path(sys.argv[1]); rows=load_suite(panel)
if len(rows)!=int(sys.argv[2]): raise ValueError('Existing panel has a different size; choose a fresh FLOW_PANEL path')
audit=panel.with_suffix('.audit.json')
if not audit.is_file() or json.loads(audit.read_text()).get('pairs_sha256')!=json.loads(Path(sys.argv[3]).read_text())['pairs_sha256']:
    raise ValueError('Existing panel has a different/unknown data source; choose a fresh FLOW_PANEL path')
print(f'[panel] Reusing {len(rows)} valid canonical prompts')
PY
else
  "$FLOW_PYTHON" -u scripts/prepare_flow_monitor.py --pairs "$FLOW_PREPARED/pairs.jsonl" \
    --out "$FLOW_PANEL" --count "$FLOW_MONITOR_COUNT" --seed 17
fi
printf '[DONE] Data preparation passed. Training input options:\n  --pairs %q --monitor-prompts %q\n' \
  "$FLOW_PREPARED/pairs.jsonl" "$FLOW_PANEL"
