#!/usr/bin/env bash
# Sequential evaluation on two already allocated GPUs. No installs/downloads.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SCD_SD15_PY:?Set original eval Python}"
: "${SCD_FLOW_PY:?Set prepared Flow Python}"
export SCD_CURVES_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
b35="${SCD_SD35_BATCH:-8}"
b15="${SCD_SD15_BATCH:-64}"
export SCD_CURVES_RUNS_ROOT="${SCD_CURVES_RUNS_ROOT:-/mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs}"
export SCD_CURVES_OUT="${SCD_CURVES_OUT:-/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-h100-b${b35}-v1}"
export SCD_CURVES_PREP="${SCD_CURVES_PREP:-${SCD_CURVES_OUT}-prepared}"
sd15="${SCD_SD15_OUT:-/mnt/aoss/xietianyu/self_correct/evaluations/sd15/main-h100-b${b15}-v1}"
mkdir -p "$sd15/logs" "$SCD_CURVES_OUT/logs"
log="$(mktemp "$SCD_CURVES_OUT/logs/two-track-$(date -u +%Y%m%dT%H%M%S)-XXXXXX.log")"
exec > >(tee -a "$log") 2>&1
"$SCD_SD15_PY" - "$SCD_CURVES_GPUS" "$b35" "$b15" <<'PY'
import sys
ids=sys.argv[1].split(',')
if len(ids)!=2 or len(set(ids))!=2 or any(not x.strip() or x.strip()=='-1' for x in ids):
    raise SystemExit('Exactly two allocated GPU IDs are required')
if any(int(x)<1 for x in sys.argv[2:]):
    raise SystemExit('Batch sizes must be positive')
PY
# Check both environments and assets before spending GPU time.
"$SCD_SD15_PY" scripts/check_eval_environment.py --profile legacy --gpu --execution standalone
"$SCD_FLOW_PY" scripts/check_eval_environment.py --profile flow --skip-assets --gpu --execution standalone
"$SCD_FLOW_PY" scripts/prepare_eval_assets.py check --profile pickscore
"$SCD_SD15_PY" evaluate_sd15.py plan --out "$sd15" --generation-batch-size "$b15"
bash scripts/run_sd35_curves.sh prepare --generation-batch-size "$b35"
bash scripts/run_sd35_curves.sh plan
bash scripts/run_sd35_curves.sh generate --gpus "$SCD_CURVES_GPUS" --check-only
# First validate an actual trained SD3.5 checkpoint at this batch size.
# These 128 images belong to the final curve plan and are reused below.
"$SCD_FLOW_PY" evaluate_curves.py generate --out "$SCD_CURVES_OUT" \
  --gpus "$SCD_CURVES_GPUS" --methods dpo --steps 3200
bash scripts/run_sd35_curves.sh generate --gpus "$SCD_CURVES_GPUS"
"$SCD_FLOW_PY" evaluate_suite.py index --out "$SCD_CURVES_OUT"
bash scripts/run_sd35_curves.sh score --gpus "$SCD_CURVES_GPUS"
bash scripts/run_sd35_curves.sh summarize
# SD1.5 smoke: one full batch each for Base and the selected DPO checkpoint.
# Keep it separate from the formal benchmark and never export as paper results.
smoke="${sd15}-smoke"
"$SCD_SD15_PY" - "$sd15" "$smoke" "$b15" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,smoke,batch=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
source=root/'prompts/partiprompts.jsonl'
rows=source.read_text().splitlines(keepends=True)[:batch]
if len(rows)!=batch: raise SystemExit('Not enough prompts for a full smoke batch')
payload=''.join(rows)
files={'partiprompts.jsonl':payload,'selection.json':json.dumps({
    'mode':'smoke-only','benchmark':'partiprompts','prompts':len(rows),
    'suite_sha256':hashlib.sha256(payload.encode()).hexdigest()})}
folder=smoke/'prompts';folder.mkdir(parents=True,exist_ok=True)
for name,text in files.items():
    p=folder/name
    if p.exists() and p.read_text()!=text: raise SystemExit(f'Smoke input changed: {p}')
    if not p.exists(): p.write_text(text)
PY
"$SCD_SD15_PY" evaluate_suite.py plan --catalog configs/evaluation-sd15-main.json \
  --config configs/evaluation-suite-sd15-main.json --methods base dpo \
  --benchmarks partiprompts --prompts-root "$smoke/prompts" \
  --generation-batch-size "$b15" --out "$smoke/output"
"$SCD_SD15_PY" evaluate_suite.py generate --out "$smoke/output" --gpus "$SCD_CURVES_GPUS"
"$SCD_SD15_PY" evaluate_sd15.py generate --out "$sd15" --gpus "$SCD_CURVES_GPUS"
"$SCD_SD15_PY" evaluate_suite.py index --out "$sd15"
"$SCD_SD15_PY" evaluate_sd15.py score --out "$sd15" --gpus "$SCD_CURVES_GPUS"
"$SCD_SD15_PY" evaluate_sd15.py summarize --out "$sd15"
"$SCD_SD15_PY" evaluate_sd15.py export --out "$sd15"
