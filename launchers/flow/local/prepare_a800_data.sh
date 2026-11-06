#!/usr/bin/env bash
# CPU data preparation; may run in manager tmux as requested by the user.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
FLOW_RAW=/irip_16t_0/xietianyu_2024/datasets/hpdv3
FLOW_DATA_PYTHON="${FLOW_DATA_PYTHON:-$PWD/.venv-flow-a800/bin/python}"
FLOW_DATA_WORKERS="${FLOW_DATA_WORKERS:-4}"
df -h /irip /irip_16t_0
"$FLOW_DATA_PYTHON" scripts/extract_hpdv3_sub.py --raw "$FLOW_RAW" --out data/hpdv3-sub-images-v1 \
  --workers "$FLOW_DATA_WORKERS"
if [[ ! -f data/hpdv3-sub-v1/ready.json ]]; then
  "$FLOW_DATA_PYTHON" scripts/prepare_hpdv3_sub.py --raw data/hpdv3-sub-images-v1 \
    --out data/hpdv3-sub-v1 --revision MizzenAI/HPDv3@5ac4b5b5f2ef42c4613dec612473170b131552db \
    --workers "$FLOW_DATA_WORKERS"
fi
if [[ ! -f data/hpdv3-pilot-64.jsonl ]]; then
  "$FLOW_DATA_PYTHON" scripts/prepare_flow_monitor.py --pairs data/hpdv3-sub-v1/pairs.jsonl \
    --out data/hpdv3-pilot-64.jsonl --count 64 --seed 17
fi
printf 'HPDv3-sub and 64-prompt panel ready.\n'
