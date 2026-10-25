#!/usr/bin/env bash
# SDXL 1024 on allocated ACP GPUs; defaults to a two-GPU pure DPO baseline.
set -eo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash launchers/acp_sdxl.sh [METHOD] [train.py arguments]
METHOD: dpo (default), multiview-B, multiview-C,
        multiview-B-gaussian, multiview-C-gaussian
Environment: SCD_WORLD_SIZE=2|4|8, SCD_VIEWS=2 (DPO always K=1),
  SCD_TRAIN_BATCH_SIZE=1, SCD_RUN_NAME=<unique name>.
Gaussian launch presets accept K=2/4/6/8; rollout accepts any positive K.
Resolution 1024, global batch 128 pairs, 10000 updates, full save every 200.
Prepare data separately: bash launchers/acp_prepare_data.sh pickapic sdxl
First bounded run: bash launchers/acp_sdxl.sh dpo --stop-after-step 2
Restart the same command without --stop-after-step to resume.
HELP
  exit 0
fi
SCD_METHOD=dpo
if [[ $# -gt 0 && "$1" != --* ]]; then
  SCD_METHOD="$1"
  shift
fi
export SCD_FAMILY=sdxl
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_train "$SCD_METHOD" "$@"
