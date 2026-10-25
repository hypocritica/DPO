#!/usr/bin/env bash
# Evaluate two retained checkpoints concurrently, one allocated GPU per run.
set -eo pipefail
export SCD_FAMILY=sdxl
export SCD_EVAL_A="${SCD_EVAL_A:-sdxl-dpo-fixed-pairs-p192-2gpu-10000-slai-v1}"
export SCD_EVAL_B="${SCD_EVAL_B:-sdxl-gaussian-C-k2-fixed-views-v192-2gpu-10000-slai-v1}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../acp_evaluate_pair.sh" "$@"
