#!/usr/bin/env bash
# Evaluate two retained checkpoints concurrently, one allocated GPU per run.
set -eo pipefail
export SCD_FAMILY=sd15
export SCD_EVAL_A="${SCD_EVAL_A:-sd15-dpo-fixed-pairs-p384-2gpu-10000-slai-v1}"
export SCD_EVAL_B="${SCD_EVAL_B:-sd15-gaussian-C-k2-fixed-views-v384-2gpu-10000-slai-v1}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../acp_evaluate_pair.sh" "$@"

