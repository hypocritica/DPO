#!/usr/bin/env bash
# One SD3.5M experiment on an already allocated four-RTX5090 server.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash launchers/flow/sd35_4x5090.sh {dpo|mvpo|linear_dpo|sft} [run_flow options]' >&2
  exit 2
fi
FLOW_ALGORITHM="$1"
shift
FLOW_ENTRY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$FLOW_ENTRY_DIR/slai_pilot.sh" sd35_medium "$FLOW_ALGORITHM" \
  --profile 5090 --world-size 4 "$@"
