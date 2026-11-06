#!/usr/bin/env bash
# One independent zimage mvpo run on allocated SLAI GPUs.
set -euo pipefail
FLOW_ENTRY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$FLOW_ENTRY_DIR/slai_pilot.sh" zimage mvpo "$@"
