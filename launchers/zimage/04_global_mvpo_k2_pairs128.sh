#!/usr/bin/env bash
set -euo pipefail
LAUNCHERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$LAUNCHERS/flow/paper_experiment.sh" zimage mvpo 2 pairs false "$@"
