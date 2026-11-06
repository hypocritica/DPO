#!/usr/bin/env bash
set -euo pipefail
LAUNCHERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$LAUNCHERS/flow/paper_experiment.sh" sd35_medium linear_dpo 1 pair_views false "$@"
