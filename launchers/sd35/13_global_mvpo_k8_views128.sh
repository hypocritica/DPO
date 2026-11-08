#!/usr/bin/env bash
set -euo pipefail
LAUNCHERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$LAUNCHERS/flow/suite.sh" sd35 global-c-k8 "$@"
