#!/usr/bin/env bash
# One coordinator uses Accelerate for DDP stages and serial checkpoint evaluation.
# Run inside allocated remote/Slurm resources. Pass --world-size 2, 4 or 8.
set -euo pipefail
SCD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$SCD_ROOT/launchers/train.sh" "$@"
