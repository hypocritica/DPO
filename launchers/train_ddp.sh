#!/usr/bin/env bash
# One coordinator starts local torchrun stages and serial checkpoint evaluation.
# Run only inside allocated CCI/ACP/Slurm resources; do not wrap this in torchrun.
set -euo pipefail
SCD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCD_CONFIG="${1:?Usage: bash launchers/train_ddp.sh CONFIG OUT [training arguments]}"
SCD_PYTHON="${SCD_PYTHON:-python}"
"$SCD_PYTHON" - "$SCD_CONFIG" <<'PY'
import json,sys
config=json.load(open(sys.argv[1]))
world=config['training'].get('world_size',1)
if type(world) is not int or world<2:
    raise SystemExit('DDP config requires training.world_size >= 2; make_config.py --world-size 2, 4 or 8')
PY
exec bash "$SCD_ROOT/launchers/train.sh" "$@"
