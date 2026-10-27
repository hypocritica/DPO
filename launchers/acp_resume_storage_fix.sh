#!/usr/bin/env bash
# Resume an existing run with its exact recorded config from an isolated checkout.
set -eo pipefail
if [[ $# -lt 1 || "${1:-}" == --help ]]; then
  echo 'Usage: bash launchers/acp_resume_storage_fix.sh RUN_NAME [--stop-after-step N]'
  exit 0
fi
SCD_RUN_NAME="$1"
shift
if [[ ! "$SCD_RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'Pass one experiment name under RUNS_ROOT, without path separators.'
  exit 1
fi
SCD_RUNTIME_ARGS=("$@")
if [[ $# -gt 0 && ( $# -ne 2 || "$1" != --stop-after-step || ! "$2" =~ ^[1-9][0-9]*$ ) ]]; then
  echo 'Only --stop-after-step N is accepted; the saved experiment config is immutable.'
  exit 1
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/acp_common.sh"
scd_log "${SCD_RUN_NAME}-storage-resume"
SCD_OUT="$RUNS_ROOT/$SCD_RUN_NAME"
SCD_CONFIG="$SCD_REPO_ROOT/configs/local-${SCD_RUN_NAME}-storage-resume.json"
python - "$SCD_OUT" "$SCD_CONFIG" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from self_correct_dpo.common import read_json, sha256, source_file
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.runtime.storage_compat import compatible_sources
out, target = map(Path, sys.argv[1:])
recorded = read_json(out / 'training.json')
current = {name: sha256(source_file(name)) for name in recorded['code']}
if not compatible_sources(recorded['code'], current):
    raise SystemExit('Source differs beyond the approved storage fix; keep the original checkout and inspect the versions.')
atomic_json(target, recorded['config'])
print('Resume existing experiment:', out)
print('Saved world_size:', recorded['config']['training'].get('world_size', 1))
print('Saved evaluation enabled:', recorded['config'].get('evaluation', {}).get('enabled', False))
PY
exec python -u train.py --config "$SCD_CONFIG" --out "$SCD_OUT" \
  --resume_from_checkpoint latest --execution standalone --run-label "$SCD_RUN_NAME" \
  "${SCD_RUNTIME_ARGS[@]}"
