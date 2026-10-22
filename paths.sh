#!/usr/bin/env bash
# Optional shell shortcuts from the same registry used by train.py.
# Usage: source paths.sh (after activating the project Python environment).
SCD_PATHS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval "$(PYTHONPATH="$SCD_PATHS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python - <<'PYTHON'
import shlex
from self_correct_dpo.resources import load_resources
for key, value in load_resources().items():
    if not key.isidentifier():
        raise ValueError(f'Invalid resource name: {key}')
    print(f'export {key}={shlex.quote(value)}')
PYTHON
)"
unset SCD_PATHS_ROOT
