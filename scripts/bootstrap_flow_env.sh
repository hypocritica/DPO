#!/usr/bin/env bash
# Clone a user-selected GPU dpo environment; never install into source or base.
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
FLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
FLOW_CONDA_ROOT=""
FLOW_SOURCE_PREFIX=""
FLOW_TARGET_PREFIX=""
FLOW_MIN_FREE_GIB=30
FLOW_DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_flow_env.sh --conda-root ABSOLUTE_PATH \
  --source-prefix ABSOLUTE_GPU_ENV [--target-prefix ABSOLUTE_PATH] \
  [--min-free-gib 30] [--dry-run]

Conda and source environment must be explicitly selected. Default target:
  <conda-root>/envs/mvpo-flow
--dry-run prints commands without validating remote paths or writing anything.
Existing targets are reusable only with this bootstrap's matching provenance.
No model downloads, CUDA execution, environment deletion or Slurm submission.
EOF
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
show() { printf '  '; printf '%q ' "$@"; printf '\n'; }
while (($#)); do
  case "$1" in
    --conda-root|--source-prefix|--target-prefix|--min-free-gib)
      (($# >= 2)) || fail "Missing value for $1"
      case "$1" in
        --conda-root) FLOW_CONDA_ROOT="$2";;
        --source-prefix) FLOW_SOURCE_PREFIX="$2";;
        --target-prefix) FLOW_TARGET_PREFIX="$2";;
        --min-free-gib) FLOW_MIN_FREE_GIB="$2";;
      esac
      shift 2;;
    --dry-run) FLOW_DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "Unknown argument: $1";;
  esac
done
[[ "$FLOW_CONDA_ROOT" == /* ]] || fail '--conda-root must be an explicit absolute path'
[[ "$FLOW_SOURCE_PREFIX" == /* ]] || fail '--source-prefix must be an explicit absolute path'
[[ "$FLOW_MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]] || fail '--min-free-gib must be a positive integer'
FLOW_CONDA_ROOT="$(realpath -m -- "$FLOW_CONDA_ROOT")"
FLOW_SOURCE_PREFIX="$(realpath -m -- "$FLOW_SOURCE_PREFIX")"
FLOW_TARGET_PREFIX="${FLOW_TARGET_PREFIX:-$FLOW_CONDA_ROOT/envs/mvpo-flow}"
[[ "$FLOW_TARGET_PREFIX" == /* ]] || fail '--target-prefix must be an absolute path'
FLOW_TARGET_PREFIX="$(realpath -m -- "$FLOW_TARGET_PREFIX")"
[[ "$FLOW_SOURCE_PREFIX" != "$FLOW_CONDA_ROOT" ]] || fail 'Do not use Conda base as the source environment'
[[ "$FLOW_TARGET_PREFIX" != "$FLOW_CONDA_ROOT" ]] || fail 'Target cannot be Conda base'
[[ "$FLOW_TARGET_PREFIX" != "$FLOW_SOURCE_PREFIX" && "$FLOW_TARGET_PREFIX" != "$FLOW_SOURCE_PREFIX/"* && "$FLOW_SOURCE_PREFIX" != "$FLOW_TARGET_PREFIX/"* ]] || fail 'Source and target must be separate, non-nested prefixes'
[[ "$FLOW_ROOT" != "$FLOW_SOURCE_PREFIX" && "$FLOW_ROOT" != "$FLOW_SOURCE_PREFIX/"* ]] || fail 'Project/log directory must be outside the source environment'
FLOW_CONDA="$FLOW_CONDA_ROOT/bin/conda"
FLOW_SOURCE_PYTHON="$FLOW_SOURCE_PREFIX/bin/python"
FLOW_TARGET_PYTHON="$FLOW_TARGET_PREFIX/bin/python"
FLOW_CONSTRAINTS="$FLOW_TARGET_PREFIX/flow-gpu-runtime.constraints.txt"
FLOW_MARKER="$FLOW_TARGET_PREFIX/.mvpo-flow-bootstrap.json"
printf 'Project: %s\nConda: %s\nSource (read-only): %s\nTarget: %s\nMinimum target free space: %s GiB\n' "$FLOW_ROOT" "$FLOW_CONDA_ROOT" "$FLOW_SOURCE_PREFIX" "$FLOW_TARGET_PREFIX" "$FLOW_MIN_FREE_GIB"
if ((FLOW_DRY_RUN)); then
  printf 'DRY RUN: no writes; source existence, GPU runtime and free space are not validated.\n'
  show df -h "$(dirname -- "$FLOW_TARGET_PREFIX")"
  show "$FLOW_CONDA" create --yes --prefix "$FLOW_TARGET_PREFIX" --clone "$FLOW_SOURCE_PREFIX"
  printf '  Capture source GPU versions as constraints; verify exact preservation after installation.\n'
  show "$FLOW_TARGET_PYTHON" -m pip install --upgrade-strategy only-if-needed -r "$FLOW_ROOT/requirements-flow.txt" -e "$FLOW_ROOT[arrow,evaluation]" -c "$FLOW_CONSTRAINTS"
  show "$FLOW_TARGET_PYTHON" -m pip check
  for FLOW_FAMILY in sd35_medium zimage; do show "$FLOW_TARGET_PYTHON" "$FLOW_ROOT/scripts/check_flow_environment.py" --family "$FLOW_FAMILY"; done
  show "$FLOW_TARGET_PYTHON" -m pip freeze --all
  exit 0
fi
[[ -x "$FLOW_CONDA" && -f "$FLOW_CONDA_ROOT/etc/profile.d/conda.sh" ]] || fail 'Conda installation not found at the selected path'
[[ -x "$FLOW_SOURCE_PYTHON" && -d "$FLOW_SOURCE_PREFIX/conda-meta" ]] || fail 'Selected source is not an existing Conda environment'
[[ -f "$FLOW_ROOT/requirements-flow.txt" ]] || fail 'Missing flow lock file'
FLOW_SPACE_PATH="$(dirname -- "$FLOW_TARGET_PREFIX")"
while [[ ! -d "$FLOW_SPACE_PATH" ]]; do FLOW_SPACE_PATH="$(dirname -- "$FLOW_SPACE_PATH")"; done
df -h "$FLOW_SPACE_PATH"
if [[ -d /irip ]]; then df -h /irip; fi
FLOW_FREE_KIB="$(df -Pk "$FLOW_SPACE_PATH" | awk 'END {print $4}')"
((FLOW_FREE_KIB >= FLOW_MIN_FREE_GIB * 1024 * 1024)) || fail 'Insufficient free space; no clone or package installation performed'
[[ -w "$FLOW_SPACE_PATH" ]] || fail 'Target parent is not writable'
mkdir -p -- "$(dirname -- "$FLOW_TARGET_PREFIX")"
# Keep new Conda package-cache writes outside base/source as well.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$FLOW_TARGET_PREFIX.conda-pkgs}"
exec 9>"$FLOW_TARGET_PREFIX.bootstrap.lock"
flock -n 9 || fail 'Another bootstrap owns this target'
FLOW_LOG_DIR="$FLOW_ROOT/logs/flow-environment/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p -- "$FLOW_LOG_DIR"
exec > >(tee -a "$FLOW_LOG_DIR/bootstrap.log") 2>&1
trap 'printf "Bootstrap stopped at line %s. Source/base unchanged. Logs: %s\n" "$LINENO" "$FLOW_LOG_DIR" >&2' ERR
# Read installed metadata and compiled CUDA version; do not allocate a GPU.
"$FLOW_SOURCE_PYTHON" - "$FLOW_LOG_DIR/source-packages.json" "$FLOW_LOG_DIR/gpu.constraints.txt" <<'PY'
import importlib.metadata as md
import json, pathlib, sys
import torch
packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions() if d.metadata['Name']}
if 'torch' not in packages or 'torchvision' not in packages or torch.version.cuda is None:
    raise SystemExit('Source must contain the user-verified CUDA torch/torchvision runtime; CPU-only sources are not supported')
names=lambda name:name in ('torch','torchvision','torchaudio','triton','pytorch-triton') or name.startswith(('nvidia-','cuda-','pytorch-triton-'))
gpu={name:version for name,version in sorted(packages.items()) if names(name)}
pathlib.Path(sys.argv[1]).write_text(json.dumps({'packages':dict(sorted(packages.items())),'gpu':gpu,'compiled_cuda':torch.version.cuda},sort_keys=True,indent=2)+'\n')
pathlib.Path(sys.argv[2]).write_text(''.join(f'{name}=={version}\n' for name,version in gpu.items()))
print('Source GPU runtime:',json.dumps(gpu,sort_keys=True),'compiled CUDA:',torch.version.cuda)
PY
"$FLOW_CONDA" list --prefix "$FLOW_SOURCE_PREFIX" --json > "$FLOW_LOG_DIR/source-conda.json"
# Existing targets must have the same source, source package fingerprint and lock.
if [[ -e "$FLOW_TARGET_PREFIX" ]]; then
  [[ -O "$FLOW_TARGET_PREFIX" && -x "$FLOW_TARGET_PYTHON" && -f "$FLOW_MARKER" ]] || fail 'Existing target is not a recognized owned bootstrap environment; choose a fresh prefix'
  "$FLOW_SOURCE_PYTHON" - "$FLOW_MARKER" "$FLOW_SOURCE_PREFIX" "$FLOW_ROOT" "$FLOW_LOG_DIR/source-packages.json" <<'PY'
import hashlib,json,pathlib,sys
marker=json.loads(pathlib.Path(sys.argv[1]).read_text())
h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
expected={'source_prefix':sys.argv[2],'project':sys.argv[3],'source_sha256':h(sys.argv[4]),'lock_sha256':h(pathlib.Path(sys.argv[3])/'requirements-flow.txt'),'project_metadata_sha256':h(pathlib.Path(sys.argv[3])/'pyproject.toml')}
if any(marker.get(k)!=v for k,v in expected.items()):raise SystemExit('Target provenance/lock differs; use a new target prefix instead of modifying an existing run environment')
PY
else
  "$FLOW_CONDA" create --yes --prefix "$FLOW_TARGET_PREFIX" --clone "$FLOW_SOURCE_PREFIX"
  "$FLOW_SOURCE_PYTHON" - "$FLOW_MARKER" "$FLOW_SOURCE_PREFIX" "$FLOW_ROOT" "$FLOW_LOG_DIR/source-packages.json" <<'PY'
import hashlib,json,pathlib,sys
h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
path=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[3])
path.write_text(json.dumps({'source_prefix':sys.argv[2],'project':str(root),'source_sha256':h(sys.argv[4]),'lock_sha256':h(root/'requirements-flow.txt'),'project_metadata_sha256':h(root/'pyproject.toml'),'status':'cloned'},indent=2)+'\n')
PY
fi
cp -- "$FLOW_LOG_DIR/gpu.constraints.txt" "$FLOW_CONSTRAINTS"
export PIP_CACHE_DIR="$FLOW_TARGET_PREFIX/.bootstrap-cache/pip"
# Constraints also apply to dependency resolution; the source interpreter is never used by pip.
# Resolve the project dependencies too (notably tensorboard), plus data/eval
# extras. The exact cloned GPU-runtime constraints forbid torch/CUDA upgrades.
"$FLOW_TARGET_PYTHON" -m pip install --upgrade-strategy only-if-needed -r "$FLOW_ROOT/requirements-flow.txt" -e "$FLOW_ROOT[arrow,evaluation]" -c "$FLOW_CONSTRAINTS"
"$FLOW_TARGET_PYTHON" -m pip check
HF_HUB_OFFLINE=1 "$FLOW_TARGET_PYTHON" "$FLOW_ROOT/scripts/check_flow_environment.py" --family sd35_medium
HF_HUB_OFFLINE=1 "$FLOW_TARGET_PYTHON" "$FLOW_ROOT/scripts/check_flow_environment.py" --family zimage
"$FLOW_TARGET_PYTHON" - "$FLOW_LOG_DIR/source-packages.json" <<'PY'
import importlib.metadata as md
import json,pathlib,sys,torch
source=json.loads(pathlib.Path(sys.argv[1]).read_text())
packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions() if d.metadata['Name']}
names=lambda name:name in ('torch','torchvision','torchaudio','triton','pytorch-triton') or name.startswith(('nvidia-','cuda-','pytorch-triton-'))
gpu={name:version for name,version in packages.items() if names(name)}
if gpu!=source['gpu'] or torch.version.cuda!=source['compiled_cuda']:raise SystemExit('GPU runtime changed; bootstrap is incomplete. Inspect logs, do not use this environment for training')
print('Torch/torchvision/CUDA/NCCL runtime versions preserved exactly.')
PY
"$FLOW_TARGET_PYTHON" -m pip freeze --all > "$FLOW_LOG_DIR/pip-freeze.txt"
"$FLOW_CONDA" list --prefix "$FLOW_TARGET_PREFIX" --json > "$FLOW_LOG_DIR/target-conda.json"
"$FLOW_TARGET_PYTHON" - "$FLOW_MARKER" "$FLOW_LOG_DIR" <<'PY'
import json,pathlib,sys
path=pathlib.Path(sys.argv[1]);marker=json.loads(path.read_text());marker.update(status='complete',last_log_dir=sys.argv[2])
temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(marker,indent=2)+'\n');temporary.replace(path)
PY
printf 'Ready: %s\nFreeze/logs: %s\nGPU training and scorers have not been validated by this bootstrap.\n' "$FLOW_TARGET_PREFIX" "$FLOW_LOG_DIR"
