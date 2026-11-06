#!/usr/bin/env bash
# Run environment installation in a login-node tmux session.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
FLOW_ENV="$PWD/.venv-flow-a800"
export CONDA_OFFLINE=true CONDA_SOLVER=classic CONDA_NO_PLUGINS=true
export CONDA_PKGS_DIRS="$FLOW_ENV.conda-pkgs,/irip/xietianyu_2024/tools/miniconda/pkgs"
export PIP_NO_INDEX=1 PIP_FIND_LINKS="$PWD/.cache/flow-wheels"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec bash scripts/bootstrap_flow_env.sh \
  --conda-root /irip/xietianyu_2024/tools/miniconda \
  --source-prefix /irip/xietianyu_2024/tools/miniconda/envs/code \
  --target-prefix "$FLOW_ENV"
