#!/usr/bin/env bash
# Offline scoring only; preserve the original PNGs and the old score directories.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-run}" in run|report|probe) ;; *) echo 'Usage: run_geneval_h100_repair.sh [run|report|probe]' >&2; exit 2;; esac
afs=/mnt/afs/L202500030/xietianyu
unset PYTHONHOME PYTHONPATH
source "$afs/miniconda3/etc/profile.d/conda.sh"
conda activate "$afs/envs/scd-eval-geneval-v1"
py="$CONDA_PREFIX/bin/python"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/src"
export SCD_RESOURCES="${SCD_RESOURCES:-$afs/workspace/DPO-mvpo-multiseed-v1/configs/resources.json}"
export SCD_COMPOSITIONAL_CONFIG="${SCD_COMPOSITIONAL_CONFIG:-$PWD/configs/compositional.json}"
export SCD_EVAL_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export GIT_CONFIG_KEY_1=safe.directory
GIT_CONFIG_VALUE_1=$("$py" -c 'from self_correct_dpo.resources import load_resources; print(load_resources()["GENEVAL_REPO"])')
export GIT_CONFIG_VALUE_1
exec "$py" scripts/reuse_geneval_h100.py "${1:-run}"
