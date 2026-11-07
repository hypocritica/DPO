#!/usr/bin/env bash
#SBATCH --job-name=zimage-paper
#SBATCH --partition=cluster-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:2
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/local-stagewise/train-%j.log
#SBATCH --error=logs/local-stagewise/train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the stagewise project root}"
case "${1:-}" in
  dpo) launcher=launchers/zimage/01_dpo_k1.sh; name=zimage-dpo-k1-irip-a800 ;;
  stagewise-k2) launcher=launchers/zimage/07_stagewise_mvpo_k2_views128.sh; name=zimage-stagewise-k2-C-irip-a800 ;;
  linear-dpo) launcher=launchers/zimage/02_linear_dpo_k1.sh; name=zimage-linear-dpo-k1-irip-a800 ;;
  global-k2) launcher=launchers/zimage/05_global_mvpo_k2_views128.sh; name=zimage-global-k2-C-irip-a800 ;;
  *) echo 'Use dpo, stagewise-k2, linear-dpo or global-k2' >&2; exit 2 ;;
esac
set +u
source /irip/xietianyu_2024/tools/miniconda/etc/profile.d/conda.sh
conda activate /irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow/.venv-flow-a800
set -u
export SCD_FLOW_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export WANDB_MODE=offline WANDB_PROJECT=mvpo-flow
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
printf 'Job=%s host=%s algorithm=%s revision=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$1" "$(git rev-parse HEAD)"
python -c 'import sys, importlib.metadata as m; print(sys.executable); print(sys.version); print({p:m.version(p) for p in ("torch","diffusers","wandb")})'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h /irip_16t_0
exec bash "$launcher" --resources configs/local-stagewise-resources.json \
  --model-report reports/zimage-model-validation.json --execution slurm \
  --run-tag irip-wandb-20261107-v1 --report-to all --wandb-mode offline \
  --wandb-project mvpo-flow --wandb-group zimage-hpdv3-pair_views128-seed17 \
  --wandb-name "$name"
