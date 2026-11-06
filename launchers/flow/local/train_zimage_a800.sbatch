#!/usr/bin/env bash
# One independent dpo, mvpo or linear_dpo run; each allocation owns two A800s.
#SBATCH --job-name=zimage-flow
#SBATCH --partition=cluster-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:2
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/local-a800/train-%j.log
#SBATCH --error=logs/local-a800/train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the flow project root}"
source /irip/xietianyu_2024/tools/miniconda/etc/profile.d/conda.sh
conda activate "$PWD/.venv-flow-a800"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
FLOW_ALGORITHM="${1:?Use dpo, mvpo or linear_dpo}"
case "$FLOW_ALGORITHM" in dpo|mvpo|linear_dpo) ;; *) exit 2;; esac
printf 'Job=%s host=%s algorithm=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$FLOW_ALGORITHM" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h /irip_16t_0
exec python -u scripts/run_flow.py --family zimage --algorithm "$FLOW_ALGORITHM" \
  --recipe paper-pilot --profile a800 --world-size 2 --batch-size 2 \
  --steps 3200 --run-tag local-a800-20261106-v1 --execution slurm \
  --resources configs/local-a800-resources.json \
  --model-revision 04cc4abb7c5069926f75c9bfde9ef43d49423021
