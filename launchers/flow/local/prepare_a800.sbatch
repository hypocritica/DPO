#!/usr/bin/env bash
# Shared data preparation only; install the environment in tmux before release.
#SBATCH --job-name=zimage-prepare
#SBATCH --partition=cluster-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/local-a800/prepare-%j.log
#SBATCH --error=logs/local-a800/prepare-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the flow project root}"
exec bash launchers/flow/local/prepare_a800_data.sh
