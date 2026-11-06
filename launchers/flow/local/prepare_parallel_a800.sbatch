#!/usr/bin/env bash
# CPU-only handoff from an existing serial preparation process.
# Stop the old writer before creating logs/local-a800/handoff-JOBID.ready.
# Positional arguments are the existing held training jobs to release on success.
#SBATCH --job-name=zimage-data-parallel
#SBATCH --partition=cluster-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/local-a800/prepare-parallel-%j.log
#SBATCH --error=logs/local-a800/prepare-parallel-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the flow project root}"
# Conda compiler deactivation hooks may reference unset backup variables.
set +u
source /irip/xietianyu_2024/tools/miniconda/etc/profile.d/conda.sh
conda activate "$PWD/.venv-flow-a800"
set -u
export FLOW_DATA_PYTHON="$PWD/.venv-flow-a800/bin/python"
export FLOW_DATA_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
printf 'Job=%s host=%s workers=%s; waiting for exclusive data handoff.\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$FLOW_DATA_WORKERS"
for ((attempt=0; attempt<120; attempt++)); do
  [[ -f "logs/local-a800/handoff-${SLURM_JOB_ID}.ready" ]] && break
  sleep 5
done
[[ -f "logs/local-a800/handoff-${SLURM_JOB_ID}.ready" ]] || { echo 'Handoff timed out; old preparation was not replaced.' >&2; exit 1; }
bash launchers/flow/local/prepare_a800_data.sh
"$FLOW_DATA_PYTHON" -c 'import json; assert json.load(open(".venv-flow-a800/.mvpo-flow-bootstrap.json"))["status"] == "complete"'
for job in "$@"; do
  [[ "$job" =~ ^[0-9]+$ ]] || exit 2
  state="$(scontrol show job "$job" -o)"
  [[ "$state" == *"UserId=${USER}("* && "$state" == *"JobState=PENDING"* && "$state" == *"Reason=JobHeldUser"* ]] || {
    printf 'Refusing to release changed/unowned job %s\n' "$job" >&2; exit 1;
  }
  scontrol release "$job"
done
printf 'Data ready; released training jobs: %s\n' "$*"
