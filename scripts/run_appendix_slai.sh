#!/usr/bin/env bash
# One invocation per allocated container; the original appendix checkout stays pinned.
set -eo pipefail
job=${1:?Pass a task ID from command.md}
launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-mvpo-appendix-22edcb6
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
source /mnt/afs/L202500030/xietianyu/miniconda3/bin/activate
conda activate eval
export SCD_RESOURCES="$PWD/configs/resources.json"
source paths.sh
export PYTHONUNBUFFERED=1
scratch=${TMPDIR:-/tmp}
loss_run="$FLOW_RUNS_ROOT/sd35-global-lossavg-k4-pv128-seed17-appendix"
loss_gen="$MVPO_APPENDIX_ROOT/lossavg-k4/generation"

case "$job" in
  G2|G3|G4|G5|G6)
    case "$job" in G2) group=seed23;; G3) group=seed42;; G4) group=step800;; G5) group=step2400;; G6) group=step3200;; esac
    python scripts/appendix_jobs.py "$group" ;;
  D[1-4]) python scripts/appendix_jobs.py sd15-gradient --group "${job#D}/4" --scratch-dir "$scratch" ;;
  F[12]) python scripts/appendix_jobs.py sd35-gradient --group "${job#F}/2" --scratch-dir "$scratch" ;;
  S[12]) python scripts/appendix_jobs.py sd35-stage --group "${job#S}/2" ;;
  prepare-FS)
    for probe in sd35-gradient sd35-stage; do
      python scripts/appendix_jobs.py "$probe" --smoke --prepare-only
      python scripts/appendix_jobs.py "$probe" --prepare-only
    done ;;
  pilot-D|pilot-F|pilot-S)
    case "$job" in pilot-D) probe=sd15-gradient;; pilot-F) probe=sd35-gradient;; pilot-S) probe=sd35-stage;; esac
    if [[ "$job" == pilot-S ]]; then
      python scripts/appendix_jobs.py "$probe" --smoke
      python scripts/appendix_stage_report.py --input "$MVPO_APPENDIX_ROOT/smoke/$probe"
    else
      python scripts/appendix_jobs.py "$probe" --smoke --scratch-dir "$scratch"
      pilot="$MVPO_APPENDIX_ROOT/smoke/$probe"
      python scripts/appendix_gradient_report.py --input "$pilot"
      python - "$pilot" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
protocol = json.loads((root / "protocol.json").read_text())
for name in ("init-check.json", "repeat-view-check.json"):
    check = json.loads((root / name).read_text())
    if check["protocol_id"] != protocol["protocol_id"] or check["passed"] is not True:
        raise SystemExit("Failed: " + name)
for path in sorted((root / "pairs").glob("*.json")):
    e = json.loads(path.read_text())["execution"]
    print(path.name, round(e["measurement_seconds"], 1), "s", round(e["peak_reserved_bytes"]/2**30, 2), "GiB reserved")
print("Pilot passed")
PY
    fi ;;
  train-lossavg)
    appendix="$PWD"
    cd "$SLAI_ACCOUNT_ROOT/workspace/DPO-flow-microbatch"
    export GIT_CONFIG_VALUE_0="$PWD" SCD_RESOURCES="$PWD/configs/resources-slai-flow.json"
    conda activate mvpo-flow
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    cfg=$(mktemp "${TMPDIR:-/tmp}/mvpo-lossavg-XXXXXX")
    trap 'rm -f "$cfg"' EXIT
    python "$launcher_dir/appendix_lossavg_config.py" "$appendix/configs/appendix-generation.json" "$FLOW_RUNS_ROOT" "$PWD" "$cfg"
    resume=()
    if [[ -f "$loss_run/checkpoints/step-0001600/checkpoint.json" && -f "$loss_run/checkpoints/step-0001600/adapter/adapter_model.safetensors" ]]; then
      echo "Step 1600 already present; use prepare-lossavg to verify and evaluate."; exit 0
    fi
    if [[ -f "$loss_run/checkpoints/latest.json" ]]; then resume=(--resume latest); fi
    python -u train.py --config "$cfg" --out "$loss_run" --execution standalone \
      --run-label sd35-global-lossavg-k4-appendix --stop-after-step 1600 "${resume[@]}"
    test -s "$loss_run/checkpoints/step-0001600/adapter/adapter_model.safetensors" ;;
  prepare-lossavg)
    python scripts/appendix_generation.py prepare --lossavg-run "$loss_run" --out "$loss_gen"
    python - "$loss_gen/plan.json" "$MVPO_APPENDIX_ROOT/generation/plan.json" <<'PY'
import json, sys
plan, old = [json.load(open(p)) for p in sys.argv[1:]]
if plan["lossavg"]["status"] != "available" or plan["noise_runtime"] != old["noise_runtime"] or plan["panels"] != old["panels"]:
    raise SystemExit("Check LossAvg eligibility, paired runtime and panel against original G1")
PY
    ;;
  eval-lossavg)
    python scripts/appendix_jobs.py direct --out "$MVPO_APPENDIX_ROOT/lossavg-k4"
    python scripts/appendix_generation.py summarize --out "$loss_gen" --section direct ;;
  report-seeds|report-curve)
    python scripts/appendix_generation.py summarize --out "$MVPO_APPENDIX_ROOT/generation" --section "${job#report-}" ;;
  report-G) python scripts/appendix_generation.py summarize --out "$MVPO_APPENDIX_ROOT/generation" ;;
  report-D) python scripts/appendix_gradient_report.py --input "$MVPO_APPENDIX_ROOT/sd15-gradient" ;;
  report-F) python scripts/appendix_gradient_report.py --input "$MVPO_APPENDIX_ROOT/sd35-gradient" ;;
  report-S) python scripts/appendix_stage_report.py --input "$MVPO_APPENDIX_ROOT/sd35-stage" ;;
  cpu-sd15|cpu-sdxl|cpu-zimage)
    metrics=(pickscore hpsv2 image_reward clip aesthetic)
    case "$job" in
      cpu-sd15) plan=sd15/main-h100-b64-v1; method=stagewise_dspo_k6; baseline=dspo; step=10000; dest=sd15-dspo-parti-10k-matched ;;
      cpu-sdxl) plan=sdxl/paper-2000-5090-v1/main/partiprompts; method=stagewise_k6; baseline=dpo; step=2000; dest=sdxl-parti ;;
      cpu-zimage) plan=zimage/formal-h800-v1/core; method=stagewise_k4; baseline=dpo; step=2300; dest=zimage-parti; metrics=(pickscore hpsv3 image_reward aesthetic) ;;
    esac
    python scripts/appendix_paired_scores.py --plan "$EVALUATIONS_ROOT/$plan/plan.json" --method "$method" --baseline "$baseline" --step "$step" --benchmark partiprompts --metrics "${metrics[@]}" --out "$MVPO_APPENDIX_ROOT/existing-rewards/$dest" ;;
  qualitative)
    out="$MVPO_APPENDIX_ROOT/qualitative-stagewise"
    python scripts/appendix_qualitative_config.py --out "$out" --gpus 4
    for phase in smoke full; do
      "$SCD_FLOW_PY" qualitative/run.py launch --config "$out/stagewise-config.json" --out "$out" --backbone sd35_medium --phase "$phase" --gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    done
    python qualitative/layout.py contacts --out "$out" --backbone sd35_medium ;;
  *) echo "Unknown appendix task: $job" >&2; exit 2 ;;
esac
