#!/usr/bin/env bash
# Run only inside a platform allocation with two H100 80GB GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SCD_SD15_PY="${SCD_SD15_PY:-/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python}"
export SCD_FLOW_PY="${SCD_FLOW_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-flow-v1/bin/python}"
export SCD_EVAL_PY="${SCD_EVAL_PY:-$SCD_FLOW_PY}"
export SCD_RESOURCES="${SCD_RESOURCES:-$PWD/configs/resources.json}"
export SCD_EVAL_GPUS="${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export SCD_Z_GPUS="$SCD_EVAL_GPUS"
export SCD_SELECTED_GPU_COUNT=2
export SCD_HPSV3_PY="${SCD_HPSV3_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-hpsv3-v1/bin/python}"
export SCD_GENEVAL_PY="${SCD_GENEVAL_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-eval-geneval-v1/bin/python}"
for executable in "$SCD_EVAL_PY" "$SCD_FLOW_PY" "$SCD_SD15_PY"; do
  [[ -x "$executable" ]] || { echo "Missing existing evaluation Python: $executable; set its SCD_*_PY path before starting." >&2; exit 2; }
done
IFS=',' read -r -a devices <<< "$SCD_EVAL_GPUS"
[[ ${#devices[@]} == 2 && "${devices[0]}" != "${devices[1]}" && "$SCD_EVAL_GPUS" != *, && "$SCD_EVAL_GPUS" != *[[:space:]]* && ",$SCD_EVAL_GPUS," != *,-1,* ]] || {
  echo 'Set SCD_EVAL_GPUS to two distinct allocated GPU tokens' >&2; exit 2;
}
export PYTHONUNBUFFERED=1
pipeline_started=$SECONDS
df -h /mnt/aoss/xietianyu
git rev-parse HEAD

# Keep the original preparation, plan, images, seeds and checkpoint set.
export SCD_Z_BATCH=4
export SCD_Z_CURVES="${SCD_Z_CURVES:-/mnt/aoss/xietianyu/self_correct/evaluations/zimage/curves-h800-b4-v1}"
export SCD_Z_PREP="${SCD_Z_PREP:-${SCD_Z_CURVES}-prepared}"
bash scripts/run_zimage_curves.sh prepare --from-prepared "${SCD_Z_SOURCE_PREP:-/mnt/aoss/xietianyu/self_correct/evaluations/zimage/curves-h800-b1-v1-prepared}"
bash scripts/run_zimage_curves.sh plan
bash scripts/run_zimage_curves.sh generate
bash scripts/run_zimage_curves.sh score
bash scripts/run_zimage_curves.sh summarize
"$SCD_EVAL_PY" scripts/report_generation_sessions.py --out "$SCD_Z_CURVES"
printf 'Z-Image resume/generate/PickScore/plots completed in %s seconds\n' "$((SECONDS-pipeline_started))"
for executable in "$SCD_HPSV3_PY" "$SCD_GENEVAL_PY"; do
  [[ -x "$executable" ]] || { echo "Z-Image curve completed. Missing formal scoring Python: $executable; set its SCD_*_PY path and rerun." >&2; exit 2; }
done

# New SD3.5 formal protocol: Base0 + seven methods1600, batch8, no offload.
export SCD_SELECTED_FAMILY=sd35_medium
export SCD_SELECTED_ROOT="${SCD_SD35_COMPACT_ROOT:-/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/compact-h100-v2}"
export SCD_SELECTED_SELECTION="$SCD_SELECTED_ROOT/selection.json"
export SCD_SELECTED_OUT="$SCD_SELECTED_ROOT/core"
if [[ ! -f "$SCD_SELECTED_SELECTION" ]]; then
  bash scripts/run_compact_evaluation.sh init \
    --generation-batch-size 8 \
    --from-prepared "${SCD_SD35_SOURCE_PREP:-/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-h100-b8-v1-prepared}" \
    --handoff configs/evaluation-sd35-main-selection.json
fi
bash scripts/run_compact_evaluation.sh check
bash scripts/run_compact_evaluation.sh plan
bash scripts/run_compact_evaluation.sh generate
"$SCD_EVAL_PY" scripts/report_generation_sessions.py --out "$SCD_SELECTED_OUT"
bash scripts/run_compact_evaluation.sh score
bash scripts/run_compact_evaluation.sh summarize --summary-name core-v1
# Use the immutable plan ID for a repeatable report name on resume.
plan_id="$("$SCD_EVAL_PY" -c 'import json,os; print(json.load(open(os.environ["SCD_SELECTED_OUT"]+"/plan.json"))["plan_id"][:12])')"
if [[ ! -f "$SCD_SELECTED_OUT/paper/core-$plan_id/export.json" ]]; then
  bash scripts/run_compact_evaluation.sh paper --report main --paper-name "core-$plan_id"
fi
printf 'Pipeline completed in %s seconds; SD3.5 results: %s\n' "$((SECONDS-pipeline_started))" "$SCD_SELECTED_OUT"
