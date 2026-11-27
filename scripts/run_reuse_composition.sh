#!/usr/bin/env bash
# Score three frozen SLAI panels on already allocated devices.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mode=${1:-help}
case "$mode" in
  geneval|dpg-smoke|report) [[ $# == 1 ]] || { echo "Unexpected arguments" >&2; exit 2; } ;;
  dpg-group)
    [[ $# == 3 && $2 =~ ^[1-9][0-9]*$ && $3 =~ ^[1-9][0-9]*$ ]] || { echo 'dpg-group GROUP TOTAL (1-based)' >&2; exit 2; }
    (( $2 <= $3 )) || { echo 'GROUP exceeds TOTAL' >&2; exit 2; } ;;
  *) echo 'Usage: run_reuse_composition.sh geneval | dpg-smoke | dpg-group GROUP TOTAL | report'; exit 2 ;;
esac
legacy=/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python
coord=${SCD_EVAL_PY:-$legacy}
[[ -x "$coord" || -n ${SCD_EVAL_PY:-} ]] || coord=python3
root=${SCD_REVIEW_ROOT:-$("$coord" -c "from self_correct_dpo.resources import load_resources; print(load_resources()['EVALUATIONS_ROOT'])")}
export SCD_REVIEW_ROOT="$root"
xl="$root/sdxl/paper-2000-5090-v1/main/geneval"
flow="$root/sd35_medium/compact-h100-v3/core"
suite=(--benchmarks geneval --metrics geneval)
if [[ "$mode" == report ]]; then
    status=0
    for plan in "$xl" "$flow"; do
        "$coord" evaluate_suite.py summarize --out "$plan" "${suite[@]}" --summary-name geneval-review-all || status=1
    done
    "$coord" scripts/reuse_dpg.py report || status=1
    exit "$status"
fi

envroot=/mnt/afs/L202500030/xietianyu/envs
if [[ "$mode" == geneval ]]; then
    scorer=${SCD_GENEVAL_PY:-$envroot/scd-eval-geneval-v1/bin/python}
else
    scorer=${SCD_DPG_PY:-$envroot/scd-eval-dpg-v1/bin/python}
fi
[[ -x "$scorer" ]] || { echo "Missing scorer Python: $scorer. Set SCD_GENEVAL_PY or SCD_DPG_PY to the installed environment." >&2; exit 2; }
count=4
[[ "$mode" != dpg-smoke ]] || count=1
gpu_list=${SCD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
if [[ -z "$gpu_list" ]]; then
    gpu_list=$("$scorer" -c 'import torch; print(",".join(map(str, range(torch.cuda.device_count()))))')
fi
IFS=, read -r -a gpus <<< "$gpu_list"
[[ ${#gpus[@]} == "$count" && "$gpu_list" != *, ]] || { echo "Expected $count allocated GPU IDs, got '$gpu_list'" >&2; exit 2; }
seen=,
for gpu in "${gpus[@]}"; do
    [[ -n "$gpu" && "$gpu" != -1 && ! "$gpu" =~ [[:space:]] && "$seen" != *",$gpu,"* ]] || { echo 'Invalid or duplicate GPU token' >&2; exit 2; }
    if [[ -n ${CUDA_VISIBLE_DEVICES:-} && ",$CUDA_VISIBLE_DEVICES," != *",$gpu,"* ]]; then
        echo "GPU $gpu is outside CUDA_VISIBLE_DEVICES" >&2; exit 2
    fi
    seen+="$gpu,"
done

if [[ "$mode" == geneval ]]; then
    if [[ -n ${SCD_COMPOSITIONAL_CONFIG:-} ]]; then
        suite+=(--compositional-config "$SCD_COMPOSITIONAL_CONFIG")
    fi
    score() {
        local plan=$1
        shift
        "$coord" evaluate_suite.py score --out "$plan" "${suite[@]}" --methods "$@" \
            --execution standalone --python "$scorer" --gpus "$gpu_list" --keep-going
    }
    status=0
    score "$xl" base dpo stagewise_k6 sft || status=$?
    (( status < 128 )) || exit "$status"
    score "$flow" base linear_dpo global_k4 sft || status=$?
    (( status < 128 )) || exit "$status"
    score "$xl" mapo global_k2 dmpo stagewise_dmpo_k6 dspo stagewise_dspo_k6 || status=$?
    (( status < 128 )) || exit "$status"
    score "$flow" dpo global_k2 stagewise_k2 stagewise_k4 || status=$?
    exit "$status"
fi
if [[ "$mode" == dpg-smoke ]]; then
    export CUDA_VISIBLE_DEVICES="${gpus[0]}"
    exec "$scorer" scripts/reuse_dpg.py smoke
fi

pids=()
stop() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait || true
    exit 130
}
trap stop INT TERM
for slot in 0 1 2 3; do
    rank=$(( ($2 - 1)*4 + slot ))
    CUDA_VISIBLE_DEVICES="${gpus[slot]}" "$scorer" scripts/reuse_dpg.py work --rank "$rank" --world "$((4 * $3))" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
