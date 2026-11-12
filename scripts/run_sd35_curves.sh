#!/usr/bin/env bash
# Use only GPUs already allocated by SLAI; no installs, downloads or job submission.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_sd35_curves.sh STAGE [OPTIONS]
STAGE: prepare | plan | generate | score | summarize | all

  --runs-root PATH       Read-only training runs root
  --catalog PATH         Optional reviewed experiment catalog
  --run-glob PATTERN     Select runs; repeat for multiple patterns
  --exclude-prompts PATH Additional prompt panel to exclude; repeat
  --prepared PATH        Frozen 128-prompt input directory
  --out PATH             Separate render/score/curve output directory
  --target-steps N       Expected final training step (default: 3200)
  --generation-batch-size N  Images per GPU generation forward (default: 4)
  --gpus IDS             Comma-separated IDs from CUDA_VISIBLE_DEVICES
  --check-only           Generate/score preflight only; no GPU forward
  --keep-going           Continue independent tasks (already the default)
  --dry-run             Print commands without executing stages or writing logs

SCD_FLOW_PY defaults to the dedicated, previously smoke-tested Flow environment.
SCD_CURVES_RUNS_ROOT, SCD_CURVES_PREP, SCD_CURVES_OUT, SCD_CURVES_GPUS,
SCD_CURVES_LOGS and SCD_RESOURCES may override paths/devices.
When no --exclude-prompts is given, prepare detects available configured files
from the five official panels. Missing exclusions are reported; full independence
from the benchmarks is not claimed. See docs/SD35_CHECKPOINT_CURVES.md.
EOF
}

stage="${1:-help}"
if [[ "$stage" == help || "$stage" == --help || "$stage" == -h ]]; then usage; exit 0; fi
shift
case "$stage" in prepare|plan|generate|score|summarize|all) ;; *) usage >&2; exit 2 ;; esac

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${SCD_FLOW_PY:-/mnt/afs/L202500030/xietianyu/envs/scd-flow-smoke-7818ade/bin/python}"
runs_root="${SCD_CURVES_RUNS_ROOT:-/mnt/aoss/xietianyu/self_correct/flow-paper-v2/runs}"
prepared="${SCD_CURVES_PREP:-/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-v1-prepared}"
out="${SCD_CURVES_OUT:-/mnt/aoss/xietianyu/self_correct/evaluations/sd35_medium/curves-v1}"
gpus="${SCD_CURVES_GPUS-${CUDA_VISIBLE_DEVICES-0,1}}"
catalog=''
target_steps=3200
generation_batch_size=4
check_only=0
dry_run=0
patterns=()
exclusions=()
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --runs-root|--prepared|--out|--gpus|--catalog|--run-glob|--exclude-prompts|--target-steps|--generation-batch-size)
      if (( $# < 2 )) || [[ "$2" == --* ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --runs-root) runs_root="$2" ;;
        --prepared) prepared="$2" ;;
        --out) out="$2" ;;
        --gpus) gpus="$2" ;;
        --catalog) catalog="$2" ;;
        --target-steps) target_steps="$2" ;;
        --generation-batch-size) generation_batch_size="$2" ;;
        --run-glob) patterns+=("$2") ;;
        --exclude-prompts) exclusions+=("$2") ;;
      esac
      shift 2 ;;
    --check-only) check_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --keep-going) shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if (( check_only )) && [[ "$stage" != generate && "$stage" != score ]]; then
  echo '--check-only is supported for generate or score; it cannot complete all stages' >&2
  exit 2
fi
if [[ "$stage" == generate || "$stage" == score || "$stage" == all ]]; then
  if [[ -z "$gpus" || "$gpus" == -1 ]] ||
     { [[ "${CUDA_VISIBLE_DEVICES+x}" == x ]] && [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == -1 ]]; }; then
    echo 'No visible GPU allocation: CUDA_VISIBLE_DEVICES/--gpus is empty or -1; use an allocated SLAI container.' >&2
    exit 2
  fi
fi
[[ -x "$python" ]] || { echo "Missing executable SCD_FLOW_PY: $python" >&2; exit 2; }
[[ -f "$repo/evaluate_curves.py" ]] || { echo "evaluate_curves.py missing; update the independent evaluation checkout" >&2; exit 2; }
cd "$repo"
export PYTHONUNBUFFERED=1

run_stage() {
  local name="$1"
  local command=("$python" -u "$repo/evaluate_curves.py" "$name")
  case "$name" in
    prepare)
      command+=(--out "$prepared" --target-steps "$target_steps" --generation-batch-size "$generation_batch_size")
      if [[ -n "$catalog" ]]; then command+=(--catalog "$catalog"); else command+=(--runs-root "$runs_root"); fi
      (( ${#patterns[@]} == 0 )) || command+=(--run-glob "${patterns[@]}")
      (( ${#exclusions[@]} == 0 )) || command+=(--exclude-prompts "${exclusions[@]}") ;;
    plan) command+=(--prepared "$prepared" --out "$out") ;;
    generate|score)
      command+=(--out "$out" --gpus "$gpus" --keep-going)
      (( check_only == 0 )) || command+=(--check-only) ;;
    summarize) command+=(--out "$out") ;;
  esac
  printf '[%s] ' "$name"
  printf '%q ' "${command[@]}"
  printf '\n'
  if (( dry_run )); then return 0; fi
  # Keep logs outside the frozen output, including failures before plan exists.
  local logs="${SCD_CURVES_LOGS:-${out}/logs}"
  mkdir -p -- "$logs"
  local log
  log="$(mktemp "$logs/$(date -u +%Y%m%dT%H%M%S)-$$-${name}-XXXXXX.log")"
  "${command[@]}" 2>&1 | tee "$log"
}

if [[ "$stage" != all ]]; then
  run_stage "$stage"
  exit "$?"
fi

status=0
ready=1
record_failure() {
  local code="$1"
  if [[ "$code" == 130 || "$code" == 143 ]]; then
    echo "Interrupted (exit $code); no further stages started. Run summarize separately when ready." >&2
    exit "$code"
  fi
  status=1
}
for name in prepare plan; do
  if run_stage "$name"; then :; else record_failure "$?"; ready=0; break; fi
done
if (( ready )); then
  if run_stage generate; then :; else record_failure "$?"; fi
  if run_stage score; then :; else record_failure "$?"; fi
fi
# Always retain a summary of complete points and missing/failed tasks.
if run_stage summarize; then :; else record_failure "$?"; fi
exit "$status"
