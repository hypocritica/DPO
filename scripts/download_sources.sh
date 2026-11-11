#!/usr/bin/env bash
# Optional: bash scripts/download_sources.sh hpsv2 /your/source/directory
set -euo pipefail
mode=download
if [[ "${1:-}" == --check || "${1:-}" == --dry-run ]]; then
  mode="${1#--}"
  shift
fi
asset="${1:?Usage: download_sources.sh [--check|--dry-run] ASSET OUT}"
out="${2:?Specify your output directory}"
[[ $# == 2 ]] || { echo 'Expected ASSET OUT only' >&2; exit 2; }
case "$asset" in
  hpsv2) repo=https://github.com/tgxs002/HPSv2.git; rev=master ;;
  aesthetic) repo=https://github.com/christophschuhmann/improved-aesthetic-predictor.git; rev=main ;;
  hpsv3) repo=https://github.com/MizzenAI/HPSv3.git; rev=bd0c5fcb5f587617b0169c07222ab78d01e2f3c2 ;;
  image-reward) repo=https://github.com/zai-org/ImageReward.git; rev=78f8051059b34fef8e0032ccb40561c23a2ddbbb ;;
  unifiedreward) repo=https://github.com/CodeGoat24/UnifiedReward.git; rev=ff6f520247a2d36e2aff7b2ba53a935afd437f66 ;;
  geneval) repo=https://github.com/djghosh13/geneval.git; rev=af4902f24d3ca90ebbb446dd9891a59e0f82725f ;;
  dpg) repo=https://github.com/TencentQQGYLab/ELLA.git; rev=3c228f1dc6c4d3cad0a47493816151a419f14db3 ;;
  mmdetection) repo=https://github.com/open-mmlab/mmdetection.git; rev=e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1 ;;
  *) echo "Unknown source: $asset" >&2; exit 2 ;;
esac
if [[ "$mode" == check ]]; then
  [[ -d "$out/.git" || -f "$out/.git" ]] || { echo "Not a Git source checkout: $out" >&2; exit 2; }
  actual="$(git -C "$out" rev-parse HEAD)"
  if [[ "$rev" =~ ^[0-9a-f]{40}$ ]]; then
    [[ "$actual" == "$rev" ]] || { echo "Wrong source revision: $actual (expected $rev)" >&2; exit 2; }
  elif [[ -f "$out/.scd-source-revision" ]]; then
    [[ "$(cat "$out/.scd-source-revision")" == "$actual" ]] || { echo 'Source receipt differs from HEAD' >&2; exit 2; }
  else
    echo 'Unpinned legacy source requires a .scd-source-revision receipt' >&2; exit 2
  fi
  # Ignore the script's own untracked receipt; reject edits to tracked evaluator code.
  git -C "$out" diff --quiet HEAD -- || { echo 'Tracked source edits found' >&2; exit 2; }
  printf 'Verified %s %s %s\n' "$asset" "$actual" "$out"
  exit 0
fi
if [[ -e "$out" ]]; then
  echo "Output already exists: $out; reuse after --check, or choose a new destination" >&2
  exit 2
fi
if [[ "$mode" == dry-run ]]; then
  printf 'git clone --no-checkout %q %q\n' "$repo" "$out"
  printf 'git -C %q checkout --detach %q\n' "$out" "$rev"
  exit 0
fi
mkdir -p -- "$(dirname -- "$out")"
git clone --no-checkout "$repo" "$out"
git -C "$out" checkout --detach "$rev"
git -C "$out" rev-parse HEAD > "$out/.scd-source-revision"
