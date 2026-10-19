#!/usr/bin/env bash
# Optional: bash scripts/download_sources.sh hpsv2 /your/source/directory
set -euo pipefail
asset="${1:?Usage: download_sources.sh ASSET OUT}"
out="${2:?Specify your output directory}"
case "$asset" in
  hpsv2) repo=https://github.com/tgxs002/HPSv2.git; rev=master ;;
  aesthetic) repo=https://github.com/christophschuhmann/improved-aesthetic-predictor.git; rev=main ;;
  hpsv3) repo=https://github.com/MizzenAI/HPSv3.git; rev=bd0c5fcb5f587617b0169c07222ab78d01e2f3c2 ;;
  image-reward) repo=https://github.com/zai-org/ImageReward.git; rev=78f8051059b34fef8e0032ccb40561c23a2ddbbb ;;
  *) echo "Unknown source: $asset" >&2; exit 2 ;;
esac
if [[ -e "$out" ]]; then
  echo "Output already exists: $out" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$out")"
git clone --no-checkout "$repo" "$out"
git -C "$out" checkout --detach "$rev"
git -C "$out" rev-parse HEAD > "$out/.scd-source-revision"
