#!/usr/bin/env bash
# Optional scorer source code; never installs dependencies or starts a GPU job.
set -euo pipefail
SCD_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCD_SCRIPT_ROOT/paths.sh"
if [[ "$SCD_STORAGE_ROOT" == *CHANGE_ME* ]]; then
  echo 'Edit paths.sh first.' >&2
  exit 2
fi
if [[ "$#" -eq 0 ]]; then
  echo 'Usage: bash scripts/download_sources.sh hpsv2 aesthetic [hpsv3 image-reward]' >&2
  exit 2
fi
for SCD_ASSET in "$@"; do
  case "$SCD_ASSET" in
    hpsv2) SCD_REPO=https://github.com/tgxs002/HPSv2.git; SCD_SOURCE_DIR="$HPSV2_SOURCE"; SCD_REV=master ;;
    aesthetic) SCD_REPO=https://github.com/christophschuhmann/improved-aesthetic-predictor.git; SCD_SOURCE_DIR="$AESTHETIC_SOURCE"; SCD_REV=main ;;
    hpsv3) SCD_REPO=https://github.com/MizzenAI/HPSv3.git; SCD_SOURCE_DIR="$HPSV3_SOURCE"; SCD_REV=bd0c5fcb5f587617b0169c07222ab78d01e2f3c2 ;;
    image-reward) SCD_REPO=https://github.com/zai-org/ImageReward.git; SCD_SOURCE_DIR="$IMAGE_REWARD_SOURCE"; SCD_REV=78f8051059b34fef8e0032ccb40561c23a2ddbbb ;;
    *) echo "Unknown source: $SCD_ASSET" >&2; exit 2 ;;
  esac
  if [[ -e "$SCD_SOURCE_DIR" ]]; then
    echo "Existing source retained: $SCD_SOURCE_DIR" >&2
    exit 2
  fi
  mkdir -p -- "$(dirname -- "$SCD_SOURCE_DIR")"
  git clone --no-checkout "$SCD_REPO" "$SCD_SOURCE_DIR"
  git -C "$SCD_SOURCE_DIR" checkout --detach "$SCD_REV"
  git -C "$SCD_SOURCE_DIR" rev-parse HEAD > "$SCD_SOURCE_DIR/.scd-source-revision"
done
