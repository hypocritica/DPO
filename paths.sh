#!/usr/bin/env bash
# Source this file before preparing data, making configs, training or evaluating.
# Change the first path to YOUR persistent, writable SLAI AFS directory.
export SCD_STORAGE_ROOT="/mnt/afs/CHANGE_ME/self-correct-assets"
export SCD_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SCD_MODELS_ROOT="$SCD_STORAGE_ROOT/models"
export SCD_RAW_ROOT="$SCD_STORAGE_ROOT/data/raw"
export SCD_IMAGES_ROOT="$SCD_STORAGE_ROOT/data/images"
export SCD_PREPARED_ROOT="$SCD_STORAGE_ROOT/data/prepared"
export SCD_RUNS_ROOT="$SCD_STORAGE_ROOT/runs"
export SCD_SOURCES_ROOT="$SCD_STORAGE_ROOT/sources"
export HF_HOME="$SCD_STORAGE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$SCD_STORAGE_ROOT/cache/torch"
export PIP_CACHE_DIR="$SCD_STORAGE_ROOT/cache/pip"
export SCD_5090_ENV="$SCD_STORAGE_ROOT/envs/sd15-5090"

export SD15_MODEL="$SCD_MODELS_ROOT/sd15"
export SDXL_MODEL="$SCD_MODELS_ROOT/sdxl-base"
export SDXL_VAE="$SCD_MODELS_ROOT/sdxl-vae-fp16-fix"
export SD35_MODEL="$SCD_MODELS_ROOT/sd35-medium"
export PICKAPIC_RAW="$SCD_RAW_ROOT/pickapic-v2"
export HPDV2_RAW="$SCD_RAW_ROOT/hpdv2"
export HPDV3_RAW="$SCD_RAW_ROOT/hpdv3"
export PICKAPIC_IMAGES="$SCD_IMAGES_ROOT/pickapic-v2"
# Separate model-specific manifests; original image bytes are shared.
export PICKAPIC_SD15_PREPARED="$SCD_PREPARED_ROOT/pickapic-v2/sd15-v1"
export PICKAPIC_SDXL_PREPARED="$SCD_PREPARED_ROOT/pickapic-v2/sdxl-v1"
export HPDV3_SD15_PREPARED="$SCD_PREPARED_ROOT/hpdv3/sd15-v1"
export HPDV3_SDXL_PREPARED="$SCD_PREPARED_ROOT/hpdv3/sdxl-v1"
# Pair-list conversion without a model token filter, for a separate SD3 trainer.
export HPDV3_UNFILTERED_PREPARED="$SCD_PREPARED_ROOT/hpdv3/unfiltered-v1"
export SCD_EVALUATION_ROOT="$SCD_PREPARED_ROOT/evaluation-v1"
export HPDV2_PROMPTS="$SCD_EVALUATION_ROOT/hpdv2-test.jsonl"
export MONITOR_PROMPTS="$SCD_EVALUATION_ROOT/pickapic-monitor.jsonl"
export PICKAPIC_TEST_PROMPTS="$SCD_EVALUATION_ROOT/pickapic-test-remaining.jsonl"
export HPDV3_PROMPTS="$SCD_EVALUATION_ROOT/hpdv3-test.jsonl"

export PICKSCORE_MODEL="$SCD_MODELS_ROOT/pickscore"
export PICKSCORE_PROCESSOR="$SCD_MODELS_ROOT/clip-vit-h14"
export CLIP_H14_CHECKPOINT="$PICKSCORE_PROCESSOR/open_clip_pytorch_model.bin"
export HPSV2_MODEL_DIR="$SCD_MODELS_ROOT/hpsv2"
export HPSV2_CHECKPOINT="$HPSV2_MODEL_DIR/HPS_v2_compressed.pt"
export AESTHETIC_CLIP_MODEL="$SCD_MODELS_ROOT/clip-vit-l14"
export AESTHETIC_SOURCE="$SCD_SOURCES_ROOT/improved-aesthetic-predictor"
export AESTHETIC_CHECKPOINT="$AESTHETIC_SOURCE/sac+logos+ava1-l14-linearMSE.pth"
export HPSV2_SOURCE="$SCD_SOURCES_ROOT/HPSv2"
export HPS_OPEN_CLIP_SOURCE="$HPSV2_SOURCE/hpsv2/src"
export SCORERS_CONFIG="$SCD_PROJECT_ROOT/configs/scorers.json"

# Optional scorers: download/install only when you actually select them.
export HPSV3_SOURCE="$SCD_STORAGE_ROOT/sources/HPSv3"
export HPSV3_BASE_MODEL="$SCD_MODELS_ROOT/Qwen2-VL-7B-Instruct"
export HPSV3_MODEL_DIR="$SCD_MODELS_ROOT/hpsv3"
export HPSV3_CHECKPOINT="$HPSV3_MODEL_DIR/HPSv3.safetensors"
export HPSV3_CONFIG="$HPSV3_SOURCE/hpsv3/config/HPSv3_7B.yaml"
export HPSV3_PYTHON="$SCD_STORAGE_ROOT/envs/hpsv3/bin/python"
export IMAGE_REWARD_SOURCE="$SCD_STORAGE_ROOT/sources/ImageReward"
export IMAGE_REWARD_MODEL_DIR="$SCD_MODELS_ROOT/ImageReward"
export IMAGE_REWARD_CHECKPOINT="$IMAGE_REWARD_MODEL_DIR/ImageReward.pt"
export IMAGE_REWARD_CONFIG="$IMAGE_REWARD_MODEL_DIR/med_config.json"
export IMAGE_REWARD_TOKENIZER="$SCD_MODELS_ROOT/bert-base-uncased"
export IMAGE_REWARD_PYTHON="$SCD_STORAGE_ROOT/envs/image-reward/bin/python"

# Each revision is resolved to an immutable commit and recorded on first download.
# Set these to known commits before first download if you need a specific release.
export PICKAPIC_REVISION="main"
export HPDV2_REVISION="main"
export HPDV3_REVISION="main"
export SD15_REVISION="main"
export SDXL_REVISION="main"
export SD35_REVISION="main"
# Existing entry points also understand these aliases. Select sd15/sdxl via
# make_config.py --family; the helper below selects the corresponding model path.
export MODEL_PATH="$SD15_MODEL"
export PAIRS_PATH="$PICKAPIC_SD15_PREPARED/pairs.jsonl"
