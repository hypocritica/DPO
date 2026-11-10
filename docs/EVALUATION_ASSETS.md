# Evaluation assets and reward conventions

This checkout prepares evaluation code and explicit asset downloads for SLAI.
No model is downloaded when importing a scorer or running evaluation. All model
loads are local/offline. The current implementation has CPU tests; real reward
inference and the complete GPU environment still require the documented smoke
run on allocated hardware. Executable setup/download/run commands belong in
the root [command.md](../command.md).

## Existing five metrics

The existing PickScore, HPS v2.0, CLIP ViT-H/14, improved LAION aesthetic and
ImageReward conventions are unchanged. PickScore/HPSv2/CLIP use paired cosine
without softmax or temperature; ImageReward uses the official standardized
reward. Do not compare HPSv3 with the HPSv2 column as if they were one metric.

## UnifiedReward: one fixed model and an explicit scalar

The CLI name is `unifiedreward`. It means
[`CodeGoat24/UnifiedReward-2.0-qwen-7b`](https://huggingface.co/CodeGoat24/UnifiedReward-2.0-qwen-7b/tree/f14142a59d75cbdf8c0de75aa942f135e4bd3a70),
revision `f14142a59d75cbdf8c0de75aa942f135e4bd3a70`, based on Qwen2.5-VL-7B.
This is the complete four-shard model, not a LoRA, a Qwen3 variant, a pairwise
ranker, or the Think/Edit variant. Downloaded metadata and all weight shards are
required by preflight.

The adapter uses the exact official [2.0 image pointwise ACS prompt](https://github.com/CodeGoat24/UnifiedReward/blob/ff6f520247a2d36e2aff7b2ba53a935afd437f66/inference_qwen/UnifiedReward-2.0-inference/point_score_ACS_image_generation.py).
The model outputs **Alignment, Coherence and Style, each on a 1–5 scale**.
There is no official single overall score in that example. Our `unifiedreward`
scalar is explicitly the arithmetic mean `(alignment + coherence + style) / 3`;
report it as **UnifiedReward 2.0 ACS mean**, and retain the three components.
This mean is a project reporting choice, not another learned reward head.

The local Transformers adapter applies the model's chat template, BF16, SDPA,
greedy decoding and at most 4096 new tokens. It preserves the official 2.0
[client](https://github.com/CodeGoat24/UnifiedReward/blob/ff6f520247a2d36e2aff7b2ba53a935afd437f66/vllm_qwen/vllm_request.py)
RGB/JPEG-quality-95 input convention. Caption tokens are not silently truncated.
The adapter uses local inference instead of the official example's vLLM server;
numerical identity with vLLM has not been established.

Only the exact three named, finite scores in `[1,5]` are accepted. Missing,
duplicate, out-of-range, nonfinite, refused or ambiguous outputs raise an error.
There is no zero fill, regex first-number fallback or silent sample deletion.
Per-image results retain `components`, `raw_output` and `aggregation` for audit.

Use a **separate Python 3.10 environment** with
[requirements-unifiedreward.txt](../requirements-unifiedreward.txt).
Transformers `4.51.3` matches the published model configuration and includes
`Qwen2_5_VLForConditionalGeneration`; `qwen-vl-utils==0.0.11` supplies image
preparation. Install the hardware-appropriate PyTorch/torchvision pair before
these requirements. Configure this environment's absolute Python path as the
scorer's `python` field in a private copy of `configs/scorers.json`; generation
and the legacy five metrics need not migrate to its Transformers version.

## HPSv3: full local bundle

The adapter follows official source revision
[`bd0c5fcb5f587617b0169c07222ab78d01e2f3c2`](https://github.com/MizzenAI/HPSv3/tree/bd0c5fcb5f587617b0169c07222ab78d01e2f3c2).
Its constructor is `HPSv3RewardInferencer(config_path=..., checkpoint_path=...,
device=..., differentiable=False)`; `reward(prompts=..., image_paths=...)`
accepts PIL images and returns `[batch, 2]`. The reported score is **column 0,
mu**, without sigmoid, softmax or conversion to percent. Column 1 is not a
second preference score to average.

Required assets are separate:

| Asset | Fixed identity / local destination |
|---|---|
| Official HPSv3 source and YAML | `HPSV3_MODEL_DIR/source`, Git revision above; `hpsv3/config/HPSv3_7B.yaml` |
| HPSv3 weights | `MizzenAI/HPSv3@4f81e3e09edd82fe3c5f636444c721b592a735ca`, `HPSV3_MODEL_DIR/HPSv3.safetensors` |
| Base weights and processor | `Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac`, `HPSV3_BASE_MODEL` |

The pinned [official YAML](https://github.com/MizzenAI/HPSv3/blob/bd0c5fcb5f587617b0169c07222ab78d01e2f3c2/hpsv3/config/HPSv3_7B.yaml)
and [reward class](https://github.com/MizzenAI/HPSv3/blob/bd0c5fcb5f587617b0169c07222ab78d01e2f3c2/hpsv3/model/qwen2vl_trainer.py)
use **Qwen2-VL**, not Qwen2.5-VL. The downloaded HPS repository's root
`config.json` is not used to instantiate the backbone. The adapter uses the
source YAML, replaces only the base path/output directory/FlashAttention choice,
and strictly loads the complete HPS checkpoint. It validates all local base
shards and tokenizer/processor files before loading.

Use [requirements-hpsv3.txt](../requirements-hpsv3.txt) in its own Python 3.10
environment. Imports come from the fixed local source, so an unrelated PyPI
`hpsv3` installation is unnecessary. The official source imports training
utilities even for inference; its pinned TRL/PEFT/DeepSpeed dependencies remain
separate from the generation environment. Current hardware compatibility has
not been proven by the CPU checks.

## Download catalog

`scripts/download_assets.py list` is offline. `plan` reads only small remote
metadata (or HTTP HEAD), displays selected/remaining/free bytes, and never
fetches weights. Only explicit `download` writes assets. HF downloads pin the
resolved commit in `.scd-download.json`; new HPSv3/UnifiedReward/Z-Image defaults
are fixed commits in the resource registry. Changing revisions requires a fresh
destination. `--out` selects a directory for HF/ModelScope assets, but a complete
filename for the two GenEval URL assets.

| Download name | Files / purpose |
|---|---|
| `zimage` | Official `Tongyi-MAI/Z-Image` base pipeline, distinct from Turbo |
| `zimage-turbo` | Official `Tongyi-MAI/Z-Image-Turbo` pipeline |
| `hpsv3` + `hpsv3-base` | Reward checkpoint/config plus all base model shards/tokenizer/processor |
| `unifiedreward` | Complete fixed UnifiedReward 2.0 7B model and processor |
| `geneval-detector` | Official OpenMMLab Mask2Former Swin-S checkpoint; official filename SHA256 prefix verified |
| `geneval-clip` | Original OpenAI ViT-L/14 JIT `.pt`; full official SHA256 verified; HF CLIP weights are not interchangeable here |
| `dpg-mplug` | Complete official ModelScope `damo/mplug_visual-question-answering_coco_large_en@v1.0.0`; every file verified against upstream SHA256 |

Source download names are `hpsv3`, `unifiedreward`, `geneval`, `dpg`, and
`mmdetection` (plus the unchanged legacy sources). GenEval uses MMDetection
`v2.28.2`, fixed commit `e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1`, including
the referenced `_base_` configs. Its legacy MMDetection/MMCV environment must
remain independent; no compatibility with the newest 5090 stack is implied.
The download script records the actual fixed source commit and refuses an
existing destination.

The GenEval detector URL comes from the official evaluator; the CLIP URL and
hash come from [OpenCLIP 2.20.0's OpenAI checkpoint registry](https://github.com/mlfoundations/open_clip/blob/v2.20.0/src/open_clip/pretrained.py).
The DPG snapshot uses the official [ModelScope release](https://modelscope.cn/models/damo/mplug_visual-question-answering_coco_large_en/files?Revision=v1.0.0).
No third-party re-upload is substituted.

Before downloads, reuse existing model/dataset caches through the resource
registry and check the destination filesystem. On irip, a mirror does not fix
blocked Xet redirects; do not disable TLS verification. Run the documented
downloads on the actual SLAI host if that is where evaluation will execute.
