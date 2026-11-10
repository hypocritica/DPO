# GenEval / DPG-Bench evaluation

This module scores **existing, completed render directories**. It never loads a
generation pipeline, a LoRA, a training checkpoint, or a reward model in place of
the benchmark evaluator. The executable workflows are collected in
[command.md](../command.md). GenEval 2 is outside this implementation.

## Pinned official inputs

| Benchmark | Repository / revision | Full prompt count | Evaluator |
|---|---|---:|---|
| GenEval | [djghosh13/geneval](https://github.com/djghosh13/geneval/tree/af4902f24d3ca90ebbb446dd9891a59e0f82725f), `af4902f24d3ca90ebbb446dd9891a59e0f82725f` | 553 | Official Mask2Former + OpenAI CLIP color classifier |
| DPG-Bench | [TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA/tree/3c228f1dc6c4d3cad0a47493816151a419f14db3), `3c228f1dc6c4d3cad0a47493816151a419f14db3` | 1,065 | Official mPLUG question scoring and dependency gating |

`scripts/prepare_compositional_prompts.py` takes `--benchmark geneval|dpg`,
`--source-dir` pointing to a local Git checkout, and a new `--out` directory.
Omitting `--source-dir` reads `GENEVAL_REPO` / `DPG_REPO` from the resource registry.
Preparation is CPU only and never downloads models or sources. It checks the
pinned HEAD and the actual Git blob bytes of the benchmark's tracked code/data.
Modified official evaluator files and prompts are refused.

The output consists of:

- `suite.jsonl`: the existing renderer's canonical-only suite schema;
- `selection.json`: revision, every source-file SHA256, suite SHA256, sample count
  and full-benchmark/smoke-only status.

GenEval retains the original structured `include`, `exclude`, `tag` and prompt
metadata; `source_id` is the original zero-based row number. DPG uses the exact
contents of `dpg_bench/prompts/<id>.txt`, including whitespace, rather than the
CSV's sometimes different `text` field. It embeds **all** original CSV question
rows, proposition IDs, dependencies and category tuples in each suite record.
No text normalization, deduplication, training-overlap filtering, prompt rewriting
or question regeneration is applied. DPG export order is lexicographic by
original filename; filenames and question IDs are preserved.

The render embeds this complete suite, so scoring and a plan's frozen prompt
snapshot do not need the original preparation directory. The pinned evaluator
checkout is still required for scoring. `--smoke-prompts N` produces an explicitly
labelled first-N prefix, never an official full-benchmark result.

## Configuration and environments

`configs/compositional.json` contains independent `geneval` and `dpg` settings.
Paths use the shared resource registry; a private copy of the config may also
provide `python` as the absolute path of each evaluator's Python interpreter.
Otherwise the current interpreter is used. The config supports an explicit
`allow_smoke: true` for a separate smoke run. Production defaults reject any
suite smaller than the full official panel.

| Resource | Required content |
|---|---|
| `GENEVAL_REPO` | Pinned, unmodified official GenEval checkout |
| `GENEVAL_MODEL_DIR` | `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth` |
| `GENEVAL_MODEL_CONFIG` | MMDetection 2.x `configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py`, with its imported `_base_` files |
| `GENEVAL_CLIP_MODEL` | Original OpenAI `ViT-L-14.pt` checkpoint, not a Hugging Face CLIP directory |
| `DPG_REPO` | Pinned, unmodified official ELLA checkout |
| `DPG_MODEL_DIR` | Complete local ModelScope snapshot of `damo/mplug_visual-question-answering_coco_large_en`, including `configuration.json` |

GenEval's published environment uses PyTorch 1.12.1/CUDA 11.3,
`mmcv-full==1.7.1`, `open-clip-torch==2.20.0`, `clip-benchmark==1.4.0` and
MMDetection 2.x. The legacy compiled detector must run in a dedicated environment,
not the modern generation/HPSv3/UnifiedReward environment.
The H100 deployment candidate in `command.md` uses Python 3.10,
PyTorch 2.0.0/cu118, torchvision 0.15.1, MMCV-full 1.7.2 and MMDetection 2.28.2.
The [official CUDA 11.8/PyTorch 2.0.0 wheel index](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html)
contains a CPython 3.10 Linux **1.7.2** wheel, but no 1.7.1 wheel.
MMDetection 2.28.2's [version check](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/mmdet/__init__.py)
accepts this version. This is a disclosed dependency-version change from the
paper's 1.7.1 environment, not a claim of bitwise reproduction.
`requirements-geneval.txt` pins the scorer's principal Python dependencies.
The installed torch/torchvision pair and compiled MMCV must match the target GPU.
In particular, the published CUDA 11.3 environment is not a validated RTX 5090
deployment recipe. The official environment and modern-GPU compatibility have
not been installed or GPU-tested in this change.

For DPG, `requirements-dpg.txt` pins a Python 3.10 deployment candidate using
ModelScope 1.13.3, Transformers 4.36.2, PyTorch 2.0.0/cu118 and a local complete
mPLUG snapshot. ModelScope's [v1.13.3 official test](https://github.com/modelscope/modelscope/blob/v1.13.3/tests/pipelines/test_mplug_tasks.py)
explicitly exercises `damo/mplug_visual-question-answering_coco_large_en`.
Its [VQA pipeline](https://github.com/modelscope/modelscope/blob/v1.13.3/modelscope/pipelines/multi_modal/visual_question_answering_pipeline.py)
also imports OFA; therefore the environment includes the old multimodal extras,
including Fairseq/OmegaConf/Lightning. Those distributions need pip 24.0 to
accept their legacy metadata. The requirements pin important direct/runtime
dependencies, not every transitive package. Installation and real GPU inference
remain untested here; keep the full resolved environment inventory with each run.

The wrapper loads GenEval's official evaluator and calls its own `load_models`
and `evaluate_image` functions. The only CLIP hook redirects the official
OpenAI checkpoint download function to `GENEVAL_CLIP_MODEL`; the `openai`
architecture, QuickGELU behavior, transforms, object thresholds, crop settings,
and color prompts are unchanged. This hook requires exactly OpenCLIP 2.20.0.
All detector configs, model artifact inventories and wrapper/evaluator source
identities are recorded. Large weights use the existing size/mtime inventory
policy; this is not a fresh cryptographic hash of all model weights.

## Sampling and official export

The renderer must produce **four distinct seeds for every prompt**, with the
same seed list and order for compared methods. There is no best-of-four selection.
The scorer verifies the render metadata, complete marker, full paired design and
SHA256 of every generated image before scoring.

For GenEval, export is `<prompt-index>/metadata.jsonl` and
`<prompt-index>/samples/00000.png` through `00003.png`, matching the official
structure. Images are copied without resampling. The export rejects non-PNG
inputs rather than silently relabelling a different format.

For DPG, four equally sized square images are assembled into a lossless PNG
`<original-id>.png`. Seed order is top-left, top-right, bottom-left, bottom-right.
`resolution` passed to the upstream scoring function is the size of **one** tile,
not the 2x2 grid. No resizing, borders or thumbnail preprocessing is added.

## Scores and a disclosed DPG upstream correction

GenEval's `mean` is in [0, 1]: each task's per-image Boolean correctness is
averaged, then those task averages are averaged with equal weight. It is **not**
the micro-average over all images and not pass-at-four. `by_category` reports the
six task scores and `micro_image_accuracy` is an explicitly separate diagnostic.

DPG's `mean` is in [0, 100]: use the unmodified upstream
`compute_dpg_one_sample` to score each tile's questions, zero child scores when a
dependency fails, average questions within each image, average four images
within each prompt, then average prompts and multiply by 100.

The pinned upstream `prepare_dpg_data` contains `if i == 0: continue` **after**
pandas has consumed the CSV header. It therefore drops the first real question,
`partiprompts97` proposition 1. That prompt's later questions reference the missing
parent; upstream `main` catches the failure and continues, potentially omitting
the prompt from the denominator. This implementation reads all CSV data rows and
constructs the same question dictionaries before calling the unmodified official
scoring function. Any missing, failed or corrupt prompt stops completion.

Consequently this protocol is named **`dpg-complete-csv-four-samples-v1`** and
always evaluates all 1,065 prompts for a full result. It must **not** be described
as an exact reproduction of historical scores produced by that omission. Compare
all methods, including the base model, with this same disclosed protocol.

The upstream function returns category answers only for the **last of the four
images**, before dependency gating. The adapter preserves that definition in
DPG `by_category` and explicitly labels it `category_definition`; it does not
present this breakdown as a four-image, dependency-aware category score.
The main `mean` does use all four images and dependency gating. Full official
per-tile answers are retained in each raw `results_detail.txt` file.

## Completion and resume

The CLI interface is `python -m self_correct_dpo.evaluation.compositional` with
`--benchmark`, `--run` (completed render), `--out`, `--config`, optional
`--resources`, optional worker `--python`, and the normal `--execution` / `--run-label` controls. `--check-only`
validates local assets, official metadata, input hashes and four-seed coverage
without importing models or requesting GPUs. It does not establish CUDA package
compatibility or verify the numerical output of the real detector/mPLUG model.

`evaluation.json` freezes the render, config, assets and code identity.
`official-images/` retains the official layout; `export.json` hashes every file.
`raw/<prompt-index>/` retains official per-image GenEval records or DPG official
score/answer text together with a structured `result.json`. Each prompt is
published by atomic directory rename only after all four images succeed.
Re-running the same command reuses verified complete prompts and recomputes only
unfinished prompts; it refuses changed images, source files, config or assets.
An interruption never promotes partial raw results to a complete score.

After every required prompt succeeds, `summary.json` and `complete.json` are
published. The latter contains the summary SHA256 and all output-file checksums.
`verify_result(out_dir, render_dir)` performs a model-free integrity check for the
batch summary stage. A changed or missing raw result invalidates completion.

## Validation boundary

Four focused CPU tests exercise real Git provenance, preserved first-question
dependencies and prompt whitespace, real PIL grid placement, task-macro versus
micro aggregation, interruption/resume, missing images, modified assets and
tampered raw results. Scorer outputs in the resume test are deterministic fixture
records; no real detector, CLIP or mPLUG inference was executed. A bounded GPU
smoke run in each isolated scorer environment remains required before reporting
benchmark scores.

The actual pinned official checkouts were also prepared in temporary directories:
553 GenEval prompts and 1,065 DPG prompts / 14,392 questions passed. DPG's
`partiprompts97` retained all 19 questions including proposition 1;
`countbench4` retained its original trailing newline. The resulting suite SHA256s
were `7df8e6b5d3220122f07092eefc4ffcdc0517063f452d451b21f3e33cbe568f39`
(GenEval) and `8a3180330127a2d2a33c14f302435c6e745ae5792aa840fcdcac524257c9930c`
(DPG). Temporary source checkouts were removed after verification.

An additional CPU check executed AST-extracted functions from the actual pinned
upstream scripts. With stub detector/model outputs and real PIL/NumPy inputs,
GenEval's `load_models` return structure, global thresholds, spatial evaluation
and JSON output passed. DPG's local mPLUG constructor, four-crop scoring,
dependency gating, final-crop category return and eight raw question answers
passed a deterministic two-question case (expected score 0.625). These are
interface/protocol checks; model inference and GPU compatibility remain untested.
