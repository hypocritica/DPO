# MVPO qualitative 30 — SLAI generation and review sheets

Current scope: generate PNGs and contact sheets on SLAI, then download and select
locally. The launcher does **not** select paper cases or a gallery. Optional layout
code is included for the later manual selection stage; proposals are explicitly
unreviewed, never a claim of measured or observed superiority.

## Fixed selection

| Backbone | DPO | MVPO checkpoint | Per-GPU batch | Allocated GPUs |
|---|---:|---|---:|---|
| SD1.5 | 10000 | Stagewise **DMPO K6**, 10000 | 16 | 4 × RTX 5090 |
| SDXL | 2000 | Stagewise **DMPO K6**, 2000 | 4 | 4 × RTX 5090 |
| SD3.5-M | 1600 | **Global K4**, 1600 | 4 | 4 × RTX 5090 |
| Z-Image Base | 2300 | Stagewise K4, 2300 | 4 | 2 × H100 |

The middle column remains **DPO** for all backbones. SDXL uses the existing main
2000-step selection, not the separate later peak-supplement selection. Exact run
paths/IDs and checkpoint IDs are in `config.json`. Z-Image imports its exact
identities from the already frozen `curves-h800-b4-v1-prepared` catalog; it never
chooses `latest`, substitutes Turbo, or picks another step. Missing retained
checkpoints are reported while available methods continue. Invalid identities
or corrupt weights stop execution rather than silently producing mislabeled data.

No dedicated seeds for this qualitative panel were found: **0, 1, 2, 3**.
The provided JSONL is copied unchanged. Only its `prompt` string enters inference.
The other metadata is retained for later selection.

| Backbone | Resolution | Scheduler | Steps | CFG | Precision / placement |
|---|---|---|---:|---:|---|
| SD1.5 | 512² | Euler | 50 | 7.5 | BF16, resident |
| SDXL | 1024² | Euler | 50 | 5 | BF16, resident |
| SD3.5-M | 1024² | FlowMatch Euler | 28 | 4.5 | BF16, FP32 VAE, model CPU offload |
| Z-Image | 1024² | FlowMatch Euler | 50 | 4 | FP32 transformer weights / latents / VAE, BF16 transformer autocast, resident |

All negative prompts are empty. Flow max sequence length is 256. Model families,
base revisions, VAE and adapter backend metadata are checked by the existing
renderer. Z-Image's raw noise is FP32 as required by the pinned Diffusers 0.37.1
[pipeline implementation](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/pipelines/z_image/pipeline_z_image.py).
The existing Z-Image padding/precision compatibility code is reused unchanged.

## Deploy and run

Copy the entire `qualitative/` folder into the existing compatible
`/mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2` checkout. The archive contains
only this new folder; it does not replace the running evaluation modules.
Do not change that checkout's renderer while its workers are running.

Use the already allocated SLAI container, not the irip login node. Resource
allocation remains through the existing SLAI platform workflow, not invented
Slurm partitions. Absolute Python paths use the existing `eval` and
`scd-eval-flow-v1` environments. The launcher does not install or download anything.

```bash
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2
export QUAL_OUT=/mnt/aoss/xietianyu/self_correct/outputs/qualitative_30

# In the allocated 4 × 5090 container. Reuse the same four GPUs sequentially:
bash qualitative/run_slai.sh sd15 smoke
bash qualitative/run_slai.sh sd15 full
bash qualitative/run_slai.sh sdxl smoke
bash qualitative/run_slai.sh sdxl full

# In the allocated 4 × 5090 SD3.5-M container:
bash qualitative/run_slai.sh sd35_medium smoke
bash qualitative/run_slai.sh sd35_medium full

# In the allocated 2 × H100 Z-Image container:
bash qualitative/run_slai.sh zimage smoke
bash qualitative/run_slai.sh zimage full
```

`smoke` generates P02/P26 at seed 0 (six images per available backbone) and their
contact sheets. Inspect weight-loading logs and these sheets before `full`.
Smoke results count toward the 360 images/backbone; full never repacks or rerenders
committed smoke samples. `all` runs smoke, audits it, then full automatically;
it does **not** constitute human visual inspection. Default phase is `smoke`.
`CUDA_VISIBLE_DEVICES` is preserved, including UUID tokens; it must expose exactly
four GPUs (SD1.5/SDXL/SD3.5-M) or two (Z-Image). `SCD_SD15_PY` and `SCD_FLOW_PY`
can override Python executable paths. Each worker loads one clean pipeline per
method, reuses it across all assigned batches, then releases it. There is no
adapter switching, merging or fusion in a live Base pipeline.

Repeat the exact command to resume. Complete PNGs are reused only with matching
frozen plan, source hashes, package versions, GPU type, sampling, checkpoint hashes,
PNG checksum and decode success. A partially completed batch is recomputed with
its original members, and only missing/corrupt PNGs are committed. Raw CPU noise
is cached per prompt/seed and cloned on the GPU. Per-image scheduler generators
and global RNG are reset for each method/batch; the scheduler is rebuilt for every
batch. Worker ownership and batch boundaries are frozen. A change of model,
checkpoint, protocol, code, environment or GPU model requires a fresh output root.

`plan` is CPU-only and reports the available methods without loading a pipeline:

```bash
/mnt/afs/L202500030/xietianyu/miniconda3/envs/eval/bin/python \
  qualitative/run.py plan --out "$QUAL_OUT" --backbone sd15
```

Planning verifies local model inventory and all files in the selected checkpoint
manifest. A successful plan is not a GPU load or memory validation. It freezes the
available set, including missing methods; use a new output root if missing weights
are later restored. Check free space on the output filesystem before generation.

## Outputs and download

```text
qualitative_30/
  sd15|sdxl|sd35_medium|zimage/
    Base|DPO|MVPO/P01/seed_0.png    # original image; adjacent JSON record
    noise/P01/seed_0.pt            # common raw noise plus cache checksum
    manifest.jsonl
    run_config.json
    MVPO_qualitative_prompts.jsonl
    prompt_list.json
    environment-*.json
    scheduler-*.json
    pipeline-defaults-*.json
    smoke_report.json / report.json
    logs/                         # exact commands and worker logs
  contact_sheets/<backbone>/P01.png
```

With all weights available: **1,440 original PNGs**, of which **480 MVPO**, and
**120 contact sheets**. Contact sheets always use four seed rows and Base/DPO/MVPO
columns, full English prompts, model and seed labels. Missing images are explicitly
marked. Pillow-only contact generation needs no new plotting dependencies.
The PNG dpi metadata is 300; original pixel resolutions remain unchanged.

To create a download bundle without duplicating the potentially large noise cache:

```bash
cd /mnt/aoss/xietianyu/self_correct/outputs
# Use a fresh filename; keep the original noise cache on SLAI.
tar --exclude='*/noise' -czf "qualitative_30_review_$(date -u +%Y%m%dT%H%M%S).tar.gz" qualitative_30
```

After download, select complete prompt–seed triplets from the sheets; do not mix
seeds between methods. For the later 20-image gallery, select MVPO images only.
All 30 prompts remain candidates. The optional `layout.py propose` creates editable
JSON proposals; it never overwrites an existing selection. If results have moved,
rerun proposals after moving old selection files aside, so image paths match their
new location. Proposals are based only on topic diversity and are unreviewed.

```bash
# Later, using a local Python with Pillow and matplotlib:
python qualitative/layout.py propose --out /absolute/path/to/qualitative_30
# Edit comparison_selection.json / gallery_selection.json after inspecting images.
python qualitative/layout.py export --out /absolute/path/to/qualitative_30
```

Export validates full paired triplets and original MVPO sources. It produces the
four-backbone 2×2 comparison (two cases each), a compact SD3.5-M/Z-Image comparison
(three cases each), SD1.5/SDXL supplement, and 4×5 MVPO gallery, all at 174 mm width
as PDF / 300 dpi PNG, plus caption drafts. Original images are resized without
retouching/cropping. `reviewed: true` is a user annotation after inspection, not an
automatic quality claim. Inspect the actual-size figures again after selecting.

## Validation performed here

CPU tests cover complete/disjoint batching, persistent paired noise including
Z-Image FP32, mutation isolation, corrupt PNG detection, plan mismatch rejection,
matched-triplet selection, and actual PDF/300 dpi PNG export at 174 mm. Color
fixtures were used only under `/tmp` for checking layout, never as generated results.
The 30 supplied prompt strings were preserved. Real SLAI GPU loading, memory usage
and visual quality require the above smoke run. No GPU generation was run on irip.

```bash
python qualitative/test_workflow.py -v
```
