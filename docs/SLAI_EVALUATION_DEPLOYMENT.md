# SLAI evaluation deployment

Use a separate evaluation checkout and persistent directories owned by your
account. SLAI GPU execution uses the devices already allocated by the platform.
The irip development host uses Slurm; SLAI deployment does not submit Slurm jobs.
The executable commands are in the first section of [command.md](../command.md).

## Environments

`scripts/setup_eval_environment.py` prints the complete installation recipe by
default. It requires an explicit profile, new prefix and runtime. Only `--apply`
creates an environment; both preview and execution reject an existing prefix,
including a partial environment. It never installs into the invoking Python,
Conda base, the existing `eval` environment or a training environment. A failed
installation leaves the partial prefix for inspection. Reuse a matching existing
environment by selecting its absolute Python path and running the checker.

| Profile | Purpose | Runtime recipe |
|---|---|---|
| `legacy` | SD1.5/SDXL generation and original five metrics | `slai-cu130` |
| `flow` | SD3.5 Medium/Z-Image Base generation | `slai-cu130` reference |
| `hpsv3` | HPSv3 only | `slai-cu130` reference |
| `unifiedreward` | UnifiedReward 2.0 ACS only | `slai-cu130` reference |
| `geneval` | Official detector evaluator | `h100-cu118` candidate |
| `dpg` | Official mPLUG VQA evaluator | `h100-cu118` candidate |

The script enforces this table: modern profiles `flow`, `hpsv3` and
`unifiedreward` accept only `slai-cu130`; the old torch 2.0/cu118 recipe is
reserved for `geneval` and `dpg`. The `legacy` profile also uses only
`slai-cu130`, as pinned in its requirements file.

The CUDA 13 recipe retains the project's existing PyTorch 2.14.1 / torchvision
0.29.1 / NCCL 2.30.7 pins. The PyTorch pair and CUDA wheels were checked against
the [official install table](https://pytorch.org/get-started/previous-versions/)
and [official wheel index](https://download.pytorch.org/whl/cu130/torch/).
Modern generation retains Diffusers 0.37.1 and Accelerate 1.14.0, whose published
versions were verified on the projects' [Diffusers release metadata](https://pypi.org/pypi/diffusers/0.37.1/json)
and [Accelerate release metadata](https://pypi.org/pypi/accelerate/1.14.0/json).
These existence checks do not prove that all reward stacks work with this CUDA
runtime. HPSv3's old DeepSpeed and GenEval's compiled MMCV particularly require
an actual allocated-GPU smoke test. GenEval/DPG cu118 recipes are H100 candidates;
do not treat them as RTX 5090 recipes.

`scripts/check_eval_environment.py --profile ...` checks pinned distribution
versions, important imports and local scorer assets. `--skip-assets` allows
dependency checks before downloads. It does not load full weights. `flow` assets
depend on the selected run/model and are checked by the generation stage's
`--check-only`. GPU checking requires explicit `--gpu` and the corresponding
execution boundary; imports and metadata alone cannot validate CUDA kernels.
The setup script runs dependency checks and `pip check` before writing its
completion receipt inside the new environment.

## Resource reuse and downloads

Keep the public resource layout in `configs/resources.json`; put machine paths
in `configs/resources.local.json`. An alternate registry is selected with
`SCD_RESOURCES` or `--resources`. Environment variables override both. The same
registry is used by evaluation and `scripts/download_assets.py`.
Point resource keys directly at existing complete local models or Hugging Face
snapshot directories; no copy is required. Check the resolved paths after moving
machines because previously exported environment variables still take priority.

Asset acquisition is explicit. `list` and `check` are offline; `plan` reads small
upstream metadata and reports destination, selected bytes, remaining bytes and
free space; `download` fetches only the requested assets. `--out` overrides one
destination. Downloading elsewhere does not update the resource registry: set
the matching resource key before generation or scoring. Never download weights
as part of model imports or evaluator execution.

`check` requires a completed acquisition receipt and validates repository,
configured fixed revision, file sizes and every referenced weight shard. For an
intentional alternate HF revision, pass the same `--revision` used at download
and update evaluator protocol settings where required. It never creates a
receipt for an untracked manual cache. Such a cache can still be reused through
resource overrides when the evaluator's own provenance requirements are met;
use that stage's `--check-only` to verify them.

The HF check validates the receipt and sizes, rather than rehashing gigabytes.
`check --full-hash` also rehashes the official URL/ModelScope files that have
published hashes. It cannot add unavailable upstream hashes for HF files.
Downloaded URL/ModelScope assets are always hash checked before publication.
Sources have separate `download_sources.sh --dry-run` and `--check` actions.
For sources pinned to a commit, checking verifies that fixed commit and rejects
tracked source modifications. Legacy `hpsv2`/`aesthetic` downloads still resolve
moving branches: their check verifies HEAD against the local acquisition receipt
and checks tracked edits, without attesting an upstream fixed SHA.
Existing source directories are never overwritten by a download command.

Check the actual destination filesystem before downloading or rendering.
Download planning reserves a 2 GiB free-space margin but excludes extracted data,
generated PNGs and checkpoints. On irip, check `/irip` before any planned write
of at least 1 GiB and inspect the existing model/dataset caches first. TLS errors
must be resolved through the available network path, without disabling certificate
verification. Gated model access must be granted on the official model page.

## First execution and validation boundary

Prepare only the needed generators, scorers, official source and benchmark
prompts. A plan can be made before reward assets are available. Run generation
preflight and a separate small-panel GPU smoke before a full render. Then check
and smoke each selected scorer against those saved images. Scoring uses images
without loading training checkpoints or the generating model. Keep full PNGs,
per-image records and source identities when transferring results.

The deployment changes were checked with small CPU fixtures and shell/Python
syntax checks. No environment was installed, no weight was downloaded, and no
GPU job was run for these changes. Complete environment resolution, GPU memory,
compiled extension compatibility and full-model numerical behavior remain SLAI
smoke-test requirements.
