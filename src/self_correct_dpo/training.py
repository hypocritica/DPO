"""Single-node LoRA B/C training with independent detached views on every rank."""
from __future__ import annotations

from self_correct_dpo.common import source_file
from contextlib import nullcontext
from functools import lru_cache
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import tempfile
import time

from self_correct_dpo.common import model_artifact_identity, digest, read_json, sha256, write_json
from self_correct_dpo.runtime.storage_compat import compatible_sources, record_compatibility

def git_revision():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


OBJECTIVES = {"sft", "dpo", "augmentation", "orbit_mean", "orbit_worst", "orbit_softmax", "orbit_consistency"}


def validate_config(config, *, files=True):
    model, data, training = (config[k] for k in ("model", "data", "training"))
    if model["family"] not in {"sd15", "sdxl", "sd35_medium"}:
        raise ValueError("model.family must be sd15, sdxl or sd35_medium")
    if not model.get("path") or not model.get("revision"):
        raise ValueError("Provide model.path and a traceable model.revision")
    if files and any(word in model["revision"].upper() for word in ("FILL_", "TODO", "UNVERIFIED")):
        raise ValueError("Replace model.revision placeholder with verified source provenance")
    if files and not (Path(model["path"]) / "model_index.json").is_file():
        raise ValueError("model.path must be a complete local Diffusers pipeline")
    if files and model["family"] == "sd15":
        if read_json(Path(model["path"]) / "model_index.json").get("_class_name") != "StableDiffusionPipeline":
            raise ValueError("SD1.5 requires a StableDiffusionPipeline checkpoint")
        unet = read_json(Path(model["path"]) / "unet/config.json")
        if unet.get("cross_attention_dim") != 768 or unet.get("in_channels") != 4:
            raise ValueError("SD1.5 UNet requires 768-dimensional CLIP conditioning and four latent channels")
    from self_correct_dpo.architecture import validate_sdxl_base, validate_vae_override
    validate_vae_override(model,files=files)
    if files and model['family']=='sdxl':validate_sdxl_base(model['path'])
    if not data.get("pairs") or data.get("split", "train") != "train":
        raise ValueError("Training requires a pair manifest and split=train")
    if data.get('validation', 'auto') not in ('auto', 'full'):
        raise ValueError('data.validation must be auto or full')
    for key, default in (('validation_workers', 8), ('validation_samples', 8)):
        if type(data.get(key, default)) is not int or data.get(key, default) < 1:
            raise ValueError(f'data.{key} must be a positive integer')
    if (not data.get("require_reviewed", True) and not data.get("require_auto_verified", False)
            and not data.get("require_human_preference", False)
            and "TECHNICAL_SYNTHETIC_NOT_RESEARCH" not in config.get("experiment_note", "")):
        raise ValueError("Unreviewed research pairs require verified automatic preference evidence")
    if training["objective"] not in OBJECTIVES:
        raise ValueError("Unknown training objective")
    for key in ("max_steps", "batch_size", "gradient_accumulation_steps", "checkpoint_every", "resolution", "lora_rank", "lora_alpha"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["resolution"] % 16 or training["resolution"] < 64:
        raise ValueError("resolution must be >=64 and divisible by 16")
    if training.get("dtype") not in {"float32", "bfloat16"}:
        raise ValueError("Training supports float32/bfloat16; FP16 requires a scaler and is deliberately excluded")
    for key in ("beta", "learning_rate", "max_grad_norm", "temperature"):
        if not math.isfinite(float(training[key])) or float(training[key]) <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    for key in ("weight_decay", "consistency_weight"):
        if not math.isfinite(float(training.get(key, 0))) or float(training.get(key, 0)) < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(training.get("orbit_size", 4)) is not int or training.get("orbit_size", 4) < 1:
        raise ValueError("orbit_size must be a positive integer")
    if training.get("prompt_mode", "canonical") not in {"canonical", "random"}:
        raise ValueError("prompt_mode must be canonical or random")
    if not 0 <= training.get("warmup_steps", 0) < training["max_steps"]:
        raise ValueError("warmup_steps must be between 0 and max_steps-1")
    if training.get("lr_schedule", "constant") not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be constant or cosine")
    for key in ("caption_dropout", "empty_prompt_probability", "proportion_empty_prompts"):
        if training.get(key, 0.0) != 0.0:
            raise ValueError(f"training.{key} must be zero for a conditioning intervention")
    if training.get("sampling", "random") not in {"random", "balanced"}:
        raise ValueError("sampling must be random or balanced")
    if training.get("sampling") == "balanced" and training["objective"].startswith("orbit_"):
        raise ValueError("balanced sampling currently supports single-prompt objectives")
    if data.get("require_human_preference"):
        if training["objective"] != "dpo" or training.get("prompt_mode", "canonical") != "canonical":
            raise ValueError("Pick-a-Pic baseline requires canonical standard DPO")
        if training.get("consistency_weight", 0) != 0:
            raise ValueError("No orbit consistency term in the human baseline")
    if training.get("checkpoint_policy") == "bounded":
        from self_correct_dpo.runtime.checkpoints import progress_steps
        progress_steps(training)
    diagnostic = training.get("diagnostic", {})
    if diagnostic and any(type(diagnostic.get(k)) is not int or diagnostic[k] < 1 for k in ("pairs", "every", "seed")):
        raise ValueError("Diagnostic pairs/every/seed must be positive integers")
    arm = training.setdefault("formal_arm", "C")
    k = training.setdefault("noise_views", 2)
    if model["family"] not in ('sd15','sdxl') or training["objective"] != "dpo":
        raise ValueError("This research release supports SD1.5/SDXL DPO only")
    if arm not in {"B", "C"}:
        raise ValueError("SD1.5/SDXL support only B/C; C is the default. Legacy A/D configs are archives.")
    if type(k) is not int or k < 1:
        raise ValueError("B/C require a positive integer noise_views (K)")
    if type(training.get('batch_views', False)) is not bool:
        raise ValueError('training.batch_views must be a boolean')
    if training.get('batch_views') and ('independent_rollout' in training or 'paired_drift' in training):
        raise ValueError('batch_views supports ordinary Gaussian views only; disable rollout/paired_drift')
    if config.get('method') in ('multiview-B-gaussian', 'multiview-C-gaussian') and (
            'independent_rollout' in training or 'paired_drift' in training):
        raise ValueError('Gaussian multiview methods require no rollout or paired_drift')
    if type(training.get("world_size", 1)) is not int or training.get("world_size", 1) < 1:
        raise ValueError("world_size must be a positive integer")
    if 'comparison_budget' in training:
        budget = training['comparison_budget']
        if (not isinstance(budget, dict) or budget.get('unit') not in ('pairs', 'pair_views')
                or type(budget.get('per_update')) is not int or budget['per_update'] < 1):
            raise ValueError('training.comparison_budget requires unit=pairs/pair_views and positive integer per_update')
        method = config.get('method')
        if (method not in ('dpo', 'multiview-B-gaussian', 'multiview-C-gaussian')
                or 'independent_rollout' in training or 'paired_drift' in training):
            raise ValueError('comparison_budget supports DPO and ordinary Gaussian multiview methods only')
        if method == 'dpo' and k != 1:
            raise ValueError('comparison_budget requires noise_views=1 for DPO')
        actual = training['batch_size'] * training['gradient_accumulation_steps'] * training.get('world_size', 1)
        if budget['unit'] == 'pair_views':
            actual *= k
        if actual != budget['per_update']:
            raise ValueError(f"comparison_budget mismatch: actual {actual} {budget['unit']} per update, expected {budget['per_update']}")
    if training.get('logging', 'tensorboard') not in ('tensorboard', 'wandb', 'all', 'none', 'jsonl'):
        raise ValueError('logging must be tensorboard, wandb, all, none or jsonl')
    if "paired_drift" in training:
        from self_correct_dpo.algorithms.paired_drift import validate_paired_drift
        validate_paired_drift(model, training)
        if files and read_json(Path(model["path"]) / "scheduler/scheduler_config.json").get("prediction_type", "epsilon") != "epsilon":
            raise ValueError("paired_drift supports epsilon prediction only")
    if "independent_rollout" in training:
        from self_correct_dpo.algorithms.independent import RolloutConfig
        total=read_json(Path(model['path'])/'scheduler/scheduler_config.json')['num_train_timesteps'] if files and model['family']=='sdxl' else 1000
        RolloutConfig(**training["independent_rollout"]).validate(total)
    return config


def eligible_prompts(row, data):
    result = list(row["prompts"])
    for field, config_key in (("template_family", "template_families"), ("variant_id", "variant_ids")):
        allowed = data.get(config_key)
        if allowed is not None:
            if not isinstance(allowed, list) or not allowed:
                raise ValueError(f"{config_key} must be null or a nonempty list")
            result = [p for p in result if p.get(field) in allowed]
    if not result:
        raise ValueError(f"No eligible prompts for pair {row['pair_id']}")
    return sorted(result, key=lambda p: p["variant_id"])


def check_rows(rows, config):
    if not rows:
        raise ValueError("Empty training split")
    t = config["training"]
    if config["data"].get("diagnostic_suite"):
        from self_correct_dpo.data.prompt_generalization import freeze_suite
        from self_correct_dpo.common import load_suite
        suite_path = Path(config["data"]["diagnostic_suite"])
        frozen = read_json(suite_path.parent / "suite.json")
        if frozen["prompts_sha256"] != sha256(suite_path):
            raise ValueError("Frozen diagnostic suite changed")
        if freeze_suite(frozen["scenes"], rows) != load_suite(suite_path):
            raise ValueError("Diagnostic suite does not match training exposure")
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Non-training row reached trainer")
        if config["data"].get("require_reviewed", True) and row.get("reviewed") is not True:
            raise ValueError(f"Pair {row['pair_id']} has no recorded semantic/preference review")
        if config["data"].get("require_human_preference", False):
            from self_correct_dpo.data.pickapic import validate_human_pair
            validate_human_pair(row)
        if config["data"].get("require_auto_verified", False):
            from self_correct_dpo.data.auto_preferences import validate_auto_pair
            validate_auto_pair(row)
        prompts = eligible_prompts(row, config["data"])
        if any(not p["prompt"].strip() for p in prompts):
            raise ValueError("Empty conditioning prompts are not allowed")
        if t["objective"].startswith("orbit_") and len(prompts) < t["orbit_size"]:
            raise ValueError(f"Pair {row['pair_id']} has fewer eligible prompts than orbit_size")
        if t["objective"] in {"sft", "dpo"} and t.get("prompt_mode", "canonical") == "canonical":
            if sum(p["variant_id"] == "canonical" for p in prompts) != 1:
                raise ValueError("Canonical objective needs canonical; use prompt_mode=random for A/B intervention")


@lru_cache(maxsize=2)
def pair_order(seed, epoch, count):
    order = list(range(count))
    random.Random(digest([seed, epoch, "pair-permutation"])).shuffle(order)
    return tuple(order)


def sample_batch(rows, config, microbatch, *, world=1, rank=0):
    """Independent stateless streams keep image exposure identical across objectives."""
    t = config["training"]
    seed = config["seed"]
    microbatch = microbatch * world + rank
    data_rng = random.Random(digest([seed, microbatch, "pairs"]))
    prompt_rng = random.Random(digest([seed, microbatch, "prompts"]))
    selected, epochs = [], []
    for offset in range(t["batch_size"]):
        if t.get("sampling") == "balanced":
            epoch, position = divmod(microbatch * t["batch_size"] + offset, len(rows))
            order = pair_order(seed, epoch, len(rows))
            selected.append(rows[order[position]])
            epochs.append(epoch)
        else:
            selected.append(rows[data_rng.randrange(len(rows))])
            epochs.append(None)
    prompts, ids = [], []
    for row, epoch in zip(selected, epochs):
        candidates = eligible_prompts(row, config["data"])
        if t["objective"].startswith("orbit_"):
            chosen = prompt_rng.sample(candidates, t["orbit_size"])
        elif t["objective"] == "augmentation" or t.get("prompt_mode") == "random":
            if epoch is None:
                chosen = [prompt_rng.choice(candidates)]
            else:
                # Each pair appears once per epoch. Every K epochs cover its K
                # expressions exactly once, independently of family A versus B.
                cycle, position = divmod(epoch, len(candidates))
                order = list(range(len(candidates)))
                random.Random(digest([seed, row["pair_id"], cycle, "prompt-permutation"])).shuffle(order)
                chosen = [candidates[order[position]]]
        else:
            chosen = [next(p for p in candidates if p["variant_id"] == "canonical")]
        prompts.append([p["prompt"] for p in chosen])
        ids.append([p["variant_id"] for p in chosen])
    return selected, prompts, ids


def exposure_summary(counts):
    values = [n for variants in counts.values() for n in variants.values()]
    return {"eligible_pair_prompts": len(values), "seen_pair_prompts": sum(n > 0 for n in values),
            "min_exposure": min(values), "max_exposure": max(values),
            "prompt_pair_evaluations": sum(values), "pair_prompt_passes": sum(values) / len(values)}


def image_batch(rows, resolution, device, workers=0):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from self_correct_dpo.data.arrow_images import image_source
    def load_pair(row):
        images=[]
        for field in ("winner", "loser"):
            with Image.open(image_source(row, field, verify_files=True)) as image:
                image=ImageOps.exif_transpose(image).convert("RGB")
                image=image.resize((resolution,resolution),Image.Resampling.LANCZOS)
                pixels=np.array(image,dtype=np.float32,copy=True)/127.5-1
            images.append(torch.from_numpy(pixels).permute(2,0,1))
        return torch.stack(images)
    if workers:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pairs=list(executor.map(load_pair,rows))
    else:
        pairs=[load_pair(row) for row in rows]
    return torch.stack(pairs).to(device)


def checkpoint_manifest(path):
    path = Path(path)
    info = read_json(path / "checkpoint.json")
    if info.get("checkpoint_id") != digest({k: v for k, v in info.items() if k != "checkpoint_id"}):
        raise ValueError("Checkpoint manifest identity mismatch")
    for name, expected in info["files"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path.resolve()) or sha256(file) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return info


def save_checkpoint(path, backend, optimizer, scheduler, generator, step, run_id, cost, *, full_state=True, staging_path=None):
    import torch
    import numpy as np
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if staging_path is None:
        temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    else:
        temporary = Path(staging_path)
        temporary.mkdir(exist_ok=False)
        write_json(temporary / "owner.json", {"run_id": run_id})
    backend.save_adapter(temporary / "adapter")
    if full_state:
        from self_correct_dpo.runtime.distributed import capture_rng
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, **capture_rng(generator),
                "rank_states": getattr(backend, "resume_rank_states", None), "cost": cost}, temporary / "state.pt")
    write_json(temporary / "exposure.json", {"step": step, **exposure_summary(cost["exposure"]),
                                           "counts": cost["exposure"]})
    info = {"run_id": run_id, "step": step, "full_state": full_state,
            "files": {str(p.relative_to(temporary)): sha256(p) for p in sorted(temporary.rglob("*")) if p.is_file()}}
    info["checkpoint_id"] = digest(info)
    write_json(temporary / "checkpoint.json", info)
    # A committed directory must contain flushed state, not only a renamed path.
    for file in temporary.rglob('*'):
        if file.is_file():
            with file.open('rb') as handle:
                os.fsync(handle.fileno())
    temporary.rename(path)
    return info


def train(config, rows, out, *, resume=None, extend_from=None, execution=None, backend_factory=None, device="cuda", validation_rows=None, stop_after_step=None):
    import torch
    import numpy as np
    from self_correct_dpo.algorithms.losses import preference_loss
    from self_correct_dpo.models import PreferenceBackend
    from self_correct_dpo.runtime.checkpoints import resolve_checkpoint, validate_extension, ManagedCheckpoints, progress_steps
    from self_correct_dpo.runtime.distributed import initialize, average_metrics, gather_rng, restore_rng
    import torch.distributed as dist
    world, rank, device = initialize(device)
    session_started = time.monotonic()
    t = config["training"]
    if stop_after_step is not None and (type(stop_after_step) is not int or not 1 <= stop_after_step <= t["max_steps"]):
        raise ValueError('stop_after_step must be an integer between 1 and configured max_steps')
    if resume and extend_from:
        raise ValueError("Choose resume or extend_from")
    parent = None
    if extend_from:
        extend_from = resolve_checkpoint(extend_from)
        parent = validate_extension(extend_from, config, rows)
    resume_requested = Path(resume).resolve() if resume else None
    if resume:
        from self_correct_dpo.runtime.checkpoints import recover_run
        if rank == 0:
            recover_run(out)
        if world > 1:
            dist.barrier()
        resume = resolve_checkpoint(resume)
    check_rows(rows, config)
    out = Path(out).resolve()
    if t.get("world_size", world) != world:
        raise ValueError("Launched world size differs from the training configuration")
    model_identity = model_artifact_identity(config["model"])
    identity = {"config": config, "pairs_id": digest(rows), "base_identity": model_identity,
                "code": {p.name: sha256(p) for p in [Path(__file__), source_file("models.py"),
                         source_file("losses.py"), source_file("formal.py"), source_file("independent.py"), source_file("joint.py"), source_file("data.py"), source_file("architecture.py"),
                         source_file("auto_preferences.py"), source_file("reward_preferences.py"),
                         source_file("pickapic.py"), source_file("checkpoints.py"),
                         source_file("diagnostics.py"), source_file("prompt_generalization.py"),
                         source_file("distributed.py"), source_file("arrow_images.py"),
                         source_file("common.py"), source_file("config.py"), source_file("train.py"), source_file("validation.py"),
                         *[source_file(name) for name in ("pipeline.py","evaluation.py","scorers.py","metrics.py","benchmarks.py","render.py","resources.py","tracking.py")]]}}
    if "paired_drift" in t:
        identity["code"]["paired_drift.py"] = sha256(source_file("paired_drift.py"))
    identity["validation_pairs_id"] = digest(validation_rows or [])
    if config.get('evaluation',{}).get('enabled'):
        from self_correct_dpo.evaluation.runner import validate_evaluation
        settings=config['evaluation']
        identity['monitor_inputs']=validate_evaluation(settings['prompts'],settings['scorers_config'],
            ['pickscore'],expected_prompts=settings.get('expected_prompts',128))
    if config["data"].get("diagnostic_suite"):
        suite_path = Path(config["data"]["diagnostic_suite"])
        identity["diagnostic_suite_sha256"] = sha256(suite_path)
    if parent:
        if parent["training"].get("diagnostic_suite_sha256") != identity.get("diagnostic_suite_sha256"):
            raise ValueError("Extension changed diagnostic suite")
        for key in ("pairs_id", "validation_pairs_id", "base_identity", "code", "monitor_inputs"):
            if parent["training"].get(key) != identity.get(key):
                raise ValueError(f"Extension changed {key}")
        identity["parent"] = {"run_id": parent["training"]["run_id"],
            "checkpoint_id": parent["checkpoint"]["checkpoint_id"],
            "path": str(extend_from), "old_budget": parent["training"]["config"]["training"]["max_steps"],
            "new_budget": t["max_steps"], "start_step": parent["checkpoint"]["step"]}
    if resume:
        recorded = read_json(out / "training.json")
        if "parent" in recorded:
            identity["parent"] = recorded["parent"]
        observed_code = identity['code']
        if compatible_sources(recorded['code'], observed_code):
            identity['code'] = recorded['code']
    run_id = digest(identity)
    if resume:
        if read_json(out / "training.json")["run_id"] != run_id:
            raise ValueError("Resume configuration, data, model inventory or training code changed")
        resume = Path(resume).resolve()
        if not resume.is_relative_to(out / "checkpoints"):
            raise ValueError("Resume must use a checkpoint from this run")
        if (out / "complete.json").exists():
            raise ValueError("Run is already complete")
        if t.get("checkpoint_policy") == "bounded":
            recovery = ManagedCheckpoints(out, run_id, t)
            if rank == 0: recovery.recover()
            if world > 1: dist.barrier()
            latest = resolve_checkpoint(out / "checkpoints" / "latest.json")
            if resume_requested.name == "latest.json":
                resume = latest
            elif resume != latest:
                raise ValueError("Resume the current rolling checkpoint; use a fresh experiment for branching")
        ckpt = checkpoint_manifest(resume)
        if ckpt["run_id"] != run_id:
            raise ValueError("Resume checkpoint belongs to a different run")
        newer = [p for p in (out / "checkpoints").glob("step-*") if p.is_dir() and int(p.name[5:]) > ckpt["step"]]
        if newer:
            raise ValueError("Resume the latest checkpoint; use a new run for branching")
        if rank == 0:
            record_compatibility(out, recorded['code'], observed_code)
    elif rank == 0 and out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    from self_correct_dpo.algorithms.formal import MultiViewBackend, sample_views, formal_loss
    arm = t.get("formal_arm", "C")
    if arm not in {"B", "C"}:
        raise ValueError("SD1.5 supports only B/C")
    sampler = sample_views
    from self_correct_dpo.algorithms.joint import JointRolloutBackend
    backend_class = JointRolloutBackend if 'independent_rollout' in t else MultiViewBackend
    if "paired_drift" in t:
        from self_correct_dpo.algorithms.paired_drift import PairedDriftBackend
        backend_class = PairedDriftBackend
    backend = (backend_factory or backend_class)(config["model"], t, device)
    source_checkpoint = resume or extend_from
    if source_checkpoint:
        backend.load_adapter(source_checkpoint / "adapter")
    if world > 1:
        backend.enable_distributed()
    parameters = list(backend.trainable_parameters())
    if not parameters:
        raise ValueError("No trainable adapter parameters")
    optimizer = torch.optim.AdamW(parameters, lr=t["learning_rate"], weight_decay=t.get("weight_decay", 0.0))
    warmup = t.get("warmup_steps", 0)

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if t.get("lr_schedule", "constant") == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, t["max_steps"] - warmup))))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1 + rank)
    step = 0
    cost = {"pairs_seen": 0, "prompt_pair_evaluations": 0, "policy_image_evaluations": 0,
            "reference_image_evaluations": 0, "rollout_model_calls": 0, "rollout_image_evaluations": 0, "training_seconds": 0.0,
            "exposure": {r["pair_id"]: {p["variant_id"]: 0 for p in eligible_prompts(r, config["data"])} for r in rows}}
    if source_checkpoint:
        state = torch.load(source_checkpoint / "state.pt", map_location="cpu", weights_only=True)
        if state["step"] != checkpoint_manifest(source_checkpoint)["step"]:
            raise ValueError("Saved training step differs from checkpoint manifest")
        parent_training = parent["training"] if parent else read_json(out / "training.json")
        for package, version in parent_training["versions"].items():
            try:
                current = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                current = "missing"
            if current != version:
                raise ValueError(f"Cannot safely resume with changed package {package}: {version} -> {current}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng = state
        if world > 1:
            if not state.get("rank_states") or len(state["rank_states"]) != world:
                raise ValueError("Distributed resume requires RNG state for the same number of ranks")
            rng = state["rank_states"][rank]
            if rng.get('rank')!=rank or rng.get('world_size')!=world:
                raise ValueError('Distributed checkpoint rank identity mismatch')
        restore_rng(rng,generator)
        step, cost = state["step"], state["cost"]
    if stop_after_step is not None and stop_after_step <= step:
        raise ValueError('stop_after_step must be greater than the resumed optimizer step')
    if not resume and rank == 0:
        out.mkdir(parents=True)
        (out / "checkpoints").mkdir()
        versions = {}
        for name in ("torch", "diffusers", "transformers", "peft", "numpy", "Pillow", "datasets", "pyarrow", "tensorboard", "protobuf"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "missing"
        write_json(out / "training.json", {**identity, "run_id": run_id, "versions": versions,
                   "execution": execution, "hostname": socket.gethostname(),
                   "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu-test",
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "created_at": time.time(),
                   "world_size": world, "cuda_runtime": torch.version.cuda,
                   "git_revision": git_revision(),
                   "preprocessing": "EXIF RGB, full-image square LANCZOS resize, [-1,1]; no crop/flip",
                   "effective_batch_size_pairs": t["batch_size"] * t["gradient_accumulation_steps"] * world,
                   "effective_batch_size_pair_views": t["batch_size"] * t["gradient_accumulation_steps"] * world * t["noise_views"],
                   "noise_views": t["noise_views"],
                   "training_pairs": len(rows), "trainable_parameters": sum(p.numel() for p in parameters),
                   "planned_pair_exposures": t["max_steps"] * t["batch_size"] * t["gradient_accumulation_steps"] * world,
                   "planned_pair_view_exposures": t["max_steps"] * t["batch_size"] * t["gradient_accumulation_steps"] * world * t["noise_views"],
                   "planned_data_passes": t["max_steps"] * t["batch_size"] * t["gradient_accumulation_steps"] * world / len(rows),
                   "seed_scope": "training seed; generation seeds configured separately"})
    if world > 1:
        dist.barrier()
    writer = None
    if rank == 0 and t.get('logging', 'jsonl') not in ('none', 'jsonl'):
        from self_correct_dpo.runtime.tracking import ExperimentTracker
        writer = ExperimentTracker(config, out, run_id, start_step=step, resume=bool(source_checkpoint))
        writer.add_text("configuration", json.dumps(config, indent=2), 0)
    stopped = [False]
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    backend.train()
    optimizer.zero_grad(set_to_none=True)
    attempt = time.time_ns()
    logfile = out / (f"metrics-{attempt}.jsonl" if rank == 0 else f"rank-{rank}-{attempt}.jsonl")
    accumulation = t["gradient_accumulation_steps"]
    last_checkpoint = resume
    managed = ManagedCheckpoints(out, run_id, t) if t.get("checkpoint_policy") == "bounded" and rank == 0 else None
    if managed:
        managed.recover()
    progress_points = progress_steps(t) if t.get("checkpoint_policy") == "bounded" else []
    diagnostic = t.get("diagnostic")
    def run_diagnostics():
        from self_correct_dpo.runtime.diagnostics import dpo_diagnostic
        diagnostics = [dpo_diagnostic(backend, rows, config, out, step, device, attempt, split="train")]
        if validation_rows:
            diagnostics.append(dpo_diagnostic(backend, validation_rows, config, out, step, device, attempt, split="validation"))
        if writer:
            for record in diagnostics:
                for key,value in record["metrics"].items():
                    writer.add_scalar(f"dpo_diagnostic/{record['split']}/{key}",value,step)
            writer.flush()
    if diagnostic and rank == 0:
        run_diagnostics()
    if world > 1: dist.barrier()
    try:
        with open(logfile, "x") as log:
            while step < t["max_steps"]:
                started = time.monotonic()
                metrics_sum = {}
                for inner in range(accumulation):
                    microbatch = step * accumulation + inner
                    batch, prompts, variant_ids = sampler(rows, config, microbatch, world=world, rank=rank)
                    for other_rank in range(world):
                        exposed, _, exposed_ids = sampler(rows, config, microbatch, world=world, rank=other_rank)
                        for row, variants in zip(exposed, exposed_ids):
                            for variant in variants: cost["exposure"][row["pair_id"]][variant] += 1
                    images = image_batch(batch, t["resolution"], device, workers=config["data"].get("num_workers", 0))
                    synchronization = backend.distributed_model.no_sync() if world > 1 and inner < accumulation - 1 else nullcontext()
                    with synchronization:
                        values = backend.score_errors(images, prompts, generator)
                        from self_correct_dpo.algorithms.losses import preference_logits
                        import torch.nn.functional as F
                        logits = preference_logits(values["policy_errors"], values["reference_errors"], t["beta"])
                        mean = logits.mean(1)
                        loss = F.softplus(-mean if arm == "C" else -logits).mean()
                        metrics = {"loss": loss.detach(), "margin_mean": mean.mean(),
                                   "margin_variance": logits.var(1, unbiased=False).mean(),
                                   "implicit_accuracy": (logits > 0).float().mean(),
                                   "aggregate_accuracy": (mean > 0).float().mean(),
                                   **{f"margin_view_{j}": logits[:, j].mean() for j in range(t["noise_views"])},
                                   **{f"loss_view_{j}": F.softplus(-logits[:, j]).mean() for j in range(t["noise_views"])},
                                   **{f"policy_mse_view_{j}_{branch}": values["policy_errors"][:, j, i].mean() for j in range(t["noise_views"]) for i, branch in enumerate(("winner", "loser"))},
                                   **{f"reference_mse_view_{j}_{branch}": values["reference_errors"][:, j, i].mean() for j in range(t["noise_views"]) for i, branch in enumerate(("winner", "loser"))}}
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        # Keep every view's policy graph and differentiate the
                        # joint objective once, without a policy scoring pass.
                        (loss / accumulation).backward()
                    for key, value in {**metrics, **values.get("rollout_diagnostics", {}), "loss": loss.detach()}.items():
                        number = float(value.detach()) if hasattr(value, "detach") else float(value)
                        metrics_sum[key] = metrics_sum.get(key, 0.0) + number / accumulation
                    b, k = len(batch) * world, len(prompts[0])
                    cost["pairs_seen"] += b
                    cost["prompt_pair_evaluations"] += b * k
                    cost["policy_image_evaluations"] += 2 * b * k
                    if 'independent_rollout' in t and t['independent_rollout'].get('state_source', 'model') == 'model':
                        cost['policy_image_evaluations'] += 2 * b * k
                        cost['rollout_model_calls'] = cost.get('rollout_model_calls', 0) + k * world
                        cost['rollout_image_evaluations'] = cost.get('rollout_image_evaluations', 0) + 2 * b * k
                    cost["reference_image_evaluations"] += 0 if t["objective"] == "sft" else 2 * b * (k)
                if world > 1:
                    metrics_sum = average_metrics(metrics_sum, device)
                norm = torch.nn.utils.clip_grad_norm_(parameters, t["max_grad_norm"], error_if_nonfinite=True)
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                cost["training_seconds"] += time.monotonic() - started
                if world > 1:
                    stop_tensor = torch.tensor(int(stopped[0]), device=device)
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                    stopped[0] = bool(stop_tensor.item())
                if diagnostic and rank == 0 and (step % diagnostic["every"] == 0 or step == t["max_steps"]):
                    run_diagnostics()
                record = {"step": step, "effective_batch_size_pairs": t["batch_size"] * accumulation * world,
                          "effective_batch_size_pair_views": t["batch_size"] * accumulation * world * t["noise_views"],
                          "noise_views": t["noise_views"], "pair_views_seen": cost["prompt_pair_evaluations"],
                          "data_passes": cost["pairs_seen"] / len(rows), "run_id": run_id, "attempt": attempt, **metrics_sum,
                          "grad_norm": float(norm), "learning_rate": used_lr, "next_learning_rate": scheduler.get_last_lr()[0],
                          "last_microbatch_pair_ids": [r["pair_id"] for r in batch],
                          "last_microbatch_variant_ids": variant_ids,
                          "cost": {k: v for k, v in cost.items() if k != "exposure"},
                          "exposure": exposure_summary(cost["exposure"])}
                if device.startswith("cuda"):
                    record["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                if rank == 0:
                    print(json.dumps({k: record[k] for k in ("step", "loss", "grad_norm", "learning_rate")}), flush=True)
                    if writer:
                        for key,value in {**metrics_sum,"grad_norm":float(norm),"learning_rate":used_lr,"pairs_seen":cost["pairs_seen"],
                                          "effective_batch_size_pair_views":record["effective_batch_size_pair_views"],
                                          "noise_views":t["noise_views"],"pair_views_seen":cost["prompt_pair_evaluations"],
                                          "data_passes":cost["pairs_seen"]/len(rows),"seconds_per_update":time.monotonic()-started}.items():
                            writer.add_scalar("training/"+key,value,step)
                        if device.startswith("cuda"):writer.add_scalar("resources/peak_cuda_GiB",record["peak_cuda_bytes"]/1024**3,step)
                        writer.flush()
                stage_end = stop_after_step is not None and step >= stop_after_step
                saving = step % t["checkpoint_every"] == 0 or step in progress_points or step == t["max_steps"] or stopped[0] or stage_end
                if world > 1 and saving:
                    backend.resume_rank_states = gather_rng(generator)
                if managed:
                    if step in managed.steps:
                        managed.save_progress(backend, optimizer, scheduler, generator, step, cost)
                    if step % t["checkpoint_every"] == 0 or step in managed.steps or step == t["max_steps"] or stopped[0] or stage_end:
                        last_checkpoint = managed.save_resume(backend, optimizer, scheduler, generator, step, cost)
                elif rank == 0 and (step % t["checkpoint_every"] == 0 or step == t["max_steps"] or stopped[0] or stage_end):
                    last_checkpoint = out / "checkpoints" / f"step-{step:07d}"
                    save_checkpoint(last_checkpoint, backend, optimizer, scheduler, generator, step, run_id, cost)
                if world > 1 and (saving or (diagnostic and step % diagnostic["every"] == 0)):
                    dist.barrier()
                    if saving:
                        last_checkpoint=resolve_checkpoint(out/'checkpoints/latest.json') if t.get('checkpoint_policy')=='bounded' else out/'checkpoints'/f'step-{step:07d}'
                if stopped[0] or stage_end:
                    break
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if writer:
            writer.close()
    sessions = out / "sessions"
    if rank == 0: sessions.mkdir(exist_ok=True)
    if world > 1: dist.barrier()
    write_json(sessions / f"rank-{rank}-{attempt}.json", {"resume": str(resume) if resume else None,
        "extend_from": str(extend_from) if extend_from else None, "elapsed_seconds": time.monotonic() - session_started,
        "optimization_seconds_cumulative": cost["training_seconds"], "end_step": step,
        "stop_after_step": stop_after_step,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    result = {"run_id": run_id, "step": step, "checkpoint": str(last_checkpoint), "cost": cost,
              "interrupted":bool(stopped[0]),
              "complete": step == t["max_steps"] and not stopped[0],
              "planned_pause": stop_after_step is not None and step == stop_after_step and step < t["max_steps"] and not stopped[0]}
    if result["complete"] and rank == 0:
        write_json(out / "complete.json", result)
    return result
