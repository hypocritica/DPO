"""Render paired SD1.5, SDXL or SD3.5 Medium experiments from local weights.

Resume checks immutable run/config/model identities and every committed image.
Sharding assigns whole scenes, never individual members of a prompt orbit.
"""
from self_correct_dpo.common import source_file
from self_correct_dpo.config import load_config
import argparse
import errno
from contextlib import contextmanager
import fcntl
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time
import uuid

from self_correct_dpo.common import canonical_scheduler_config
from self_correct_dpo.common import (artifact_identity, model_artifact_identity, digest, load_suite, read_json, require_gpu_allocation,
                    sha256, verify_images, write_json)
from self_correct_dpo.runtime.storage_compat import compatible_sources, record_compatibility

FAMILIES = {"sd15": ("StableDiffusionPipeline", "DDIMScheduler", "unet"),
            "sdxl": ("StableDiffusionXLPipeline", "EulerDiscreteScheduler", "unet"),
            "sd35_medium": ("StableDiffusion3Pipeline", "FlowMatchEulerDiscreteScheduler", "transformer")}


@contextmanager
def render_lock(root):
    with open(root / "render.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is rendering or resuming this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def model_spec(config):
    if "model" in config:
        model = config["model"]
        if not model.get("path") or not model.get("revision"):
            raise ValueError("Set model.path and a traceable model.revision")
        result={"family": model.get("family", "sdxl"), "path": model["path"],
                "revision": model["revision"]}
        if model.get('vae'):result['vae']=dict(model['vae'])
        return result
    return {"family": "sdxl", "path": config["base_model"],
            "revision": config.get("base_revision")}


def effective_sampling(config):
    sampling = dict(config["sampling"])
    if model_spec(config)["family"] == "sd35_medium":
        model_length = config.get("model", {}).get("max_sequence_length")
        if model_length is not None and "max_sequence_length" in sampling and model_length != sampling["max_sequence_length"]:
            raise ValueError("model and sampling max_sequence_length disagree")
        sampling.setdefault("max_sequence_length", model_length if model_length is not None else 256)
    return sampling


def validate_sd35_medium_config(config):
    # Keep this CPU-only preflight aligned with robustdpo.models' runtime validator.
    expected = {"num_layers": 24, "num_attention_heads": 24,
                "qk_norm": "rms_norm", "pos_embed_max_size": 384}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"sd35_medium architecture mismatch: {key}={config.get(key)!r}, expected {value!r}")
    if list(config.get("dual_attention_layers", [])) != list(range(13)):
        raise ValueError("sd35_medium requires dual_attention_layers=0..12")


def validate_adapter_backend(path, model, sampling, component):
    sidecar = path / "backend.json"
    if not sidecar.is_file():
        raise ValueError("PEFT inference requires backend.json with verified model family/base revision provenance")
    backend = read_json(sidecar)
    trained = backend.get("model", {})
    if backend.get("family") != model["family"] or trained.get("family") != model["family"]:
        raise ValueError("Adapter backend model family mismatch")
    if backend.get("component") != component:
        raise ValueError("Adapter backend denoiser component mismatch")
    if not model.get("revision") or trained.get("revision") != model["revision"]:
        raise ValueError("Adapter/base model revision mismatch")
    from self_correct_dpo.architecture import same_vae_revision
    if not same_vae_revision(trained,model):
        raise ValueError('Adapter/render VAE revision mismatch')
    if model["family"] == "sd35_medium":
        if trained.get("max_sequence_length", 256) != sampling["max_sequence_length"]:
            raise ValueError("Adapter/render T5 max_sequence_length mismatch")
        if backend.get("error_parameterization") != "flow_velocity_mse_uniform_schedule_index":
            raise ValueError("SD3.5 adapter was not trained with the supported flow velocity parameterization")
    elif backend.get("error_parameterization") not in {"diffusion_epsilon_mse", "diffusion_v_prediction_mse"}:
        raise ValueError("Diffusion adapter has incompatible denoising parameterization")


def validate_config(config, arm):
    model = model_spec(config)
    if model["family"] not in FAMILIES:
        raise ValueError("model.family must be sd15, sdxl or sd35_medium")
    pipeline_class, scheduler_class, component = FAMILIES[model["family"]]
    sampling = effective_sampling(config)
    seeds = sampling["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Use at least one distinct, nonnegative integer seed")
    allowed_schedulers = {'DDIMScheduler', 'PNDMScheduler'} if model['family']=='sd15' else {'EulerDiscreteScheduler','DDIMScheduler'} if model['family']=='sdxl' else {scheduler_class}
    if sampling['scheduler'] not in allowed_schedulers:
        raise ValueError(f"Unsupported scheduler for {model['family']}: {sampling['scheduler']}")
    if sampling["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported dtype")
    multiple = 16 if model["family"] == "sd35_medium" else 8
    for name in ("height", "width"):
        if type(sampling[name]) is not int or sampling[name] < 64 or sampling[name] % multiple:
            raise ValueError(f"Image dimensions must be at least 64 and multiples of {multiple}")
    if type(sampling["num_inference_steps"]) is not int or sampling["num_inference_steps"] < 1:
        raise ValueError("num_inference_steps must be a positive integer")
    scale = sampling["guidance_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    if not isinstance(sampling["negative_prompt"], str):
        raise ValueError("negative_prompt must be a string")
    max_length = sampling.get("max_sequence_length", 256)
    if model["family"] == "sd35_medium":
        if type(max_length) is not int or not 1 <= max_length <= 512:
            raise ValueError("SD3 max_sequence_length must be an integer in [1, 512]")
    elif "max_sequence_length" in sampling:
        raise ValueError("max_sequence_length is an SD3-only sampler option")
    index_path = Path(model["path"]) / "model_index.json"
    if not index_path.is_file():
        raise ValueError("Model path must be a local Diffusers pipeline")
    index = read_json(index_path)
    if index.get("_class_name") != pipeline_class:
        raise ValueError(f"Model index does not declare {pipeline_class}")
    if model["family"] == "sd15":
        unet = read_json(index_path.parent / "unet/config.json")
        if unet.get("cross_attention_dim") != 768 or unet.get("in_channels") != 4:
            raise ValueError("SD1.5 requires 768-dimensional conditioning and four latent channels")
    from self_correct_dpo.architecture import validate_sdxl_base, validate_vae_override
    validate_vae_override(model)
    if model['family']=='sdxl':validate_sdxl_base(model['path'])
    if model["family"] == "sd35_medium":
        transformer_config = index_path.parent / "transformer/config.json"
        if not transformer_config.is_file():
            raise ValueError("SD3.5 pipeline must include transformer/config.json")
        validate_sd35_medium_config(read_json(transformer_config))
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3",
                     "tokenizer", "tokenizer_2", "tokenizer_3"):
            if not index.get(name) or index[name][-1] is None or not (index_path.parent / name).is_dir():
                raise ValueError(f"SD3.5 requires all three text encoders/tokenizers: missing {name}")
    if arm == "aligned":
        aligned = config["aligned"]
        if aligned["kind"] not in {component, "peft"}:
            raise ValueError(f"{model['family']} aligned.kind must be {component} or peft")
        if not aligned.get("path") or not aligned.get("revision"):
            raise ValueError("Set aligned.path and a traceable aligned.revision")
        path = Path(aligned["path"])
        if not path.is_dir():
            raise ValueError("aligned.path must be an existing local directory")
        if path.resolve() in {Path(model["path"]).resolve(), (Path(model["path"]) / component).resolve()}:
            raise ValueError("Aligned model points to the base model")
        if aligned["kind"] == "peft":
            if not (path / "adapter_config.json").is_file():
                raise ValueError("PEFT directory must contain adapter_config.json")
            if read_json(path / "adapter_config.json").get("peft_type") != "LORA":
                raise ValueError("Only standard PEFT LoRA adapters are supported")
            if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
                raise ValueError("Missing standard PEFT adapter_model weights")
            validate_adapter_backend(path, model, sampling, component)
        elif model["family"] == "sd35_medium":
            validate_sd35_medium_config(read_json(path / "config.json"))
    return model


def select_shard(suite, index=0, count=1):
    scenes = sorted({row["scene_id"] for row in suite})
    if type(count) is not int or count < 1 or count > len(scenes):
        raise ValueError("shard-count must be between 1 and the number of scenes")
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("shard-index must be in [0, shard-count)")
    selected = set(scenes[index::count])
    return [row for row in suite if row["scene_id"] in selected]


def check_dependencies(family, peft=False):
    names = ["torch", "diffusers", "transformers", "safetensors"]
    if family == "sd35_medium":
        names += ["sentencepiece"]
    if peft or importlib.util.find_spec("peft") is not None:
        names += ["peft"]
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing render dependencies: " + ", ".join(missing))
    return {name: importlib.metadata.version(name) for name in names}


def check_prompt_lengths(pipe, suite, sampling, family):
    tokenizers = [("CLIP-1", pipe.tokenizer, pipe.tokenizer.model_max_length)]
    if family != "sd15":
        tokenizers.append(("CLIP-2", pipe.tokenizer_2, pipe.tokenizer_2.model_max_length))
    if family == "sd35_medium":
        if pipe.text_encoder_3 is None or pipe.tokenizer_3 is None:
            raise ValueError("SD3.5 T5 text encoder and tokenizer must not be disabled")
        tokenizers.append(("T5", pipe.tokenizer_3, sampling.get("max_sequence_length", 256)))
    prompts = [(f"{r['scene_id']} / {r['variant_id']}", r["prompt"]) for r in suite]
    prompts.append(("negative_prompt", sampling["negative_prompt"]))
    truncated=[]
    policy=sampling.get('prompt_truncation','error')
    if policy not in ('error','truncate'):
        raise ValueError('prompt_truncation must be error or truncate')
    for label, prompt in prompts:
        for name, tokenizer, limit in tokenizers:
            length = len(tokenizer(prompt, truncation=False).input_ids)
            if length > limit:
                if policy=='error':
                    raise ValueError(f"Prompt truncation in {name}: {label} ({length} > {limit})")
                truncated.append({'prompt':label,'tokenizer':name,'tokens':length,'limit':limit})
    return truncated


def load_pipeline(config, arm, torch, *, device='cuda'):
    import diffusers
    model = model_spec(config)
    pipeline_name, scheduler_name, component = FAMILIES[model["family"]]
    kwargs = {"local_files_only": True, "torch_dtype": getattr(torch, config["sampling"]["dtype"])}
    if model['family']=='sd15' and config['sampling'].get('safety_checker')=='disabled':
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    if arm == "aligned" and config["aligned"]["kind"] == component:
        cls = diffusers.UNet2DConditionModel if component == "unet" else diffusers.SD3Transformer2DModel
        kwargs[component] = cls.from_pretrained(config["aligned"]["path"], **kwargs)
    if model["family"] == "sdxl":
        kwargs["add_watermarker"] = False
        if model.get('vae'):
            kwargs['vae']=diffusers.AutoencoderKL.from_pretrained(model['vae']['path'],
                local_files_only=True,torch_dtype=getattr(torch,config['sampling']['dtype']))
    pipe = getattr(diffusers, pipeline_name).from_pretrained(model["path"], **kwargs)
    if arm == "aligned" and config["aligned"]["kind"] == "peft":
        from peft import PeftModel
        denoiser = PeftModel.from_pretrained(getattr(pipe, component), config["aligned"]["path"],
                                           is_trainable=False, local_files_only=True)
        if not any("lora_" in name for name, _ in denoiser.named_parameters()):
            raise ValueError("No LoRA parameters loaded")
        setattr(pipe, component, denoiser)
    pipe.scheduler = getattr(diffusers, config["sampling"].get("scheduler",scheduler_name)).from_config(pipe.scheduler.config)
    getattr(pipe, component).eval()
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.eval()
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def expected_records(run):
    records = {}
    for index, row in enumerate(run["suite"]):
        for seed_index, seed in enumerate(run["sampling"]["seeds"]):
            key = (row["scene_id"], row["variant_id"], seed)
            records[key] = {"scene_id": row["scene_id"], "variant_id": row["variant_id"],
                            "tag": row["tag"], "seed": seed, "prompt": row["prompt"],
                            "image": f"images/{index:05d}/samples/{seed_index:04d}.png"}
    return records


def validate_record(record, root, run, expected, *, image_path=None):
    key = (record["scene_id"], record["variant_id"], record["seed"])
    if key not in expected or any(record.get(k) != v for k, v in expected[key].items()):
        raise ValueError("Resume manifest does not match the recorded paired design")
    payload = {k: v for k, v in record.items() if k != "image_id"}
    if record["image_id"] != digest({"run_id": run["run_id"], **payload}):
        raise ValueError("Resume manifest image identity mismatch")
    path = Path(image_path) if image_path is not None else root / record["image"]
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != record["image_sha256"]:
        raise ValueError(f"Resume image checksum/path mismatch: {record['image']}")
    return key


def read_resume_manifest(root, run, *, repair=False):
    """Preserve and remove only an incomplete final JSON line; never repair corruption."""
    path = root / "manifest.jsonl"
    content = path.read_bytes() if path.exists() else b""
    records, offset, bad_tail = [], 0, b""
    expected = expected_records(run)
    seen = set()
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Corrupt manifest JSON; only an unfinished final line is recoverable")
            bad_tail = line
            break
        key = validate_record(record, root, run, expected)
        if key in seen:
            raise ValueError("Duplicate resume manifest record")
        seen.add(key)
        records.append(record)
        offset += len(line)
    if repair and bad_tail:
        recovery = root / "recovery"
        recovery.mkdir(exist_ok=True)
        backup = recovery / f"manifest-tail-{uuid.uuid4().hex}.bin"
        with open(backup, "xb") as f:
            f.write(bad_tail)
        with open(path, "r+b") as f:
            f.truncate(offset)
            f.flush()
            os.fsync(f.fileno())
        print(f"Preserved incomplete manifest tail at {backup}; resuming committed images", flush=True)
    if repair and offset and not content[:offset].endswith(b"\n"):
        with open(path, "ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
    return records, bool(bad_tail)


def append_record(manifest, record):
    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest.flush()
    os.fsync(manifest.fileno())


def publish_image(staged, destination):
    """Publish under render_lock; AOSS supports rename but not hard links.

    The journal and flushed PNG already exist. A rename interrupted before the
    manifest append is recovered from the destination by recover_transactions.
    The output lock serializes the existence check and rename for this renderer.
    """
    try:
        os.link(staged, destination)
    except OSError as error:
        if error.errno not in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        if os.path.lexists(destination):
            raise FileExistsError(errno.EEXIST, 'Refusing to overwrite a rendered image', str(destination)) from error
        os.rename(staged, destination)


def recover_transactions(root, run, records, manifest):
    """Finish image commits journaled before a crash; never overwrite an image."""
    expected = expected_records(run)
    by_key = {(r["scene_id"], r["variant_id"], r["seed"]): r for r in records}
    for journal in sorted((root / "pending").glob("*.json")):
        transaction = read_json(journal)
        record = transaction["record"]
        staged = root / transaction["staged_image"]
        destination = root / record["image"]
        if not staged.resolve().is_relative_to((root / "pending").resolve()):
            raise ValueError("Unsafe pending transaction path")
        available = destination if destination.exists() else staged
        key = validate_record(record, root, run, expected, image_path=available)
        if key in by_key:
            if record != by_key[key]:
                raise ValueError("Pending transaction disagrees with committed manifest")
        else:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                publish_image(staged, destination)
            append_record(manifest, record)
            records.append(record)
            by_key[key] = record
        if staged.exists():
            staged.unlink()
        journal.unlink()
    return by_key


def commit_image(image, root, run, record, manifest):
    pending = root / "pending"
    pending.mkdir(exist_ok=True)
    transaction_id = uuid.uuid4().hex
    staged = pending / f"{transaction_id}.png"
    with open(staged, "xb") as f:
        image.save(f, format="PNG")
        f.flush()
        os.fsync(f.fileno())
    record = {**record, "image_sha256": sha256(staged)}
    record["image_id"] = digest({"run_id": run["run_id"], **record})
    journal = pending / f"{transaction_id}.json"
    write_json(journal, {"record": record, "staged_image": str(staged.relative_to(root))})
    publish_image(staged, root / record["image"])
    append_record(manifest, record)
    staged.unlink(missing_ok=True)
    journal.unlink()
    return record


def validate_resume_run(root, *, config, arm, suite, full_suite_id, shard, base_identity, model_identity, suite_indices=None):
    run = read_json(root / "run.json")
    if run["run_id"] != digest({k: v for k, v in run.items() if k != "run_id"}):
        raise ValueError("Run metadata checksum mismatch")
    wanted = {"config": config, "arm": arm, "suite": suite, "suite_id": digest(suite),
              "full_suite_id": full_suite_id, "shard": shard, "suite_indices": suite_indices, "base_identity": base_identity,
              "model_identity": model_identity, "render_sha256": sha256(__file__),
              "common_sha256": sha256(source_file("common.py"))}
    for key, value in wanted.items():
        if run.get(key) != value:
            if key == 'render_sha256' and compatible_sources({'render.py': run.get(key)}, {'render.py': value}):
                continue
            raise ValueError(f"Cannot resume: {key} changed")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--arm", choices=["base", "aligned"], required=True)
    parser.add_argument("--out", required=True, help="Unique run directory; use --resume to continue it")
    parser.add_argument("--check-only", action="store_true", help="CPU preflight, including dependency discovery")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--execution", choices=["slurm", "standalone"], default="slurm")
    parser.add_argument("--run-label", help="Required for explicitly authorized standalone GPU execution")
    args = parser.parse_args()
    if args.execution == "standalone" and not args.run_label:
        parser.error("--execution standalone requires --run-label")
    config, full_suite = load_config(args.config), load_suite(args.prompts)
    model = validate_config(config, args.arm)
    suite = select_shard(full_suite, args.shard_index, args.shard_count)
    selected_scenes = {row["scene_id"] for row in suite}
    suite_indices = [i for i, row in enumerate(full_suite) if row["scene_id"] in selected_scenes]
    shard = {"index": args.shard_index, "count": args.shard_count,
             "assignment": "sorted-scene-round-robin-v1"}
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f"Use a new run directory or --resume: {out}")
    if args.resume and not out.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {out}")
    sampling = effective_sampling(config)
    expected = len(suite) * len(sampling["seeds"])
    base_identity = model_artifact_identity(model)
    model_identity = base_identity if args.arm == "base" else artifact_identity(config["aligned"]["path"])
    run = None
    records = []
    if args.resume:
        run = validate_resume_run(out, config=config, arm=args.arm, suite=suite,
                                  full_suite_id=digest(full_suite), shard=shard,
                                  base_identity=base_identity, model_identity=model_identity, suite_indices=suite_indices)
        records, _ = read_resume_manifest(out, run)
        if (out / "complete.json").exists():
            verify_images(out)
            print(f"Already complete and verified: {out}")
            return
    versions = check_dependencies(model["family"], args.arm == "aligned" and config["aligned"]["kind"] == "peft")
    print(json.dumps({"arm": args.arm, "family": model["family"], "images": expected,
                      "committed_images": len(records), "sampling": sampling, "shard": shard,
                      "output": str(out), "dependencies": versions}))
    if args.check_only:
        return
    execution = {"mode": args.execution, "run_label": args.run_label}
    require_gpu_allocation(execution)
    ancestor = out.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    needed = 10 * 1024**3 + (expected - len(records)) * sampling["height"] * sampling["width"] * 4
    if shutil.disk_usage(ancestor).free < needed:
        raise RuntimeError("Insufficient output filesystem space for images plus 10 GiB margin")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    started = time.time()
    pipe = load_pipeline(config, args.arm, torch)
    truncated=check_prompt_lengths(pipe, suite, sampling, model["family"])
    scheduler_config = canonical_scheduler_config(pipe.scheduler.config)
    if run is not None:
        if run["versions"] != versions or canonical_scheduler_config(run["scheduler_config"]) != scheduler_config:
            raise ValueError("Cannot resume with changed package versions or scheduler configuration")
    else:
        run = {"arm": args.arm, "config": config, "model": model, "suite": suite,
               "suite_id": digest(suite), "full_suite_id": digest(full_suite), "shard": shard, "suite_indices": suite_indices,
               "sampling": sampling, "base_identity": base_identity, "model_identity": model_identity,
               "scheduler_config": scheduler_config, "versions": versions, "execution": execution,
               "truncated_prompts":truncated, "gpu": torch.cuda.get_device_name(), "hostname": socket.gethostname(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "expected_images": expected,
               "render_sha256": sha256(__file__), "common_sha256": sha256(source_file("common.py"))}
        run["run_id"] = digest(run)
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "run.json", run)
    with render_lock(out):
        if args.resume:
            record_compatibility(out, {'render.py': run['render_sha256']}, {'render.py': sha256(__file__)})
        if (out / "complete.json").exists():
            verify_images(out)
            print(f"Already complete and verified: {out}")
            return
        records, _ = read_resume_manifest(out, run, repair=True)
        initial_images = len(records)
        with open(out / "manifest.jsonl", "a") as manifest:
            completed = recover_transactions(out, run, records, manifest)
            expected_map = expected_records(run)
            for key, spec in expected_map.items():
                if (out / spec["image"]).exists() and key not in completed:
                    raise ValueError(f"Uncommitted image without a recovery journal: {spec['image']}")
            for index, row in enumerate(suite):
                folder = out / "images" / f"{index:05d}"
                (folder / "samples").mkdir(parents=True, exist_ok=True)
                metadata = folder / "metadata.jsonl"
                if metadata.exists():
                    if read_json(metadata) != row["metadata"]:
                        raise ValueError("Canonical scene metadata changed")
                else:
                    write_json(metadata, row["metadata"])
                for seed in sampling["seeds"]:
                    key = (row["scene_id"], row["variant_id"], seed)
                    if key in completed:
                        continue
                    kwargs = {"prompt": row["prompt"], "negative_prompt": sampling["negative_prompt"],
                              "height": sampling["height"], "width": sampling["width"],
                              "num_inference_steps": sampling["num_inference_steps"],
                              "guidance_scale": sampling["guidance_scale"],
                              "generator": torch.Generator("cuda").manual_seed(seed)}
                    if model["family"] == "sd35_medium":
                        kwargs["max_sequence_length"] = sampling.get("max_sequence_length", 256)
                    torch.cuda.synchronize()
                    image_started = time.perf_counter()
                    with torch.inference_mode():
                        image = pipe(**kwargs).images[0]
                    torch.cuda.synchronize()
                    record = {**expected_map[key], "generation_seconds": time.perf_counter() - image_started}
                    completed[key] = commit_image(image, out, run, record, manifest)
                print(f"{args.arm}: {index + 1}/{len(suite)} prompts", flush=True)
        sessions = out / "sessions"
        sessions.mkdir(exist_ok=True)
        write_json(sessions / f"{uuid.uuid4().hex}.json", {
            "execution": execution, "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_unix": started, "elapsed_seconds": time.time() - started,
            "resumed": args.resume, "images_at_start": initial_images, "images_at_end": len(completed),
            "gpu": torch.cuda.get_device_name(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        write_json(out / "complete.json", {"images": expected, "manifest_sha256": sha256(out / "manifest.jsonl")})
        verify_images(out)


if __name__ == "__main__":
    main()
