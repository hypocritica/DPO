"""Fixed-checkpoint diffusion and Flow-DPO diagnostics for the MVPO appendix."""
import argparse
from contextlib import nullcontext
import hashlib
import heapq
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

from self_correct_dpo.common import digest, model_artifact_identity, output_lock, read_json, sha256
from self_correct_dpo.evaluation.checkpoint_catalog import resolve_checkpoint
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json


CHECKPOINTS = {
    "sd15": (
        "MVPO_APPENDIX_SD15_DPO_RUN", 1000,
        "11c4dbedab00cd8d6ad32f9299733a61cf9766a203b9e9f6e7a07f0218c817af",
    ),
    "sd35_medium": (
        "MVPO_APPENDIX_SD35_DPO_RUN", 1600,
        "3d6cc23e4e808ec0d131a840bcccdda14a5b7e4f35ab24bd9ba9dbea314d8fff",
    ),
}


def load_checkpoint(family, path=None, resources=None, expected=None, method=None):
    resource, step, expected_id = CHECKPOINTS[family]
    if expected is not None:
        step, expected_id = expected["step"], expected["checkpoint_id"]
    checkpoint_name = f"step-{step:07d}"
    path = (Path(path).expanduser() if path else
            Path(load_resources(resources)[resource]) / "checkpoints" / checkpoint_name)
    if path.name != checkpoint_name or path.parent.name != "checkpoints":
        raise ValueError(f"Expected the retained checkpoint directory checkpoints/{checkpoint_name}: {path}")
    run = path.parent.parent.resolve()
    model, adapter, identity = resolve_checkpoint(run, step)
    if identity["checkpoint_id"] != expected_id or model["family"] != family:
        raise ValueError(f"Wrong checkpoint identity: {identity['checkpoint_id']}, expected {expected_id}")

    if expected is not None and identity["run_id"] != expected["run_id"]:
        raise ValueError("Checkpoint run differs from frozen scan catalog")
    training = read_json(run / "training.json")
    keys = ("config", "pairs_id", "base_identity", "code", "validation_pairs_id",
            "monitor_inputs", "diagnostic_suite_sha256", "parent")
    if digest({key: training[key] for key in keys if key in training}) != identity["run_id"]:
        raise ValueError(f"Training metadata does not match the checkpoint run ID: {run}")
    settings = training["config"]["training"]
    if method is not None:
        from self_correct_dpo.evaluation.experiment_catalog import validate_method
        from self_correct_dpo.evaluation.appendix_native_gradients import ARCHIVED_OBJECTIVES_SHA, objective_spec
        if expected is None or family != "sd15" or method not in ("dspo", "dmpo", "stagewise_dspo_k6", "stagewise_dmpo_k6"):
            raise ValueError("native method requires a frozen SD1.5 checkpoint catalog")
        validate_method(training["config"], method, str(run))
        objective_spec(settings)
        if training["code"].get("objectives.py") != ARCHIVED_OBJECTIVES_SHA:
            raise ValueError("native loss source differs from the audited training implementation")
        if settings["prompt_mode"] != "canonical" or settings.get("reference_mode", "base") != "base":
            raise ValueError("native probe requires canonical prompts and frozen base reference")
    elif (settings["objective"] != "dpo" or settings["noise_views"] != 1
            or settings["prompt_mode"] != "canonical" or settings.get("reference_mode", "base") != "base"
            or settings.get("stagewise") or settings.get("independent_rollout")):
        raise ValueError("Expected canonical single-view DPO training with a frozen base reference")
    if model_artifact_identity(model) != training["base_identity"]:
        raise ValueError(f"Base model inventory differs from the checkpoint training record: {model['path']}")
    backend_metadata = read_json(adapter / "backend.json")
    if backend_metadata["training"] != settings or backend_metadata["model"] != model:
        raise ValueError("Adapter backend settings disagree with immutable training metadata")
    manifest = read_json(adapter.parent / "checkpoint.json")
    identity.update(
        path=str(adapter.parent), adapter_path=str(adapter),
        adapter_sha256=manifest["files"]["adapter/adapter_model.safetensors"],
        adapter_config_sha256=manifest["files"]["adapter/adapter_config.json"],
        adapter_backend_sha256=manifest["files"]["adapter/backend.json"],
    )
    return model, adapter, training, identity


def manifest_stat(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def choose_pairs(training, manifest, count, seed, excluded_pair_ids=()):
    from self_correct_dpo.data.pairs import semantic_scene_key, validate_pair_rows
    from self_correct_dpo.training import check_rows, eligible_prompts

    config = training["config"]
    data = config["data"]
    split = "train"
    if data["split"] != split:
        raise ValueError("The diagnostic is defined on the checkpoint's training split")
    if ((manifest.parent / "preparing.json").exists() and not (manifest.parent / "ready.json").exists()):
        raise ValueError(f"Training data preparation did not finish: {manifest.parent}")

    # Flow receipts bound the raw manifest bytes; the older DDPM run bound
    # the JSON digest of full normalized training rows. Hash either form
    # while retaining only the preselected pairs in memory.
    receipt_backed = bool(data.get("validation_report"))
    before = manifest_stat(manifest)
    raw, normalized = hashlib.sha256(), hashlib.sha256(b"[")
    training_pairs = 0
    excluded_pair_ids = set(excluded_pair_ids)

    def training_rows():
        nonlocal training_pairs
        with manifest.open("rb") as handle:
            for line in handle:
                raw.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] != split:
                    continue
                if not receipt_backed:
                    if row.get("storage", {}).get("kind") != "arrow":
                        for side in ("winner", "loser"):
                            # The legacy startup uses lexical absolute paths,
                            # preserving symlinks and avoiding per-image stat.
                            path = Path(row[side]).expanduser()
                            row[side] = os.path.abspath(manifest.parent / path)
                    row["semantic_scene_key"] = semantic_scene_key(row)
                    if training_pairs:
                        normalized.update(b", ")
                    normalized.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
                training_pairs += 1
                if row["pair_id"] not in excluded_pair_ids:
                    yield row

    rows = heapq.nsmallest(count, training_rows(), key=lambda row: (digest([seed, row["pair_id"]]), row["pair_id"]))
    raw_sha = raw.hexdigest()
    normalized.update(b"]")
    pairs_id = digest({"manifest_sha256": raw_sha, "split": split}) if receipt_backed else normalized.hexdigest()
    if before != manifest_stat(manifest):
        raise ValueError(f"Pair manifest changed during verification: {manifest}")
    if pairs_id != training["pairs_id"] or training_pairs != training["training_pairs"]:
        raise ValueError(f"Pair manifest does not match the checkpoint training split: {manifest}")
    if len(rows) != count:
        raise ValueError(f"Need exactly {count} pairs, found {len(rows)}")
    rows = validate_pair_rows(rows, manifest.parent, verify_files=False, resolve_paths=False, in_place=True)
    check_rows(rows, config)
    prompts = [next(p["prompt"] for p in eligible_prompts(row, data)
                    if p["variant_id"] == "canonical") for row in rows]
    identity = {
        "path": str(manifest), "manifest_sha256": raw_sha, "pairs_id": pairs_id,
        "identity_method": "manifest_sha256_and_split" if receipt_backed else "normalized_training_rows",
        "split": split, "split_size": training_pairs, "manifest_stat": before,
    }
    return rows, prompts, identity


def seed_for(*parts):
    return int(digest(["mvpo-appendix-v1", *parts])[:16], 16) % (2 ** 63)


def conditions_for(family):
    if family == "sd15":
        return [{"condition": "sd15_t499", "index_start": 499, "index_stop": 500, "probability": 1.0}]
    return [
        {"condition": "sd35_stage0", "stage_index": 0, "index_start": 0,
         "index_stop": 500, "probability": 0.5},
        {"condition": "sd35_stage1", "stage_index": 1, "index_start": 500,
         "index_stop": 1000, "probability": 0.5},
    ]


def schedule_identity(backend):
    import torch

    scheduler = backend.scheduler
    total = int(scheduler.config.num_train_timesteps)
    if total != 1000:
        raise ValueError(f"Expected 1000 training schedule indices, received {total}")
    if backend.family == "sd15":
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("Expected the archived SD1.5 epsilon-prediction scheduler")
        return {"axis": "diffusion_timestep", "num_train_timesteps": total, "prediction_type": "epsilon"}
    sigmas = scheduler.sigmas.detach().cpu().double()
    timesteps = scheduler.timesteps.detach().cpu().double()
    unshifted = 1 - torch.arange(total, dtype=torch.float64) / total
    shifted = 3 * unshifted / (1 + 2 * unshifted)
    if (float(scheduler.config.shift) != 3 or timesteps.shape != shifted.shape
            or not torch.allclose(sigmas[:total], shifted, atol=2e-7, rtol=0)
            or not torch.allclose(timesteps / total, shifted, atol=2e-7, rtol=0)):
        raise ValueError("SD3.5 scheduler differs from the archived static shift-3 training schedule")
    return {
        "axis": "descending_flow_schedule_index", "num_train_timesteps": total,
        "shift": 3, "boundary_index": 500, "boundary_sigma": float(sigmas[500]),
        "native_timesteps": timesteps.tolist(), "native_sigmas": sigmas[:total].tolist(),
        "interpolation_dtype": str(backend.dtype).removeprefix("torch."),
    }


class PairProbe:
    """A fixed posterior sample and fixed canonical text across a pair's views."""

    def __init__(self, backend, parameters, beta, row, prompt, latent_seed):
        import torch
        from self_correct_dpo.training import image_batch

        self.backend = backend
        self.parameters = parameters
        self.beta = beta
        self.latents = backend._encode_images(
            image_batch([row], backend.resolution, backend.device),
            torch.Generator(device=backend.device).manual_seed(latent_seed),
        )
        self.condition = backend._condition([prompt, prompt])
        self.truncations = dict(backend.last_prompt_truncations)

    def margins(self, condition, seeds):
        import torch
        from self_correct_dpo.algorithms.formal import _repeat_condition
        from self_correct_dpo.algorithms.losses import preference_logits

        start, stop = condition["index_start"], condition["index_stop"]
        noise, indices = [], []
        latent = self.latents[0, 0]
        for seed in seeds:
            generator = torch.Generator(device=latent.device).manual_seed(seed)
            noise.append(torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator))
            index = (torch.full((1,), start, device=latent.device, dtype=torch.long) if stop - start == 1
                     else torch.randint(start, stop, (1,), device=latent.device, generator=generator))
            indices.append(index)
        count = len(seeds)
        latents = self.latents.expand(count, *self.latents.shape[1:])
        indices = torch.cat(indices)
        noisy, timestep, target = self.backend.pair_view_inputs(
            latents, torch.stack(noise)[:, None].expand_as(latents), indices[:, None].expand(-1, 2),
        )
        text = self.condition if count == 1 else _repeat_condition(self.condition, count)
        reference = self.backend.pair_mse(noisy, timestep, target, text, reference=True)
        policy = self.backend.pair_mse(noisy, timestep, target, text)
        return preference_logits(policy, reference, self.beta).flatten(), indices

    def gradient(self, condition, seed):
        import torch

        margins, indices = self.margins(condition, [seed])
        gradients = torch.autograd.grad(margins[0], self.parameters)
        return float(margins[0].detach()), torch.cat([g.detach().reshape(-1).float() for g in gradients]), int(indices[0])

    def gradients(self, condition, seeds):
        """Batch the forward only; each yielded gradient belongs to one margin."""
        import torch

        margins, indices = self.margins(condition, seeds)
        for i in range(len(seeds)):
            grads = torch.autograd.grad(margins[i], self.parameters, retain_graph=i + 1 < len(seeds))
            yield float(margins[i].detach()), torch.cat([g.detach().reshape(-1).float() for g in grads]), int(indices[i])


def batch_check(probe, conditions, size, seed):
    """Record batch-vs-single numerical drift and exact-batch repeatability."""
    import torch

    checks = []
    for condition in conditions:
        seeds = [seed_for("batch-check", seed, condition["condition"], i) for i in range(size)]
        single = [(m, g.cpu(), index) for m, g, index in (probe.gradient(condition, s) for s in seeds)]
        batched = [(m, g.cpu(), index) for m, g, index in probe.gradients(condition, seeds)]
        repeated = probe.gradients(condition, seeds)
        for i, ((m, g, index), (bm, bg, bi), (rm, rg, ri)) in enumerate(zip(single, batched, repeated)):
            # Long LoRA vectors need FP64 reductions: FP32 cosine can exceed 1.
            g, bg, rg = g.double(), bg.double(), rg.cpu().double()
            delta, norm = float((bg - g).norm()), float(g.norm())
            cosine = float(torch.nn.functional.cosine_similarity(bg[None], g[None]).clamp(-1, 1)) if norm and float(bg.norm()) else None
            repeat_ok = (bi == ri and math.isclose(bm, rm, rel_tol=1e-5, abs_tol=1e-7)
                         and torch.allclose(bg, rg, rtol=1e-5, atol=1e-7))
            # Frozen operational limits; BF16 batch kernels need not match bitwise.
            criteria = dict(same_index=index == bi, batch_repeat=repeat_ok,
                            finite=all(math.isfinite(v) for v in (m, bm, rm, delta, norm)),
                            gradient_l2_close=delta <= 1e-7 + 0.01 * norm,
                            margin_close=math.isclose(m, bm, rel_tol=0.01, abs_tol=0.001))
            passed = all(criteria.values())
            checks.append(dict(condition=condition["condition"], view=i, seed=seeds[i],
                               single_margin=m, batch_margin=bm, relative_gradient_l2=delta / max(norm, 1e-30),
                               cosine=cosine, batch_repeat_passed=repeat_ok, criteria=criteria, passed=passed))
    return dict(view_batch_size=size, tolerances=dict(gradient_relative_l2=0.01, gradient_absolute_l2=1e-7,
                margin_rtol=0.01, margin_atol=0.001, repeat_rtol=1e-5, repeat_atol=1e-7),
                checks=checks, passed=all(row["passed"] for row in checks))


def stage_measurement(probe, conditions, sampling, checkpoint_id, pair_id, schedule):
    import torch

    result = []
    count, batch = sampling["stage_views"], sampling["stage_batch_size"]
    with torch.inference_mode():
        for condition in conditions:
            seeds = [
                seed_for("sd35-stage", checkpoint_id, sampling["view_seed"], pair_id,
                         condition["condition"], "stage", 0, index)
                for index in range(count)
            ]
            if len(set(seeds)) != len(seeds):
                raise ValueError("Stage view seed collision")
            views = []
            for start in range(0, count, batch):
                group = seeds[start:start + batch]
                margins, indices = probe.margins(condition, group)
                for view, (seed, index, margin) in enumerate(zip(group, indices.tolist(), margins.tolist()), start):
                    views.append({
                        "view_index": view, "seed": seed, "scheduler_index": index,
                        "timestep": schedule["native_timesteps"][index],
                        "sigma": schedule["native_sigmas"][index], "margin": margin,
                    })
            result.append({**condition, "views": views})
    return {"stages": result}


def gradient_measurement(probe, condition, sampling, experiment, checkpoint_id, pair_id, scratch_dir, scan=None):
    from self_correct_dpo.evaluation.appendix_gradients import collect_gradient_statistics

    def bank(role):
        if scan:
            from self_correct_dpo.evaluation.appendix_scan import scan_seed
            return [[scan_seed(scan["rng_scope"], "view", sampling["view_seed"], pair_id,
                               condition["condition"], role, block, index)
                     for index in range(sampling[f"{role}_views_per_block"])]
                    for block in range(sampling[f"{role}_blocks"])]
        return [
            [seed_for(experiment, checkpoint_id, sampling["view_seed"], pair_id, condition["condition"],
                      role, block, index) for index in range(sampling[f"{role}_views_per_block"])]
            for block in range(sampling[f"{role}_blocks"])
        ]

    observed = {}

    source, audit = bank("source"), bank("audit")
    size = scan["view_batch_size"] if scan else 1

    def observations():
        # Never cross a bank/block boundary. The collector consumes in this order.
        for block in source + audit:
            for start in range(0, len(block), size):
                chunk = block[start:start + size]
                values = (probe.gradients(condition, chunk) if size > 1 else
                          (probe.gradient(condition, seed) for seed in chunk))
                for seed, (margin, gradient, index) in zip(chunk, values):
                    yield seed, margin, gradient, index

    stream = observations()

    def observe(seed):
        actual, margin, gradient, index = next(stream)
        if actual != seed:
            raise ValueError("collector consumed views outside the frozen batch order")
        observed[seed] = {"seed": seed, "scheduler_index": index, "margin": margin}
        return margin, gradient
    statistics = collect_gradient_statistics(
        observe, source_seeds=source, audit_seeds=audit, ks=sampling["ks"],
        source_views_per_block=sampling["source_views_per_block"],
        audit_views_per_block=sampling["audit_views_per_block"], scratch_dir=scratch_dir,
    )
    return {
        "statistics": statistics,
        "source_views": [[observed[seed] for seed in block] for block in source],
        "audit_views": [[observed[seed] for seed in block] for block in audit],
    }


def repeat_view_check(probe, condition, seed):
    import torch

    # Reset only the explicit view generator inside gradient(). Leave the global RNG advancing.
    def rng_state():
        return (torch.get_rng_state(), torch.cuda.get_rng_state(probe.backend.device))

    before = rng_state()
    m1, g1, t1 = probe.gradient(condition, seed)
    between = rng_state()
    m2, g2, t2 = probe.gradient(condition, seed)
    after = rng_state()
    atol, rtol = 1e-7, 1e-5
    diff = g2 - g1
    norm, norm2 = float(g1.norm()), float(g2.norm())
    diff_norm = float(diff.norm())
    finite = all(math.isfinite(x) for x in (m1, m2, norm, norm2, diff_norm))
    criteria = dict(finite=finite, same_index=t1 == t2,
                    margin_close=finite and math.isclose(m1, m2, abs_tol=atol, rel_tol=rtol),
                    gradient_elementwise_close=finite and torch.allclose(g1, g2, atol=atol, rtol=rtol),
                    gradient_l2_close=finite and diff_norm <= atol + rtol * norm)
    return dict(condition=condition["condition"], seed=seed, indices=[t1, t2], finite=finite,
                margins=[x if math.isfinite(x) else None for x in (m1, m2)],
                gradient_norm=norm if finite else None, second_gradient_norm=norm2 if finite else None,
                gradient_difference_norm=diff_norm if finite else None,
                relative_l2_difference=diff_norm / norm if finite and norm > 0 else None,
                max_abs_difference=float(diff.abs().max()) if finite else None,
                default_rng_changed={name: [not torch.equal(before[j], between[j]),
                                            not torch.equal(between[j], after[j])]
                                     for j, name in enumerate(("cpu", "cuda"))},
                atol=atol, rtol=rtol, criteria=criteria, passed=all(criteria.values()))


def initialization_check(backend, beta, rows, pairs, conditions, checkpoint_id, view_seed, init_seed):
    import torch

    pair, condition = pairs[0], conditions[0]
    parameters = tuple(p for _, p in sorted((name, p) for name, p in backend.model.named_parameters()
                                            if p.requires_grad))
    probe = PairProbe(backend, parameters, beta, rows[0], pair["prompt"], pair["latent_seed"])
    seeds = [seed_for("initialization", checkpoint_id, view_seed, pair["pair_id"], i) for i in range(2)]
    observations = [probe.gradient(condition, seed) for seed in seeds]
    margins, gradients, indices = zip(*observations)
    margins_tensor = torch.tensor(margins, device=backend.device, dtype=torch.float32)
    h0, h1 = gradients
    weights = -torch.sigmoid(-margins_tensor)
    loss_avg = (weights[0] * h0 + weights[1] * h1) / 2
    mvpo = -torch.sigmoid(-margins_tensor.mean()) * ((h0 + h1) / 2)
    a, b, difference = [float(torch.linalg.vector_norm(vector)) for vector in (loss_avg, mvpo, loss_avg - mvpo)]
    margin_atol, gradient_rtol = 1e-5, 1e-5
    nonzero = min(a, b) > 0
    relative = difference / max(a, b) if nonzero else None
    return {
        "pair_id": pair["pair_id"], "pair_index": pair["pair_index"], "condition": condition["condition"],
        "initial_lora_seed": init_seed, "latent_seed": pair["latent_seed"], "beta": beta,
        "views": [{"seed": seed, "scheduler_index": index, "margin": margin}
                  for seed, index, margin in zip(seeds, indices, margins)],
        "lossavg_gradient_norm": a, "mvpo_gradient_norm": b, "gradient_difference_norm": difference,
        "relative_gradient_difference": relative, "margin_atol": margin_atol, "gradient_rtol": gradient_rtol,
        "passed": max(map(abs, margins)) <= margin_atol and nonzero and relative <= gradient_rtol,
    }


def probe_settings(args, parser):
    def missing(value, default):
        return default if value is None else value

    flow = args.experiment != "sd15-gradient"
    stage = args.experiment == "sd35-stage"
    args.pair_count = missing(args.pair_count, 128 if stage else (8 if flow else 16))
    if not 0 <= args.shard_index < args.shard_count <= args.pair_count:
        parser.error("require 0 <= shard-index < shard-count <= pair-count")
    sampling = {"pair_seed": args.pair_seed, "view_seed": args.view_seed}
    if stage:
        if args.check_initialization:
            parser.error("--check-initialization applies to gradient experiments")
        if args.stage_views < 2 or args.stage_batch_size < 1:
            parser.error("require at least two stage views and a positive stage-batch-size")
        return {**sampling, "stage_views": args.stage_views, "stage_batch_size": args.stage_batch_size}
    ks = [1, 2, 4] if flow else [1, 2, 4, 6, 8]
    source_blocks = missing(args.source_blocks, 4 if flow else 8)
    source_views = missing(args.source_views_per_block, 24 if flow else 96)
    audit_blocks = missing(args.audit_blocks, 4 if flow else 8)
    audit_views = missing(args.audit_views_per_block, 32 if flow else 128)
    if source_blocks < 2 or audit_blocks < 2 or audit_blocks % 2 or audit_views < 1:
        parser.error("require at least two source blocks, an even number of audit blocks, and positive audit views")
    if source_views < max(ks) or any(source_views % k for k in ks):
        parser.error(f"source views per block must be positive and divisible by {ks}")
    return {
        **sampling, "ks": ks, "source_blocks": source_blocks,
        "source_views_per_block": source_views, "audit_blocks": audit_blocks,
        "audit_views_per_block": audit_views,
    }


def publish_protocol(output, protocol):
    import fcntl

    path = output / "protocol.json"
    with (output / ".protocol.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if read_json(path) != protocol:
                raise ValueError(f"Output directory belongs to a different probe protocol: {path}")
        else:
            atomic_json(path, protocol)


def prepare_inputs(args, sampling, training, checkpoint, output):
    import fcntl

    manifest = Path(args.pairs or training["config"]["data"]["pairs"]).expanduser().resolve(strict=True)
    request = {
        "experiment": args.experiment, "checkpoint": checkpoint, "manifest": str(manifest),
        "pair_count": args.pair_count, "sampling": sampling, "source_sha256": sha256(__file__),
    }
    scan = getattr(args, "scan", None)
    if scan:
        request["scan"] = scan
    path = output / "prepared.json"
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = read_json(path)
            actual_id = digest({key: value for key, value in saved.items() if key != "prepared_id"})
            if (saved["prepared_id"] != actual_id or saved["request"] != request
                    or saved["dataset"]["manifest_stat"] != manifest_stat(manifest)):
                raise ValueError(f"Frozen pair manifest differs from the current inputs: {path}")
            return saved

        rows, prompts, dataset = choose_pairs(training, manifest, args.pair_count, sampling["pair_seed"],
                                              scan.get("excluded_pair_ids", ()) if scan else ())
        def latent_seed(pair_id):
            if scan:
                from self_correct_dpo.evaluation.appendix_scan import scan_seed
                return scan_seed(scan["rng_scope"], "posterior", scan["latent_seed"], pair_id)
            return seed_for("posterior", checkpoint["checkpoint_id"], sampling["view_seed"], pair_id)
        pairs = [
            {"pair_index": index, "pair_id": row["pair_id"], "scene_id": row["scene_id"], "prompt": prompt,
             "winner": row["winner"], "winner_sha256": row["winner_sha256"],
             "loser": row["loser"], "loser_sha256": row["loser_sha256"],
             "latent_seed": latent_seed(row["pair_id"])}
            for index, (row, prompt) in enumerate(zip(rows, prompts))
        ]
        saved = {"schema_version": 1, "request": request, "dataset": dataset, "pairs": pairs, "rows": rows}
        saved["prepared_id"] = digest(saved)
        atomic_json(path, saved)
        return saved


def run(args, sampling):
    family = "sd15" if args.experiment == "sd15-gradient" else "sd35_medium"
    stage = args.experiment == "sd35-stage"
    scan = getattr(args, "scan", None)
    model, adapter, training, checkpoint = load_checkpoint(
        family, args.checkpoint, args.resources, scan["checkpoint"] if scan else None,
        scan.get("method") if scan else None)
    settings = training["config"]["training"]
    native = bool(scan and scan.get("method"))
    probe_class = PairProbe
    if native:
        from self_correct_dpo.evaluation.appendix_native_probe import NativePairProbe, initialization_native, native_measurement
        probe_class = NativePairProbe
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_inputs(args, sampling, training, checkpoint, output)
    if args.prepare_only:
        print(json.dumps({"prepared_id": prepared["prepared_id"], "manifest": str(output / "prepared.json"),
                          "pairs": len(prepared["pairs"])}), flush=True)
        return

    import torch
    from self_correct_dpo.evaluation.appendix_backend import AppendixBackend

    rows, pairs = prepared["rows"], prepared["pairs"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU assigned to this shard is required")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    init = None
    conditions = scan["conditions"] if scan else conditions_for(family)
    check_initial = args.check_initialization and args.shard_index == 0
    load_started = time.perf_counter()
    rng_context = torch.random.fork_rng(devices=[device.index]) if check_initial else nullcontext()
    with rng_context:
        if check_initial:
            init_seed = seed_for("initial_lora", checkpoint["checkpoint_id"], sampling["view_seed"])
            torch.manual_seed(init_seed)
        backend = AppendixBackend(model, settings, device)
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    if check_initial:
        init = (initialization_native if native else initialization_check)(
            backend, float(settings["beta"]), rows, pairs, conditions,
            checkpoint["checkpoint_id"], sampling["view_seed"], init_seed,
        )
    adapter_started = time.perf_counter()
    backend.load_adapter(adapter)
    torch.cuda.synchronize(device)
    adapter_load_seconds = time.perf_counter() - adapter_started
    if stage:
        backend.eval()
    else:
        backend.train()
    if backend.error_parameterization != read_json(adapter / "backend.json")["error_parameterization"]:
        raise ValueError("Loaded model uses a different training error parameterization")
    named = sorted((name, p) for name, p in backend.model.named_parameters() if p.requires_grad)
    parameter_count = sum(p.numel() for _, p in named)
    if parameter_count != training["trainable_parameters"] or any(p.dtype != torch.float32 for _, p in named):
        raise ValueError("Actual trainable FP32 LoRA parameter dimensions differ from training")
    parameters = tuple(p for _, p in named)
    schedule = schedule_identity(backend)
    relevant_sources = [
        "evaluation/appendix_probe.py", "evaluation/appendix_backend.py", "models.py", "training.py",
        "algorithms/losses.py", "algorithms/formal.py", "data/pairs.py", "data/arrow_images.py",
        "evaluation/checkpoint_catalog.py", "common.py",
    ]
    if not stage:
        relevant_sources.append("evaluation/appendix_gradients.py")
    if native:
        relevant_sources.extend(["evaluation/appendix_native_probe.py", "evaluation/appendix_native_gradients.py",
                                 "evaluation/experiment_catalog.py"])
    root = Path(__file__).resolve().parents[1]
    protocol = {
        "schema_version": 1, "experiment": args.experiment, "checkpoint": checkpoint,
        "prepared_id": prepared["prepared_id"], "dataset": prepared["dataset"],
        "pairs": pairs, "conditions": conditions, "sampling": sampling,
        "model": {
            "config": model, "base_identity": training["base_identity"], "schedule": schedule,
            "error_parameterization": backend.error_parameterization, "lora_parameter_count": parameter_count,
            "lora_parameters": [{"name": name, "shape": list(p.shape), "dtype": "float32"} for name, p in named],
        },
        "training": {"config": settings, "preprocessing": training["preprocessing"],
                     "reference": "frozen_base_adapter_disabled",
                     "margin": "(-beta/2)*((policy_w-reference_w)-(policy_l-reference_l)), fp32 per-image latent MSE"},
        "software": {
            "probe": {**{name: version(name) for name in ("torch", "diffusers", "peft", "transformers", "numpy", "Pillow")},
                      "cuda": torch.version.cuda},
            "training": training["versions"],
        },
        "source": {"probe": {name: sha256(root / name) for name in relevant_sources}, "training": training["code"]},
        "seed_derivation": {
            "function": "first 16 hex digits of UTF-8 canonical JSON SHA256 modulo 2^63, prefix mvpo-appendix-v1",
            "pair_order": "sort by SHA256([pair_seed,pair_id]) then pair_id",
            "posterior": ["posterior", "checkpoint_id", "view_seed", "pair_id"],
            "view": ["experiment", "checkpoint_id", "view_seed", "pair_id", "condition", "bank", "block", "view_index"],
            "draw_order": "torch CUDA generator: shared Gaussian noise in latent dtype, then uniform integer index within condition",
        },
    }
    if not stage:
        protocol["numerics"] = {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "sdpa_backend": "math",
        }
    if scan:
        protocol["scan"] = scan
        protocol["seed_derivation"].update(
            function="SHA256 canonical JSON first 16 hex modulo 2^63; prefix mvpo-scan-v1",
            posterior=["rng_scope", "posterior", "latent_seed", "pair_id"],
            view=["rng_scope", "view", "view_seed", "pair_id", "condition", "bank", "block", "view_index"],
        )
        protocol["source"]["probe"]["evaluation/appendix_scan.py"] = sha256(root / "evaluation/appendix_scan.py")
    if native:
        from self_correct_dpo.evaluation.appendix_native_gradients import objective_spec
        protocol["native_objective"] = objective_spec(settings)
        protocol["training"]["margin"] = ("u=beta*((reference_w-policy_w)-(reference_l-policy_l))"
            if settings["objective"] == "dmpo" else "m=beta/2*((reference_w-policy_w)-(reference_l-policy_l)); differentiable winner MSE and prediction gap")
        protocol["native_interpretation"] = "Local fixed-index native objective; alignment is to the independent audit loss gradient. Not a whole-stage training-gradient claim."
    protocol["protocol_id"] = digest(protocol)
    (output / "pairs").mkdir(parents=True, exist_ok=True)
    publish_protocol(output, protocol)
    print(json.dumps({"protocol_id": protocol["protocol_id"], "shard_index": args.shard_index,
                      "shard_count": args.shard_count, "event": "ready",
                      "model_load_seconds": model_load_seconds, "adapter_load_seconds": adapter_load_seconds}), flush=True)

    execution = {"gpu": torch.cuda.get_device_name(device),
                 "compute_capability": list(torch.cuda.get_device_capability(device))}
    if init is not None:
        check_path = output / "init-check.json"
        record = {**init, "protocol_id": protocol["protocol_id"], "execution": execution}
        record["result_id"] = digest(record)
        with output_lock(check_path):
            atomic_json(check_path, record)
        if not init["passed"]:
            raise RuntimeError(f"Initialized policy/reference gradient check failed: {check_path}")
    if check_initial:
        pair = pairs[0]
        probe = probe_class(backend, parameters, float(settings["beta"]), rows[0], pair["prompt"], pair["latent_seed"])
        checks = [repeat_view_check(probe, c, seed_for("repeat", checkpoint["checkpoint_id"], sampling["view_seed"],
                                                       pair["pair_id"], c["condition"])) for c in conditions]
        record = dict(protocol_id=protocol["protocol_id"], pair_id=pair["pair_id"], execution=execution,
                      checks=checks, passed=all(c["passed"] for c in checks))
        record["result_id"] = digest(record)
        check_path = output / "repeat-view-check.json"
        with output_lock(check_path):
            atomic_json(check_path, record)
        print(json.dumps({"event": "repeat_view", **record}), flush=True)
        if not record["passed"]:
            raise RuntimeError(f"Loaded checkpoint failed the train-mode repeat-view check: {check_path}")
        if scan and scan["view_batch_size"] > 1:
            batch = batch_check(probe, conditions, scan["view_batch_size"], sampling["view_seed"])
            batch.update(protocol_id=protocol["protocol_id"], execution=execution)
            batch["result_id"] = digest(batch)
            with output_lock(output / "batch-check.json"):
                atomic_json(output / "batch-check.json", batch)
            print(json.dumps({"event": "batch_check", **batch}), flush=True)
            if not batch["passed"]:
                raise RuntimeError(f"Batch numerical check failed: {output / 'batch-check.json'}")
        del probe
    for index in range(args.shard_index, len(rows), args.shard_count):
        pair = pairs[index]
        shared = {"protocol_id": protocol["protocol_id"], "pair_index": index, "pair_id": pair["pair_id"]}
        jobs = [("", None)] if stage else [(f"-{c['condition']}", c) for c in conditions]
        probe = None
        for suffix, condition in jobs:
            path = output / "pairs" / f"{index:04d}{suffix}.json"
            expected = shared if stage else {**shared, "condition": condition["condition"]}
            with output_lock(path):
                if path.exists():
                    existing = read_json(path)
                    saved_id = existing.pop("result_id")
                    if any(existing.get(k) != v for k, v in expected.items()) or digest(existing) != saved_id:
                        raise ValueError(f"Existing result is inconsistent with the probe: {path}")
                    continue
                torch.cuda.reset_peak_memory_stats(device)
                pair_preparation_seconds = 0.0
                if probe is None:
                    pair_started = time.perf_counter()
                    probe = probe_class(backend, parameters, float(settings["beta"]), rows[index],
                                      pair["prompt"], pair["latent_seed"])
                    torch.cuda.synchronize(device)
                    pair_preparation_seconds = time.perf_counter() - pair_started
                started = time.perf_counter()
                if stage:
                    measurement = stage_measurement(probe, conditions, sampling, checkpoint["checkpoint_id"],
                                                    pair["pair_id"], schedule)
                elif native:
                    measurement = native_measurement(probe, condition, sampling, pair["pair_id"], args.scratch_dir, scan)
                else:
                    measurement = gradient_measurement(probe, condition, sampling, args.experiment,
                                                       checkpoint["checkpoint_id"], pair["pair_id"], args.scratch_dir, scan)
                torch.cuda.synchronize(device)
                work = {
                    **execution, "measurement_seconds": time.perf_counter() - started,
                    "measurement_unit": "pair_two_stages" if stage else "pair_condition",
                    "pair_preparation_seconds": pair_preparation_seconds,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
                result = {**expected, **measurement, "prompt_truncations": probe.truncations,
                          "execution": work}
                result["result_id"] = digest(result)
                atomic_json(path, result)
                print(json.dumps({"event": "pair_saved", **expected}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("sd15-gradient", "sd35-gradient", "sd35-stage"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scan-job", type=Path, help="Immutable scan job produced by gradient_scan.py")
    parser.add_argument("--checkpoint", type=Path, help="Override the retained step directory; full ID still verified")
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--pairs", type=Path, help="Override the training manifest; training split identity still verified")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--pair-seed", type=int, default=20261121)
    parser.add_argument("--view-seed", type=int, default=17)
    parser.add_argument("--prepare-only", action="store_true", help="Verify inputs and freeze the pair manifest on CPU")
    parser.add_argument("--check-initialization", action="store_true",
                        help="On gradient shard 0 compare the objectives before loading the trained adapter")
    parser.add_argument("--source-blocks", type=int)
    parser.add_argument("--source-views-per-block", type=int)
    parser.add_argument("--audit-blocks", type=int)
    parser.add_argument("--audit-views-per-block", type=int)
    parser.add_argument("--stage-views", type=int, default=96)
    parser.add_argument("--stage-batch-size", type=int, default=4)
    parser.add_argument("--scratch-dir", type=Path)
    args = parser.parse_args()
    args.scan = None
    if args.scan_job:
        from self_correct_dpo.evaluation.appendix_scan import validate_job
        family = "sd15" if args.experiment == "sd15-gradient" else "sd35_medium"
        if args.experiment == "sd35-stage":
            parser.error("scan jobs use the gradient experiments, including optional stage intervals")
        args.scan = validate_job(read_json(args.scan_job), family)
        if args.checkpoint and args.checkpoint.resolve() != Path(args.scan["checkpoint"]["path"]).resolve():
            parser.error("--checkpoint differs from frozen scan job")
        args.checkpoint = Path(args.scan["checkpoint"]["path"])
    sampling = probe_settings(args, parser)
    if args.experiment in ("sd15-gradient", "sd35-gradient") and not args.prepare_only:
        # Set before importing the GPU stack and creating the first CUDA context.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with sdpa_kernel(SDPBackend.MATH):
            run(args, sampling)
    else:
        run(args, sampling)
