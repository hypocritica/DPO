"""Small shared helpers; preparing and analysing experiments needs only Python."""
import hashlib
import json
import os
import socket
from pathlib import Path
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parent


@contextmanager
def output_lock(path):
    import fcntl
    path=Path(path).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path.parent/('.'+path.name+'.lock'),'a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process owns {path}') from error
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    with open(path, "x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    with open(path, "x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def artifact_identity(path):
    """Fingerprint local configs and weight inventories, without reading GB of weights.

    Weight size + mtime is an inventory fingerprint, NOT a cryptographic weight hash.
    Record the upstream revision separately in the model config.
    """
    path = Path(path).resolve(strict=True)
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    rows = []
    for p in files:
        small_config_suffixes = {".json", ".txt", ".model", ".vocab", ".merges", ".tiktoken"}
        if not p.is_file() or p.suffix not in small_config_suffixes | {".safetensors", ".bin", ".pth"}:
            continue
        stat = p.stat()
        row = {"path": str(p.relative_to(path)) if p != path else p.name,
               "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if p.suffix in small_config_suffixes:
            # Tokenizer files (including SD3 T5 spiece.model) affect conditioning.
            # Weight tensors retain inventory-only fingerprints to avoid GB reads.
            row["sha256"] = sha256(p)
        rows.append(row)
    if not rows:
        raise ValueError(f"No model artifacts found at {path}")
    return {"path": str(path), "inventory_sha256": digest(rows), "files": rows}


def model_artifact_identity(model):
    base=artifact_identity(model['path'])
    if model.get('vae'):
        return {'pipeline':base,'vae':artifact_identity(model['vae']['path'])}
    return base


def canonical_only_marker(group):
    return (len(group) == 1 and group[0].get("suite_mode") == "canonical-only-v1"
            and group[0].get("variant_id") == "canonical"
            and group[0].get("transform_type") == "canonical"
            and group[0].get("template_family") == "canonical")


def load_suite(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty prompt suite")
    groups = {}
    seen = set()
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Prompts must be nonempty strings")
        key = (row["scene_id"], row["variant_id"])
        if key in seen:
            raise ValueError(f"Duplicate prompt: {key}")
        seen.add(key)
        groups.setdefault(row["scene_id"], []).append(row)
        if row["tag"] != row["metadata"]["tag"]:
            raise ValueError("Tag does not match evaluation metadata")
    for scene, group in groups.items():
        explicit = any("transform_type" in r or "template_family" in r for r in group)
        if explicit and any(not isinstance(r.get(k), str) or not r[k].strip()
                            for r in group for k in ("transform_type", "template_family")):
            raise ValueError("Every explicit-orbit prompt needs transform_type and template_family")
        if sum(r["variant_id"] == "canonical" for r in group) != 1:
            raise ValueError("Each orbit needs exactly one canonical prompt")
        if explicit and len(group) < 2 and canonical_only_marker(group) is False:
            raise ValueError("An explicit orbit needs at least one noncanonical prompt")
        if not explicit and {r["variant_id"] for r in group} != {"canonical", "lexical", "sentence", "structure"}:
            raise ValueError(f"Incomplete orbit: {scene}")
        if len({digest(r["metadata"]) for r in group}) != 1:
            raise ValueError("All variants must use identical evaluation metadata")
        canonical = next(r for r in group if r["variant_id"] == "canonical")
        if canonical["prompt"] != canonical["metadata"]["prompt"]:
            raise ValueError("Canonical prompt mismatch")
        if explicit and any((r["transform_type"] == "canonical") != (r["variant_id"] == "canonical")
                            for r in group):
            raise ValueError("Only canonical variant may have canonical transform_type")
        if len({r["prompt"].strip() for r in group}) != len(group):
            raise ValueError("Duplicate text inside an orbit")
    return rows


def require_gpu_allocation(execution=None):
    """Require Slurm by default; explicit labelled standalone runs are external only."""
    execution = execution or {"mode": "slurm"}
    mode = execution.get("mode", "slurm")
    if mode == "standalone":
        host = socket.gethostname().lower()
        if "irip-c1-manager" in host or "login" in host:
            raise RuntimeError("Standalone GPU execution is forbidden on a manager/login node")
        if not isinstance(execution.get("run_label"), str) or not execution["run_label"].strip():
            raise ValueError("Standalone execution requires an explicit nonempty run_label")
    elif mode != "slurm":
        raise ValueError("execution.mode must be slurm or standalone")
    elif not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU rendering/evaluation must run inside a Slurm allocation")


def verify_images(run_dir):
    root = Path(run_dir).resolve()
    run = read_json(root / "run.json")
    if run["run_id"] != digest({k: v for k, v in run.items() if k != "run_id"}):
        raise ValueError("Run metadata checksum mismatch")
    if run["suite_id"] != digest(run["suite"]):
        raise ValueError("Run suite checksum mismatch")
    complete = read_json(root / "complete.json")
    manifest_path = root / "manifest.jsonl"
    if complete["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Manifest checksum mismatch")
    rows = read_jsonl(manifest_path)
    if len(rows) != run["expected_images"] or complete["images"] != len(rows):
        raise ValueError("Incomplete rendering")
    expected = {(r["scene_id"], r["variant_id"], seed)
                for r in run["suite"] for seed in run["sampling"]["seeds"]}
    observed = [(r["scene_id"], r["variant_id"], r["seed"]) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Render manifest does not cover the full paired design")
    prompts = {(r["scene_id"], r["variant_id"]): r for r in run["suite"]}
    for row in rows:
        prompt = prompts[(row["scene_id"], row["variant_id"])]
        if row["prompt"] != prompt["prompt"] or row["tag"] != prompt["tag"]:
            raise ValueError("Manifest prompt/tag differs from the recorded suite")
        payload = {k: v for k, v in row.items() if k != "image_id"}
        if row["image_id"] != digest({"run_id": run["run_id"], **payload}):
            raise ValueError("Manifest image identity mismatch")
        image = (root / row["image"]).resolve()
        if not image.is_relative_to(root) or sha256(image) != row["image_sha256"]:
            raise ValueError(f"Image checksum/path mismatch: {row['image']}")
    return run, rows


def canonical_scheduler_config(config):
    """Normalize Diffusers' unordered default-name bookkeeping, retaining all values."""
    result = dict(config)
    if "_use_default_values" in result:
        values = result["_use_default_values"]
        if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
            raise ValueError("Invalid scheduler default-name metadata")
        result["_use_default_values"] = sorted(values)
    return result
