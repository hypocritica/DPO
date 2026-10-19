"""Sequential, hash-verified node-local copies for cold NFS model weights."""
import hashlib
from pathlib import Path
import shutil
import tempfile

from self_correct_dpo.common import sha256, write_json


def stage_judge_model(source, record):
    source = Path(source).resolve()
    files = [p for p in sorted(source.iterdir()) if p.is_file()
             and p.suffix in {".json", ".safetensors", ".txt", ".model"}]
    required = sum(p.stat().st_size for p in files)
    if not files or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Expected a local Transformers safetensors model")
    if shutil.disk_usage("/tmp").free < required + 2 * 1024 ** 3:
        raise RuntimeError("Insufficient node-local /tmp space for the judge model")
    destination = Path(tempfile.mkdtemp(prefix="robustdpo-qwen-", dir="/tmp"))
    hashes = {}
    for path in files:
        h = hashlib.sha256()
        target = destination / path.name
        with path.open("rb") as src, target.open("xb") as dst:
            while block := src.read(8 * 1024 ** 2):
                h.update(block)
                dst.write(block)
        if sha256(target) != h.hexdigest():
            raise ValueError("Staged model checksum mismatch")
        hashes[path.name] = h.hexdigest()
        print(f"staged {path.name}: {target.stat().st_size} bytes", flush=True)
    write_json(record, {"source": str(source), "loaded_from": str(destination),
                        "files_sha256": hashes, "bytes": required})
    return str(destination)
