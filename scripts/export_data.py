"""Materialize an existing validated manifest into portable image files.

Reads ordinary files or the original local Arrow storage; retains existing
splits, captions and labels. Never copies models or experiment outputs.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import sha256, write_json, write_jsonl
from self_correct_dpo.data import load_pairs
from self_correct_dpo.arrow_images import image_source


def export_data(manifest, out):
    rows = load_pairs(manifest)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    # Dry size estimate is based on the manifest's original image identities.
    sources = {}
    for row in rows:
        for side in ("winner", "loser"):
            sources[row[side + "_sha256"]] = (row, side)
    required = 0
    for row, side in sources.values():
        source = image_source(row, side)
        required += len(source.getbuffer()) if hasattr(source, "getbuffer") else Path(source).stat().st_size
    ancestor = out.resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < required + 1024**3:
        raise RuntimeError(f"Insufficient destination space for {required} bytes plus 1 GiB margin")
    out.mkdir(parents=True)
    (out / "images").mkdir()
    for checksum, (row, side) in sources.items():
        source = image_source(row, side)
        target = out / "images" / (checksum + ".img")
        if hasattr(source, "read"):
            payload = source.read()
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError("Arrow image checksum mismatch")
            with target.open("xb") as f:
                f.write(payload)
        else:
            shutil.copyfile(source, target)
        if sha256(target) != checksum:
            raise ValueError("Export checksum mismatch")
    exported = []
    for original in rows:
        row = dict(original)
        row.pop("storage", None)
        row.pop("semantic_scene_key", None)
        for side in ("winner", "loser"):
            row[side] = "images/" + row[side + "_sha256"] + ".img"
        exported.append(row)
    write_jsonl(out / "pairs.jsonl", exported)
    load_pairs(out / "pairs.jsonl")
    write_json(out / "export.json", {"source_manifest_sha256":sha256(manifest),
               "pairs":len(rows), "unique_images":len(sources), "image_bytes":required,
               "manifest_sha256":sha256(out / "pairs.jsonl"),
               "note":"Original labels, splits and captions retained; no experiment results bundled."})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    args=p.parse_args()
    export_data(args.manifest, args.out)


if __name__ == "__main__":
    main()
