"""CPU-only preference manifests and semantic split leakage checks.

Each JSONL row explicitly labels winner/loser; this module never infers preference.
Paths are relative to the manifest unless absolute. No torch dependency is needed.
"""
import copy
from pathlib import Path

from self_correct_dpo.common import digest, read_jsonl, sha256

SPLITS = {"train", "validation", "test"}


def semantic_scene_key(row):
    """Ignore wording and object order, and canonicalize inverse spatial edges.

    Identical object descriptors intentionally coalesce for split isolation: this
    may conservatively group ambiguous graphs, but never separates their aliases.
    Without metadata we can only enforce the supplied scene_id, not semantics.
    """
    metadata = row.get("metadata")
    if metadata is None:
        return "id:" + str(row["scene_id"])
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    graph = {k: v for k, v in metadata.items() if k not in {"prompt", "tag", "include"}}
    if "include" in metadata:
        objects = metadata["include"]
        if not isinstance(objects, list) or not all(isinstance(o, dict) for o in objects):
            raise ValueError("metadata.include must be a list of objects")
        nodes = [{k: v for k, v in obj.items() if k != "position"} for obj in objects]
        labels = [digest(node) for node in nodes]
        edges = []
        for index, obj in enumerate(objects):
            if "position" not in obj:
                continue
            relation = obj["position"]
            if not isinstance(relation, (list, tuple)) or len(relation) != 2:
                raise ValueError("Invalid scene relation")
            rel, target = relation
            if type(target) is not int or not 0 <= target < len(nodes):
                raise ValueError("Invalid scene relation target")
            left, right = labels[index], labels[target]
            if rel in {"right of", "below"}:
                left, right = right, left
                rel = {"right of": "left of", "below": "above"}[rel]
            edges.append([left, rel, right])
        graph["nodes"] = sorted(nodes, key=digest)
        graph["edges"] = sorted(edges)
    if "exclude" in graph:
        graph["exclude"] = sorted(graph["exclude"], key=digest)
    if not graph:
        raise ValueError("metadata must contain scene semantics beyond prompt/tag")
    return "semantic:" + digest(graph)


def validate_split_disjointness(rows):
    """Reject scene aliases or any shared image across train/validation/test."""
    assignments = {}
    for row in rows:
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split!r}")
        keys = [("scene_id", str(row["scene_id"])), ("scene_semantics", semantic_scene_key(row))]
        if row.get("human_preference") or row.get("prompt_group_id"):
            from self_correct_dpo.data.pickapic import prompt_key
            keys.extend(("prompt", prompt_key(p["prompt"])) for p in row["prompts"])
            if row.get("prompt_group_id"):
                keys.append(("prompt_group", row["prompt_group_id"]))
        for side in ("winner", "loser"):
            keys.append(("image_path", str(row[side])))
            if row.get(side + "_sha256"):
                keys.append(("image_sha256", row[side + "_sha256"]))
        for key in keys:
            previous = assignments.setdefault(key, split)
            if previous != split:
                raise ValueError(f"Cross-split leakage for {key[0]}: {previous} vs {split}")
    return True


def validate_pair_rows(rows, base_dir, *, verify_files=True):
    """Validate complete rows and resolve paths; return independent dictionaries."""
    if not rows:
        raise ValueError("Empty preference manifest")
    base_dir = Path(base_dir).resolve()
    result, pair_ids, hashes = [], set(), {}
    for original in rows:
        row = copy.deepcopy(original)
        for key in ("pair_id", "scene_id", "winner", "loser"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Pair requires nonempty {key}")
        if row["pair_id"] in pair_ids:
            raise ValueError("Duplicate pair_id")
        pair_ids.add(row["pair_id"])
        if row.get("split") not in SPLITS:
            raise ValueError("Pair split must be train, validation, or test")
        if "reviewed" in row and type(row["reviewed"]) is not bool:
            raise ValueError("reviewed must be a boolean")
        prompts = row.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("Pair requires a nonempty prompts list")
        ids, texts = set(), set()
        for prompt in prompts:
            if not isinstance(prompt, dict):
                raise ValueError("Each prompt must be an object")
            for key in ("variant_id", "prompt", "template_family", "transform_type"):
                if not isinstance(prompt.get(key), str) or not prompt[key].strip():
                    raise ValueError(f"Pair prompt requires nonempty {key}")
            if prompt["variant_id"] in ids or prompt["prompt"].strip() in texts:
                raise ValueError("Duplicate variant or prompt text in pair orbit")
            ids.add(prompt["variant_id"])
            texts.add(prompt["prompt"].strip())
        storage = row.get("storage", {})
        if storage.get("kind") == "arrow":
            shard = Path(storage["shard"]).expanduser().absolute()
            if type(storage.get("row")) is not int or storage["row"] < 0:
                raise ValueError("Invalid Arrow row index")
            for side in ("winner", "loser"):
                if storage.get(side + "_column") not in {"jpg_0", "jpg_1"} or not row.get(side + "_sha256"):
                    raise ValueError("Arrow preference requires explicit columns and image checksums")
                expected = f"{shard}#row={storage['row']}:{storage[side + '_column']}"
                if row[side] != expected:
                    raise ValueError("Arrow image identity does not match storage descriptor")
            if storage["winner_column"] == storage["loser_column"]:
                raise ValueError("Winner and loser must be distinct Arrow columns")
            if verify_files:
                stat = shard.stat()
                if stat.st_size != storage["shard_size"] or stat.st_mtime_ns != storage["shard_mtime_ns"]:
                    raise ValueError("Arrow shard inventory changed")
        for side in ("winner", "loser"):
            if storage.get("kind") == "arrow":
                continue
            path = Path(row[side]).expanduser()
            path = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
            row[side] = str(path)
            if verify_files:
                if not path.is_file():
                    raise FileNotFoundError(path)
                if str(path) not in hashes:
                    hashes[str(path)] = sha256(path)
                actual = hashes[str(path)]
                if row.get(side + "_sha256") not in (None, actual):
                    raise ValueError(f"{side} image checksum mismatch")
                row[side + "_sha256"] = actual
        if row["winner"] == row["loser"] or (
            row.get("winner_sha256") is not None and row.get("winner_sha256") == row.get("loser_sha256")
        ):
            raise ValueError("Winner and loser must be distinct images")
        row["semantic_scene_key"] = semantic_scene_key(row)
        result.append(row)
    validate_split_disjointness(result)
    return result


def load_pairs(path, split=None, verify_files=True):
    """Validate ALL splits before filtering, so selecting train cannot hide leakage."""
    if split is not None and split not in SPLITS:
        raise ValueError("Unknown requested split")
    path = Path(path).resolve()
    rows = validate_pair_rows(read_jsonl(path), path.parent, verify_files=verify_files)
    return rows if split is None else [row for row in rows if row["split"] == split]
