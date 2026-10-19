"""Optional local CLIP/PickScore alignment and image-feature diversity diagnostics.

All expressions are scored against the original scene caption. These diagnostics
do not replace GenEval/human semantic evaluation. No model is downloaded.
"""
import argparse
from collections import defaultdict
import json
import importlib.metadata
from pathlib import Path
import statistics

from self_correct_dpo.common import artifact_identity, digest, require_gpu_allocation, sha256, verify_images, write_json, write_jsonl


def feature_diversity(features):
    """Mean unordered-pair cosine distance; features must already be normalized."""
    import torch
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Diversity requires at least two images")
    similarities = features @ features.T
    mask = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
    return float((1 - similarities[mask]).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True, help="Local Transformers CLIPModel/PickScore checkpoint")
    p.add_argument("--processor", required=True, help="Local matching CLIP processor/tokenizer")
    p.add_argument("--kind", choices=["clip", "pickscore"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--execution", choices=["slurm", "standalone"], default="slurm")
    p.add_argument("--run-label")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    run, manifest = verify_images(args.run)
    versions = {}
    for package in ("torch", "transformers", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    identity = {"kind": args.kind, "versions": versions, "model": artifact_identity(args.model),
                "processor": artifact_identity(args.processor), "script_sha256": sha256(__file__)}
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if args.check_only:
        print(json.dumps({"images": len(manifest), "evaluator_id": digest(identity)}))
        return
    execution = {"mode": args.execution, "run_label": args.run_label}
    require_gpu_allocation(execution)
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from transformers import AutoModel, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("Quality scoring requires an allocated CUDA GPU")
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to("cuda")
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    captions = {row["scene_id"]: row["metadata"]["prompt"] for row in run["suite"]}
    max_tokens = model.config.text_config.max_position_embeddings
    for caption in set(captions.values()):
        if len(processor.tokenizer(caption, truncation=False)["input_ids"]) > max_tokens:
            raise ValueError("Canonical caption would be truncated by quality evaluator")
    feature_groups, scores = defaultdict(list), []
    with torch.inference_mode():
        for start in range(0, len(manifest), args.batch_size):
            batch = manifest[start:start + args.batch_size]
            images = []
            for row in batch:
                with Image.open(Path(args.run) / row["image"]) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB"))
            image_inputs = processor(images=images, return_tensors="pt").to("cuda")
            text_inputs = processor(text=[captions[r["scene_id"]] for r in batch],
                                    padding=True, truncation=False, return_tensors="pt").to("cuda")
            image_features = F.normalize(model.get_image_features(**image_inputs).float(), dim=-1)
            text_features = F.normalize(model.get_text_features(**text_inputs).float(), dim=-1)
            cosine = (image_features * text_features).sum(-1)
            values = cosine * model.logit_scale.exp() if args.kind == "pickscore" else cosine
            for row, feature, value in zip(batch, image_features.cpu(), values.cpu()):
                scores.append({**row, "score": float(value)})
                feature_groups[(row["scene_id"], row["variant_id"])].append(feature)
    diversity = [{"scene_id": scene, "variant_id": variant,
                  "mean_pairwise_cosine_distance": feature_diversity(torch.stack(features)),
                  "images": len(features)} for (scene, variant), features in feature_groups.items()]
    # Explicit scene/category macro aggregation, not a pooled image average.
    by_scene, tags = defaultdict(list), {}
    for row in scores:
        by_scene[row["scene_id"]].append(row["score"])
        tags[row["scene_id"]] = row["tag"]
    by_tag = defaultdict(list)
    for scene, values in by_scene.items():
        by_tag[tags[scene]].append(statistics.mean(values))
    out.mkdir(parents=True)
    write_jsonl(out / "scores.jsonl", scores)
    write_jsonl(out / "diversity.jsonl", diversity)
    write_json(out / "quality.json", {"run_id": run["run_id"], "manifest_sha256": sha256(Path(args.run) / "manifest.jsonl"), "evaluator": identity,
               "evaluator_id": digest(identity), "execution": execution,
               "scores_sha256": sha256(out / "scores.jsonl"),
               "macro_score": statistics.mean(statistics.mean(v) for v in by_tag.values()),
               "interpretation": "raw canonical-caption cosine or scaled PickScore; no batch-dependent softmax",
               "limitations": "Feature diversity is descriptive; different content/errors can also raise it. No standalone quality claim."})


if __name__ == "__main__":
    main()
