"""Select unchanged held-out captions from the validated preference manifest."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import digest, write_jsonl
from self_correct_dpo.data import load_pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument("--split", choices=["validation","test"], default="validation")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--out", required=True)
    args=p.parse_args()
    if args.limit < 1:
        p.error("--limit must be positive")
    rows=load_pairs(args.pairs, split=args.split)
    prompts={}
    for row in sorted(rows, key=lambda r:digest(r["pair_id"])):
        text=next(p["prompt"] for p in row["prompts"] if p["variant_id"] == "canonical")
        scene=row["scene_id"]
        prompts.setdefault(scene, {"scene_id":scene, "variant_id":"canonical", "prompt":text,
            "tag":"heldout", "metadata":{"tag":"heldout", "prompt":text},
            "suite_mode":"canonical-only-v1", "transform_type":"canonical", "template_family":"canonical"})
    if not prompts:
        raise ValueError("No held-out prompts")
    write_jsonl(args.out, list(prompts.values())[:args.limit])


if __name__ == "__main__":
    main()
