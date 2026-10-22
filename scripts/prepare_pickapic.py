"""Adapt a local image-path JSONL export using an audited caption-group map."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import read_json, read_jsonl, write_json, write_jsonl
from self_correct_dpo.data.pickapic import adapt_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--groups", required=True, help="JSON mapping each caption to its audited semantic group")
    p.add_argument("--revision", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", required=True)
    args=p.parse_args()
    raw=Path(args.raw).resolve()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    pairs, exclusions, report=adapt_rows(read_jsonl(raw), raw.parent, read_json(args.groups),
                                       source_revision=args.revision, seed=args.seed)
    out.mkdir(parents=True)
    write_jsonl(out / "pairs.jsonl", pairs)
    write_jsonl(out / "exclusions.jsonl", exclusions)
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
