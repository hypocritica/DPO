"""Prepare exact official GenEval or DPG prompts from a pinned local checkout."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.evaluation.compositional import SOURCES, prepare
from self_correct_dpo.resources import load_resources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=SOURCES, required=True)
    parser.add_argument('--source-dir', help='Pinned official Git checkout; defaults to resource registry')
    parser.add_argument('--resources')
    parser.add_argument('--out', required=True, help='New directory containing suite.jsonl and selection.json')
    parser.add_argument('--smoke-prompts', type=int, help='First N prompts, labelled smoke-only')
    args = parser.parse_args(argv)
    source = args.source_dir or load_resources(args.resources)[args.benchmark.upper() + '_REPO']
    print(json.dumps(prepare(source, args.benchmark, args.out,
        smoke_prompts=args.smoke_prompts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
