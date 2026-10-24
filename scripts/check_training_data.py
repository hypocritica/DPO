"""Audit/cache a pair manifest without models, GPUs or training outputs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.data.validation import load_training_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--mode', choices=['auto', 'full'], default='auto')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    load_training_pairs(args.pairs, mode=args.mode, workers=args.workers, samples=args.samples)


if __name__ == '__main__':
    main()
