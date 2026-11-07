"""Identify existing HPDv3 files offline; print the preparation source identity."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.data.hpdv3_source import source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    path, record = source_record(args.raw, create_inventory=True)
    print(f'[source] Using {path}', file=sys.stderr, flush=True)
    print(record['repo_id'] + '@' + record['revision'], flush=True)


if __name__ == '__main__':
    main()
