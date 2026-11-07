"""Identify existing HPDv3 files offline; print the preparation source identity."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.data.hpdv3_source import source_record, source_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--diagnose', action='store_true',
                        help='Read-only validation in this container; emit runtime context and result as JSON')
    args = parser.parse_args()
    context = source_context(args.raw)
    if not args.diagnose:
        print('[source-context] ' + json.dumps(context), file=sys.stderr, flush=True)
    try:
        path, record = source_record(args.raw, create_inventory=not args.diagnose)
    except (ValueError, OSError) as error:
        diagnostic = {**context, **getattr(error, 'diagnostic', {}),
                      'passed': False, 'error': str(error)}
        if args.diagnose:
            print(json.dumps(diagnostic), flush=True)
        else:
            print('[source-diagnostic] ' + json.dumps(diagnostic), file=sys.stderr, flush=True)
        raise SystemExit(1)
    if args.diagnose:
        print(json.dumps({**context, 'passed': True, 'source_record': str(path),
                          'revision': record['revision'],
                          'stat_validation': record.get('stat_validation')}), flush=True)
        return
    print(f'[source] Using {path}', file=sys.stderr, flush=True)
    print(record['repo_id'] + '@' + record['revision'], flush=True)


if __name__ == '__main__':
    main()
