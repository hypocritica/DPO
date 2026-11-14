"""Freeze manual checkpoint choices, then use evaluate_suite.py for each evaluation stage."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('--family', required=True, choices=('sd35_medium', 'zimage'))
    init.add_argument('--from-prepared')
    init.add_argument('--runs-root', nargs='+')
    init.add_argument('--catalog')
    init.add_argument('--run-glob', nargs='+', help='Explicit discovery paths/globs; resolve smoke/backup ambiguity')
    init.add_argument('--config', default=str(ROOT / 'configs/evaluation-suite.json'))
    init.add_argument('--out', required=True)
    init.add_argument('--generation-batch-size', type=int, help='Frozen images per GPU forward (SD3.5 default 8; Z-Image default 1)')
    for name in ('check', 'plan'):
        sub = commands.add_parser(name)
        sub.add_argument('--selection', required=True)
        if name == 'plan':
            sub.add_argument('--out', required=True)
            sub.add_argument('--dry-run', action='store_true')
            sub.add_argument('--benchmarks', nargs='+')
    for sub in (init, *[commands.choices[k] for k in ('check', 'plan')]):
        sub.add_argument('--resources')
    args = parser.parse_args(argv)
    from self_correct_dpo.evaluation import selected
    if args.command == 'init':
        result = selected.initialize(args)
        print(json.dumps({'selection': str(Path(args.out).resolve()), 'pending_methods': [c['method'] for c in result['choices']]}, indent=2))
        return 0
    if args.command == 'check':
        _, _, gaps = selected.check(args.selection, args.resources)
        print(json.dumps({'ready': not gaps, 'gaps': gaps}, indent=2))
        return 1 if gaps else 0
    return selected.plan(args)


if __name__ == '__main__':
    raise SystemExit(main())
