"""SD3.5M checkpoint curves: fixed 128 development prompts, seed 17, PickScore."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
METHODS = ('base', 'sft', 'dpo', 'linear_dpo', 'global_k2', 'stagewise_k2', 'global_k4', 'stagewise_k4')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze runs, permanent checkpoints and 128 validation prompts')
    prepare.add_argument('--runs-root', nargs='+')
    prepare.add_argument('--catalog', help='Explicit reviewed catalog to resolve duplicate method candidates')
    prepare.add_argument('--run-glob', nargs='+')
    prepare.add_argument('--resources')
    prepare.add_argument('--config', default=str(ROOT / 'configs/evaluation-suite.json'))
    prepare.add_argument('--target-steps', type=int, default=3200)
    prepare.add_argument('--generation-batch-size', type=int, default=4)
    prepare.add_argument('--exclude-prompts', nargs='+', action='extend', default=[])
    prepare.add_argument('--out', dest='prepare_dir', required=True)
    plan = commands.add_parser('plan', help='Create immutable generation/scoring plan')
    plan.add_argument('--prepared', required=True)
    plan.add_argument('--out', required=True)
    plan.add_argument('--resources')
    plan.add_argument('--dry-run', action='store_true')
    for stage in ('generate', 'score'):
        p = commands.add_parser(stage)
        p.add_argument('--out', required=True)
        p.add_argument('--resources')
        p.add_argument('--gpus', help='Allocated CUDA_VISIBLE_DEVICES IDs, e.g. 0,1 or 0,1,2,3')
        p.add_argument('--python', help='Explicit stage environment interpreter')
        p.add_argument('--check-only', action='store_true')
        p.add_argument('--keep-going', action='store_true')
        p.add_argument('--methods', nargs='+', choices=METHODS)
        p.add_argument('--steps', nargs='+', type=int)
    summary = commands.add_parser('summarize', help='Update points.csv/json and PNG/PDF curves, including pending points')
    summary.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if getattr(args, 'resources', None):
        os.environ['SCD_RESOURCES'] = str(Path(args.resources).expanduser().resolve())
    if args.command == 'prepare':
        from self_correct_dpo.evaluation.curve_prepare import prepare_curve
        args.prompt_count, args.seed = 128, 17
        print(json.dumps(prepare_curve(args), ensure_ascii=False, indent=2))
        return 0
    from self_correct_dpo.evaluation import suite
    if args.command == 'plan':
        from self_correct_dpo.evaluation.curve_prepare import load_prepared
        load_prepared(args.prepared)
        directory = Path(args.prepared).expanduser().resolve()
        prepared = {'catalog': str(directory / 'catalog.json'), 'config': str(directory / 'suite.json'),
                    'prompts_root': str(directory / 'prompts')}
        command = ['plan', '--catalog', prepared['catalog'], '--config', prepared['config'],
                   '--prompts-root', prepared['prompts_root'], '--families', 'sd35_medium',
                   '--methods', *METHODS, '--benchmarks', 'pickv2', '--seeds', '17', '--out', args.out]
        if args.resources:
            command += ['--resources', args.resources]
        if args.dry_run:
            command += ['--dry-run']
        return suite.main(command)
    if args.command == 'summarize':
        from self_correct_dpo.evaluation.curves import summarize_curves
        result = summarize_curves(args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['complete'] else 1
    # The existing scheduler splits checkpoints across one process per allocated
    # GPU. Within-checkpoint image journals and score receipts retain resume state
    # independently of which GPU or number of workers handles the next attempt.
    from self_correct_dpo.evaluation.curves import validate_curve_plan
    plan, out = suite.read_plan(args.out)
    validate_curve_plan(plan, out)
    command = [args.command, '--out', args.out]
    for name in ('resources', 'gpus', 'python'):
        if getattr(args, name):
            command += ['--' + name, getattr(args, name)]
    for name in ('check_only', 'keep_going'):
        if getattr(args, name):
            command += ['--' + name.replace('_', '-')]
    for name in ('methods', 'steps'):
        if getattr(args, name):
            command += ['--' + name, *map(str, getattr(args, name))]
    if args.command == 'score':
        command += ['--metrics', 'pickscore']
    return suite.main(command)


if __name__ == '__main__':
    raise SystemExit(main())
