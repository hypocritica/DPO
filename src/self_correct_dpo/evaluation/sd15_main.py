"""Execute the predeclared eleven-version SD1.5 handoff without selecting scores."""
import argparse
import copy
import json
import os
import subprocess
from pathlib import Path
import sys

from self_correct_dpo.common import digest, read_json
from self_correct_dpo.evaluation.checkpoint_catalog import resolve_checkpoint
from self_correct_dpo.evaluation.paper import export_paper
from self_correct_dpo.evaluation.runner import run_command
from self_correct_dpo.evaluation.suite import (freeze_json, make_plan, parse_args as suite_args,
    read_plan, safe_target, summarize, validate_task_render)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = '/mnt/aoss/xietianyu/self_correct/evaluations/sd15/main-v1'
CATALOG = ROOT / 'configs/evaluation-sd15-main.json'
SELECTION = ROOT / 'configs/evaluation-sd15-main-selection.json'
SUITE = ROOT / 'configs/evaluation-suite-sd15-main.json'
PANELS = ('partiprompts', 'pickv2', 'hpsv2')
COUNTS = {'partiprompts': 1632, 'pickv2': 487, 'hpsv2': 3200}
METRICS = ['pickscore', 'hpsv2', 'image_reward', 'clip', 'aesthetic']


def load_selection(path=SELECTION):
    selection = read_json(path)
    variants = selection['variants']
    keys = {(v['method'], v['step']) for v in variants}
    if (len(variants) != 11 or len(keys) != 11 or selection['benchmarks'] != COUNTS
            or selection['benchmark_order'] != list(PANELS)):
        raise ValueError('SD1.5 main requires the eleven-version, three-panel handoff selection')
    if next(v for v in variants if v['method'] == 'kto')['label'] != 'Pair-derived KTO':
        raise ValueError('The Pair-derived KTO label must be retained')
    if {v['step'] for v in variants if v['method'] == 'stagewise_dspo_k6'} != {6000, 10000}:
        raise ValueError('Stage-C DSPO requires both predeclared checkpoints')
    return selection


def validate_plan(plan, selection):
    expected = {(v['method'], v['step'], panel): v for v in selection['variants'] for panel in PANELS}
    tasks = plan['tasks']
    observed = {(t['method'], t['step'], t['benchmark']) for t in tasks}
    if len(tasks) != 33 or observed != set(expected):
        raise ValueError('Frozen SD1.5 plan must contain exactly all 33 handoff tasks')
    for panel, count in COUNTS.items():
        if plan['panels'][panel]['prompts'] != count or plan['panels'][panel].get('mode') != 'full-benchmark':
            raise ValueError(f'SD1.5 main requires the complete official {panel} panel')
    if plan['metric_groups']['legacy'] != METRICS:
        raise ValueError('SD1.5 main requires the original five metrics')
    for task in tasks:
        variant = expected[(task['method'], task['step'], task['benchmark'])]
        if (task['family'] != 'sd15' or task['model'] != selection['model']
                or task['sampling'] != selection['sampling'] or task['prompts'] != COUNTS[task['benchmark']]
                or task['images'] != COUNTS[task['benchmark']]):
            raise ValueError('SD1.5 main model, sampling or image count differs from the handoff')
        identity = task['checkpoint']
        if identity['checkpoint_id'] != variant['checkpoint_id']:
            raise ValueError(f'Historical checkpoint_id changed: {variant["selection_id"]}')
        if task['step'] and (identity.get('run_id') != variant['run_id']
                or task['run_path'] != variant['run_path'] or task['run'] != variant['catalog_id']):
            raise ValueError(f'Historical run identity changed: {variant["selection_id"]}')
        if task['method_label'] != variant['label']:
            raise ValueError(f'Handoff method label changed: {variant["selection_id"]}')


def validate_live_checkpoints(plan):
    """Recheck exact immutable identities and every registered adapter-file SHA256."""
    seen = set()
    for task in plan['tasks']:
        if task['step'] == 0 or (task['run_path'], task['step']) in seen:
            continue
        seen.add((task['run_path'], task['step']))
        model, _, identity = resolve_checkpoint(task['run_path'], task['step'])
        if model != task['model'] or identity != task['checkpoint']:
            raise ValueError(f'Current checkpoint differs from frozen handoff: {task["output"]}')


def plan_main(args, selection):
    flags = ['plan', '--catalog', args.catalog, '--config', args.suite_config,
             '--families', 'sd15', '--benchmarks', *PANELS, '--out', args.out, '--dry-run']
    if args.resources:
        flags += ['--resources', args.resources]
    native = suite_args(flags)
    proposal = make_plan(native)
    validate_plan(proposal, selection)
    validate_live_checkpoints(proposal)
    if not args.dry_run:
        native.dry_run = False
        proposal = make_plan(native)
        validate_plan(proposal, selection)
        freeze_json(Path(args.out).resolve() / 'sd15-selection.json', selection)
    return {'plan_id': proposal['plan_id'], 'tasks': len(proposal['tasks']),
            'images': sum(t['images'] for t in proposal['tasks']), 'dry_run': args.dry_run}


def report_plan(plan, role, benchmarks=None):
    """Subset a copy; the original plan and all generated pixels remain unchanged."""
    chosen = copy.deepcopy(plan)
    chosen['tasks'] = [t for t in chosen['tasks'] if
        (not benchmarks or t['benchmark'] in benchmarks) and
        ((t['method'], t['step']) != ('stagewise_dspo_k6', 10000) if role == 'main' else
         t['method'] == 'stagewise_dspo_k6' and t['step'] in (6000, 10000))]
    chosen['report_selection'] = {'role': role, 'source_plan_id': plan['plan_id'],
        'rule': ('Exclude only stagewise_dspo_k6 step 10000' if role == 'main' else
                 'Only stagewise_dspo_k6 steps 6000 and 10000'),
        'policy': 'Predeclared method/step selection; no per-metric checkpoint choice',
        'selected_task_ids': [t['task_id'] for t in chosen['tasks']]}
    return chosen


def allocated_devices(value):
    value = value if value is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    devices = [v.strip() for v in value.split(',')]
    if len(devices) != 4 or len(set(devices)) != 4 or any(not v or v == '-1' for v in devices):
        raise ValueError('SD1.5 main requires four distinct allocated GPUs; set --gpus to their actual IDs')
    return ','.join(devices)


def stage_commands(args):
    panels = [p for p in PANELS if not args.benchmarks or p in args.benchmarks]
    devices = allocated_devices(args.gpus)
    commands = []
    for panel in panels:
        command = [sys.executable, str(ROOT / 'evaluate_suite.py'), args.stage,
                   '--out', args.out, '--benchmarks', panel, '--gpus', devices]
        if args.stage == 'score':
            command += ['--metric-group', 'legacy']
        for key in ('resources', 'python'):
            if getattr(args, key):
                command += ['--' + key, getattr(args, key)]
        if args.check_only:
            command.append('--check-only')
        if args.keep_going:
            command.append('--keep-going')
        commands.append(command)
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'generate', 'score', 'summarize', 'export', 'gallery'))
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--catalog', default=str(CATALOG))
    parser.add_argument('--selection', default=str(SELECTION))
    parser.add_argument('--suite-config', default=str(SUITE))
    parser.add_argument('--resources')
    parser.add_argument('--benchmarks', nargs='+', choices=PANELS)
    parser.add_argument('--gpus')
    parser.add_argument('--python', help='Isolated generation or scorer Python')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate metadata and print commands; no GPU execution')
    parser.add_argument('--summary-name', default='summary')
    parser.add_argument('--report', choices=('main', 'sensitivity', 'all'), default='all')
    parser.add_argument('--paper-name', default='main-v1')
    parser.add_argument('--sensitivity-name', default='dspo-stage-sensitivity-v1')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--gallery-name', default='sd15-preview-v1')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--scene-id', nargs='+')
    parser.add_argument('--seed', nargs='+', type=int)
    parser.add_argument('--copy-images', action='store_true')
    args = parser.parse_args(argv)
    if args.check_only and args.stage not in ('generate', 'score'):
        parser.error('--check-only is only supported for generate/score')
    if args.dry_run and args.stage not in ('plan', 'generate', 'score'):
        parser.error('--dry-run is only supported for plan/generate/score')
    try:
        selection = load_selection(args.selection)
        if args.stage == 'plan':
            result = plan_main(args, selection)
        else:
            plan, out = read_plan(args.out)
            validate_plan(plan, selection)
            if read_json(out / 'sd15-selection.json') != selection:
                raise ValueError('Frozen SD1.5 handoff selection changed')
            if args.stage in ('generate', 'score'):
                if args.stage == 'generate':
                    validate_live_checkpoints(plan)
                elif not args.dry_run:
                    # Keep generation and scoring separate: all 33 planned image
                    # panels must be committed before any scorer is launched.
                    for task in plan['tasks']:
                        validate_task_render(task, plan, out)
                commands = stage_commands(args)
                failures = []
                for command in commands:
                    if args.dry_run:
                        print(json.dumps(command))
                    else:
                        try:
                            run_command(command)
                        except subprocess.CalledProcessError as error:
                            if not args.keep_going or error.returncode < 0 or error.returncode in (75, 130, 143):
                                raise
                            failures.append({'benchmark': command[command.index('--benchmarks') + 1],
                                             'exit_code': error.returncode})
                result = {'stage': args.stage, 'panels': len(commands), 'dry_run': args.dry_run,
                          'complete': not failures, 'failures': failures}
            elif args.stage == 'summarize':
                native = suite_args(['summarize', '--out', args.out, '--metric-group', 'legacy',
                                     '--summary-name', args.summary_name])
                native.benchmarks = args.benchmarks
                result = summarize(plan, out, native)
            elif args.stage == 'export':
                exports = []
                for role in ('main', 'sensitivity') if args.report == 'all' else [args.report]:
                    native = argparse.Namespace(families=None, methods=None, run_glob=None, steps=None,
                        benchmarks=None, metrics=METRICS, metric_group='legacy', allow_incomplete=args.allow_incomplete,
                        paper_name=args.paper_name if role == 'main' else args.sensitivity_name)
                    exports.append(export_paper(report_plan(plan, role, args.benchmarks), out, native))
                result = {'exports': exports, 'complete': all(r['complete'] for r in exports)}
            else:
                from self_correct_dpo.evaluation.gallery import make_gallery
                native = suite_args(['gallery', '--out', args.out])
                for key in ('gallery_name', 'limit', 'scene_id', 'seed', 'copy_images'):
                    setattr(native, key, getattr(args, key))
                native.benchmarks = args.benchmarks or ['partiprompts']
                result = {'gallery': make_gallery(plan, out, native)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get('complete', True) else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'SD1.5 handoff evaluation failed: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
