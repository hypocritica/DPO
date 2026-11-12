"""Frozen checkpoint plans with independent generation and scoring stages.

Planning needs only prompt/checkpoint metadata. Scoring never reloads the
generator, its checkpoint, or training run. Existing evaluate_batch.py is kept
for the original SD1.5 protocol.
"""
import argparse
import copy
import csv
import fnmatch
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from self_correct_dpo.common import digest, load_suite, output_lock, read_json, read_jsonl, sha256
from self_correct_dpo.config import load_config
from self_correct_dpo.resources import CONFIG_DIR, load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.evaluation.runner import run_command
from self_correct_dpo.evaluation.experiment_catalog import (METHODS, base_record, discover_roots,
                                                          draft_catalog, load_catalog, model_spec, roots_from_args)

FAMILIES = ('sd15', 'sdxl', 'sd35_medium', 'zimage')
PANELS = ('pickv2', 'hpsv2', 'partiprompts', 'geneval', 'dpg')
COMPOSITIONAL = ('geneval', 'dpg')


def safe_target(root, relative):
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()) or target == Path(root).resolve():
        raise ValueError(f'Unsafe output path: {relative}')
    return target


def freeze_json(path, data):
    path = Path(path)
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f'Inputs changed; use a new output directory: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, data)


def selected_steps(available, *, steps=None, latest=None, every=None, minimum=1, maximum=None, base=False):
    available = sorted(set(available) - {0})
    if steps is not None:
        if not steps or len(set(steps)) != len(steps) or any(s < 0 for s in steps):
            raise ValueError('--steps must contain distinct nonnegative steps')
        missing = set(steps) - set(available) - {0}
        if missing:
            raise ValueError(f'Requested permanent checkpoints are missing: {sorted(missing)}')
        chosen = sorted(steps)
    else:
        chosen = [s for s in available if s >= minimum and (maximum is None or s <= maximum)
                  and (every is None or s % every == 0)]
        if latest is not None:
            chosen = chosen[-latest:]
    if base:
        chosen = sorted(set(chosen) | {0})
    return chosen


def panel_provenance(source, name, checksum, count):
    """Freeze prepared-panel scope only after checking its current bytes/count."""
    selection_path = Path(source).parent / 'selection.json'
    if not selection_path.is_file():
        return {'mode': 'custom-unverified',
                'provenance': {'status': 'unverified', 'reason': 'No adjacent selection.json'}}
    import hashlib
    payload = selection_path.read_bytes()
    try:
        selection = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f'Invalid prompt selection provenance: {selection_path}') from error
    if not isinstance(selection, dict):
        raise ValueError(f'Invalid prompt selection provenance: {selection_path}')
    if 'panels' in selection:
        panel = selection['panels'].get(name) if isinstance(selection['panels'], dict) else None
        valid = isinstance(panel, dict) and panel.get('sha256') == checksum and panel.get('prompts') == count
    else:
        valid = (selection.get('benchmark') == name and selection.get('suite_sha256') == checksum
                 and selection.get('prompts') == count)
    mode = selection.get('mode')
    if not valid or mode not in ('full-benchmark', 'smoke-only', 'development-curve'):
        raise ValueError(f'Prompt selection hash/count/benchmark/mode mismatch: {selection_path}')
    return {'mode': mode, 'provenance': {'status': 'selection-verified',
            'selection_path': str(selection_path), 'selection_sha256': hashlib.sha256(payload).hexdigest(),
            'selection': selection}}


def make_plan(args):
    registry = load_resources(args.resources)
    catalog = None
    if args.catalog:
        methods = args.methods
        if args.include_base and methods and 'base' not in methods:
            methods = [*methods, 'base']
        runs, roots, catalog = load_catalog(args.catalog, registry, families=args.families,
                                           methods=methods, patterns=args.run_glob)
    else:
        roots = roots_from_args(args.runs_root, registry)
        runs = discover_roots(roots, patterns=args.run_glob, families=args.families)
    out = Path(args.out).expanduser().resolve()
    if any(out == root or out.is_relative_to(root) or root.is_relative_to(out) for root in roots):
        raise ValueError('Evaluation output must be separate from the training runs tree')
    config = load_config(args.config, benchmarks=args.benchmarks, resources=args.resources)
    if not runs:
        raise ValueError('No matching training runs; inspect the list command first')
    panels = {}
    for name in args.benchmarks:
        source = Path(args.prompts_root) / f'{name}.jsonl' if args.prompts_root else Path(config['benchmarks'][name])
        source = source.expanduser().resolve()
        rows = load_suite(source)
        panels[name] = {'source': str(source), 'snapshot': f'prompts/{name}.jsonl',
                        'sha256': sha256(source), 'prompts': len(rows)}
        panels[name].update(panel_provenance(source, name, panels[name]['sha256'], len(rows)))
    tasks, bases = [], set()
    for run in runs:
        checkpoints = {r['step']: r for r in run['checkpoints']}
        steps = (run.get('selected_steps', run['steps']) if args.catalog else
                 selected_steps(run['steps'], steps=args.steps, latest=args.latest, every=args.every,
                                minimum=args.min_step, maximum=args.max_step, base=args.include_base))
        for step in steps:
            chosen = run
            if step == 0:
                key = digest(model_spec(run['model']))
                if key in bases:
                    continue
                bases.add(key)
                chosen = run if 'base_identity' in run else base_record(run['model'])
                identity = chosen['base_identity']
            else:
                saved = checkpoints[step]
                identity = {'training_sha256': run['training_sha256'], 'run_id': run['run_id'], 'step': step,
                            'checkpoint_id': saved['checkpoint_id'],
                            'checkpoint_sha256': sha256(Path(saved['path']) / 'checkpoint.json')}
            for panel, info in panels.items():
                sampling = copy.deepcopy(config['protocols'][run['family']])
                if getattr(args, 'generation_batch_size', None) is not None:
                    sampling['generation_batch_size'] = args.generation_batch_size
                batch_size = sampling.get('generation_batch_size', 1)
                if type(batch_size) is not int or batch_size < 1:
                    raise ValueError('generation_batch_size must be a positive integer')
                if panel in config.get('benchmark_seeds', {}):
                    sampling['seeds'] = config['benchmark_seeds'][panel]
                if args.seeds is not None:
                    sampling['seeds'] = args.seeds
                seeds = sampling['seeds']
                if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
                    raise ValueError('Seeds must be distinct nonnegative integers')
                if panel in COMPOSITIONAL and len(seeds) != 4:
                    raise ValueError(f'{panel} requires four fixed seeds per prompt')
                relative = str(Path(chosen['name']) / f'step-{step:07d}' / panel)
                safe_target(out, relative)
                task = {'run': chosen['name'], 'run_path': chosen['path'], 'family': chosen['family'], 'step': step,
                        'method': chosen.get('method', 'unregistered'),
                        'method_label': chosen.get('method_label', 'Unregistered'),
                        'experiment_id': chosen.get('experiment_id', chosen['name']),
                        'model': chosen['model'], 'checkpoint': identity, 'benchmark': panel,
                        'sampling': sampling, 'prompts': info['prompts'], 'images': info['prompts'] * len(seeds),
                        'output': relative}
                if chosen.get('model_audit'):
                    task['model_audit'] = chosen['model_audit']
                if chosen.get('method_config'):
                    task['method_config'] = chosen['method_config']
                task['task_id'] = digest(task)
                tasks.append(task)
    if not tasks:
        raise ValueError('No checkpoints match the selection; use --include-base or change step filters')
    outputs = [t['output'] for t in tasks]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Duplicate output paths in plan')
    comp = Path(config.get('compositional_config', 'compositional.json'))
    if not comp.is_absolute():
        comp = Path(args.config).resolve().parent / comp
    plan = {'version': 1, 'runs_root': str(roots[0]) if len(roots) == 1 else None,
            'runs_roots': [str(root) for root in roots], 'catalog': catalog, 'resources': str(Path(args.resources).resolve()) if args.resources else None,
            'config_sha256': sha256(args.config), 'panels': panels, 'tasks': tasks,
            'scorers_config': config['scorers_config'], 'compositional_config': str(comp.resolve()),
            'metric_groups': config['metric_groups'], 'all_scheduled_samples_required': True}
    plan['plan_id'] = digest(plan)
    if not args.dry_run:
        with output_lock(str(out) + '.plan'):
            if (out / 'plan.json').exists() and read_json(out / 'plan.json') != plan:
                raise ValueError('This directory has a different plan; choose a new --out')
            if out.exists() and not (out / 'plan.json').exists() and any(out.iterdir()):
                raise ValueError('Refusing to place a plan in a nonempty unrelated directory')
            out.mkdir(parents=True, exist_ok=True)
            for info in panels.values():
                target = safe_target(out, info['snapshot'])
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if sha256(target) != info['sha256']:
                        raise ValueError('Frozen prompt snapshot changed')
                else:
                    payload = Path(info['source']).read_bytes()
                    import hashlib
                    if hashlib.sha256(payload).hexdigest() != info['sha256']:
                        raise ValueError('Prompt source changed while planning')
                    target.write_bytes(payload)
            freeze_json(out / 'plan.json', plan)
    return plan


def read_plan(out):
    out = Path(out).expanduser().resolve()
    plan = read_json(out / 'plan.json')
    if plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('Plan checksum mismatch')
    for panel in plan['panels'].values():
        if sha256(safe_target(out, panel['snapshot'])) != panel['sha256']:
            raise ValueError('Frozen prompt snapshot checksum mismatch')
    return plan, out


def filter_tasks(plan, args):
    return [t for t in plan['tasks'] if
            (not args.families or t['family'] in args.families) and
            (not getattr(args, 'methods', None) or t.get('method') in args.methods) and
            (not args.run_glob or any(fnmatch.fnmatchcase(t['run'], p) or fnmatch.fnmatchcase(Path(t['run']).name, p)
                                     for p in args.run_glob)) and
            (not args.steps or t['step'] in args.steps) and
            (not args.benchmarks or t['benchmark'] in args.benchmarks)]


def metrics_for(task, plan, args):
    from self_correct_dpo.evaluation.scorers import METRICS
    if args.metrics:
        metrics = args.metrics
    elif task['benchmark'] in COMPOSITIONAL:
        metrics = [task['benchmark']]
    else:
        group = args.metric_group or ('legacy' if task['family'] in ('sd15', 'sdxl') else 'modern')
        metrics = plan['metric_groups'][group]
    if len(set(metrics)) != len(metrics) or any(m not in (*METRICS, *COMPOSITIONAL) for m in metrics):
        raise ValueError('Use distinct supported metrics')
    for metric in metrics:
        if metric in COMPOSITIONAL and metric != task['benchmark']:
            raise ValueError(f'{metric} requires its own prompt panel, not {task["benchmark"]}')
    return metrics


def execution_flags(args):
    return ['--execution', args.execution, '--run-label', args.run_label]


def generate_task(task, plan, out, args, command_runner=run_command):
    from self_correct_dpo.evaluation.checkpoint_catalog import resolve_checkpoint
    if task['step'] == 0 and 'base_model_id' in task['checkpoint']:
        model, adapter = task['model'], None
        identity = task['checkpoint']
        if identity['base_model_id'] != digest(model):
            raise ValueError('Base model identity differs from the frozen plan')
    else:
        model, adapter, identity = resolve_checkpoint(task['run_path'], task['step'])
        if model != task['model'] or any(identity.get(k) != v for k, v in task['checkpoint'].items()):
            raise ValueError('Training run or checkpoint changed after planning')
    target = safe_target(out, task['output'])
    config = {'model': model, 'sampling': task['sampling']}
    arm = 'base'
    if adapter:
        arm = 'aligned'
        config['aligned'] = {'kind': 'peft', 'path': str(adapter), 'revision': identity['checkpoint_id']}
    with output_lock(str(target) + '.generate'):
        # Check-only may publish the tiny invocation config, never images/models.
        freeze_json(target / 'render-config.json', config)
        command = [args.python or sys.executable, '-m', 'self_correct_dpo.evaluation.render',
                   '--config', str(target / 'render-config.json'), '--arm', arm,
                   '--prompts', str(safe_target(out, plan['panels'][task['benchmark']]['snapshot'])),
                   '--out', str(target / 'render'), *execution_flags(args)]
        if (target / 'render').exists():
            command.append('--resume')
        if args.check_only:
            command.append('--check-only')
        command_runner(command)


def score_task(task, metric, plan, out, args, command_runner=run_command):
    target = safe_target(out, task['output'])
    render = target / 'render'
    validate_task_render(task, plan, out)
    if metric in COMPOSITIONAL:
        # Config selects this evaluator's isolated Python environment.
        from self_correct_dpo.evaluation.compositional import load_settings
        config_path = args.compositional_config or plan['compositional_config']
        resources = args.resources or os.environ.get('SCD_RESOURCES')
        spec = load_settings(config_path, metric, resources=resources)
        command = [args.python or spec.get('python') or sys.executable, '-m',
                   'self_correct_dpo.evaluation.compositional', '--benchmark', metric,
                   '--run', str(render), '--out', str(target / 'scores' / metric),
                   '--config', str(Path(config_path).resolve()), *execution_flags(args)]
        if resources:
            command += ['--resources', resources]
        if args.python:
            command += ['--python', args.python]
    else:
        config_path = args.scorers_config or plan['scorers_config']
        spec = load_config(config_path, scorers=[metric])['scorers'][metric]
        command = [args.python or spec.get('python') or sys.executable, '-m',
                   'self_correct_dpo.evaluation.metrics', '--metric', metric,
                   '--run', str(render), '--out', str(target / 'scores' / metric),
                   '--scorers-config', str(Path(config_path).resolve()), *execution_flags(args)]
    if args.check_only:
        command.append('--check-only')
    command_runner(command)


def render_header(target):
    """Check the committed journal without re-reading every PNG during summary."""
    render = Path(target) / 'render'
    run = read_json(render / 'run.json')
    complete = read_json(render / 'complete.json')
    if run['run_id'] != digest({k: v for k, v in run.items() if k != 'run_id'}):
        raise ValueError('Render identity checksum mismatch')
    if run['suite_id'] != digest(run['suite']):
        raise ValueError('Render prompt checksum mismatch')
    manifest = render / 'manifest.jsonl'
    if complete['manifest_sha256'] != sha256(manifest):
        raise ValueError('Render manifest checksum mismatch')
    rows = read_jsonl(manifest)
    if complete['images'] != len(rows) or len(rows) != run['expected_images']:
        raise ValueError('Incomplete render manifest')
    return run, complete


def validate_task_render(task, plan, out):
    """Match completed pixels to this plan without accessing training/model files."""
    run, complete = render_header(safe_target(out, task['output']))
    expected_config = {'model': task['model'], 'sampling': task['sampling']}
    arm = 'base'
    if task['step']:
        arm = 'aligned'
        adapter = Path(task['run_path']) / 'checkpoints' / f'step-{task["step"]:07d}' / 'adapter'
        expected_config['aligned'] = {'kind': 'peft', 'path': str(adapter),
                                      'revision': task['checkpoint']['checkpoint_id']}
    suite = load_suite(safe_target(out, plan['panels'][task['benchmark']]['snapshot']))
    if (run['config'] != expected_config or run['arm'] != arm or run['suite'] != suite
            or run['full_suite_id'] != digest(suite) or run['expected_images'] != task['images']
            or run['sampling']['seeds'] != task['sampling']['seeds']):
        raise ValueError('Generated model, prompts, checkpoint or sampling differ from the frozen plan')
    return run, complete


def score_summary(target, metric, run, manifest_sha256):
    directory = target / 'scores' / metric
    complete = read_json(directory / 'complete.json')
    if complete['summary_sha256'] != sha256(directory / 'summary.json'):
        raise ValueError('Score summary checksum mismatch')
    if metric not in COMPOSITIONAL:
        if complete['scores_sha256'] != sha256(directory / 'scores.jsonl'):
            raise ValueError('Scores checksum mismatch')
        identity = read_json(directory / 'identity.json')
        if identity['evaluation_id'] != digest(identity['identity']):
            raise ValueError('Score identity checksum mismatch')
        if identity['identity']['scorer'].get('metric') != metric:
            raise ValueError('Score directory is labelled with a different metric')
        if identity['identity']['run_id'] != run['run_id'] or identity['identity']['manifest_sha256'] != manifest_sha256:
            raise ValueError('Score belongs to a different generated image panel')
    else:
        from self_correct_dpo.evaluation.compositional import verify_result
        verify_result(directory, target / 'render')
    summary = read_json(directory / 'summary.json')
    if summary.get('metric') != metric:
        raise ValueError('Summary metric differs from the requested score')
    if metric not in COMPOSITIONAL and summary.get('evaluation_id') != identity['evaluation_id']:
        raise ValueError('Score summary belongs to a different evaluation')
    if summary.get('images') != run['expected_images']:
        raise ValueError('Score does not cover every scheduled image')
    return summary


def summarize(plan, out, args):
    rows = []
    for task in filter_tasks(plan, args):
        row = {k: task[k] for k in ('run', 'family', 'step', 'benchmark', 'prompts', 'images')}
        row.update({k: task.get(k, '') for k in ('experiment_id', 'method', 'method_label')})
        row.update(seeds=','.join(map(str, task['sampling']['seeds'])), status='pending_generation', error='',
                   output=str(safe_target(out, task['output'])))
        try:
            target = safe_target(out, task['output'])
            run, complete = validate_task_render(task, plan, out)
            row['status'] = 'complete'
            for metric in metrics_for(task, plan, args):
                try:
                    score = score_summary(target, metric, run, complete['manifest_sha256'])
                    row[metric] = score['mean']
                    row[metric + '_images'] = score['images']
                    row[metric + '_protocol'] = score.get('protocol', score.get('convention'))
                except FileNotFoundError:
                    row[metric] = None
                    if row['status'] != 'failed':
                        row['status'] = 'pending_scores'
                failure = target / 'failures' / f'score-{metric}.json'
                if failure.exists():
                    row['status'] = 'failed'
                    row['error'] += read_json(failure)['error'] + '\n'
        except FileNotFoundError:
            failure = safe_target(out, task['output']) / 'failures/generate.json'
            if failure.exists():
                row.update(status='failed', error=read_json(failure)['error'])
        except (ValueError, KeyError, TypeError) as error:
            row.update(status='invalid', error=str(error))
        rows.append(row)
    result = {'plan_id': plan['plan_id'], 'rows': rows,
              'complete': bool(rows) and all(r['status'] == 'complete' for r in rows)}
    name = args.summary_name
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('--summary-name must be a simple file stem')
    with output_lock(out / name):
        atomic_json(out / f'{name}.json', result)
        stream = io.StringIO()
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields or ['status'])
        writer.writeheader()
        writer.writerows(rows)
        temporary = out / f'.{name}.csv.tmp'
        temporary.write_text(stream.getvalue())
        temporary.replace(out / f'{name}.csv')
    return result


def run_stage(plan, out, args):
    tasks = filter_tasks(plan, args)
    work = [(t, None) for t in tasks] if args.command == 'generate' else [
        (t, metric) for t in tasks for metric in metrics_for(t, plan, args)]
    if not work:
        raise ValueError('No planned tasks match these filters')
    failures = []
    selected = work[args.shard_index::args.shard_count]
    for task, metric in selected:
        label = task['output'] + (f'/{metric}' if metric else '')
        failure_path = safe_target(out, task['output']) / 'failures' / (f'score-{metric}.json' if metric else 'generate.json')
        print(f'{args.command}: {label}', flush=True)
        try:
            if metric:
                score_task(task, metric, plan, out, args)
            else:
                generate_task(task, plan, out, args)
            if not args.check_only:
                failure_path.unlink(missing_ok=True)
        except (Exception,) as error:
            failure = {'task': label, 'error': f'{type(error).__name__}: {error}'}
            failures.append(failure)
            if not args.check_only:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(failure_path, failure)
            print(json.dumps(failure), file=sys.stderr, flush=True)
            # Cancellation must never start another expensive job.
            if isinstance(error, subprocess.CalledProcessError) and (error.returncode < 0 or error.returncode in (75, 130, 143)):
                return 130
            if not args.keep_going:
                break
    print(json.dumps({'stage': args.command, 'selected': len(selected), 'failures': failures}))
    return int(bool(failures))


def launch_workers(args, argv):
    devices = [s.strip() for s in args.gpus.split(',')]
    if any(not s for s in devices) or len(set(devices)) != len(devices):
        raise ValueError('--gpus requires distinct allocated device IDs')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and not set(devices).issubset(visible.split(',')):
        raise ValueError('--gpus must select devices from CUDA_VISIBLE_DEVICES')
    at = next(i for i, v in enumerate(argv) if v == '--gpus' or v.startswith('--gpus='))
    argv = list(argv)
    del argv[at:at + (1 if '=' in argv[at] else 2)]
    children, stopped = [], [False]
    def forward(signum, frame):
        stopped[0] = True
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)
    previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, device in enumerate(devices):
            if stopped[0]:
                break
            children.append(subprocess.Popen([sys.executable, str(CONFIG_DIR.parent / 'evaluate_suite.py'), *argv,
                '--shard-index', str(index), '--shard-count', str(len(devices))],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=device)))
        failed = any([c.wait() != 0 for c in children])
        return 130 if stopped[0] else int(failed)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('list', 'catalog', 'plan', 'generate', 'score', 'summarize', 'gallery', 'index'):
        p = commands.add_parser(name)
        p.add_argument('--resources', help='SLAI resource registry (also inherited by subprocesses)')
        p.add_argument('--run-glob', nargs='+', help='Quoted run paths/globs relative to runs root')
        p.add_argument('--families', nargs='+', choices=FAMILIES)
        if name != 'catalog':
            p.add_argument('--methods', nargs='+', choices=tuple(METHODS), help='Explicit catalog methods; never inferred from run names')
        if name in ('list', 'catalog', 'plan'):
            p.add_argument('--runs-root', nargs='+', help='One or more read-only training roots')
            if name != 'catalog':
                p.add_argument('--catalog', help='Explicit models/methods/runs/exact-steps JSON registry')
        if name == 'list':
            p.add_argument('--json', action='store_true', help='Full metadata instead of a compact checkpoint table')
        if name == 'catalog':
            p.add_argument('--out', required=True, help='New experiment catalog JSON file to review')
            p.add_argument('--dry-run', action='store_true', help='Print the imported draft; write nothing')
        if name not in ('list', 'catalog'):
            p.add_argument('--out', required=True, help='Independent evaluation directory containing plan.json')
            p.add_argument('--steps', nargs='+', type=int)
            p.add_argument('--benchmarks', nargs='+', choices=PANELS, default=None)
        if name == 'plan':
            p.add_argument('--config', default=str(CONFIG_DIR / 'evaluation-suite.json'))
            p.add_argument('--prompts-root', help='Directory containing selected {benchmark}.jsonl files')
            p.add_argument('--latest', type=int, help='Take latest N permanent snapshots per run after filters')
            p.add_argument('--every', type=int, help='Only steps divisible by this interval')
            p.add_argument('--min-step', type=int, default=1)
            p.add_argument('--max-step', type=int)
            p.add_argument('--include-base', action='store_true')
            p.add_argument('--seeds', nargs='+', type=int, help='Explicit protocol override; use a fresh output')
            p.add_argument('--generation-batch-size', type=int, help='Images per GPU forward; frozen in sampling, use a fresh output')
            p.add_argument('--dry-run', action='store_true', help='Print image counts and plan; write nothing')
        if name in ('score', 'summarize'):
            p.add_argument('--metrics', nargs='+')
            p.add_argument('--metric-group', choices=('legacy', 'modern'))
        if name in ('generate', 'score'):
            p.add_argument('--python', help='Python for generation/scoring; overrides scorer-specific Python')
            p.add_argument('--check-only', action='store_true', help='Check local assets/dependencies without GPU execution')
            p.add_argument('--execution', choices=('standalone', 'slurm'), default='standalone')
            p.add_argument('--run-label', default='self-correct-offline-evaluation')
            p.add_argument('--gpus', help='Comma-separated allocated GPU IDs; one worker per GPU')
            p.add_argument('--shard-index', type=int, default=0)
            p.add_argument('--shard-count', type=int, default=1)
            p.add_argument('--keep-going', action='store_true', help='Record failures and continue independent tasks; exit nonzero')
        if name == 'score':
            p.add_argument('--scorers-config', help='May be prepared after generation; score identity remains immutable')
            p.add_argument('--compositional-config')
        if name == 'summarize':
            p.add_argument('--summary-name', default='summary', help='Separate names for different metric/filter tables')
        if name == 'gallery':
            p.add_argument('--gallery-name', default='gallery')
            p.add_argument('--copy-images', action='store_true', help='Copy selected originals for a portable gallery')
            p.add_argument('--limit', type=int, default=200, help='Maximum images in HTML (never affects evaluation)')
            p.add_argument('--scene-id', nargs='+', help='Choose specific prompt IDs for matched qualitative panels')
            p.add_argument('--seed', type=int, nargs='+')
    args = parser.parse_args(argv)
    if args.command in ('list', 'plan'):
        if args.catalog and args.runs_root:
            parser.error('--catalog declares its own runs_roots; do not pass --runs-root')
        if args.methods and not args.catalog:
            parser.error('--methods requires --catalog for list/plan')
    if args.command == 'plan':
        if args.benchmarks is None:
            args.benchmarks = list(PANELS if args.catalog else PANELS[:3])
        if args.catalog and (args.steps is not None or args.latest is not None or args.every is not None
                             or args.min_step != 1 or args.max_step is not None):
            parser.error('--catalog supplies exact per-experiment steps; edit the catalog instead of step filters')
        if args.steps is not None and (args.latest is not None or args.every is not None or args.min_step != 1 or args.max_step is not None):
            parser.error('Use explicit --steps OR interval/latest filters')
        if any(v is not None and v <= 0 for v in (args.latest, args.every)):
            parser.error('--latest and --every must be positive')
        if args.min_step < 0 or (args.max_step is not None and args.max_step < args.min_step):
            parser.error('Invalid step range')
    if args.command in ('generate', 'score'):
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            parser.error('Invalid worker shard')
        if args.gpus and (args.shard_count != 1 or args.shard_index != 0):
            parser.error('--gpus cannot be combined with manual shards')
    if args.command not in ('list', 'catalog') and args.benchmarks and len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error('Select each benchmark once')
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.resources:
        os.environ['SCD_RESOURCES'] = str(Path(args.resources).expanduser().resolve())
    if args.command == 'catalog':
        registry = load_resources(args.resources)
        draft = draft_catalog(roots_from_args(args.runs_root, registry), families=args.families,
                              patterns=args.run_glob)
        if not args.dry_run:
            target = Path(args.out).expanduser().resolve()
            roots = [Path(root) for root in draft['runs_roots'].values()]
            if any(target == root or target.is_relative_to(root) for root in roots):
                raise ValueError('Catalog output must be separate from the read-only training roots')
            with output_lock(str(target) + '.catalog'):
                freeze_json(target, draft)
        print(json.dumps(draft, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'list':
        registry = load_resources(args.resources)
        if args.catalog:
            runs, _, _ = load_catalog(args.catalog, registry, families=args.families,
                                      methods=args.methods, patterns=args.run_glob)
        else:
            runs = discover_roots(roots_from_args(args.runs_root, registry),
                                  patterns=args.run_glob, families=args.families)
        if args.json:
            print(json.dumps(runs, ensure_ascii=False, indent=2))
        else:
            print('RUN\tFAMILY\tMETHOD\tPERMANENT STEPS (0 = base)')
            for run in runs:
                print(f'{run["name"]}\t{run["family"]}\t{run.get("method_label", "Unregistered")}\t'
                      + ','.join(map(str, run.get('selected_steps', run['steps']))))
            print(f'{len(runs)} runs; metadata only, checkpoint tensor hashes checked at generation.')
        return 0
    if args.command == 'plan':
        plan = make_plan(args)
        print(json.dumps({'plan_id': plan['plan_id'], 'tasks': len(plan['tasks']),
                          'total_images': sum(t['images'] for t in plan['tasks']),
                          'output': str(Path(args.out).resolve()), 'dry_run': args.dry_run,
                          'selection': [{k: t[k] for k in ('run', 'family', 'method', 'method_label', 'experiment_id', 'step', 'benchmark', 'prompts', 'images')}
                                        for t in plan['tasks']]}, ensure_ascii=False, indent=2))
        return 0
    plan, out = read_plan(args.out)
    if not args.resources and plan.get('resources'):
        os.environ['SCD_RESOURCES'] = plan['resources']
    if args.command == 'summarize':
        result = summarize(plan, out, args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['complete'] else 1
    if args.command == 'index':
        from self_correct_dpo.evaluation.image_index import export_image_index
        result = export_image_index(plan, out, args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['complete'] else 1
    if args.command == 'gallery':
        from self_correct_dpo.evaluation.gallery import make_gallery
        print(make_gallery(plan, out, args))
        return 0
    if args.gpus:
        return launch_workers(args, argv)
    return run_stage(plan, out, args)


if __name__ == '__main__':
    raise SystemExit(main())
