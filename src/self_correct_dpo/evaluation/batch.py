"""Evaluate exact retained checkpoints on explicit prompt panels, with resumable outputs."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from self_correct_dpo.common import digest, load_suite, output_lock, read_json, sha256
from self_correct_dpo.config import load_config
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.evaluation.runner import evaluate_checkpoint, validate_evaluation
from self_correct_dpo.evaluation.render import validate_config

METRICS = ['pickscore', 'hpsv2', 'aesthetic', 'clip', 'image_reward']
PANELS = ['pickv2', 'hpsv2', 'partiprompts']


class MissingCheckpointError(FileNotFoundError):
    pass


def exact_checkpoint(run, step):
    """Never substitute latest: benchmark steps must have their retained directory."""
    from self_correct_dpo.training import checkpoint_manifest
    run = Path(run).resolve()
    if not (run / 'training.json').is_file():
        raise MissingCheckpointError(f'Missing training run: {run}')
    info = read_json(run / 'training.json')
    checkpoint = run / 'checkpoints' / f'step-{step:07d}'
    if not checkpoint.is_dir():
        raise MissingCheckpointError(f'Missing retained checkpoint: {checkpoint}')
    try:
        saved = checkpoint_manifest(checkpoint)
    except FileNotFoundError as error:
        raise ValueError(f'Committed checkpoint contains missing files: {checkpoint}: {error}') from error
    if saved['run_id'] != info['run_id'] or saved['step'] != step:
        raise ValueError(f'Checkpoint run/step identity mismatch: {checkpoint}')
    if info['config']['model']['family'] != 'sd15':
        raise ValueError(f'Expected an SD1.5 training run: {run}')
    if not (checkpoint / 'adapter').is_dir():
        raise ValueError(f'Checkpoint adapter missing: {checkpoint}')
    return info['config']['model'], checkpoint / 'adapter', {
        'training_sha256': sha256(run / 'training.json'), 'checkpoint_id': saved['checkpoint_id']}


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Cannot resume with changed inputs: {path}')
        return
    with output_lock(str(path) + '.identity'):
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f'Cannot resume with changed inputs: {path}')
        else:
            atomic_json(path, value)


def build_request(args):
    config = load_config(args.config, benchmarks=args.benchmarks, resources=args.resources)
    registry = load_resources(args.resources)
    runs_root = Path(args.runs_root or registry['RUNS_ROOT']).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if out == runs_root or out.is_relative_to(runs_root):
        raise ValueError('--out must be outside RUNS_ROOT')
    runs = args.runs or config['runs']
    if not runs or len(set(runs)) != len(runs) or any(Path(r).name != r or r in ('.', '..') for r in runs):
        raise ValueError('Use distinct run directory names, without paths')
    steps = args.steps or config['steps']
    if not steps or len(set(steps)) != len(steps) or any(s <= 0 for s in steps):
        raise ValueError('Use distinct positive retained checkpoint steps')
    sampling = config['sampling']
    if args.seeds is not None:
        sampling['seeds'] = args.seeds
    if not sampling['seeds'] or len(set(sampling['seeds'])) != len(sampling['seeds']) or any(s < 0 for s in sampling['seeds']):
        raise ValueError('Use distinct nonnegative seeds')
    panels = {name: str((Path(args.prompts_root).expanduser() / f'{name}.jsonl').resolve())
              if args.prompts_root else config['benchmarks'][name] for name in args.benchmarks}
    if args.plan_only:
        inputs = {name: {'prompts': len(load_suite(path)), 'prompts_sha256': sha256(path)} for name, path in panels.items()}
    else:
        inputs = {name: validate_evaluation(path, config['scorers_config'], METRICS) for name, path in panels.items()}
    request = {'version': 1, 'runs_root': str(runs_root), 'runs': runs, 'steps': steps,
               'panels': panels, 'sampling': sampling, 'metrics': METRICS,
               'scorers_config': config['scorers_config'], 'inputs': inputs,
               'batch_source_sha256': sha256(__file__)}
    return request, out


def persist_summary(request, out, rows, suffix):
    result = {'batch_id': digest(request), 'rows': rows}
    atomic_json(out / f'summary{suffix}.json', result)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['run', 'step', 'benchmark', 'status'])
    writer.writeheader()
    writer.writerows(rows)
    temporary = out / f'.summary{suffix}.csv.tmp'
    temporary.write_text(stream.getvalue())
    temporary.replace(out / f'summary{suffix}.csv')


def run_batch(request, out, *, check_only=False, strict_missing=False, shard_index=0, shards=1,
              execution=None, summarize_only=False, plan_only=False, evaluator=evaluate_checkpoint):
    out = Path(out)
    if not check_only:
        immutable(out / 'batch.json', {'request': request, 'batch_id': digest(request)})
    rows = []
    suffix = '' if summarize_only or shards == 1 else f'-worker-{shard_index}'
    for index, (name, step) in enumerate((r, s) for r in request['runs'] for s in request['steps']):
        if not summarize_only and index % shards != shard_index:
            continue
        run = Path(request['runs_root']) / name
        checkpoint_error = None
        try:
            model, adapter, identity = exact_checkpoint(run, step)
            if not plan_only:
                validate_config({'model': model, 'sampling': request['sampling'],
                    'aligned': {'kind': 'peft', 'path': str(adapter),
                                'revision': identity['checkpoint_id']}}, 'aligned')
        except MissingCheckpointError as error:
            checkpoint_error = ('missing_checkpoint', str(error))
        except (ValueError, KeyError, FileNotFoundError) as error:
            checkpoint_error = ('invalid_checkpoint', str(error))
        for panel, prompts in request['panels'].items():
            target = out / name / f'step-{step:07d}' / panel
            row = {'run': name, 'step': step, 'benchmark': panel,
                   'prompts': request['inputs'][panel]['prompts'], 'images': 0,
                   'status': 'planned', 'error': '', 'output': str(target)}
            for metric in METRICS:
                row[metric] = None
                row[metric + '_count'] = 0
            if checkpoint_error:
                row['status'], row['error'] = checkpoint_error
            else:
                try:
                    if not check_only:
                        immutable(target.parent / 'checkpoint-identity.json', identity)
                    elif (target.parent / 'checkpoint-identity.json').exists() and read_json(target.parent / 'checkpoint-identity.json') != identity:
                        raise ValueError('Checkpoint identity changed since evaluation')
                    result = {} if plan_only else evaluator(model, adapter, prompts, request['scorers_config'], METRICS,
                        request['sampling'], target, step=step, execution=execution,
                        check_only=check_only or summarize_only)
                    if 'metrics' in result and isinstance(result['metrics'], dict):
                        row.update(status='complete', images=result['images'])
                        for metric in METRICS:
                            row[metric] = result['metrics'][metric]['mean']
                            row[metric + '_count'] = result['metrics'][metric]['images']
                    elif summarize_only:
                        failure = target.parent / f'{panel}-failure.json'
                        row['status'] = 'failed' if failure.exists() else 'incomplete'
                        if failure.exists():
                            row['error'] = read_json(failure)['error']
                    if row['status'] == 'complete' and not check_only:
                        (target.parent / f'{panel}-failure.json').unlink(missing_ok=True)
                except subprocess.CalledProcessError as error:
                    row.update(status='failed', error=f'Subprocess failed with exit code {error.returncode}')
                    rows.append(row)
                    if not check_only:
                        atomic_json(target.parent / f'{panel}-failure.json', {'error': row['error']})
                        persist_summary(request, out, rows, suffix)
                    raise
                except Exception as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}')
                    if not check_only:
                        atomic_json(target.parent / f'{panel}-failure.json', {'error': row['error']})
            rows.append(row)
            if not check_only:
                persist_summary(request, out, rows, suffix)
    result = {'batch_id': digest(request), 'rows': rows}
    failures = any(r['status'] in ('failed', 'invalid_checkpoint') or
                   (strict_missing and r['status'] == 'missing_checkpoint') for r in rows)
    return result, failures


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).resolve().parents[3] / 'configs/evaluate-selected-sd15.json'))
    p.add_argument('--resources')
    p.add_argument('--runs-root')
    p.add_argument('--prompts-root', help='Use selected {benchmark}.jsonl panels from this directory')
    p.add_argument('--runs', nargs='+')
    p.add_argument('--steps', nargs='+', type=int)
    p.add_argument('--benchmarks', nargs='+', choices=PANELS, default=PANELS)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--out', required=True)
    p.add_argument('--check-only', '--plan', action='store_true')
    p.add_argument('--plan-only', action='store_true', help='CPU manifest/prompt plan without model or scorer asset checks; writes nothing')
    p.add_argument('--strict-missing', action='store_true')
    p.add_argument('--execution', choices=['standalone', 'slurm'], default='standalone')
    p.add_argument('--run-label', default='sd15-checkpoint-benchmarks')
    p.add_argument('--gpus', help='Comma-separated allocated CUDA device IDs; one independent worker per device')
    p.add_argument('--shard-index', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--shards', type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument('--summarize-only', action='store_true', help='Verify completed outputs and rebuild aggregate summaries without generating')
    a = p.parse_args(argv)
    if len(set(a.benchmarks)) != len(a.benchmarks):
        p.error('Select each benchmark once')
    if a.shards < 1 or not 0 <= a.shard_index < a.shards:
        p.error('Invalid worker shard')
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.resources:
        os.environ['SCD_RESOURCES'] = str(Path(args.resources).expanduser().resolve())
    if args.plan_only:
        args.check_only = True
    request, out = build_request(args)
    execution = {'mode': args.execution, 'run_label': args.run_label}
    if args.gpus and not args.check_only and not args.summarize_only:
        devices = args.gpus.split(',')
        if any(not d.strip() for d in devices) or len(set(devices)) != len(devices):
            raise ValueError('--gpus must list distinct allocated devices')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None and not set(devices).issubset(visible.split(',')):
            raise ValueError('--gpus must select devices from CUDA_VISIBLE_DEVICES')
        immutable(out / 'batch.json', {'request': request, 'batch_id': digest(request)})
        original = list(sys.argv[1:] if argv is None else argv)
        at = next(i for i, value in enumerate(original) if value == '--gpus' or value.startswith('--gpus='))
        del original[at:at + (1 if '=' in original[at] else 2)]
        children = []
        interrupted = [False]
        def forward(signum, frame):
            interrupted[0] = True
            for child in children:
                if child.poll() is None:
                    child.send_signal(signum)
        previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            for i, device in enumerate(devices):
                if interrupted[0]:
                    break
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
                children.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[3] / 'evaluate_batch.py'),
                    *original, '--shard-index', str(i), '--shards', str(len(devices))], env=environment))
                if interrupted[0] and children[-1].poll() is None:
                    children[-1].terminate()
            failed = any([child.wait() != 0 for child in children])
        except BaseException:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
            raise
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
        if interrupted[0]:
            return 130
        result, errors = run_batch(request, out, strict_missing=args.strict_missing,
            summarize_only=True, execution=execution)
        print(json.dumps(result))
        return int(failed or errors)
    result, failed = run_batch(request, out, check_only=args.check_only,
        strict_missing=args.strict_missing, shard_index=args.shard_index, shards=args.shards,
        summarize_only=args.summarize_only, plan_only=args.plan_only, execution=execution)
    print(json.dumps(result))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
