"""Sync finished offline segments while training continues in another process.

This command alone uses the network. --dry-run only inspects local records.
The fixed W&B version matches the offline format and sync --append implementation.
"""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys


def inspect_segment(path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    store = DataStore()
    info = {'path': path, 'first': float('inf'), 'last': -1,
            'finished': False, 'base_evaluation': False, 'run_id': None, 'project': None}
    saw_exit = False
    core_header = False
    last_kind = None
    before = path.stat()
    try:
        store.open_for_scan(str(path))
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            kind = record.WhichOneof('record_type')
            last_kind = kind
            if kind == 'header':
                core_header = True
            elif kind == 'run':
                info.update(run_id=record.run.run_id, project=record.run.project)
            elif kind == 'history':
                row = {item.key or '.'.join(item.nested_key): json.loads(item.value_json)
                       for item in record.history.item}
                if '_step' in row:
                    step = row['_step']
                    if type(step) is not int or step < 0 or step <= info['last']:
                        raise ValueError(f'Nonmonotonic offline history: {path}')
                    info['first'] = min(info['first'], step)
                    info['last'] = step
                if row.get('optimizer_step') == 0 and any(k.startswith('evaluation/') for k in row):
                    info['base_evaluation'] = True
            elif kind == 'exit':
                saw_exit = True
        # wandb-core terminates with exit; the Python backend adds final/footer.
        info['finished'] = saw_exit and (last_kind in ('final', 'footer')
                                        or (core_header and last_kind == 'exit'))
    except (AssertionError, EOFError, OSError):
        # A currently open record may end between any two bytes. Never upload it.
        info['finished'] = False
    finally:
        store.close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        info['finished'] = False
    return info


def history_start(path):
    return inspect_segment(path)['first']


def plan_sync(run, project):
    metadata = json.loads((run / 'training.json').read_text())
    config = metadata['config']
    settings = config['training'].get('wandb', {})
    if settings.get('project') != project or settings.get('mode', 'offline') != 'offline':
        raise ValueError(f'Project/offline mode differs: {run}')
    expected_id = metadata['run_id'][:32]
    records = [inspect_segment(path) for path in run.glob('wandb/offline-run-*/run-*.wandb')]
    for record in records:
        if record['run_id'] is not None and (record['run_id'] != expected_id or record['project'] != project):
            raise ValueError(f'Offline run identity differs: {record["path"]}')
    evaluation = config.get('evaluation', {})
    if evaluation.get('enabled') and evaluation.get('include_base', True):
        if not any(r['finished'] and r['base_evaluation'] for r in records):
            return [], 'waiting for the finished step-0 evaluation (created after first training segment)'
    # Do not upload newer records ahead of an unfinished older record.
    barrier = min((r['first'] for r in records if not r['finished']), default=float('inf'))
    finished = sorted((r for r in records if r['finished'] and r['last'] < barrier),
                      key=lambda r: (r['first'], str(r['path'])))
    previous = -1
    plans = []
    for record in finished:
        if record['last'] < 0:
            continue
        if record['first'] <= previous:
            raise ValueError(f'Overlapping offline history after restart: {record["path"]}. '
                             'Keep records intact and review the restart before uploading.')
        previous = record['last']
        marker = record['path'].with_suffix('.wandb.synced')
        if not marker.exists():
            plans.append(record)
    active = sum(not r['finished'] for r in records)
    return plans, f'{len(plans)} finished unsynced segment(s), {active} unfinished segment(s) skipped'


def sync_run(run, entity, project, *, dry_run=False, runner=subprocess.run):
    run = run.expanduser().resolve()
    state_path = run / '.flow-wandb-sync.json'
    destination = {'entity': entity, 'project': project}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('destination') != destination:
            raise ValueError(f'This directory was synced to a different destination: {run}')
    plans, message = plan_sync(run, project)
    print(f'{run.name}: {message}', flush=True)
    for record in plans:
        command = [sys.executable, '-m', 'wandb', 'sync', '--append', '--mark-synced',
                   '--entity', entity, '--project', project, str(record['path'].parent)]
        print(shlex.join(command), flush=True)
        if dry_run:
            continue
        runner(command, check=True)
        if not record['path'].with_suffix('.wandb.synced').is_file():
            raise RuntimeError('W&B did not mark the finished segment as synced; inspect its CLI log')
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'destination': destination,
            'last_history_step': record['last']}, indent=2) + '\n')
        temporary.replace(state_path)
    return len(plans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity', required=True, help='Your W&B user or team; use the same on both servers')
    parser.add_argument('--project', default='mvpo-flow')
    parser.add_argument('--dry-run', action='store_true', help='No login, uploads, synced markers or state writes')
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    if importlib.metadata.version('wandb') != '0.18.7':
        parser.error('Use the tested offline/sync stack: python -m pip install wandb==0.18.7')
    for run in args.runs:
        if args.dry_run:
            sync_run(run, args.entity, args.project, dry_run=True)
        else:
            # Serialize uploaders for this experiment without locking the trainer.
            with (run / '.flow-wandb-sync.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                sync_run(run, args.entity, args.project)


if __name__ == '__main__':
    main()
