"""Read immutable training snapshots without importing a training or GPU stack.

Discovery validates small JSON identities only. Resolution verifies every file
listed in the published checkpoint before the adapter is used for evaluation.
Rolling resume slots, latest pointers, pending writes and symlinked snapshots
are never substituted for a requested step.
"""
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import re

from self_correct_dpo.common import digest, sha256


FAMILIES = ('sd15', 'sdxl', 'sd35_medium', 'zimage')
_STEP = re.compile(r'step-(\d{7,})\Z')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')


class MissingCheckpointError(FileNotFoundError):
    """The requested immutable snapshot has not been published/retained."""


def _read(path):
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f'Invalid JSON metadata: {path}') from error
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _training(run):
    run = Path(run).expanduser().resolve()
    path = run / 'training.json'
    if not path.is_file():
        raise MissingCheckpointError(f'Missing training run: {run}')
    info = _read(path)
    model = info.get('config', {}).get('model', {})
    if not isinstance(info.get('run_id'), str) or not info['run_id']:
        raise ValueError(f'Missing training run_id: {path}')
    if not isinstance(model, dict) or model.get('family') not in FAMILIES:
        raise ValueError(f'Unsupported training model family: {path}')
    if not isinstance(model.get('path'), str) or not model['path']:
        raise ValueError(f'Missing training model path: {path}')
    return run, info, model


def _manifest(run, step, run_id, *, verify_files):
    checkpoint = run / 'checkpoints' / f'step-{step:07d}'
    if checkpoint.is_symlink():
        raise ValueError(f'An immutable checkpoint must not be a symlink: {checkpoint}')
    if not checkpoint.is_dir() or not (checkpoint / 'checkpoint.json').is_file():
        raise MissingCheckpointError(f'Missing published retained checkpoint: {checkpoint}')
    if (checkpoint / 'checkpoint.json').is_symlink():
        raise ValueError(f'An immutable manifest must not be a symlink: {checkpoint}')
    root = checkpoint.resolve()
    if not root.is_relative_to(run):
        raise ValueError(f'Checkpoint directory escapes training run: {checkpoint}')
    info = _read(checkpoint / 'checkpoint.json')
    if info.get('checkpoint_id') != digest({k: v for k, v in info.items() if k != 'checkpoint_id'}):
        raise ValueError(f'Checkpoint manifest identity mismatch: {checkpoint}')
    if info.get('run_id') != run_id or type(info.get('step')) is not int or info['step'] != step:
        raise ValueError(f'Checkpoint run/step identity mismatch: {checkpoint}')
    files = info.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError(f'Checkpoint manifest has no files: {checkpoint}')
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or '..' in Path(name).parts or name != Path(name).as_posix()
                or not isinstance(expected, str) or not _SHA256.fullmatch(expected)):
            raise ValueError(f'Invalid checkpoint file entry: {name!r}')
        file = checkpoint / name
        if file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError(f'Checkpoint file escapes immutable snapshot: {name}')
        if verify_files and (not file.is_file() or sha256(file) != expected):
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    adapter = checkpoint / 'adapter'
    if adapter.is_symlink() or not adapter.is_dir() or not any(name.startswith('adapter/') for name in files):
        raise ValueError(f'Checkpoint adapter missing from manifest: {checkpoint}')
    if verify_files:
        observed = {p.relative_to(checkpoint).as_posix() for p in adapter.rglob('*') if p.is_file()}
        if observed != {name for name in files if name.startswith('adapter/')}:
            raise ValueError(f'Checkpoint contains unregistered adapter files: {checkpoint}')
    return checkpoint, info


def discover_runs(root, patterns=None, families=None):
    """List runs recursively, filtering relative paths or names with shell globs.

    Returned ``steps`` includes baseline step zero. ``files_verified=False`` is
    deliberate: listing does not read tensor weights or optimizer state. Call
    :func:`resolve_checkpoint` before rendering any selected snapshot. A corrupt
    published manifest raises ValueError rather than silently dropping that run.
    """
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Run root must be a directory')
    patterns = [patterns] if isinstance(patterns, str) else list(patterns or ['*'])
    families = [families] if isinstance(families, str) else list(families or FAMILIES)
    if any(f not in FAMILIES for f in families):
        raise ValueError(f'Unsupported families: {families}')
    if any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError('Run patterns must be nonempty strings')
    records = []
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith('.') and d != 'checkpoints')
        if 'training.json' not in files:
            continue
        path = Path(folder)
        directories[:] = []  # A run's own outputs never define nested training runs.
        relative = path.relative_to(root).as_posix() if path != root else path.name
        if not any(fnmatchcase(relative, p) or fnmatchcase(path.name, p) for p in patterns):
            continue
        # Filter families before inspecting snapshots of unrelated experiments.
        info = _read(path / 'training.json')
        if info.get('config', {}).get('model', {}).get('family') not in families:
            continue
        path, info, model = _training(path)
        snapshots = []
        checkpoint_root = path / 'checkpoints'
        if checkpoint_root.is_dir():
            for candidate in sorted(checkpoint_root.iterdir()):
                match = _STEP.fullmatch(candidate.name)
                if not match or not (candidate / 'checkpoint.json').is_file():
                    continue
                step = int(match.group(1))
                if step <= 0 or candidate.name != f'step-{step:07d}':
                    continue
                checkpoint, saved = _manifest(path, step, info['run_id'], verify_files=False)
                snapshots.append({'step': step, 'path': str(checkpoint),
                                  'checkpoint_id': saved['checkpoint_id'],
                                  'full_state': saved.get('full_state', False)})
        snapshots.sort(key=lambda item: item['step'])
        records.append({'name': relative, 'path': str(path), 'run_id': info['run_id'],
                        'family': model['family'], 'model': model,
                        'training_sha256': sha256(path / 'training.json'),
                        'steps': [0, *(item['step'] for item in snapshots)],
                        'checkpoints': snapshots, 'files_verified': False})
    return sorted(records, key=lambda item: item['name'])


def resolve_checkpoint(run, step):
    """Return (model, adapter Path or None, immutable identity), fully verified."""
    if type(step) is not int or step < 0:
        raise ValueError('Checkpoint step must be a nonnegative integer')
    run, info, model = _training(run)
    identity = {'training_sha256': sha256(run / 'training.json'),
                'run_id': info['run_id'], 'step': step, 'checkpoint_id': 'base',
                'checkpoint_sha256': None}
    if step == 0:
        return model, None, identity
    checkpoint, saved = _manifest(run, step, info['run_id'], verify_files=True)
    identity.update(checkpoint_id=saved['checkpoint_id'],
                    checkpoint_sha256=sha256(checkpoint / 'checkpoint.json'))
    return model, checkpoint / 'adapter', identity
