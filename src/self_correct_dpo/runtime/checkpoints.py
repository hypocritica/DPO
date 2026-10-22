"""Bounded, manifest-owned adapter snapshots and one atomic rolling resume slot."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import uuid
from self_correct_dpo.common import digest, read_json


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '-' + uuid.uuid4().hex)
    with open(temporary, 'x') as f:
        json.dump(data, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def progress_steps(training):
    if 'lora_checkpoint_every' in training:
        every = training['lora_checkpoint_every']
        if type(every) is not int or every < 1:
            raise ValueError('lora_checkpoint_every must be a positive integer')
        limit = training.get('max_lora_checkpoints')
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('max_lora_checkpoints must be null or a positive integer')
        return sorted(set(range(every, training['max_steps'] + 1, every)) | {training['max_steps']})
    fractions = training.get('checkpoint_fractions', [.1, .3, .6, 1.])
    limit = training.get('max_lora_checkpoints', 4)
    if type(limit) is not int or limit < 1 or not isinstance(fractions, list) or not fractions:
        raise ValueError('Set a positive adapter retention limit and progress fractions')
    if len(fractions) > limit or sorted(set(fractions)) != fractions or fractions[-1] != 1.:
        raise ValueError('Fractions must be increasing, end at 1, and fit retention limit')
    if any(not math.isfinite(x) or not 0 < x <= 1 for x in fractions):
        raise ValueError('Checkpoint fractions must be in (0,1]')
    return sorted({max(1, math.ceil(x * training['max_steps'])) for x in fractions})


def resolve_checkpoint(path):
    from self_correct_dpo.training import checkpoint_manifest
    path = Path(path).resolve()
    if path.is_file():
        pointer = read_json(path)
        if path.name != 'latest.json' or Path(pointer['directory']).name != pointer['directory']:
            raise ValueError('Unsafe rolling checkpoint pointer')
        path = path.parent / pointer['directory']
        info = checkpoint_manifest(path)
        if info['checkpoint_id'] != pointer['checkpoint_id']:
            raise ValueError('Rolling checkpoint pointer checksum mismatch')
    info = checkpoint_manifest(path)
    if not info.get('full_state', (path / 'state.pt').exists()) or not (path / 'state.pt').is_file():
        raise ValueError('Adapter-only snapshot cannot resume optimizer state; use rolling/latest.json')
    return path


def recover_run(out):
    """Finish an owned checkpoint transaction before resolving latest.json."""
    out = Path(out)
    recorded = read_json(out / 'training.json')
    training = recorded['config']['training']
    if training.get('checkpoint_policy') == 'bounded':
        ManagedCheckpoints(out, recorded['run_id'], training).recover()


def validate_extension(path, config, rows):
    from self_correct_dpo.training import checkpoint_manifest
    path = Path(path)
    parent_root = path.parent.parent
    parent = read_json(parent_root / 'training.json')
    checkpoint = checkpoint_manifest(path)
    identity = {k: parent[k] for k in ('config', 'pairs_id', 'base_identity', 'code')}
    for key in ('parent', 'diagnostic_suite_sha256', 'validation_pairs_id', 'monitor_inputs'):
        if key in parent: identity[key] = parent[key]
    if parent['run_id'] != digest(identity):
        raise ValueError('Parent experiment identity mismatch')
    if checkpoint['run_id'] != parent['run_id']:
        raise ValueError('Checkpoint does not belong to recorded parent')
    old, new = copy.deepcopy(parent['config']), copy.deepcopy(config)
    old_budget = old['training'].pop('max_steps'); new_budget = new['training'].pop('max_steps')
    if old != new or new_budget <= old_budget or new_budget <= checkpoint['step']:
        raise ValueError('Extension permits only a larger max_steps, with all other settings fixed')
    if config['training'].get('lr_schedule', 'constant') != 'constant':
        raise ValueError('Budget extension currently requires constant LR to preserve scheduler continuity')
    if digest(rows) != parent['pairs_id']:
        raise ValueError('Extension changed data')
    return {'training': parent, 'checkpoint': checkpoint}


class ManagedCheckpoints:
    def __init__(self, out, run_id, training):
        self.root = Path(out) / 'checkpoints'
        self.run_id = run_id
        self.steps = progress_steps(training)
        self.limit = training.get('max_lora_checkpoints', None if 'lora_checkpoint_every' in training else 4)
        self.index = self.root / 'managed.json'
        self.data = read_json(self.index) if self.index.exists() else {'run_id': run_id, 'progress': [], 'rolling': None, 'garbage': [], 'pending': None}
        if self.data['run_id'] != run_id:
            raise ValueError('Retention manifest belongs to another experiment')

    def recover(self):
        from self_correct_dpo.training import checkpoint_manifest
        # A journal registers staging paths before a write. Complete writes roll
        # forward; incomplete owned staging is removed without touching other files.
        pending = self.data.get('pending')
        if pending:
            name = pending['directory']
            if Path(name).name != name or not (name.startswith('step-') or name.startswith('resume-')):
                raise ValueError('Unsafe pending checkpoint path')
            path = self.root / name
            staging = self.root / ('.pending-' + name)
            if path.exists():
                info = checkpoint_manifest(path)
                if info['run_id'] != self.run_id:
                    raise ValueError('Pending checkpoint belongs to another run')
                if pending['full_state']:
                    old = self.data['rolling']
                    if old and old != name: self.data['garbage'].append(old)
                    self.data['rolling'] = name
                elif name not in self.data['progress']:
                    self.data['progress'].append(name)
                self.data['progress'].sort()
                while self.limit is not None and len(self.data['progress']) > self.limit:
                    self.data['garbage'].append(self.data['progress'].pop(0))
            if staging.exists():
                if staging.is_symlink(): raise ValueError('Unsafe staging symlink')
                marker = staging / 'owner.json'
                if marker.exists():
                    if read_json(marker)['run_id'] != self.run_id: raise ValueError('Staging belongs to another run')
                    shutil.rmtree(staging)
                else:
                    staging.rmdir()  # Only an empty directory can predate its owner marker.
            self.data['pending'] = None
            atomic_json(self.index, self.data)
        # Commit the new pointer before deleting the old slot, including crash recovery.
        if self.data['rolling']:
            self._pointer(self.data['rolling'])
        # Only paths already registered by this run may be removed. Never glob-delete.
        for name in list(self.data['garbage']):
            path = self.root / name
            if Path(name).name != name or not (name.startswith('step-') or name.startswith('resume-')):
                raise ValueError('Unsafe managed checkpoint path')
            if path.exists():
                if checkpoint_manifest(path)['run_id'] != self.run_id:
                    raise ValueError('Refusing to remove a checkpoint owned by another run')
                shutil.rmtree(path)
            self.data['garbage'].remove(name)
        atomic_json(self.index, self.data)

    def _begin(self, name, full_state):
        staging = self.root / ('.pending-' + name)
        if staging.exists() or self.data.get('pending'):
            raise ValueError('Recover pending checkpoint transaction before saving')
        self.data['pending'] = {'directory': name, 'full_state': full_state}
        atomic_json(self.index, self.data)
        return staging

    def _pointer(self, name):
        from self_correct_dpo.training import checkpoint_manifest
        info = checkpoint_manifest(self.root / name)
        atomic_json(self.root / 'latest.json', {'directory': name, 'checkpoint_id': info['checkpoint_id']})

    def save_progress(self, backend, optimizer, scheduler, generator, step, cost):
        from self_correct_dpo.training import save_checkpoint, checkpoint_manifest
        name = f'step-{step:07d}'
        path = self.root / name
        if path.exists():
            if checkpoint_manifest(path)['run_id'] != self.run_id:
                raise ValueError('Progress snapshot belongs to another run')
        else:
            staging = self._begin(name, False)
            save_checkpoint(path, backend, optimizer, scheduler, generator, step, self.run_id, cost, full_state=False, staging_path=staging)
            self.data['pending'] = None
        if name not in self.data['progress']: self.data['progress'].append(name)
        self.data['progress'].sort()
        while self.limit is not None and len(self.data['progress']) > self.limit:
            self.data['garbage'].append(self.data['progress'].pop(0))
        atomic_json(self.index, self.data)
        self.recover()

    def save_resume(self, backend, optimizer, scheduler, generator, step, cost):
        from self_correct_dpo.training import save_checkpoint
        name = f'resume-{step:07d}-{uuid.uuid4().hex}'
        path = self.root / name
        staging = self._begin(name, True)
        save_checkpoint(path, backend, optimizer, scheduler, generator, step, self.run_id, cost, staging_path=staging)
        self.data['pending'] = None
        if self.data['rolling']: self.data['garbage'].append(self.data['rolling'])
        self.data['rolling'] = name
        atomic_json(self.index, self.data)
        self._pointer(name)
        self.recover()
        return path
