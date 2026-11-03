"""Fixed-global-batch resume for every objective on the batch-resume branch."""
import copy
from pathlib import Path

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.runtime.storage_compat import compatible_sources


ENVIRONMENT_FLAG = 'SCD_ALLOW_BATCH_RESIZE'


def batch_settings(config):
    t = config['training']
    return {key: t.get(key, 1) for key in
            ('batch_size', 'gradient_accumulation_steps', 'world_size')}


def validate_batch_change(original, current):
    old, new = copy.deepcopy(original), copy.deepcopy(current)
    for config in (old, new):
        t = config['training']
        for key in ('batch_size', 'gradient_accumulation_steps'):
            if type(t.get(key)) is not int or t[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
    old_settings, new_settings = batch_settings(old), batch_settings(new)
    if (old_settings['batch_size'] * old_settings['gradient_accumulation_steps']
            != new_settings['batch_size'] * new_settings['gradient_accumulation_steps']):
        raise ValueError('Batch resizing must preserve the effective global batch')
    for config in (old, new):
        config['training'].pop('batch_size')
        config['training'].pop('gradient_accumulation_steps')
    if old != new:
        raise ValueError('Batch resizing permits only batch_size and gradient_accumulation_steps changes')


def compatible_batch_sources(recorded, current):
    # This branch changes the training hook and CLI batch repartitioning.
    # All model, algorithm, data, evaluation and checkpoint code stays checked.
    if not isinstance(recorded, dict) or not isinstance(current, dict) or recorded.keys() != current.keys():
        return False
    adjusted = dict(current)
    for name in ('training.py', 'config.py'):
        if name in recorded:
            adjusted[name] = recorded[name]
    return compatible_sources(recorded, adjusted)


def resume_identity(recorded, current):
    validate_batch_change(recorded['config'], current['config'])
    if not compatible_batch_sources(recorded['code'], current['code']):
        raise ValueError('Resume training code changed beyond the batch-resume training/config hooks')
    identity = dict(current, config=recorded['config'], code=recorded['code'])
    if digest(identity) != recorded['run_id']:
        raise ValueError('Batch resume changed model, data, validation or other run identity')
    return identity


def batch_behavior_notes(config):
    t = config['training']
    notes = ['Random-number assignment and floating-point accumulation may change.']
    if t.get('sampling') != 'balanced':
        notes.append('Non-balanced sampling may select different future samples after repartitioning.')
    if t.get('objective') == 'kto':
        notes.append('KTO estimates its KL baseline per global microbatch; changing B changes this estimate.')
    if t.get('objective') == 'dspo' and t.get('dspo_recipe') == 'official_sdxl':
        notes.append('DSPO official_sdxl thresholds depend on the global microbatch prediction-gap mean.')
    return notes


def record_batch_resume(out, recorded, config, code, checkpoint):
    from self_correct_dpo.runtime.checkpoints import atomic_json
    info = read_json(Path(checkpoint) / 'checkpoint.json')
    report = {'policy': 'all-objectives-fixed-global-batch-v2', 'run_id': recorded['run_id'],
              'checkpoint_id': info['checkpoint_id'], 'start_step': info['step'],
              'original_batch': batch_settings(recorded['config']),
              'active_batch': batch_settings(config), 'active_config': config,
              'original_code': recorded['code'], 'active_code': code,
              'policy_sha256': sha256(__file__),
              'reproduces_original_random_trajectory': False,
              'behavior_notes': batch_behavior_notes(config)}
    path = Path(out) / ('batch-resume-' + digest(report)[:16] + '.json')
    if path.exists():
        if read_json(path) != report:
            raise ValueError('Batch resume audit changed')
    else:
        atomic_json(path, report)
    print(f'Batch resume ({config["training"]["objective"]}): {report["active_batch"]}, '
          f'step={info["step"]}; audit: {path}', flush=True)
    if report['original_batch'] != report['active_batch']:
        for note in report['behavior_notes']:
            print(f'Batch change: {note}', flush=True)
