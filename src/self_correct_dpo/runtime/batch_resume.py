"""Fixed-global-batch resume for every objective on the batch-resume branch."""
import copy
from pathlib import Path

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.runtime.storage_compat import compatible_sources


ENVIRONMENT_FLAG = 'SCD_ALLOW_BATCH_RESIZE'

# Only these reviewed entry/data-loader revisions may replace the published
# startup checks. Algorithm, model, data and checkpoint identities stay checked.
FAST_RESUME_SOURCES = {
    'train.py': ('abf78ba2d6cf69f659c0635820e350833a128a8e7aa9b44f19b9a623bf869b06', 'b11fb91caae20d3b8eca85c1f17e3794fbd5bf79a481295286f4f353fba8b300'),
    'validation.py': ('b4343fb6947527dddc444b93bda7dc104e0d74a06102a5ee7c339f21a79ab36d', '8b46cd75ce285133e88bdc86f16cafb00d202a0d93ff52bd6949db88b7034f7a'),
}


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
    for name, versions in FAST_RESUME_SOURCES.items():
        if name in recorded and (recorded[name], current[name]) == versions:
            adjusted[name] = recorded[name]
    return compatible_sources(recorded, adjusted)


def resume_identity(recorded, current):
    validate_batch_change(recorded['config'], current['config'])
    if not compatible_batch_sources(recorded['code'], current['code']):
        raise ValueError('Resume training code changed beyond the batch-resume hooks and approved fast data loader')
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


def record_batch_resume(out, recorded, config, code, checkpoint, *, fast_resume=False):
    from self_correct_dpo.runtime.checkpoints import atomic_json
    info = read_json(Path(checkpoint) / 'checkpoint.json')
    report = {'policy': 'all-objectives-fixed-global-batch-v2', 'run_id': recorded['run_id'],
              'checkpoint_id': info['checkpoint_id'], 'start_step': info['step'],
              'original_batch': batch_settings(recorded['config']),
              'active_batch': batch_settings(config), 'active_config': config,
              'startup_data_checks': 'saved-train-validation-fingerprints' if fast_resume else 'standard',
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
