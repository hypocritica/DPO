"""Resume an existing training run with a new per-GPU batch and unchanged global batch."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from self_correct_dpo.common import read_json, sha256, source_file
from self_correct_dpo.runtime.batch_resume import (
    ENVIRONMENT_FLAG, batch_settings, compatible_batch_sources, validate_batch_change, batch_behavior_notes)
from self_correct_dpo.training import validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--batch-size', '--train_batch_size', type=int, help='New per-GPU B; default: SCD_TRAIN_BATCH_SIZE or saved B')
    p.add_argument('--resume', '--resume_from_checkpoint', default='latest')
    p.add_argument('--execution', choices=['standalone', 'slurm'], default='standalone')
    p.add_argument('--world-size', type=int)
    p.add_argument('--run-label')
    p.add_argument('--expected-objective', help='Optional launcher objective check')
    p.add_argument('--resources')
    p.add_argument('--stop-after-step', type=int, help='Absolute optimizer step at which to pause')
    p.add_argument('--config-only', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.resources:
        os.environ['SCD_RESOURCES'] = str(Path(a.resources).expanduser().resolve())
    if (a.batch_size is not None and a.batch_size < 1) or (a.stop_after_step is not None and a.stop_after_step < 1):
        p.error('Batch size and stop step must be positive')
    out = a.run.expanduser().resolve()
    recorded = read_json(out / 'training.json')
    if (out / 'complete.json').exists():
        p.error('Training is already complete')
    config = copy.deepcopy(recorded['config'])
    t = config['training']
    if a.world_size is not None and a.world_size != t.get('world_size', 1):
        p.error('Batch resume retains the original world_size')
    if a.expected_objective and t.get('objective') != a.expected_objective:
        p.error('The selected launcher objective differs from the recorded experiment')
    if a.batch_size is None:
        a.batch_size = int(os.environ.get('SCD_TRAIN_BATCH_SIZE', t['batch_size']))
    if a.batch_size < 1:
        p.error('Batch size must be positive')
    per_rank = t['batch_size'] * t['gradient_accumulation_steps']
    if per_rank % a.batch_size:
        p.error(f'Batch size must divide the saved per-GPU effective batch {per_rank}')
    t.update(batch_size=a.batch_size, gradient_accumulation_steps=per_rank // a.batch_size)
    validate_batch_change(recorded['config'], config)
    validate_config(config, files=False)
    current = {name: sha256(source_file(name)) for name in recorded['code']}
    if not compatible_batch_sources(recorded['code'], current):
        p.error('Source differs beyond training/config hooks; use a checkout matching the original experiment')
    target = ROOT / 'configs' / f'local-{out.name}-resume-b{a.batch_size}.json'
    if target.exists():
        if read_json(target) != config:
            p.error(f'Existing batch-resume config differs: {target}')
    else:
        with target.open('x') as handle:
            json.dump(config, handle, indent=2)
            handle.write('\n')
    settings = batch_settings(config)
    print(json.dumps({'run': str(out), 'config': str(target), **settings,
                      'effective_global_batch': settings['batch_size'] *
                      settings['gradient_accumulation_steps'] * settings['world_size'],
                      'objective': t['objective'], 'noise_views': t['noise_views'],
                      'resume': a.resume, 'behavior_notes': batch_behavior_notes(config),
                      'evaluation': config.get('evaluation', {})}), flush=True)
    if a.config_only:
        return
    command = [sys.executable, '-u', str(ROOT / 'train.py'), '--config', str(target),
               '--out', str(out), '--resume_from_checkpoint', a.resume,
               '--execution', a.execution, '--run-label', a.run_label or out.name]
    if a.check_only:
        command += ['--check-only']
    if a.stop_after_step is not None:
        command += ['--stop-after-step', str(a.stop_after_step)]
    if a.resources:
        command += ['--resources', a.resources]
    os.environ[ENVIRONMENT_FLAG] = '1'
    from self_correct_dpo.evaluation.runner import run_command
    run_command(command)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
