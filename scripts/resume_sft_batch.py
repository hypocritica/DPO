"""Resume an existing SFT run with a new per-GPU batch and unchanged global batch."""
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
    ENVIRONMENT_FLAG, batch_settings, compatible_batch_sources, validate_batch_change)
from self_correct_dpo.training import validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--stop-after-step', type=int, help='Absolute optimizer step at which to pause')
    p.add_argument('--config-only', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.batch_size < 1 or (a.stop_after_step is not None and a.stop_after_step < 1):
        p.error('Batch size and stop step must be positive')
    out = a.run.expanduser().resolve()
    recorded = read_json(out / 'training.json')
    if (out / 'complete.json').exists():
        p.error('Training is already complete')
    config = copy.deepcopy(recorded['config'])
    t = config['training']
    per_rank = t['batch_size'] * t['gradient_accumulation_steps']
    if per_rank % a.batch_size:
        p.error(f'Batch size must divide the saved per-GPU effective batch {per_rank}')
    t.update(batch_size=a.batch_size, gradient_accumulation_steps=per_rank // a.batch_size)
    validate_batch_change(recorded['config'], config)
    validate_config(config, files=False)
    current = {name: sha256(source_file(name)) for name in recorded['code']}
    if not compatible_batch_sources(recorded['code'], current):
        p.error('Source differs beyond training.py; use a checkout matching the original experiment')
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
                      'resume': 'latest', 'evaluation': config.get('evaluation', {})}), flush=True)
    if a.config_only:
        return
    command = [sys.executable, '-u', str(ROOT / 'train.py'), '--config', str(target),
               '--out', str(out), '--resume_from_checkpoint', 'latest',
               '--execution', 'standalone', '--run-label', out.name]
    if a.check_only:
        command += ['--check-only']
    if a.stop_after_step is not None:
        command += ['--stop-after-step', str(a.stop_after_step)]
    os.environ[ENVIRONMENT_FLAG] = '1'
    from self_correct_dpo.evaluation.runner import run_command
    run_command(command)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
