"""Run one fixed SDXL experiment on two already allocated H100 GPUs."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import read_json
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json, recover_run, resolve_checkpoint
from self_correct_dpo.train import training_lock

RECIPES = {
    'dmpo': ('dmpo', 'dpo', 1),
    'dmpo-k6-stage': ('dmpo', 'stagewise-C-gaussian', 6),
    'dspo': ('dspo', 'dpo', 1),
}


def prepare_output(output, config):
    """Recover committed transactions; preserve attempts that never saved state."""
    if not output.exists():
        return False
    recorded = output / 'training.json'
    if recorded.is_file():
        metadata = read_json(recorded)
        if metadata['config'] != config:
            raise ValueError('Existing run has a different configuration; use a new run tag')
        recover_run(output)
        completion = output / 'evaluation-complete.json'
        if completion.is_file():
            training_done = read_json(output / 'complete.json')
            if (not read_json(completion).get('complete') or not training_done.get('complete')
                    or training_done['run_id'] != metadata['run_id']
                    or training_done['step'] != config['training']['max_steps']):
                raise ValueError('Completion records do not match this run')
            print(f'Experiment already complete: {output}', flush=True)
            return True
    pointer = output / 'checkpoints/latest.json'
    if pointer.is_file():
        checkpoint = resolve_checkpoint(pointer)
        print(f'Resume optimizer step {read_json(checkpoint / "checkpoint.json")["step"]}: {output}', flush=True)
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archived = output.with_name(f'{output.name}.interrupted-{stamp}-{uuid.uuid4().hex[:8]}')
        output.rename(archived)
        print(f'No committed checkpoint; preserved previous attempt at {archived}. Restart from base.', flush=True)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=RECIPES)
    p.add_argument('--run-tag', default=os.environ.get('SCD_H100_RUN_TAG', 'v1'))
    p.add_argument('--config-only', action='store_true')
    p.add_argument('--stop-after-step', type=int)
    p.add_argument('--runs-root', type=Path)
    p.add_argument('--config-dir', type=Path, default=ROOT / 'configs')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', a.run_tag):
        p.error('run-tag must be a simple directory name')
    if a.stop_after_step is not None and not 1 <= a.stop_after_step <= 10000:
        p.error('stop-after-step must be between 1 and 10000')
    os.chdir(ROOT)
    objective, method, views = RECIPES[a.experiment]
    name = f'sdxl-{a.experiment}-pv192-2h100-10000-{a.run_tag}'
    runs = (a.runs_root or Path(load_resources()['RUNS_ROOT'])).resolve()
    configs = a.config_dir.resolve()
    output = runs / name
    frozen = configs / f'local-{name}.json'
    command = [sys.executable, '-u', str(ROOT / 'scripts/compare_objectives.py'),
        '--family', 'sdxl', '--objective', objective, '--method', method,
        '--views', str(views), '--world-size', '2', '--batch-size', '1',
        '--budget-unit', 'pair_views', '--skip-startup-image-checks', '--no-base-evaluation',
        '--run-name', name, '--runs-root', str(runs), '--config-dir', str(configs)]
    # Hold a launcher lock through training, independently of the trainer's locks.
    with training_lock(str(output) + '.launcher'):
        if not frozen.exists() and (output / 'training.json').is_file():
            configs.mkdir(parents=True, exist_ok=True)
            atomic_json(frozen, read_json(output / 'training.json')['config'])
        subprocess.run(command + ['--config-only'], check=True)
        if a.config_only:
            return
        if prepare_output(output, read_json(frozen)):
            return
        if a.stop_after_step is not None:
            command += ['--stop-after-step', str(a.stop_after_step)]
        from self_correct_dpo.evaluation.runner import run_command
        run_command(command)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
