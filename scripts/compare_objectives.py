"""Freeze and launch one objective comparison using the existing training lifecycle.

Run inside an allocated remote container with the project's Python environment.
Budget counts pair groups or pair-views, not FLOPs (SFT uses only winners).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.resources import load_resources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', required=True, choices=['sd15', 'sdxl'])
    p.add_argument('--objective', required=True, choices=['dpo', 'sft', 'kto', 'mapo', 'dmpo', 'dspo'])
    p.add_argument('--method', default='dpo', choices=['dpo', 'multiview-B-gaussian',
                   'multiview-C-gaussian', 'stagewise-C-gaussian'])
    p.add_argument('--views', type=int, choices=[1, 2, 4, 6])
    p.add_argument('--world-size', type=int, choices=[2, 4, 8], default=2)
    p.add_argument('--budget-unit', choices=['pairs', 'pair_views'], default='pairs')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--beta', type=float, help='Explicit objective scale; frozen with the experiment')
    p.add_argument('--dspo-recipe', choices=['score_residual', 'official_sdxl'])
    p.add_argument('--dspo-sdxl-threshold', type=float)
    p.add_argument('--resources')
    p.add_argument('--run-name')
    p.add_argument('--runs-root', type=Path)
    p.add_argument('--config-dir', type=Path, default=ROOT / 'configs')
    p.add_argument('--mode', choices=['train', 'smoke', 'benchmark'], default='train')
    p.add_argument('--config-only', action='store_true', help='Freeze/print config without checking data or launching')
    p.add_argument('--check-only', action='store_true', help='Run the existing CPU training preflight only')
    p.add_argument('--stop-after-step', type=int)
    p.add_argument('--resume', dest='resume_from_checkpoint')
    a = p.parse_args()
    if a.objective != 'dspo' and (a.dspo_recipe is not None or a.dspo_sdxl_threshold is not None):
        p.error('DSPO recipe options require --objective dspo')
    if a.objective == 'kto' and a.family != 'sd15':
        p.error('KTO comparisons are SD1.5 only')
    if a.objective in ('sft', 'kto', 'mapo') and a.method != 'dpo':
        p.error('SFT/KTO/MaPO comparisons require single-view method=dpo')
    k = a.views if a.views is not None else (1 if a.method == 'dpo' else 2)
    if (a.method == 'dpo' and k != 1) or (a.method != 'dpo' and k == 1):
        p.error('Use K=1 for single view and K=2/4/6 for Gaussian multiview')
    batch = a.batch_size if a.batch_size is not None else int(os.environ.get(
        'SCD_TRAIN_BATCH_SIZE', 2 if a.family == 'sd15' and k <= 2 else 1))
    if batch < 1 or (a.stop_after_step is not None and a.stop_after_step < 1):
        p.error('Batch size and stop step must be positive')
    budget = 384 if a.family == 'sd15' else 192
    name = a.run_name or f'{a.family}-{a.objective}-{a.method}-k{k}-{a.budget_unit}{budget}-{a.world_size}gpu-{a.mode}-v1'
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in name) or name in ('.', '..'):
        p.error('run-name must be a simple directory name')
    registry = load_resources(a.resources)
    output = (a.runs_root or Path(registry['RUNS_ROOT'])) / name
    if (output / 'training.json').is_file():
        resume_command = [sys.executable, '-u', str(ROOT / 'scripts/resume_batch.py'),
                          '--run', str(output), '--batch-size', str(batch),
                          '--expected-objective', a.objective]
        for flag, enabled in (('--config-only', a.config_only), ('--check-only', a.check_only)):
            if enabled:
                resume_command.append(flag)
        for flag, value in (('--stop-after-step', a.stop_after_step),
                            ('--resume', a.resume_from_checkpoint), ('--resources', a.resources)):
            if value is not None:
                resume_command += [flag, str(value)]
        from self_correct_dpo.evaluation.runner import run_command
        run_command(resume_command)
        return
    a.config_dir.mkdir(parents=True, exist_ok=True)
    frozen = a.config_dir / f'local-{name}.json'
    if output.exists() and not frozen.is_file():
        p.error(f'Missing frozen config for existing output: {frozen}')
    command = [sys.executable, str(ROOT / 'scripts/make_config.py'),
        '--family', a.family, '--objective', a.objective, '--method', a.method,
        '--views', str(k), '--world-size', str(a.world_size), '--batch-size', str(batch),
        '--max_train_steps', '10000', '--mixed_precision', 'bf16',
        '--checkpointing_steps', '200', '--report_to', 'tensorboard', '--launcher', 'accelerate']
    if a.beta is not None:
        command += ['--beta', str(a.beta)]
    if a.dspo_recipe is not None:
        command += ['--set', f'training.dspo_recipe={a.dspo_recipe}']
    if a.dspo_sdxl_threshold is not None:
        command += ['--set', f'training.dspo_sdxl_threshold={a.dspo_sdxl_threshold}']
    if a.resources:
        command += ['--resources', a.resources]
    # Smoke keeps one accumulation/update; benchmark/train preserve the declared budget.
    if a.mode == 'smoke':
        command += ['--smoke', '--max_train_steps', '2']
    else:
        command += ['--budget-unit', a.budget_unit, '--budget-per-update', str(budget)]
        if a.mode == 'benchmark':
            command += ['--benchmark', '--max_train_steps', '100']
    with tempfile.TemporaryDirectory() as tmp:
        expected = Path(tmp) / 'expected.json'
        subprocess.run(command + ['--out', str(expected)], cwd=ROOT, check=True)
        config = json.loads(expected.read_text())
        if frozen.exists():
            if json.loads(frozen.read_text()) != config:
                p.error('Frozen config differs; choose a new --run-name')
        else:
            with frozen.open('x') as handle:
                handle.write(expected.read_text())
    print(json.dumps({'config': str(frozen), 'output': str(output),
                      'objective': a.objective, 'method': a.method, 'views': k,
                      'budget_is_equal_flops': False}), flush=True)
    if a.config_only:
        return
    train = [sys.executable, str(ROOT / 'train.py'), '--config', str(frozen),
             '--out', str(output), '--execution', 'standalone', '--run-label', name]
    if a.check_only:
        train += ['--check-only']
    else:
        resume = a.resume_from_checkpoint or ('latest' if output.exists() else None)
        if resume:
            train += ['--resume_from_checkpoint', resume]
        if a.stop_after_step:
            train += ['--stop-after-step', str(a.stop_after_step)]
    subprocess.run(train, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
