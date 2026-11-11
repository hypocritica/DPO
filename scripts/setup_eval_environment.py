"""Print an isolated SLAI environment recipe; --apply explicitly creates it.

Never modifies an existing prefix. GPU compatibility still needs a smoke test
on allocated hardware. No model weights are downloaded by this script.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'legacy': 'requirements-eval.txt',
    'flow': 'requirements-eval-flow.txt',
    'hpsv3': 'requirements-hpsv3.txt',
    'unifiedreward': 'requirements-unifiedreward.txt',
    'geneval': 'requirements-geneval.txt',
    'dpg': 'requirements-dpg.txt',
}


def recipe(profile, prefix, runtime, conda='conda'):
    prefix = Path(prefix).expanduser().absolute()
    if prefix.exists() or prefix.is_symlink():
        raise FileExistsError(f'Environment prefix already exists; reuse/check it or choose a new prefix: {prefix}')
    if profile in ('geneval', 'dpg') and runtime != 'h100-cu118':
        raise ValueError('GenEval/DPG require the isolated H100 cu118 candidate; not a 5090 recipe')
    if profile == 'legacy' and runtime != 'slai-cu130':
        raise ValueError('requirements-eval.txt fixes the legacy SLAI cu130 runtime')
    if profile in ('flow', 'hpsv3', 'unifiedreward') and runtime != 'slai-cu130':
        raise ValueError(f'{profile} requires the modern slai-cu130 recipe; torch 2.0/cu118 is reserved for GenEval/DPG')
    python = str(prefix / 'bin/python')
    pip = [python, '-m', 'pip', 'install']
    commands = [[conda, 'create', '--prefix', str(prefix), 'python=3.10', 'pip=24.0', '-y']]
    if runtime == 'slai-cu130':
        commands.append(pip + ['--extra-index-url', 'https://download.pytorch.org/whl/cu130',
                              'torch==2.14.1+cu130', 'torchvision==0.29.1+cu130', 'nvidia-nccl-cu13==2.30.7'])
    else:
        commands.append(pip + ['torch==2.0.0', 'torchvision==0.15.1', '--index-url',
                              'https://download.pytorch.org/whl/cu118'])
    if profile == 'legacy':
        commands.append(pip + ['setuptools==80.9.0', 'wheel'])
    if profile == 'geneval':
        commands.append(pip + ['--only-binary=mmcv-full', 'mmcv-full==1.7.2', '-f',
                              'https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html'])
    if profile == 'dpg':
        commands.append(pip + ['pip==24.0', 'setuptools==68.2.2', 'wheel==0.41.3',
                              'Cython==0.29.37', 'numpy==1.26.4'])
    options = ['--no-build-isolation'] if profile in ('legacy', 'dpg') else []
    commands.append(pip + options + ['-r', str(ROOT / PROFILES[profile])])
    commands.append([python, '-m', 'pip', 'check'])
    commands.append([python, str(ROOT / 'scripts/check_eval_environment.py'), '--profile', profile, '--skip-assets'])
    return prefix, commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--prefix', required=True, help='New absolute environment prefix; never overwritten')
    parser.add_argument('--runtime', choices=('slai-cu130', 'h100-cu118'), required=True)
    parser.add_argument('--conda', default='conda', help='Conda executable, without shell activation')
    parser.add_argument('--dry-run', action='store_true', help='Print recipe only (the default)')
    parser.add_argument('--apply', action='store_true', help='Explicitly create the new environment and install dependencies')
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error('--apply and --dry-run are mutually exclusive')
    try:
        prefix, commands = recipe(args.profile, args.prefix, args.runtime, args.conda)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.apply:
        print('Dry run: no environment created. Use --apply (without --dry-run) to execute this recipe.')
        return
    # Do not delete partial environments after failure: retain evidence and refuse
    # another install over them. The user may choose a new prefix after inspection.
    env = {**os.environ, 'DS_BUILD_OPS': '0'}
    for command in commands:
        subprocess.run(command, check=True, env=env, cwd=ROOT)
    (prefix / '.scd-eval-environment.json').write_text(json.dumps({
        'profile': args.profile, 'runtime': args.runtime, 'commands': commands,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
