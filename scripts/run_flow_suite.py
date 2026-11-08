"""Run named Flow experiments on allocated GPUs; list/plan without loading models.

Default: B4, 128 pair-views/update, continuous training and separate evaluation.
Every experiment has its own optimizer, RNG and output; reruns resume/skip through
the existing launcher. Each independent experiment loads its base once.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
DEFAULT_TAG = 'flow-b4-v1'


def catalogue():
    cases, groups = {}, {}

    def add(name, algorithm='mvpo', k=2, arm='C', stage=False, boundary=500,
            unit='pair_views', overrides=None, suffix=''):
        cases[name] = dict(algorithm=algorithm, k=k, arm=arm, stage=stage,
                           boundary=boundary, unit=unit, overrides=overrides or [], suffix=suffix)
        return name

    add('dpo', 'dpo', 1)
    add('linear-dpo', 'linear_dpo', 1)
    add('sft', 'sft', 1)
    for k in (2, 4, 8):
        for stage in (False, True):
            for arm in ('B', 'C'):
                add(f'{"stage" if stage else "global"}-{arm.lower()}-k{k}',
                    k=k, arm=arm, stage=stage)
    groups['main'] = ['dpo', 'stage-c-k2', 'stage-c-k4', 'linear-dpo']
    groups['baselines'] = ['dpo', 'linear-dpo', 'sft']
    groups['aggregation'] = [f'{scope}-{arm}-k{k}' for k in (2, 4)
                             for scope in ('global', 'stage') for arm in ('b', 'c')]
    groups['views'] = ['dpo'] + [f'{scope}-c-k{k}' for k in (2, 4, 8)
                                for scope in ('global', 'stage')]
    groups['boundary'] = []
    for k in (2, 4):
        for boundary in (250, 500, 750):
            name = f'stage-c-k{k}'
            if boundary != 500:
                name = add(f'{name}-boundary{boundary}', k=k, stage=True, boundary=boundary)
            groups['boundary'].append(name)
    # Apply the same tuning/repetition grid to both aggregations and baselines.
    controlled = ['dpo', 'global-b-k2', 'global-c-k2', 'stage-b-k2',
                  'stage-c-k2', 'stage-c-k4', 'linear-dpo']
    for group, flag, values, default in (('beta', '--beta', (100, 500, 2000), 500),
                                        ('seeds', '--seed', (17, 42, 123), 17)):
        groups[group] = []
        for base in controlled:
            for value in values:
                name = base
                if value != default:
                    name = f'{base}-{flag[2:]}{value}'
                    cases[name] = dict(copy.deepcopy(cases[base]),
                                       overrides=[flag, str(value)], suffix=f'-{flag[2:]}{value}')
                groups[group].append(name)
    groups['pairs'] = []
    for base in ['dpo', 'linear-dpo'] + [f'{scope}-{arm}-k{k}' for k in (2, 4, 8)
                                       for scope in ('global', 'stage') for arm in ('b', 'c')]:
        name = base + '-pairs128'
        cases[name] = dict(copy.deepcopy(cases[base]), unit='pairs')
        groups['pairs'].append(name)
    groups['all'] = list(cases)
    return cases, groups


CASES, GROUPS = catalogue()


def command(family, name, tag=DEFAULT_TAG, forwarded=()):
    case = CASES[name]
    cmd = ['bash', str(ROOT / 'launchers/flow/paper_experiment.sh'),
           'sd35_medium' if family == 'sd35' else 'zimage', case['algorithm'],
           str(case['k']), case['unit'], str(case['stage']).lower()]
    if case['algorithm'] == 'mvpo':
        cmd += ['--internal-aggregation', case['arm']]
    if case['stage']:
        cmd += ['--internal-boundary', str(case['boundary'])]
    cmd += ['--run-tag', tag + case['suffix'], *case['overrides'], *forwarded]
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=('sd35', 'zimage'))
    parser.add_argument('selection', nargs='?', default='main',
                        help='Group or case ID; comma-separated IDs/groups are allowed')
    parser.add_argument('--run-tag', default=DEFAULT_TAG)
    parser.add_argument('--list', action='store_true', help='List selected cases, no writes or model loads')
    parser.add_argument('--plan-only', action='store_true', help='Print exact commands as JSON, no writes')
    args, forwarded = parser.parse_known_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', args.run_tag):
        parser.error('run-tag must be 1..80 letters/digits/dots/underscores/hyphens')
    selected = []
    for name in args.selection.split(','):
        if name in GROUPS:
            selected.extend(GROUPS[name])
        elif name in CASES:
            selected.append(name)
        else:
            parser.error(f'Unknown case/group: {name}; groups: {", ".join(GROUPS)}')
    selected = list(dict.fromkeys(selected))
    plan = [{'case': name, **CASES[name],
             'command': command(args.family, name, args.run_tag, forwarded)} for name in selected]
    if args.list:
        print('Groups: ' + ', '.join(f'{g}({len(v)})' for g, v in GROUPS.items()))
        print('Defaults: B4 per GPU; SD3.5 4x5090 / ZImage 2xH100; 128 pair-views; resident frozen modules.')
        print('case | objective | K | aggregation | sampling | budget')
        for row in plan:
            print(f'{row["case"]} | {row["algorithm"]} | {row["k"]} | {row["arm"]} | '
                  f'{"stage@" + str(row["boundary"]) if row["stage"] else "global"} | {row["unit"]}128')
        return 0
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    from self_correct_dpo.evaluation.runner import run_command
    for index, row in enumerate(plan, 1):
        print(f'[{index}/{len(plan)}] {row["case"]}: {shlex.join(row["command"])}', flush=True)
        try:
            run_command(row['command'])
        except subprocess.CalledProcessError as error:
            print(f'Stopped at {row["case"]}, exit={error.returncode}; rerun the same selection/tag to resume.', flush=True)
            return error.returncode if error.returncode > 0 else 128 - error.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
