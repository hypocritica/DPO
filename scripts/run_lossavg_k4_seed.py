#!/usr/bin/env python3
"""Run one existing L/LossAvg K4 generation seed on four allocated GPUs."""
import argparse
import json
import os
from pathlib import Path

import appendix_flexible as flex


def select_job(plan, seed):
    jobs = [j for j in plan['jobs'] if j.get('old_pack') == 'L'
            and j['method'] == 'lossavg_k4' and j['step'] == 1600
            and j['panel'] == 'full' and j['seed'] == seed]
    if len(jobs) != 1 or jobs[0]['shards'] != 4:
        raise ValueError('Need exactly one prepared L/LossAvg K4 step1600 full-panel job with four shards')
    return [('sd35_medium', jobs[0])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'score', 'status'))
    parser.add_argument('--seed', type=int, required=True, choices=(17, 23, 42))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus')
    args = parser.parse_args(argv)
    try:
        registry = flex.load_resources()
        args.output = (args.output or Path(os.environ.get('MVPO_PARALLEL_ROOT')
                        or registry['MVPO_PARALLEL_ROOT'])).expanduser().resolve()
        args.config = flex.ROOT / 'configs/appendix-parallel-5090.json'
        plan = flex.load_plan(args.output, 'sd35_medium')
        if plan['spec'] != flex.read_json(args.config):
            raise ValueError('Configuration differs from the frozen parallel plan')
        selected = select_job(plan, args.seed)
        plans = {'sd35_medium': plan}
        sizes = [len(flex.wanted_rows(args.output, plan, selected[0][1], i)) for i in range(4)]
        if sizes != [408] * 4:
            raise ValueError(f'Expected 1632 prompts split into four groups of 408, got {sizes}')
        if args.action == 'status':
            print(json.dumps(flex.status(args.output, plans, selected), indent=2))
            return
        devices = flex.gpu_tokens(args.gpus or os.environ.get('SCD_EVAL_GPUS'),
                                  os.environ, plan['interpreters']['render'])
        if len(devices) != 4:
            raise ValueError('Allocate four GPUs for each seed task')
        args.gpus = ','.join(devices)
        args.batch_size = None  # Frozen parallel spec: SD3.5 on 5090 uses batch 2.
        args.tasks = [f'L-lossavg_k4-s{args.seed}']
        print(f'LossAvg K4 / step1600 / seed{args.seed}: 1632 slots, 4 x 408; {args.action}', flush=True)
        with flex.output_lock(args.output / ('launch-' + flex.digest(sorted(args.tasks))[:16])):
            flex.launch(args, plans, selected)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
