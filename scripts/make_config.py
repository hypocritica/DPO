"""Write a resolved experiment config using the same overrides as train.py."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.config import add_training_arguments, training_config
from self_correct_dpo.resources import CONFIG_DIR, load_resources
from self_correct_dpo.training import validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '--template', dest='config', default=str(CONFIG_DIR / 'multiview-C.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', choices=['B', 'C'])
    parser.add_argument('--effective-batch', type=int)
    parser.add_argument('--smoke', action='store_true', help='Two updates without evaluation; keeps model resolution')
    parser.add_argument('--benchmark', action='store_true', help='100 updates without evaluation')
    parser.add_argument('--dataset', choices=['pickapic', 'hpdv3'], default='pickapic')
    add_training_arguments(parser)
    args = parser.parse_args()
    if args.smoke and args.benchmark:
        parser.error('Choose --smoke or --benchmark')
    config = training_config(args)
    t = config['training']
    if args.arm:
        t['formal_arm'] = args.arm
    if args.dataset == 'hpdv3' and not args.pairs:
        key = 'HPDV3_' + config['model']['family'].upper() + '_PREPARED'
        config['data']['pairs'] = str(Path(load_resources(args.resources)[key]) / 'pairs.jsonl')
    if args.smoke or args.benchmark:
        t.update(max_steps=args.max_steps or (2 if args.smoke else 100), warmup_steps=0,
                 diagnostic={}, logging='jsonl')
        config['evaluation'] = {'enabled': False}
        if args.smoke:
            t.update(batch_size=args.batch_size or 1,
                     gradient_accumulation_steps=args.gradient_accumulation_steps or 1)
        t['checkpoint_every'] = 1 if args.smoke else t['max_steps']
    if args.effective_batch is not None:
        if args.gradient_accumulation_steps is not None:
            parser.error('Choose --effective-batch or --gradient_accumulation_steps')
        unit = t['batch_size'] * t.get('world_size', 1)
        if unit < 1 or args.effective_batch < 1 or args.effective_batch % unit:
            parser.error('Effective batch must be positive and divisible by batch size * world size')
        t['gradient_accumulation_steps'] = args.effective_batch // unit
    validate_config(config, files=False)
    with open(args.out, 'x') as handle:
        json.dump(config, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'config': args.out, 'world_size': t.get('world_size', 1),
        'effective_batch': t['batch_size'] * t['gradient_accumulation_steps'] * t.get('world_size', 1),
        'steps': t['max_steps']}))


if __name__ == '__main__':
    main()
