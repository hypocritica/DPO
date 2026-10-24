"""Compose defaults, model settings, method settings and command overrides."""
import copy
import json
import os
import re
from pathlib import Path
from self_correct_dpo.common import read_json
from self_correct_dpo.resources import CONFIG_DIR, load_resources


def merge_config(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, *, scorers=None, benchmarks=None, resources=None, _parents=()):
    path = Path(path).expanduser().resolve()
    if path in _parents:
        raise ValueError(f'Configuration inheritance cycle: {path}')
    config = read_json(path)
    registry = load_resources(resources)
    inherited = {}
    for parent in config.pop('extends', []):
        inherited = merge_config(inherited, load_config(path.parent / parent, scorers=scorers,
            benchmarks=benchmarks, resources=resources, _parents=(*_parents, path)))
    def resolve(value, label):
        from string import Template
        value = os.path.expanduser(Template(value).safe_substitute({**os.environ, **registry}))
        if re.search(r"\$\{|\$[A-Za-z_]", value):
            raise ValueError(f"Set the environment variable used by {label}: {value}")
        target=Path(value)
        return str((target if target.is_absolute() else path.parent/target).resolve())
    fields = (("model", "path"), ("data", "pairs"), ("data", "diagnostic_suite"),
              ("aligned", "path"))
    for section, key in fields:
        value = config.get(section, {}).get(key)
        if value is None:
            continue
        config[section][key] = resolve(value, f'{section}.{key}')
    if config.get('model',{}).get('vae'):
        config['model']['vae']['path']=resolve(config['model']['vae']['path'],'model.vae.path')
    if config.get('evaluation'):
        for key in ('prompts','scorers_config'):
            if key in config['evaluation']:
                config['evaluation'][key]=resolve(config['evaluation'][key],f'evaluation.{key}')
    if config.get('scorers_config'):
        config['scorers_config']=resolve(config['scorers_config'],'scorers_config')
    if 'benchmarks' in config:
        selected=benchmarks if benchmarks is not None else config['benchmarks'].keys()
        config['benchmarks']={key:resolve(config['benchmarks'][key],f'benchmarks.{key}') for key in selected}
    if 'scorers' in config:
        selected = scorers if scorers is not None else config['scorers'].keys()
        config['scorers']={name:{key:resolve(value,f'scorers.{name}.{key}')
            if key in ('model','processor','checkpoint','open_clip_source','source','config','tokenizer','python') else value
            for key,value in config['scorers'][name].items()} for name in selected}
    return merge_config(inherited, config)


def add_training_arguments(parser):
    parser.add_argument('--family', choices=['sd15', 'sdxl'])
    parser.add_argument('--resources', help='Resource registry; default configs/resources.json')
    parser.add_argument('--method', choices=['dpo', 'multiview-B', 'multiview-C',
                                           'multiview-B-gaussian', 'multiview-C-gaussian'])
    parser.add_argument('--pretrained_model_name_or_path', '--model', dest='model_path')
    parser.add_argument('--model-revision')
    parser.add_argument('--data_path', '--pairs', dest='pairs')
    parser.add_argument('--monitor-prompts')
    parser.add_argument('--scorers-config')
    parser.add_argument('--mixed_precision', choices=['no', 'bf16'])
    for flags, key, kind in [
        (['--train_batch_size', '--batch-size'], 'batch_size', int),
        (['--gradient_accumulation_steps', '--accumulation'], 'gradient_accumulation_steps', int),
        (['--max_train_steps', '--steps'], 'max_steps', int),
        (['--learning_rate'], 'learning_rate', float),
        (['--checkpointing_steps'], 'checkpoint_every', int),
        (['--rank'], 'lora_rank', int), (['--lora_alpha'], 'lora_alpha', int),
        (['--beta'], 'beta', float), (['--resolution'], 'resolution', int),
        (['--views', '--K'], 'noise_views', int), (['--world-size'], 'world_size', int),
    ]:
        parser.add_argument(*flags, dest=key, type=kind)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument('--report_to', choices=['tensorboard', 'wandb', 'all', 'none', 'jsonl'])
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-mode', choices=['offline', 'online', 'disabled'])
    parser.add_argument('--launcher', choices=['accelerate', 'torchrun'])
    parser.add_argument('--no-evaluation', action='store_true')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Override a dotted config key; value is JSON or a string')


def training_config(args):
    if args.resources:
        # Scorer/renderer subprocesses must resolve the same resource table.
        os.environ['SCD_RESOURCES'] = str(Path(args.resources).expanduser().resolve())
    config = load_config(args.config, resources=args.resources)
    family = args.family or config['model']['family']
    if args.family and family != config['model']['family']:
        config = merge_config(config, load_config(CONFIG_DIR / 'models' / f'{family}.json', resources=args.resources))
    if args.method:
        config['method'] = args.method
        settings = load_config(CONFIG_DIR / f'{args.method}.json', resources=args.resources)['training']
        for key in ('formal_arm', 'noise_views', 'independent_rollout', 'batch_views'):
            config['training'].pop(key, None)
            if key in settings:
                config['training'][key] = settings[key]
    t = config['training']
    original_effective = t['batch_size'] * t['gradient_accumulation_steps'] * t.get('world_size', 1)
    runtime_world = int(os.environ.get('WORLD_SIZE', '1'))
    if args.world_size is None and runtime_world > 1:
        args.world_size = runtime_world
    for key in ('batch_size', 'gradient_accumulation_steps', 'max_steps', 'learning_rate',
                'checkpoint_every', 'lora_rank', 'lora_alpha', 'beta', 'resolution', 'noise_views', 'world_size', 'launcher'):
        if getattr(args, key, None) is not None:
            t[key] = getattr(args, key)
    # Changing GPU count alone preserves the default global pair batch.
    if args.world_size is not None and args.gradient_accumulation_steps is None:
        unit = t['batch_size'] * t['world_size']
        if unit < 1 or original_effective % unit:
            raise ValueError('Global batch must divide batch_size * world_size; set --gradient_accumulation_steps explicitly')
        t['gradient_accumulation_steps'] = original_effective // unit
    precision = args.mixed_precision or os.environ.get('ACCELERATE_MIXED_PRECISION')
    if precision:
        if precision not in ('no', 'bf16'):
            raise ValueError('Supported mixed precision: no, bf16')
        t['dtype'] = {'no': 'float32', 'bf16': 'bfloat16'}[precision]
    if args.seed is not None:
        config['seed'] = args.seed
    for value, section, key in ((args.model_path, 'model', 'path'), (args.pairs, 'data', 'pairs'),
            (args.monitor_prompts, 'evaluation', 'prompts'), (args.scorers_config, 'evaluation', 'scorers_config')):
        if value is not None:
            config.setdefault(section, {})[key] = str(Path(value).expanduser().resolve())
    receipt = Path(config['model']['path']) / '.scd-download.json'
    if receipt.is_file() and not args.model_revision:
        identity = read_json(receipt)
        if not identity.get('complete'):
            raise ValueError('Model download is incomplete')
        config['model']['revision'] = identity['revision']
    if args.model_revision:
        config['model']['revision'] = args.model_revision
    if args.dataloader_num_workers is not None:
        config['data']['num_workers'] = args.dataloader_num_workers
    if args.report_to:
        t['logging'] = args.report_to
    if args.wandb_project:
        t.setdefault('wandb', {})['project'] = args.wandb_project
    if args.wandb_mode:
        t.setdefault('wandb', {})['mode'] = args.wandb_mode
    if args.no_evaluation:
        config['evaluation'] = {'enabled': False}
    if args.resolution is not None and config.get('evaluation', {}).get('enabled'):
        config['evaluation']['sampling'].update(height=args.resolution, width=args.resolution)
    for override in args.set:
        key, separator, raw = override.partition('=')
        if not separator:
            raise ValueError('--set expects section.key=value')
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = config
        parts = key.split('.')
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                raise ValueError(f'Unknown config section in --set {key}')
            target = target[part]
        target[parts[-1]] = value
    if config.get('method') == 'dpo' and (t['noise_views'] != 1 or 'independent_rollout' in t):
        raise ValueError('Pure DPO requires K=1 and no rollout')
    return config
