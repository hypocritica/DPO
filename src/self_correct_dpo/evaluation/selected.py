"""One manually selected permanent checkpoint per method, sharing suite execution."""
import copy
from pathlib import Path

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.config import load_config
from self_correct_dpo.resources import load_resources
from self_correct_dpo.evaluation import suite
from self_correct_dpo.evaluation.experiment_catalog import draft_catalog, load_catalog

METHODS = {
    'sd35_medium': ('base', 'sft', 'dpo', 'linear_dpo', 'global_k2', 'stagewise_k2', 'global_k4', 'stagewise_k4'),
    'zimage': ('base', 'dpo', 'stagewise_k4'),
}
PANELS = {'sd35_medium': suite.PANELS, 'zimage': ('partiprompts', 'geneval')}


def identity(selection):
    """Only selected_step is editable after importing training identities."""
    result = copy.deepcopy(selection)
    result.pop('identity_sha256', None)
    for entry in result['choices']:
        entry.pop('selected_step', None)
    return digest(result)


def initialize(args):
    family = args.family
    registry = load_resources(args.resources)
    provenance = {'selection_policy': 'Human choice of exactly one retained step per method; no automatic ranking.'}
    if args.from_prepared:
        if args.catalog or args.runs_root:
            raise ValueError('--from-prepared is exclusive with catalog/runs-root')
        from self_correct_dpo.evaluation.curve_prepare import load_prepared
        source = Path(args.from_prepared).expanduser().resolve()
        manifest = load_prepared(source)
        catalog_path = source / 'catalog.json'
        catalog = read_json(catalog_path)
        if set(catalog['models']) != {family}:
            raise ValueError('Prepared curve family differs from requested family')
        provenance.update(curve_prepared=str(source), curve_manifest=manifest,
                          curve_selection=read_json(source / 'prompts/selection.json'),
                          limitation='Checkpoint selection used development prompts; independence is limited to recorded exclusion panels.')
    else:
        if family == 'sd35_medium':
            raise ValueError('SD3.5 formal selection requires --from-prepared to preserve curve run identities')
        if args.catalog:
            catalog_path = Path(args.catalog).expanduser().resolve()
            catalog = read_json(catalog_path)
        else:
            if not args.runs_root:
                raise ValueError('Z-Image requires --runs-root or an explicit --catalog')
            catalog = draft_catalog([Path(p).expanduser().resolve() for p in args.runs_root], families=[family], patterns=getattr(args, 'run_glob', None))
            catalog_path = Path(args.out).expanduser().resolve()
    # Resolve resource variables and relative paths through the authoritative loader.
    # For discovery the draft already contains absolute roots/model paths.
    if not args.from_prepared and not args.catalog:
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / 'catalog.json'
            suite.freeze_json(candidate, catalog)
            records, _, _ = load_catalog(candidate, registry, families=[family], methods=METHODS[family])
    else:
        records, _, _ = load_catalog(catalog_path, registry, families=[family], methods=METHODS[family])
    trained = [r for r in records if r['method'] != 'base']
    for method in METHODS[family][1:]:
        if sum(r['method'] == method for r in trained) != 1:
            raise ValueError(f'Require exactly one {method} run; resolve ambiguity with --catalog')
    # Rebuild an absolute catalog retaining the exact imported experiment IDs.
    frozen = {'version': 1, 'models': {family: records[0]['model']}, 'runs_roots': {}, 'experiments': []}
    choices = []
    training_data = {}
    for index, run in enumerate(trained):
        root = f'run-{index}'
        frozen['runs_roots'][root] = str(Path(run['path']).parent)
        steps = sorted(run.get('selected_steps', run['steps']))
        frozen['experiments'].append({'id': run['experiment_id'], 'family': family, 'method': run['method'],
                                     'root': root, 'run': Path(run['path']).name, 'steps': steps})
        choices.append({'method': run['method'], 'experiment_id': run['experiment_id'],
                        'run_path': run['path'], 'training_sha256': run['training_sha256'],
                        'available_steps': steps, 'selected_step': None,
                        'checkpoints': run['checkpoints']})
        metadata = read_json(Path(run['path']) / 'training.json')
        training_data[run['method']] = {key: value for key, value in metadata.items() if 'data' in key.lower() or key in ('pairs_id', 'pairs_sha256', 'validation_pairs_id')}
        training_data[run['method']]['config_data'] = metadata['config'].get('data')
    provenance['training_budgets'] = {run['method']: read_json(Path(run['path']) / 'training.json')['config'].get('training', {}) for run in trained}
    provenance['budget_limitation'] = 'Same corpus or pair-view budget does not imply equal image-pair exposure; selected steps and training budgets must be reported.'
    if family == 'zimage':
        left, right = training_data['dpo'], training_data['stagewise_k4']
        known = bool(left.get('pairs_id') and right.get('pairs_id'))
        semantic_keys = ('pairs', 'split', 'sampling', 'human_only', 'human_filter', 'require_human_preference', 'require_reviewed', 'prompt_mode')
        left_data = {k: v for k, v in (left.get('config_data') or {}).items() if k in semantic_keys}
        right_data = {k: v for k, v in (right.get('config_data') or {}).items() if k in semantic_keys}
        budgets = provenance['training_budgets']
        for key in ('sampling', 'prompt_mode'):
            if budgets['dpo'].get(key) != budgets['stagewise_k4'].get(key):
                raise ValueError(f'Z-Image DPO/StagewiseK4 training {key} differs')
        if (known and left['pairs_id'] != right['pairs_id']) or (left_data and right_data and left_data != right_data):
            raise ValueError('Z-Image DPO/StagewiseK4 training dataset identities differ')
        provenance['dataset_comparability'] = ('matching recorded metadata; content independence not established' if known
                                               else 'unknown: training dataset identity missing; fairness not established')
    config = load_config(args.config, benchmarks=list(PANELS[family]), resources=args.resources)
    # Resolve config-relative scorer/source paths before embedding the config.
    for key in ('scorers_config', 'compositional_config'):
        if config.get(key) and not Path(config[key]).is_absolute():
            config[key] = str((Path(args.config).resolve().parent / config[key]).resolve())
    config['protocols'][family]['cpu_offload'] = False
    config['protocols'][family]['generation_batch_size'] = ((8 if family == 'sd35_medium' else 1) if args.generation_batch_size is None else args.generation_batch_size)
    if family == 'zimage':
        config['protocols'][family]['seeds'] = [17]
        config['metric_groups']['modern'] = ['hpsv3', 'unifiedreward', 'pickscore', 'image_reward']
    if config['protocols'][family]['generation_batch_size'] < 1:
        raise ValueError('Generation batch size must be positive')
    provenance['training_data'] = training_data
    result = {'version': 1, 'family': family, 'methods': list(METHODS[family]), 'choices': choices,
              'catalog': frozen, 'suite': config, 'provenance': provenance}
    result['identity_sha256'] = identity(result)
    path = Path(args.out).expanduser().resolve()
    if path.exists():
        raise ValueError('Selection already exists; use a new --out')
    suite.freeze_json(path, result)
    return result


def check(path, resources=None):
    selection = read_json(path)
    if selection.get('version') != 1 or selection.get('family') not in METHODS:
        raise ValueError('Unsupported selected checkpoint file')
    if selection.get('identity_sha256') != identity(selection):
        raise ValueError('Imported identities/protocol changed; edit only selected_step or initialize a fresh selection')
    seal = Path(str(Path(path).resolve()) + '.frozen.json')
    if seal.exists() and read_json(seal) != selection:
        raise ValueError('Selected steps already frozen by a formal plan; copy to a new selection file and use a new output root')
    family = selection['family']
    if selection['methods'] != list(METHODS[family]):
        raise ValueError('Formal method set changed')
    gaps = []
    catalog = copy.deepcopy(selection['catalog'])
    for choice, entry in zip(selection['choices'], catalog['experiments']):
        step = choice.get('selected_step')
        if type(step) is not int or step not in choice['available_steps']:
            gaps.append(f"{choice['method']}: select exactly one available positive integer step")
        else:
            entry['steps'] = [step]
    if gaps:
        return selection, catalog, gaps
    # Recheck exact permanent checkpoints, current training identity and adapter manifest.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / 'catalog.json'
        suite.freeze_json(candidate, catalog)
        records, _, _ = load_catalog(candidate, load_resources(resources), families=[family], methods=METHODS[family])
    by_method = {r['method']: r for r in records}
    for choice in selection['choices']:
        run = by_method[choice['method']]
        if run['training_sha256'] != choice['training_sha256'] or run['path'] != choice['run_path']:
            raise ValueError(f"Training identity changed: {choice['method']}")
        step = choice['selected_step']
        original = next(c for c in choice['checkpoints'] if c['step'] == step)
        current = next(c for c in run['checkpoints'] if c['step'] == step)
        if current != original:
            raise ValueError(f"Retained checkpoint identity changed: {choice['method']} step {step}")
    return selection, catalog, []


def _suite_plan(command, selection):
    import json
    parsed = suite.parse_args(command)
    parsed.report_selection = {'mode': 'formal-selected-checkpoints', 'selection': selection,
                               'selection_id': digest(selection)}
    result = suite.make_plan(parsed)
    print(json.dumps({'plan_id': result['plan_id'], 'tasks': len(result['tasks']),
                      'total_images': sum(t['images'] for t in result['tasks']),
                      'output': parsed.out, 'dry_run': parsed.dry_run,
                      'selection': [{**{k: t[k] for k in ('method', 'step', 'benchmark', 'prompts', 'images')},
                                     'seeds': t['sampling']['seeds'],
                                     'generation_batch_size': t['sampling'].get('generation_batch_size', 1)}
                                    for t in result['tasks']]}, indent=2))
    return 0


def plan(args):
    selection, catalog, gaps = check(args.selection, args.resources)
    if gaps:
        raise ValueError('; '.join(gaps))
    family = selection['family']
    panels = args.benchmarks or list(PANELS[family])
    if not panels or set(panels) - set(PANELS[family]):
        raise ValueError('Benchmark subset is outside the frozen family protocol')
    out = Path(args.out).expanduser().resolve()
    if getattr(args, 'dry_run', False):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary)
            suite.freeze_json(inputs / 'catalog.json', catalog)
            suite.freeze_json(inputs / 'suite.json', selection['suite'])
            command = ['plan', '--dry-run', '--catalog', str(inputs / 'catalog.json'),
                       '--config', str(inputs / 'suite.json'), '--families', family,
                       '--methods', *METHODS[family], '--benchmarks', *panels, '--out', str(out)]
            if args.resources:
                command += ['--resources', args.resources]
            return _suite_plan(command, selection)
    # Sidecar inputs remain outside suite output, which enforces nonempty protection.
    inputs = out.with_name(out.name + '.selected-inputs')
    suite.freeze_json(inputs / 'catalog.json', catalog)
    suite.freeze_json(inputs / 'suite.json', selection['suite'])
    suite.freeze_json(inputs / 'selection.json', selection)
    command = ['plan', '--catalog', str(inputs / 'catalog.json'), '--config', str(inputs / 'suite.json'),
               '--families', family, '--methods', *METHODS[family], '--benchmarks', *panels, '--out', str(out)]
    if args.resources:
        command += ['--resources', args.resources]
    with suite.output_lock(str(Path(args.selection).resolve()) + '.selected'):
        # Validate prompts and exact checkpoint metadata before sealing choices.
        preview = suite.parse_args([*command, '--dry-run'])
        preview.report_selection = {'mode': 'formal-selected-checkpoints', 'selection': selection, 'selection_id': digest(selection)}
        candidate = suite.make_plan(preview)
        if out.exists():
            if (out / 'plan.json').exists():
                if read_json(out / 'plan.json') != candidate:
                    raise ValueError('This directory has a different plan; choose a new --out')
            elif any(p.name != 'logs' or p.is_symlink() or not p.is_dir() for p in out.iterdir()):
                raise ValueError('Refusing to place a plan in a nonempty unrelated directory')
        seal = Path(str(Path(args.selection).resolve()) + '.frozen.json')
        if seal.exists() and read_json(seal) != selection:
            raise ValueError('Selected steps already frozen by a formal plan; copy to a new selection file and use a new output root')
        suite.freeze_json(seal, selection)
        result = _suite_plan(command, selection)
        if result == 0:
            suite.freeze_json(out / 'selected-checkpoints.json', {'selection': selection,
                              'selection_sha256': sha256(inputs / 'selection.json'), 'benchmarks': panels,
                              'plan_id': read_json(out / 'plan.json')['plan_id']})
        return result
