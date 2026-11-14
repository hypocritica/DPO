"""Freeze an SD3.5 or Z-Image checkpoint development curve without loading model weights.

This panel is for checkpoint selection, never an official benchmark. Preparation
reads caption manifests only; a completed preparation is reused without rescans.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile

from self_correct_dpo.common import digest, output_lock, read_json, sha256, write_json, write_jsonl
from self_correct_dpo.config import load_config
from self_correct_dpo.data.benchmarks import normalize_panel, prompt_key
from self_correct_dpo.evaluation.experiment_catalog import (discover_roots, load_catalog, model_spec,
                                                            roots_from_args, validate_method)
from self_correct_dpo.resources import CONFIG_DIR, load_resources

CURVE_METHODS = ('base', 'sft', 'dpo', 'linear_dpo', 'global_k2', 'stagewise_k2', 'global_k4', 'stagewise_k4')
ZIMAGE_CURVE_METHODS = ('base', 'dpo', 'stagewise_k4')

def curve_methods(family):
    return ZIMAGE_CURVE_METHODS if family == 'zimage' else CURVE_METHODS


_BAD_RUN = re.compile(r'(^|[./_-])(smoke|backup|recovery)([./_-]|$)', re.I)


def _path(value):
    return str(Path(value).expanduser().resolve())


def _records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSONL: {path}:{index}') from error
            if not isinstance(row, dict):
                raise ValueError(f'Expected JSONL object: {path}:{index}')
            yield row


def _texts(row):
    if 'prompts' in row:
        values = row['prompts']
        if not isinstance(values, list):
            raise ValueError('prompts must be an array')
        for value in values:
            if isinstance(value, dict):
                value = value.get('prompt', value.get('caption'))
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Each prompt must contain nonempty text')
            yield value
    else:
        value = row.get('prompt', row.get('caption'))
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Caption manifest row requires prompts, prompt or caption')
        yield value


def _selected_runs(args, registry):
    family = getattr(args, 'family', 'sd35_medium')
    methods_for_family = curve_methods(family)
    explicit = getattr(args, 'catalog', None)
    if explicit:
        records, roots, _ = load_catalog(explicit, registry, families=[family],
                                        methods=list(methods_for_family[1:]), patterns=getattr(args, 'run_glob', None))
    else:
        roots = roots_from_args(getattr(args, 'runs_root', None), registry)
        records = discover_roots(roots, patterns=getattr(args, 'run_glob', None), families=[family])
    chosen = {}
    for run in records:
        training_path = Path(run['path']) / 'training.json'
        config = read_json(training_path)['config']
        if sha256(training_path) != run['training_sha256']:
            raise ValueError(f'Training metadata changed during curve preparation: {training_path}')
        if not explicit and (config.get('training', {}).get('max_steps') != getattr(args, 'target_steps', 3200)
                             or _BAD_RUN.search(run['name'])):
            continue
        methods = []
        for method in methods_for_family[1:]:
            try:
                validate_method(config, method, run['name'])
            except ValueError:
                continue
            methods.append(method)
        if len(methods) != 1:
            continue
        method = methods[0]
        if method in chosen:
            raise ValueError(f'Multiple curve candidates for {method}: {chosen[method][0]["path"]}, {run["path"]}; narrow --run-glob or provide --catalog')
        if not any(step > 0 for step in run['steps']):
            raise ValueError(f'No permanent checkpoints for curve method {method}: {run["path"]}')
        chosen[method] = (run, config)
    missing = set(methods_for_family[1:]) - chosen.keys()
    if missing:
        raise ValueError(f'Missing {"SD3.5" if family == "sd35_medium" else "Z-Image"} curve methods: {sorted(missing)}; provide current runs roots or an explicit catalog')
    models = [run['model'] for run, _ in chosen.values()]
    if any(model_spec(model) != model_spec(models[0]) for model in models[1:]):
        raise ValueError('Curve methods use different Base model identities')
    return chosen, roots, models[0]


def _panel(chosen, exclusions, count=128, seed=17):
    sources = sorted({_path(config['data']['pairs']) for _, config in chosen.values()})
    forbidden, receipts = set(), []
    for path in sources:
        checksum = sha256(path)
        train_rows = 0
        for row in _records(path):
            if row.get('split') not in ('train', 'validation', 'validation_unique', 'test', 'test_unique'):
                raise ValueError(f'Caption source has unknown/missing split: {path}')
            if row['split'] == 'train':
                forbidden.update(prompt_key(text) for text in _texts(row))
                train_rows += 1
        receipts.append({'path': path, 'sha256': checksum, 'train_rows': train_rows})
    train_keys = len(forbidden)
    exclude_receipts = []
    for path in exclusions:
        checksum = sha256(path)
        for row in _records(path):
            forbidden.update(prompt_key(text) for text in _texts(row))
        if sha256(path) != checksum:
            raise ValueError(f'Exclusion source changed while preparing: {path}')
        exclude_receipts.append({'path': path, 'sha256': checksum})
    # Keep the smallest seeded key hashes. Memory for candidate texts is bounded
    # by 128; selection is invariant to manifest order and repeated captions.
    selected, validation_rows = {}, 0
    for receipt in receipts:
        for row in _records(receipt['path']):
            if row['split'] not in ('validation', 'validation_unique'):
                continue
            validation_rows += 1
            for text in _texts(row):
                key = prompt_key(text)
                if key in forbidden:
                    continue
                rank = hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()
                candidate = (rank, key)
                if key in selected:
                    selected[key] = min(selected[key], text.strip())
                elif len(selected) < count:
                    selected[key] = text.strip()
                else:
                    worst = max(selected, key=lambda k: (hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest(), k))
                    if candidate < (hashlib.sha256(f'{seed}:{worst}'.encode()).hexdigest(), worst):
                        del selected[worst]
                        selected[key] = text.strip()
        if sha256(receipt['path']) != receipt['sha256']:
            raise ValueError(f'Caption source changed while preparing: {receipt["path"]}')
    if len(selected) != count:
        raise ValueError(f'Need {count} unique validation captions disjoint from train and exclusions; found {len(selected)}. No training/test fallback is allowed.')
    rows = normalize_panel([{'prompt': selected[key], 'split': 'validation'} for key in sorted(selected)], 'pickv2')
    return rows, {'sources': receipts, 'exclude_sources': exclude_receipts,
                  'training_prompt_keys': train_keys, 'total_excluded_prompt_keys': len(forbidden),
                  'validation_rows_examined': validation_rows,
                  'selection_rule': '128 lowest SHA256(seed17 + normalized caption key); NFKC/casefold/whitespace deduplication',
                  'allowed_splits': ['validation', 'validation_unique']}


def load_prepared(path):
    """Verify frozen artifacts only; do not revisit runs, datasets or assets."""
    out = Path(path).expanduser().resolve()
    manifest = read_json(out / 'prepare.json')
    if manifest.get('preparation_id') != digest({k: v for k, v in manifest.items() if k != 'preparation_id'}):
        raise ValueError('Curve preparation manifest checksum mismatch')
    required = {'catalog.json', 'suite.json', 'scorers.json', 'prompts/pickv2.jsonl', 'prompts/selection.json'}
    if set(manifest.get('files', {})) != required:
        raise ValueError('Curve preparation artifact set mismatch')
    for name, checksum in manifest['files'].items():
        file = (out / name).resolve()
        if not file.is_relative_to(out) or sha256(file) != checksum:
            raise ValueError(f'Frozen curve preparation changed: {name}')
    return manifest


def fork_prepared(source, out, batch_size, cpu_offload):
    source = Path(source).expanduser().resolve()
    parent = load_prepared(source)
    request = {'from_prepared': str(source), 'parent_preparation_id': parent['preparation_id'],
               'generation_batch_size': batch_size, 'cpu_offload': cpu_offload}
    with output_lock(str(out) + '.prepare'):
        if out.exists():
            manifest = load_prepared(out)
            if manifest['request'] != request:
                raise ValueError('Curve preparation inputs changed; choose a new --prepare-dir')
        else:
            suite = read_json(source / 'suite.json')
            suite['scorers_config'] = str(out / 'scorers.json')
            suite['benchmarks']['pickv2'] = str(out / 'prompts/pickv2.jsonl')
            family = next(iter(read_json(source / 'catalog.json')['models']))
            suite['protocols'][family]['generation_batch_size'] = batch_size
            if cpu_offload is not None:
                suite['protocols'][family]['cpu_offload'] = cpu_offload == 'true'
            out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
                stage = Path(temporary) / 'prepared'
                (stage / 'prompts').mkdir(parents=True)
                for name in parent['files']:
                    if name != 'suite.json':
                        (stage / name).write_bytes((source / name).read_bytes())
                write_json(stage / 'suite.json', suite)
                manifest = {k: copy.deepcopy(v) for k, v in parent.items() if k not in ('preparation_id', 'request', 'files')}
                manifest.update(request=request, files={name: sha256(stage / name) for name in parent['files']})
                manifest['preparation_id'] = digest(manifest)
                write_json(stage / 'prepare.json', manifest)
                stage.rename(out)
    return {'prepare_dir': str(out), 'catalog': str(out / 'catalog.json'),
            'config': str(out / 'suite.json'), 'prompts_root': str(out / 'prompts'),
            'manifest': str(out / 'prepare.json'), 'preparation_id': manifest['preparation_id']}


def prepare_curve(args):
    """Return frozen catalog/config/prompt paths for the independent suite plan.

    Args: prepare_dir, runs_root (str/list, optional), catalog (optional),
    run_glob (optional list), resources (optional), config (optional suite template),
    target_steps (default 3200), exclude_prompts (optional JSONL list).
    The panel is always 128 prompts with generation seed 17; Z-Image may freeze HPSv3 instead of PickScore.
    """
    family = getattr(args, 'family', 'sd35_medium')
    methods_for_family = curve_methods(family)
    requested_metric = getattr(args, 'selection_metric', None)
    metric = requested_metric or 'pickscore'
    if family not in ('sd35_medium', 'zimage') or metric not in ('pickscore', 'hpsv3') or (family == 'sd35_medium' and metric != 'pickscore'):
        raise ValueError('Unsupported curve family/selection metric')
    out = Path(args.prepare_dir).expanduser().resolve()
    batch_size = getattr(args, 'generation_batch_size', 4)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('SD3.5 curve generation_batch_size must be a positive integer')
    cpu_offload = getattr(args, 'cpu_offload', None)
    if cpu_offload not in (None, 'true', 'false'):
        raise ValueError('cpu_offload must be true or false')
    if getattr(args, 'from_prepared', None):
        load_prepared(args.from_prepared)
        frozen_metric = read_json(Path(args.from_prepared) / 'suite.json')['metric_groups']['modern'][0]
        if requested_metric is not None and requested_metric != frozen_metric:
            raise ValueError('Fork selection metric differs from frozen preparation; create a fresh preparation')
        source_family = next(iter(read_json(Path(args.from_prepared) / 'catalog.json')['models']))
        if source_family != family:
            raise ValueError('Prepared curve family differs from requested family')
        return fork_prepared(args.from_prepared, out, batch_size, cpu_offload)
    registry = load_resources(getattr(args, 'resources', None))
    roots = roots_from_args(getattr(args, 'runs_root', None), registry)
    exclusions = getattr(args, 'exclude_prompts', None) or []
    exclusions = [exclusions] if isinstance(exclusions, str) else exclusions
    exclusions = sorted({_path(path) for path in exclusions})
    config_path = _path(getattr(args, 'config', None) or CONFIG_DIR / 'evaluation-suite.json')
    request = {'runs_roots': [str(root) for root in roots], 'catalog': _path(args.catalog) if getattr(args, 'catalog', None) else None,
               'run_glob': getattr(args, 'run_glob', None), 'resources': _path(args.resources) if getattr(args, 'resources', None) else None,
               'config': config_path, 'target_steps': getattr(args, 'target_steps', 3200),
               'exclude_prompts': exclusions, 'prompts': 128, 'seeds': [17], 'metrics': ['pickscore'],
               'generation_batch_size': batch_size}
    if family == 'zimage':
        request.update(family=family, metrics=[metric])
    if cpu_offload is not None:
        request['cpu_offload'] = cpu_offload
    def result(manifest):
        return {'prepare_dir': str(out), 'catalog': str(out / 'catalog.json'),
                'config': str(out / 'suite.json'), 'prompts_root': str(out / 'prompts'),
                'manifest': str(out / 'prepare.json'), 'preparation_id': manifest['preparation_id']}
    with output_lock(str(out) + '.prepare'):
        if out.exists():
            manifest = load_prepared(out)
            if manifest['request'] != request:
                raise ValueError('Curve preparation inputs changed; choose a new --prepare-dir')
            return result(manifest)
        chosen, selected_roots, model = _selected_runs(args, registry)
        training_datasets = {}
        if family == 'zimage':
            for method, (run, config) in chosen.items():
                metadata = read_json(Path(run['path']) / 'training.json')
                training_datasets[method] = {'pairs_id': metadata.get('pairs_id'),
                                             'data': config.get('data', {}),
                                             'training': config.get('training', {})}
            ids = [value['pairs_id'] for value in training_datasets.values()]
            if all(ids) and len(set(ids)) != 1:
                raise ValueError('Z-Image curve methods have different training pairs_id content identities')
        if any(out == root or out.is_relative_to(root) for root in selected_roots):
            raise ValueError('Curve preparation must be outside read-only training roots')
        template = load_config(config_path, resources=getattr(args, 'resources', None))
        official = template.get('benchmarks', {})
        missing_panels = [name for name in ('pickv2', 'hpsv2', 'partiprompts', 'geneval', 'dpg')
                          if name not in official or not Path(official[name]).is_file()]
        available_panels = {name: official[name] for name in ('pickv2', 'hpsv2', 'partiprompts', 'geneval', 'dpg')
                            if name in official and Path(official[name]).is_file()}
        rows, provenance = _panel(chosen, sorted(set(exclusions) | set(available_panels.values())))
        for run, _ in chosen.values():
            if sha256(Path(run['path']) / 'training.json') != run['training_sha256']:
                raise ValueError('Training metadata changed during curve preparation')
        if family == 'zimage':
            provenance['training_datasets'] = training_datasets
            provenance['dataset_comparability'] = ('matching pairs_id content identities; pair-view budgets do not imply equal image-pair exposure'
                                                    if all(v['pairs_id'] for v in training_datasets.values()) else
                                                    'unknown: missing pairs_id; matched training corpus fairness is not established')
        provenance.update(missing_exclusion_panels=missing_panels, excluded_formal_panels=available_panels,
                          independence='not-established' if missing_panels else 'text-disjoint-from-configured-formal-panels')
        if missing_panels:
            print('Curve panel independence from formal tests is not established; missing exclusion panels: ' + ', '.join(missing_panels), flush=True)
        catalog = {'version': 1, 'runs_roots': {f'root-{i + 1}': str(root) for i, root in enumerate(selected_roots)},
                   'models': {family: model}, 'experiments': []}
        for method in methods_for_family[1:]:
            run, config = chosen[method]
            index = next(i for i, root in enumerate(selected_roots) if Path(run['path']).is_relative_to(root))
            catalog['experiments'].append({'id': f'{"sd35" if family == "sd35_medium" else family}-{method}', 'family': family, 'method': method,
                'root': f'root-{index + 1}', 'run': Path(run['path']).relative_to(selected_roots[index]).as_posix(),
                'steps': [step for step in run['steps'] if step > 0]})
        suite = copy.deepcopy(template)
        scorer_config = load_config(suite['scorers_config'], scorers=(['pickscore', 'hpsv3'] if family == 'zimage' else ['pickscore']), resources=getattr(args, 'resources', None))
        scorer_config['scorers']['pickscore']['batch_size'] = 1
        suite['scorers_config'] = str(out / 'scorers.json')
        sampling = copy.deepcopy(suite['protocols'][family])
        sampling['seeds'] = [17]
        sampling['generation_batch_size'] = batch_size
        if cpu_offload is not None:
            sampling['cpu_offload'] = cpu_offload == 'true'
        suite['protocols'] = {family: sampling}
        suite['benchmarks'] = {'pickv2': str(out / 'prompts/pickv2.jsonl')}
        suite['metric_groups'] = {'legacy': [metric], 'modern': [metric]}
        suite['benchmark_seeds'] = {}
        if suite.get('compositional_config') and not Path(suite['compositional_config']).is_absolute():
            suite['compositional_config'] = str(Path(config_path).parent / suite['compositional_config'])
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
            stage = Path(temporary) / 'prepared'
            (stage / 'prompts').mkdir(parents=True)
            write_json(stage / 'catalog.json', catalog)
            write_json(stage / 'suite.json', suite)
            write_json(stage / 'scorers.json', scorer_config)
            write_jsonl(stage / 'prompts/pickv2.jsonl', rows)
            selection = {'mode': 'development-curve', 'purpose': 'Checkpoint selection only; not an official benchmark or paper test result',
                'seed': 17, 'normalization': 'NFKC/casefold/whitespace', **provenance,
                'panels': {'pickv2': {'prompts': 128, 'sha256': sha256(stage / 'prompts/pickv2.jsonl')}}}
            write_json(stage / 'prompts/selection.json', selection)
            manifest = {'request': request, 'mode': 'development-curve',
                'independence': provenance['independence'], 'missing_exclusion_panels': missing_panels,
                'training_sources': [{'path': run['path'], 'training_sha256': run['training_sha256'],
                                      'checkpoints': run['checkpoints']} for run, _ in chosen.values()],
                'files': {name: sha256(stage / name) for name in ('catalog.json', 'suite.json', 'scorers.json', 'prompts/pickv2.jsonl', 'prompts/selection.json')}}
            manifest['preparation_id'] = digest(manifest)
            write_json(stage / 'prepare.json', manifest)
            stage.rename(out)
        return result(manifest)
