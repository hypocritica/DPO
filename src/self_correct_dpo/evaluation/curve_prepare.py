"""Freeze an SD3.5 checkpoint development curve without loading model weights.

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
    explicit = getattr(args, 'catalog', None)
    if explicit:
        records, roots, _ = load_catalog(explicit, registry, families=['sd35_medium'],
                                        methods=list(CURVE_METHODS[1:]), patterns=getattr(args, 'run_glob', None))
    else:
        roots = roots_from_args(getattr(args, 'runs_root', None), registry)
        records = discover_roots(roots, patterns=getattr(args, 'run_glob', None), families=['sd35_medium'])
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
        for method in CURVE_METHODS[1:]:
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
    missing = set(CURVE_METHODS[1:]) - chosen.keys()
    if missing:
        raise ValueError(f'Missing SD3.5 curve methods: {sorted(missing)}; provide current runs roots or an explicit catalog')
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


def prepare_curve(args):
    """Return frozen catalog/config/prompt paths for the independent suite plan.

    Args: prepare_dir, runs_root (str/list, optional), catalog (optional),
    run_glob (optional list), resources (optional), config (optional suite template),
    target_steps (default 3200), exclude_prompts (optional JSONL list).
    The panel is always 128 prompts with generation seed 17 and PickScore only.
    """
    out = Path(args.prepare_dir).expanduser().resolve()
    registry = load_resources(getattr(args, 'resources', None))
    roots = roots_from_args(getattr(args, 'runs_root', None), registry)
    exclusions = getattr(args, 'exclude_prompts', None) or []
    exclusions = [exclusions] if isinstance(exclusions, str) else exclusions
    exclusions = sorted({_path(path) for path in exclusions})
    config_path = _path(getattr(args, 'config', None) or CONFIG_DIR / 'evaluation-suite.json')
    request = {'runs_roots': [str(root) for root in roots], 'catalog': _path(args.catalog) if getattr(args, 'catalog', None) else None,
               'run_glob': getattr(args, 'run_glob', None), 'resources': _path(args.resources) if getattr(args, 'resources', None) else None,
               'config': config_path, 'target_steps': getattr(args, 'target_steps', 3200),
               'exclude_prompts': exclusions, 'prompts': 128, 'seeds': [17], 'metrics': ['pickscore']}
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
        provenance.update(missing_exclusion_panels=missing_panels, excluded_formal_panels=available_panels,
                          independence='not-established' if missing_panels else 'text-disjoint-from-configured-formal-panels')
        if missing_panels:
            print('Curve panel independence from formal tests is not established; missing exclusion panels: ' + ', '.join(missing_panels), flush=True)
        catalog = {'version': 1, 'runs_roots': {f'root-{i + 1}': str(root) for i, root in enumerate(selected_roots)},
                   'models': {'sd35_medium': model}, 'experiments': []}
        for method in CURVE_METHODS[1:]:
            run, config = chosen[method]
            index = next(i for i, root in enumerate(selected_roots) if Path(run['path']).is_relative_to(root))
            catalog['experiments'].append({'id': f'sd35-{method}', 'family': 'sd35_medium', 'method': method,
                'root': f'root-{index + 1}', 'run': Path(run['path']).relative_to(selected_roots[index]).as_posix(),
                'steps': [step for step in run['steps'] if step > 0]})
        suite = copy.deepcopy(template)
        scorer_config = load_config(suite['scorers_config'], scorers=['pickscore'], resources=getattr(args, 'resources', None))
        scorer_config['scorers']['pickscore']['batch_size'] = 1
        suite['scorers_config'] = str(out / 'scorers.json')
        sampling = copy.deepcopy(suite['protocols']['sd35_medium'])
        sampling['seeds'] = [17]
        suite['protocols'] = {'sd35_medium': sampling}
        suite['benchmarks'] = {'pickv2': str(out / 'prompts/pickv2.jsonl')}
        suite['metric_groups'] = {'legacy': ['pickscore'], 'modern': ['pickscore']}
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
