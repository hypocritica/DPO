"""Explicit paper experiment selection; directory names never imply a method.

A catalog freezes the model, method, run and exact retained steps together.
Paths are relative to the catalog, and may use the resource registry variables.
Discovery is metadata-only and ambiguous globs fail instead of choosing a run.
"""
import copy
from fnmatch import fnmatchcase
from pathlib import Path
import re
from string import Template

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.evaluation.checkpoint_catalog import FAMILIES, discover_runs

METHODS = {
    'base': 'Base', 'sft': 'SFT', 'dpo': 'DPO', 'linear_dpo': 'LinearDPO',
    'global_k2': 'Global K2', 'stagewise_k2': 'Stagewise K2',
    'global_k4': 'Global K4', 'stagewise_k4': 'Stagewise K4',
    'dmpo': 'DMPO', 'dspo': 'DSPO', 'kto': 'KTO', 'mapo': 'MAPO',
    'global_k6': 'Global K6', 'stagewise_k6': 'Stagewise K6',
    'stagewise_dmpo_k6': 'Stagewise DMPO K6', 'stagewise_dspo_k6': 'Stagewise DSPO K6',
    'gaussian_b_k2': 'Gaussian B K2',
    'stagewise_constant_k2': 'Stagewise Constant K2',
    'stagewise_phase_k2': 'Stagewise Phase K2', 'stagewise_reverse_k2': 'Stagewise Reverse K2',
}
_SLUG = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')


def roots_from_args(values, registry):
    values = [values] if isinstance(values, str) else values
    return [Path(p).expanduser().resolve() for p in (values or [registry['RUNS_ROOT']])]


def discover_roots(roots, patterns=None, families=None):
    """Keep legacy single-root names, qualify names for multiple roots."""
    if len(set(roots)) != len(roots):
        raise ValueError('Runs roots must be distinct')
    result, paths, names = [], set(), set()
    for index, root in enumerate(roots):
        for run in discover_runs(root, patterns=patterns, families=families):
            if run['path'] in paths:
                raise ValueError(f'Overlapping runs roots discover the same run: {run["path"]}')
            paths.add(run['path'])
            if len(roots) > 1:
                run['name'] = f'root-{index + 1}/{run["name"]}'
            if run['name'] in names:
                raise ValueError(f'Duplicate run name: {run["name"]}')
            names.add(run['name'])
            result.append(run)
    return result


def model_spec(model):
    """Report locations are provenance; all generation controls remain identity."""
    return {key: value for key, value in model.items() if key != 'validation_report'}


def base_record(model):
    """A baseline belongs to the model identity, independent of training runs."""
    key = digest(model_spec(model))
    return {'name': f'_base/{model["family"]}-{key[:12]}', 'path': None,
            'family': model['family'], 'model': copy.deepcopy(model_spec(model)),
            'model_audit': copy.deepcopy(model), 'steps': [0],
            'checkpoints': [], 'method': 'base', 'method_label': METHODS['base'],
            'experiment_id': f'{model["family"]}-base-{key[:12]}',
            'base_identity': {'base_model_id': key, 'step': 0, 'checkpoint_id': 'base',
                              'checkpoint_sha256': None}, 'files_verified': False}


def validate_method(config, method, experiment):
    """Validate the declared paper label against frozen training semantics."""
    if method not in METHODS or method == 'base':
        raise ValueError(f'Unsupported trained method: {method}')
    training = config.get('training', {})
    actual = {'method': config.get('method'), 'objective': training.get('objective'),
              'noise_views': training.get('noise_views'), 'formal_arm': training.get('formal_arm'),
              'stagewise': training.get('stagewise')}
    if method in ('sft', 'dpo', 'linear_dpo', 'dmpo', 'dspo', 'kto', 'mapo'):
        valid = (actual['method'] == 'dpo' and actual['objective'] == method
                 and actual['noise_views'] == 1 and not actual['stagewise'])
    else:
        stage = method.startswith('stagewise_')
        objective = next((o for o in ('dmpo', 'dspo') if f'_{o}_' in method), 'dpo')
        stage_objective = next((o for o in ('constant', 'phase', 'reverse') if f'_{o}_' in method), 'dpo')
        arm = 'B' if method == 'gaussian_b_k2' else 'C'
        valid = (actual['method'] == (f'stagewise-{arm}-gaussian' if stage else f'multiview-{arm}-gaussian')
                 and actual['objective'] == objective and actual['noise_views'] == int(method[-1])
                 and actual['formal_arm'] == arm
                 and (isinstance(actual['stagewise'], dict) and actual['stagewise'].get('objective') == stage_objective
                      if stage else not actual['stagewise']))
    if not valid or training.get('independent_rollout'):
        raise ValueError(f'Experiment {experiment} declares {method}, but training method/objective/views disagree: {actual}')
    return actual


def load_catalog(path, registry, *, families=None, methods=None, patterns=None):
    path = Path(path).expanduser().resolve()
    catalog = read_json(path)
    if not isinstance(catalog, dict) or catalog.get('version') != 1:
        raise ValueError('Experiment catalog requires version 1')
    if not isinstance(catalog.get('models'), dict) or not catalog['models']:
        raise ValueError('Experiment catalog requires explicit models')

    def location(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Catalog paths must be nonempty strings')
        try:
            value = Template(value).substitute(registry)
        except KeyError as error:
            raise ValueError(f'Unknown catalog resource: {error.args[0]}') from error
        result = Path(value).expanduser()
        return (result if result.is_absolute() else path.parent / result).resolve()

    if families and set(families) - set(catalog['models']):
        raise ValueError(f'Families missing from catalog models: {sorted(set(families) - set(catalog["models"]))}')
    def expand(value):
        if isinstance(value, str):
            try:
                return Template(value).substitute(registry)
            except KeyError as error:
                raise ValueError(f'Unknown catalog resource: {error.args[0]}') from error
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value
    models = expand(copy.deepcopy(catalog['models']))
    for family, model in models.items():
        if family not in FAMILIES or not isinstance(model, dict) or model.get('family') != family:
            raise ValueError(f'Invalid catalog model: {family}')
        model['path'] = str(location(model.get('path')))
        if model.get('vae'):
            model['vae'] = str(location(model['vae']))
    roots = catalog.get('runs_roots', {})
    if not isinstance(roots, dict):
        raise ValueError('runs_roots must map names to paths')
    roots = {name: location(value) for name, value in roots.items()}
    if any(not _SLUG.fullmatch(name) for name in roots):
        raise ValueError('runs_roots names must be simple identifiers')
    experiments = catalog.get('experiments', [])
    if not isinstance(experiments, list):
        raise ValueError('experiments must be a list')
    selected, ids, selections = [], set(), set()
    for entry in experiments:
        if not isinstance(entry, dict):
            raise ValueError('Each experiment must be an object')
        name, family, method = entry.get('id'), entry.get('family'), entry.get('method')
        if not isinstance(name, str) or not _SLUG.fullmatch(name) or name in ids:
            raise ValueError(f'Experiment id must be unique and a simple identifier: {name}')
        ids.add(name)
        if family not in models or method not in METHODS or method == 'base':
            raise ValueError(f'Invalid experiment model/method: {name}; select Base via --methods base')
        steps = entry.get('steps')
        if (not isinstance(steps, list) or not steps or any(type(s) is not int or s <= 0 for s in steps)
                or len(set(steps)) != len(steps)):
            raise ValueError(f'Experiment {name} requires distinct positive exact steps')
        if families and family not in families or methods and method not in methods:
            continue
        if patterns and not any(fnmatchcase(name, p) for p in patterns):
            continue
        root_name = entry.get('root')
        if root_name not in roots:
            raise ValueError(f'Unknown runs root for {name}: {root_name}')
        if ('run' in entry) == ('run_glob' in entry):
            raise ValueError(f'{name} requires exactly one of run or run_glob')
        root = roots[root_name]
        selector = entry.get('run', entry.get('run_glob'))
        if not isinstance(selector, str) or not selector:
            raise ValueError(f'Experiment {name} run selector must be a nonempty string')
        if 'run' in entry:
            run_path = location(str(root / entry['run']))
            if not run_path.is_relative_to(root):
                raise ValueError(f'Experiment run escapes its root: {name}')
            found = discover_runs(run_path, families=[family])
        else:
            found = discover_runs(root, patterns=[entry['run_glob']], families=[family])
        if len(found) != 1:
            raise ValueError(f'Experiment {name} matched {len(found)} runs; require exactly one explicit run')
        run = copy.deepcopy(found[0])
        if 'run' in entry and Path(run['path']) != run_path:
            raise ValueError(f'Experiment {name} points to a parent directory, not a training run')
        method_config = validate_method(read_json(Path(run['path']) / 'training.json')['config'], method, name)
        if model_spec(run['model']) != model_spec(models[family]):
            raise ValueError(f'Experiment {name} model differs from the registered Base identity')
        missing = set(steps) - set(run['steps'])
        if missing:
            raise ValueError(f'Experiment {name} missing exact permanent checkpoints: {sorted(missing)}')
        for step in steps:
            key = (family, method, step)
            if key in selections:
                raise ValueError(f'Ambiguous family/method/step selection: {key}')
            selections.add(key)
        run.update(name=name, experiment_id=name, method=method, method_label=METHODS[method],
                   selected_steps=sorted(steps), method_config=method_config)
        selected.append(run)
    if methods:
        requested = {(family, method) for family in (families or models) for method in methods if method != 'base'}
        missing = requested - {(run['family'], run['method']) for run in selected}
        if missing:
            raise ValueError(f'Requested family/method selections missing from catalog: {sorted(missing)}')
    if not methods or 'base' in methods:
        selected = [base_record(model) for family, model in models.items()
                    if not families or family in families] + selected
    if not selected:
        raise ValueError('No catalog experiments match the selection')
    return selected, list(roots.values()), {'path': str(path), 'sha256': sha256(path)}


def draft_catalog(roots, *, families=None, patterns=None):
    """Import current local training metadata, retaining every valid candidate.

    Multiple candidates for a method are deliberately retained. Planning rejects
    overlapping family/method/step selections until the user narrows the draft.
    """
    records = discover_roots(roots, patterns=patterns, families=families)
    if not records:
        raise ValueError('No matching training runs to import')
    models, experiments, unknown = {}, [], []
    for run in records:
        family = run['family']
        if family in models and model_spec(models[family]) != model_spec(run['model']):
            raise ValueError(f'Multiple model identities for {family}; select --families and narrow --run-glob before importing')
        models.setdefault(family, copy.deepcopy(run['model']))
        config = read_json(Path(run['path']) / 'training.json')['config']
        methods = []
        for method in METHODS:
            if method == 'base':
                continue
            try:
                validate_method(config, method, run['name'])
            except ValueError:
                continue
            methods.append(method)
        if len(methods) != 1:
            unknown.append({'run': run['path'], 'reason': 'Unsupported or ambiguous training semantics',
                            'training_method': config.get('method'),
                            'training_objective': config.get('training', {}).get('objective')})
            continue
        steps = [step for step in run['steps'] if step > 0]
        if not steps:
            unknown.append({'run': run['path'], 'reason': 'No retained permanent checkpoints yet'})
            continue
        root_index = next(i for i, root in enumerate(roots) if Path(run['path']).is_relative_to(root))
        # Include a path hash so two runs with the same basename stay distinct.
        name = re.sub(r'[^A-Za-z0-9_.-]', '-', Path(run['path']).name)
        name = f'run-{name}-{digest(run["path"])[:8]}'
        experiments.append({'id': name, 'family': family, 'method': methods[0],
                            'root': f'root-{root_index + 1}',
                            'run': Path(run['path']).relative_to(roots[root_index]).as_posix(),
                            'steps': steps})
    return {'version': 1, 'runs_roots': {f'root-{i + 1}': str(root) for i, root in enumerate(roots)},
            'models': models, 'experiments': experiments, 'unregistered_runs': unknown,
            'notes': ['Imported from current local metadata, without loading weights.',
                      'Review exact steps and keep the intended run for each family/method/step.',
                      'No candidate is chosen automatically when multiple runs match.']}
