"""Freeze four supplemental SDXL points from permanent LoRA checkpoints only."""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import digest, read_json
from self_correct_dpo.evaluation import suite
from self_correct_dpo.evaluation.checkpoint_catalog import resolve_checkpoint
from self_correct_dpo.evaluation.experiment_catalog import load_catalog, validate_method
from self_correct_dpo.resources import load_resources

METHODS = ['dspo', 'stagewise_dspo_k6', 'dmpo', 'stagewise_dmpo_k6']


def manifest(path, run_id):
    if path.is_symlink() or (path / 'checkpoint.json').is_symlink():
        raise ValueError('Checkpoint or manifest is a symlink')
    info = read_json(path / 'checkpoint.json')
    if info['checkpoint_id'] != digest({k: v for k, v in info.items() if k != 'checkpoint_id'}):
        raise ValueError('Checkpoint manifest checksum mismatch')
    if info['run_id'] != run_id or type(info['step']) is not int or not 0 < info['step'] <= 10000:
        raise ValueError('Checkpoint run/step mismatch')
    return info


def latest_source(run, run_id):
    final = run / 'checkpoints/step-0010000'
    if (final / 'checkpoint.json').exists():
        return final, manifest(final, run_id)
    candidates = []
    for path in (run / 'checkpoints').glob('step-*'):
        if (path / 'checkpoint.json').is_file():
            info = manifest(path, run_id)
            if path.name != f'step-{info["step"]:07d}':
                raise ValueError('Permanent checkpoint directory/step mismatch')
            candidates.append((path, info))
    if not candidates:
        raise ValueError('No retained DMPO + MVPO checkpoint available')
    return max(candidates, key=lambda item: item[1]['step'])


def prepare(root, template, resources=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'catalog.json'
    with (root / '.peak-selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            catalog = copy.deepcopy(read_json(template))
            catalog['selection'] = {'role': 'development-selected-supplement',
                'decision': 'DSPO pair3000, DMPO10000, Stagewise DMPO latest permanent LoRA up to10000; rolling checkpoints excluded; fixed once before formal scoring.'}
            for entry in catalog['experiments']:
                run = Path(catalog['runs_roots'][entry['root']]) / entry['run']
                if root.is_relative_to(run.resolve()):
                    raise ValueError('Supplement output must be outside training runs')
                training = read_json(run / 'training.json')
                if training['run_id'] != entry['expected_run_id']:
                    raise ValueError(f'Historical run identity mismatch: {run}')
                validate_method(training['config'], entry['method'], entry['id'])
                if entry['method'] == 'stagewise_dmpo_k6':
                    source, info = latest_source(run, training['run_id'])
                    step = info['step']
                    catalog['selection']['stagewise_dmpo_source'] = {'path': str(source), 'checkpoint_id': info['checkpoint_id'], 'step': step}
                else:
                    step = 10000 if entry['method'] == 'dmpo' else 3000
                _, _, identity = resolve_checkpoint(run, step)
                if step == 3000 and entry['method'] != 'stagewise_dmpo_k6' and identity['checkpoint_id'] != entry['expected_checkpoint_ids']['3000']:
                    raise ValueError('Historical 3000 checkpoint identity mismatch')
                entry['steps'] = [step]
                entry['expected_checkpoint_ids'] = {str(step): identity['checkpoint_id']}
            # Validate all four selected identities before publishing the catalog.
            with tempfile.TemporaryDirectory(dir=root) as directory:
                check = Path(directory) / 'catalog.json'
                suite.freeze_json(check, catalog)
                load_catalog(check, load_resources(resources), families=['sdxl'], methods=METHODS)
            suite.freeze_json(target, catalog)
        runs, _, _ = load_catalog(target, load_resources(resources), families=['sdxl'], methods=METHODS)
        for run in runs:
            for step in run['selected_steps']:
                resolve_checkpoint(run['path'], step)
                print(f"{run['method_label']}: step={step}; {run['path']}", flush=True)
    return target


def reuse(out, previous_root):
    """Import compatible completed artifacts, then let normal resume validate them."""
    plan, out = suite.read_plan(out)
    # Enforce the original 2k panel's protocol whenever its frozen plan exists.
    for panel in plan['panels']:
        reference = Path(previous_root) / 'main' / panel
        if (reference / 'plan.json').is_file():
            original, _ = suite.read_plan(reference)
            if (original['panels'][panel]['sha256'] != plan['panels'][panel]['sha256']
                    or original['metric_groups']['legacy'] != plan['metric_groups']['legacy']
                    or any(t['sampling'] != original['tasks'][0]['sampling'] for t in plan['tasks'])):
                raise ValueError('Supplement prompt/sampling/metric protocol differs from original 2k plan')
    for role in ('supplement', 'main'):
        for panel in plan['panels']:
            previous = Path(previous_root) / role / panel
            if not (previous / 'plan.json').is_file():
                continue
            old_plan, previous = suite.read_plan(previous)
            if old_plan['panels'][panel]['sha256'] != plan['panels'][panel]['sha256']:
                continue
            for task in plan['tasks']:
                target = suite.safe_target(out, task['output'])
                if (target / 'render').exists():
                    continue
                keys = ('experiment_id', 'method', 'step', 'benchmark', 'model', 'sampling', 'run_path', 'checkpoint')
                matched = [t for t in old_plan['tasks'] if all(t.get(k) == task.get(k) for k in keys)]
                if len(matched) != 1:
                    continue
                old = matched[0]
                try:
                    old_run, old_complete = suite.validate_task_render(old, old_plan, previous)
                except FileNotFoundError:
                    continue
                source = suite.safe_target(previous, old['output'])
                target.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=target, prefix='.reuse-') as temporary:
                    staging = Path(temporary)
                    shutil.copytree(source / 'render', staging / 'render')
                    # Scores are independently checked against their exact scorer on resume.
                    if (source / 'scores').is_dir() and old_plan['scorers_config'] == plan['scorers_config']:
                        for directory in (source / 'scores').iterdir():
                            if not directory.is_dir() or not (directory / 'complete.json').is_file():
                                continue
                            suite.score_summary(source, directory.name, old_run, old_complete['manifest_sha256'])
                            shutil.copytree(directory, staging / 'scores' / directory.name)
                    for item in staging.iterdir():
                        if not (target / item.name).exists():
                            item.rename(target / item.name)
                suite.validate_task_render(task, plan, out)
                suite.freeze_json(target / 'reuse-source.json', {'source': str(source),
                    'source_plan_id': old_plan['plan_id'], 'manifest_sha256': old_complete['manifest_sha256']})
                print(f"Reused {task['method']} step={task['step']} {panel} from {source}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--template', default=str(ROOT / 'configs/evaluation-sdxl-parti-supplement.json'))
    parser.add_argument('--resources')
    parser.add_argument('--reuse-out')
    parser.add_argument('--previous-root')
    args = parser.parse_args()
    if args.reuse_out:
        if not args.previous_root:
            parser.error('--reuse-out requires --previous-root')
        reuse(args.reuse_out, args.previous_root)
    else:
        prepare(args.root, args.template, args.resources)
