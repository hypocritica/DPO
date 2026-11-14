"""Freeze reviewed SDXL inputs once, then plan one independent preference panel."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import read_json
from self_correct_dpo.config import load_config
from self_correct_dpo.resources import load_resources
from self_correct_dpo.evaluation import suite
from self_correct_dpo.evaluation.experiment_catalog import load_catalog

PANELS = ('partiprompts', 'pickv2', 'hpsv2')


def prepare(args):
    registry = load_resources(args.resources)
    records, _, _ = load_catalog(args.catalog, registry, families=['sdxl'])
    catalog = {'version': 1, 'models': {'sdxl': records[0]['model']}, 'runs_roots': {}, 'experiments': []}
    identities = []
    for i, run in enumerate(records):
        if run['method'] == 'base':
            identities.append({'method': 'base', 'model': run['model']})
            continue
        key = f'run-{i}'
        catalog['runs_roots'][key] = str(Path(run['path']).parent)
        catalog['experiments'].append({'id': run['experiment_id'], 'family': 'sdxl', 'method': run['method'],
            'method_label': run['method_label'], 'root': key, 'run': Path(run['path']).name,
            'steps': run['selected_steps']})
        identities.append({'method': run['method'], 'experiment_id': run['experiment_id'],
            'training_sha256': run['training_sha256'], 'checkpoints': [c for c in run['checkpoints'] if c['step'] in run['selected_steps']]})
    config = load_config(args.config, benchmarks=list(PANELS), resources=args.resources)
    config['protocols'] = {'sdxl': copy.deepcopy(config['protocols']['sdxl'])}
    config['protocols']['sdxl']['generation_batch_size'] = args.batch_size
    config['protocols']['sdxl']['cpu_offload'] = False
    if config.get('compositional_config'):
        config['compositional_config'] = str((Path(args.config).resolve().parent / config['compositional_config']).resolve())
    receipt = {'version': 1, 'catalog_sha256': hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest(),
               'catalog': catalog, 'suite': config, 'checkpoint_identities': identities}
    root = Path(args.root).resolve()
    shared = root / 'shared'

    def plan(folder):
        return suite.main(['plan', '--catalog', str(folder/'catalog.json'), '--config', str(folder/'suite.json'),
            '--families', 'sdxl', '--benchmarks', args.panel, '--out', str(root/args.panel),
            *(['--resources', args.resources] if args.resources else []), *(['--dry-run'] if args.dry_run else [])])

    if args.dry_run:
        if (shared/'selection.json').exists() and read_json(shared/'selection.json') != receipt:
            raise ValueError('Shared SDXL catalog/protocol/checkpoint identities changed; use a new root')
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            suite.freeze_json(temporary/'catalog.json', catalog)
            suite.freeze_json(temporary/'suite.json', config)
            return plan(temporary)
    shared.mkdir(parents=True, exist_ok=True)
    # A short blocking lock allows all three independently launched panel jobs to
    # initialize the same root. Generation/scoring never hold this shared lock.
    with (shared/'.selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        suite.freeze_json(shared/'selection.json', receipt)
        suite.freeze_json(shared/'catalog.json', catalog)
        suite.freeze_json(shared/'suite.json', config)
    return plan(shared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--panel', choices=PANELS, required=True)
    parser.add_argument('--config', default=str(ROOT/'configs/evaluation-suite.json'))
    parser.add_argument('--resources')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    return prepare(args)


if __name__ == '__main__':
    raise SystemExit(main())
