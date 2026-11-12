"""Explicit scorer asset stages; checks reuse local legacy/PickScore caches.

check is offline and never loads a reward model. plan queries only metadata;
download fetches selected weights; sources explicitly clones missing source.
No action installs environments or submits GPU work.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.config import load_config
from self_correct_dpo.resources import load_resources
import download_assets as downloads

PROFILES = {
    'pickscore': {'metrics': ['pickscore'], 'assets': ['pickscore', 'pickscore-processor']},
    'legacy': {'metrics': ['pickscore', 'hpsv2', 'clip', 'aesthetic', 'image_reward'],
               'assets': ['pickscore', 'pickscore-processor', 'hpsv2', 'clip-h14', 'clip-l14',
                          'image-reward', 'image-reward-tokenizer']},
    'hpsv3': {'metrics': ['hpsv3'], 'assets': ['hpsv3', 'hpsv3-base']},
    'unifiedreward': {'metrics': ['unifiedreward'], 'assets': ['unifiedreward']},
    'modern': {'metrics': ['hpsv3', 'unifiedreward'], 'assets': ['hpsv3', 'hpsv3-base', 'unifiedreward']},
}


def _positive(path):
    return path.is_file() and path.stat().st_size > 0


def local_legacy_payload(asset, target):
    """Structural reuse only: no invented download/revision receipt for manual files."""
    target = Path(target)
    required = {
        'hpsv2': ['HPS_v2_compressed.pt'],
        'image-reward': ['ImageReward.pt', 'med_config.json'],
        'image-reward-tokenizer': ['vocab.txt', 'tokenizer_config.json'],
        'pickscore-processor': ['preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'],
        'clip-h14': ['open_clip_pytorch_model.bin', 'preprocessor_config.json',
                     'tokenizer_config.json', 'vocab.json', 'merges.txt'],
    }.get(asset)
    if required is not None:
        return all(_positive(target / name) for name in required)
    if asset not in ('pickscore', 'clip-l14'):
        return False
    if not _positive(target / 'config.json'):
        return False
    indexes = list(target.glob('*.index.json'))
    if indexes:
        for index in indexes:
            shards = set(json.loads(index.read_text()).get('weight_map', {}).values())
            if not shards or any(not (target / name).resolve().is_relative_to(target.resolve())
                                 or not _positive(target / name) for name in shards):
                return False
    elif not any(_positive(target / name) for name in ('model.safetensors', 'pytorch_model.bin')):
        return False
    if asset == 'clip-l14':
        return all(_positive(target / name) for name in
                   ('preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'))
    return True


def check_metrics(metrics, specs):
    from self_correct_dpo.evaluation.scorers import scorer_identity
    errors = []
    for metric in metrics:
        try:
            # Legacy/PickScore payloads intentionally do not require our download
            # receipt; scorer_identity records their actual local inventory.
            scorer_identity(metric, specs[metric])
            if metric in ('pickscore', 'aesthetic'):
                model_asset = 'pickscore' if metric == 'pickscore' else 'clip-l14'
                if not local_legacy_payload(model_asset, specs[metric]['model']):
                    raise ValueError('Incomplete local model/tokenizer/processor payload')
                if not local_legacy_payload('pickscore-processor', specs[metric]['processor']):
                    raise ValueError('Incomplete local processor payload')
            if metric == 'image_reward' and not local_legacy_payload('image-reward-tokenizer', specs[metric]['tokenizer']):
                raise ValueError('Incomplete local BERT tokenizer payload')
            print(f'[OK] {metric}: local assets present; no model weights loaded; no downloads')
        except (OSError, ValueError, KeyError, ImportError) as error:
            errors.append(metric)
            print(f'[FAIL] {metric}: {error}', file=sys.stderr)
    return int(bool(errors))


def source_destinations(metrics, specs):
    sources = {}
    if 'hpsv2' in metrics or 'clip' in metrics:
        spec = specs['hpsv2'] if 'hpsv2' in metrics else specs['clip']
        # Official checkout contains hpsv2/src/open_clip; source is two levels
        # above the configured open_clip_source (not its hpsv2 package parent).
        sources['hpsv2'] = Path(spec['open_clip_source']).parents[1]
    if 'image_reward' in metrics:
        sources['image-reward'] = Path(specs['image_reward']['source'])
    if 'aesthetic' in metrics:
        sources['aesthetic'] = Path(specs['aesthetic']['checkpoint']).parent
    for metric in ('hpsv3', 'unifiedreward'):
        if metric in metrics:
            sources[metric] = Path(specs[metric]['source'])
    return sources


def asset_destinations(specs):
    paths = {}
    if 'pickscore' in specs:
        paths.update({'pickscore': Path(specs['pickscore']['model']),
                      'pickscore-processor': Path(specs['pickscore']['processor'])})
    if 'hpsv2' in specs:
        paths['hpsv2'] = Path(specs['hpsv2']['checkpoint']).parent
    if 'clip' in specs:
        paths['clip-h14'] = Path(specs['clip']['checkpoint']).parent
    if 'aesthetic' in specs:
        paths['clip-l14'] = Path(specs['aesthetic']['model'])
    if 'image_reward' in specs:
        paths.update({'image-reward': Path(specs['image_reward']['checkpoint']).parent,
                      'image-reward-tokenizer': Path(specs['image_reward']['tokenizer'])})
    if 'hpsv3' in specs:
        paths.update({'hpsv3': Path(specs['hpsv3']['checkpoint']).parent,
                      'hpsv3-base': Path(specs['hpsv3']['model'])})
    if 'unifiedreward' in specs:
        paths['unifiedreward'] = Path(specs['unifiedreward']['model'])
    return paths


def prepare_sources(metrics, specs):
    script = ROOT / 'scripts/download_sources.sh'
    for asset, target in source_destinations(metrics, specs).items():
        if target.exists():
            if asset in ('hpsv3', 'unifiedreward'):
                # Existing modern checkouts must be clean and exactly pinned.
                subprocess.run(['bash', str(script), '--check', asset, str(target)], check=True)
            else:
                # Preserve manual legacy source caches; actual required package
                # files are checked by check_metrics after preparation.
                print(f'[REUSE] {asset} source exists: {target}; run check to verify required payload')
            continue
        subprocess.run(['bash', str(script), asset, str(target)], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'plan', 'download', 'sources'))
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--resources', help='Same registry/local/environment override mechanism as evaluation')
    parser.add_argument('--scorers-config', default=str(ROOT / 'configs/scorers.json'))
    args = parser.parse_args(argv)
    downloads.DEFAULT_PATHS = load_resources(args.resources)
    profile = PROFILES[args.profile]
    specs = load_config(args.scorers_config, scorers=profile['metrics'], resources=args.resources)['scorers']
    if args.action == 'check':
        return check_metrics(profile['metrics'], specs)
    if args.action == 'sources':
        prepare_sources(profile['metrics'], specs)
        return 0
    reuse_manual = args.profile in ('pickscore', 'legacy')
    targets = asset_destinations(specs)
    for asset in profile['assets']:
        target = targets[asset]
        if reuse_manual and local_legacy_payload(asset, target):
            print(f'[REUSE] {asset}: {target}; local payload present (no acquisition receipt required)')
            continue
        plan = downloads.plan_asset(asset, out=target)
        print(json.dumps({key: value for key, value in plan.items() if key != 'files'}, indent=2), flush=True)
        if args.action == 'download':
            downloads.download(plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
