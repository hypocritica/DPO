"""Read-only checks for isolated evaluation environments; never downloads weights."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
METRICS = ('pickscore', 'hpsv2', 'aesthetic', 'clip', 'image_reward')
from setup_eval_environment import PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='legacy')
    parser.add_argument('--skip-assets', action='store_true', help='Check dependencies before downloading assets')
    parser.add_argument('--resources', help='Machine resource registry')
    parser.add_argument('--scorers-config', default=str(ROOT / 'configs/scorers.json'))
    parser.add_argument('--compositional-config', default=str(ROOT / 'configs/compositional.json'))
    parser.add_argument('--gpu', action='store_true', help='Explicit tiny CUDA operation; requires allocated GPU')
    parser.add_argument('--execution', choices=('slurm', 'standalone'), default='slurm',
                        help='GPU execution boundary; standalone is for externally allocated SLAI GPUs')
    parser.add_argument('--run-label', default='eval-preflight')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    errors = []

    def check(label, operation):
        try:
            detail = operation()
            print(f'[OK] {label}: {detail or "available"}')
            return True
        except Exception as exc:
            errors.append(label)
            print(f'[FAIL] {label}: {type(exc).__name__}: {exc}')
            return False

    print(f'Python {sys.version.split()[0]}: {sys.executable}')
    check('Python 3.10', lambda: require(sys.version_info[:2] == (3, 10), 'Use conda eval with Python 3.10'))
    # Read pins without invoking pip, importing CUDA, or contacting package indexes.
    from packaging.requirements import Requirement
    for line in (ROOT / PROFILES[args.profile]).read_text().splitlines():
        line = line.split(' #', 1)[0].strip()
        if not line or line.startswith(('#', '--')):
            continue
        req = Requirement(line)

        def installed(req=req):
            version = metadata.version(req.name)
            require(not req.specifier or req.specifier.contains(version), f'{version} does not satisfy {req.specifier}')
            if req.url:
                direct = json.loads(metadata.distribution(req.name).read_text('direct_url.json') or '{}')
                expected = req.url.rsplit('@', 1)[-1]
                require(direct.get('vcs_info', {}).get('commit_id') == expected,
                        f'Install {req.name} from the pinned official Git revision {expected}')
            return version
        check(req.name, installed)

    imports = {
        'legacy': ('torch', 'torchvision', 'transformers', 'diffusers', 'peft', 'accelerate',
                   'timm', 'fairscale', 'clip', 'datasets', 'webdataset'),
        'flow': ('torch', 'torchvision', 'diffusers', 'transformers', 'peft', 'sentencepiece'),
        'hpsv3': ('torch', 'torchvision', 'transformers', 'peft', 'trl', 'deepspeed', 'qwen_vl_utils'),
        'unifiedreward': ('torch', 'torchvision', 'transformers', 'qwen_vl_utils'),
        'geneval': ('torch', 'torchvision', 'mmcv.ops', 'mmdet', 'open_clip', 'clip_benchmark'),
        'dpg': ('torch', 'torchvision', 'modelscope', 'transformers', 'fairseq'),
    }
    for name in imports[args.profile]:
        check(f'import {name}', lambda name=name: importlib.import_module(name).__file__)

    if args.profile == 'flow':
        from self_correct_dpo.evaluation.render import check_dependencies
        check('Z-Image API versions', lambda: check_dependencies('zimage', peft=True))
    if args.profile == 'unifiedreward':
        check('Qwen2.5-VL API', lambda: getattr(importlib.import_module('transformers'),
                                               'Qwen2_5_VLForConditionalGeneration').__name__)

    if not args.skip_assets:
        check('local assets', lambda: profile_assets(args))
    if args.gpu:
        check('tiny CUDA operation', lambda: cuda_check(args))
    print(f'{len(errors)} failed checks. Full model loading/scoring still requires a small evaluation run.')
    return bool(errors)


def profile_assets(args):
    if args.profile == 'flow':
        return 'Generation assets are run-specific; use evaluate_suite.py generate --check-only'
    if args.profile in ('geneval', 'dpg'):
        from self_correct_dpo.evaluation.compositional import load_settings, asset_identity, source_identity
        settings = load_settings(args.compositional_config, args.profile, resources=args.resources)
        source_identity(settings['repo'], args.profile)
        asset_identity(settings, args.profile)
        return 'local evaluator source/config/weights present'

    from self_correct_dpo.config import load_config
    from self_correct_dpo.evaluation.scorers import scorer_identity, _open_clip
    from self_correct_dpo.evaluation.optional_scorers import source_package
    try:
        metrics = METRICS if args.profile == 'legacy' else (args.profile,)
        specs = load_config(args.scorers_config, scorers=metrics, resources=args.resources)['scorers']
    except Exception as exc:
        raise ValueError(f'Scorer configuration: {exc}') from exc
    for metric in metrics:
        spec = specs[metric]
        if spec.get('python'):
            require(
                Path(spec['python']).resolve() == Path(sys.executable).resolve(),
                'Selected scorer uses a different Python; run the check in that environment')
        if metric in ('hpsv3', 'unifiedreward'):
            scorer_identity(metric, spec)
        else:
            asset_check(metric, spec, scorer_identity)
    if args.profile == 'legacy':
        _open_clip(specs['hpsv2']['open_clip_source'])
        source_package('ImageReward', specs['image_reward']['source'])
    return 'local files and source identities present (weights not loaded)'


def cuda_check(args):
    from self_correct_dpo.common import require_gpu_allocation
    require_gpu_allocation({'mode': args.execution, 'run_label': args.run_label})
    import torch
    require(torch.cuda.is_available(), 'CUDA is unavailable')
    value = torch.ones((16, 16), device='cuda')
    require(float((value @ value).sum().item()) == 4096, 'CUDA matrix multiplication failed')
    torch.cuda.synchronize()
    return f'{torch.cuda.get_device_name()}; torch={torch.__version__}, CUDA={torch.version.cuda}'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def asset_check(metric, spec, identity):
    identity(metric, spec)
    # Identity checks files/source provenance; also check essential local HF payloads.
    for key in ('model', 'processor', 'tokenizer'):
        if key not in spec:
            continue
        path = Path(spec[key])
        if key == 'model':
            require((path / 'config.json').is_file(), f'{path}/config.json missing')
            require(any(path.glob('*.safetensors')) or any(path.glob('*.bin')), f'{path}: model weights missing')
        elif key == 'processor':
            require((path / 'preprocessor_config.json').is_file(), f'{path}: image processor config missing')
            require((path / 'tokenizer_config.json').is_file(), f'{path}: tokenizer config missing')
        else:
            require((path / 'vocab.txt').is_file(), f'{path}: BERT vocab.txt missing')
    return 'local files and source identity present (weights not loaded)'


if __name__ == '__main__':
    raise SystemExit(main())
