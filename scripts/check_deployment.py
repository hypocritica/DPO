"""Read-only Shangtang deployment checks; CUDA is used only with --gpu."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.resources import load_resources
from prepare_downloaded_data import pickapic_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources')
    parser.add_argument('--family', choices=['sd15', 'sdxl'], default='sd15')
    parser.add_argument('--stage', choices=['assets', 'prepared'], default='assets')
    parser.add_argument('--revision', help='Real dataset provenance if no download receipt exists')
    parser.add_argument('--gpu', action='store_true', help='Run tiny CUDA checks on allocated GPUs')
    parser.add_argument('--world-size', type=int, default=2)
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error('--world-size must be positive')
    resources = load_resources(args.resources)
    errors = []

    def check(label, condition, detail):
        print(f'[{"OK" if condition else "MISSING"}] {label}: {detail}')
        if not condition:
            errors.append(label)

    print('Python:', sys.version.split()[0], sys.executable)
    for name in ('torch', 'torchvision', 'diffusers', 'transformers', 'peft', 'accelerate',
                 'pyarrow', 'tensorboard'):
        try:
            print(f'{name}: {importlib.metadata.version(name)}')
        except importlib.metadata.PackageNotFoundError:
            check(name, False, 'Install the project requirements in this environment')
    for key in ('MODELS_ROOT', 'DATASETS_ROOT', 'RUNS_ROOT'):
        path = Path(resources[key])
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free / 1024**3
        print(f'{key}={path} (free on {ancestor}: {free:.1f} GiB)')

    model = Path(resources[args.family.upper() + '_MODEL'])
    check('model metadata', (model / 'model_index.json').is_file(), model)
    components = ['unet', 'vae', 'text_encoder'] + (['text_encoder_2'] if args.family == 'sdxl' else [])
    for name in components:
        path = model / name
        weights = [file for pattern in ('*.safetensors', '*.bin') for file in path.glob(pattern)]
        check(f'model/{name}', (path / 'config.json').is_file() and bool(weights), path)
    for name in ['tokenizer'] + (['tokenizer_2'] if args.family == 'sdxl' else []):
        check(f'model/{name}', (model / name / 'tokenizer_config.json').is_file(), model / name)
    check('scheduler', (model / 'scheduler/scheduler_config.json').is_file(), model / 'scheduler')
    marker = model / '.scd-download.json'
    if marker.is_file():
        receipt = json.loads(marker.read_text())
        check('model download', bool(receipt.get('complete')), receipt.get('revision', 'unknown revision'))

    scorer = Path(resources['PICKSCORE_MODEL'])
    weights = list(scorer.glob('*.safetensors')) + list(scorer.glob('*.bin'))
    check('PickScore', (scorer / 'config.json').is_file() and bool(weights), scorer)
    processor = Path(resources['PICKSCORE_PROCESSOR'])
    check('PickScore processor', (processor / 'preprocessor_config.json').is_file()
          and (processor / 'tokenizer_config.json').is_file(), processor)

    if args.stage == 'assets':
        try:
            files, revision, _, marker = pickapic_source(resources['PICKAPIC_RAW'], args.revision)
            check('Pick-a-Pic source', True, f'{resources["PICKAPIC_RAW"]}: {len(files)} Parquet files')
            print('Dataset revision:', revision)
            print('Dataset download receipt:', marker or 'manually supplied revision')
        except (OSError, ValueError, KeyError) as exc:
            check('Pick-a-Pic source', False, str(exc))
    else:
        prepared = Path(resources['PICKAPIC_' + args.family.upper() + '_PREPARED'])
        check('pairs.jsonl', (prepared / 'pairs.jsonl').is_file(), prepared / 'pairs.jsonl')
        monitor = Path(resources['MONITOR_PROMPTS_' + args.family.upper()])
        check('monitor prompts', monitor.is_file(), monitor)
        if monitor.is_file():
            with monitor.open() as handle:
                count = sum(bool(line.strip()) for line in handle)
            check('monitor count', count == 128, f'{count} (default expected: 128)')
    if args.gpu:
        try:
            import torch
            check('CUDA available', torch.cuda.is_available(), f'torch CUDA={torch.version.cuda}')
            count = torch.cuda.device_count()
            check('allocated GPU count', count >= args.world_size, f'visible={count}, requested={args.world_size}')
            if torch.cuda.is_available() and count >= args.world_size:
                for index in range(args.world_size):
                    with torch.cuda.device(index):
                        check(f'GPU {index} BF16', torch.cuda.is_bf16_supported(), torch.cuda.get_device_name(index))
                        layer = torch.nn.Conv2d(4, 4, 3, padding=1).to(index)
                        x = torch.ones((1, 4, 16, 16), device=f'cuda:{index}', requires_grad=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            loss = layer(x).float().square().mean()
                        loss.backward()
                        torch.cuda.synchronize(index)
                        check(f'GPU {index} CUDA forward/backward', bool(torch.isfinite(loss)), 'small BF16 convolution')
        except (ImportError, RuntimeError) as exc:
            check('CUDA execution', False, str(exc))
    print('File checks use metadata. Next: train.py --check-only, then the two-step training smoke.')
    if errors:
        print('Fix the items marked MISSING before continuing.')
        return 1
    print('Deployment checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
