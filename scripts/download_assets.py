"""Explicit SLAI downloads. list is offline; plan/download query upstream metadata.

No model imports, resource submissions, automatic extraction, or training.
Defaults use the SLAI workspace below; --out overrides any destination.
HF_ENDPOINT can override the default https://hf-mirror.com mirror.
"""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.common import output_lock

# Edit these paths if the workspace moves. No paths.sh is required.
WORKSPACE = Path('/mnt/afs/L202500030/xietianyu/workspace')
MODELS = WORKSPACE / 'models'
DATASETS = WORKSPACE / 'datasets'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

DEFAULT_PATHS = {
    'SD15_MODEL': MODELS / 'stable_diffusion/sd15',
    'SDXL_MODEL': MODELS / 'stable_diffusion/sdxl',
    'SDXL_VAE': MODELS / 'stable_diffusion/sdxl-vae',
    'SD35_MODEL': MODELS / 'stable_diffusion/sd35',
    'PICKAPIC_RAW': DATASETS / 'pickapic-v2',
    'HPDV2_RAW': DATASETS / 'hpdv2',
    'HPDV3_RAW': DATASETS / 'hpdv3',
    'PICKSCORE_MODEL': MODELS / 'pickscore',
    'PICKSCORE_PROCESSOR': MODELS / 'clip-h14',
    'AESTHETIC_CLIP_MODEL': MODELS / 'clip-l14',
    'HPSV2_MODEL_DIR': MODELS / 'hpsv2',
    'HPSV3_MODEL_DIR': MODELS / 'hpsv3',
    'HPSV3_BASE_MODEL': MODELS / 'Qwen2-VL-7B-Instruct',
    'IMAGE_REWARD_MODEL_DIR': MODELS / 'image-reward',
    'IMAGE_REWARD_TOKENIZER': MODELS / 'bert-base-uncased',
}

# repo, repo type, destination variable, revision variable, file patterns
ASSETS = {
    'pickapic-v2': ('yuvalkirstain/pickapic_v2', 'dataset', 'PICKAPIC_RAW', 'PICKAPIC_REVISION', ['*.parquet', 'README.md']),
    'hpdv2': ('ymhao/HPDv2', 'dataset', 'HPDV2_RAW', 'HPDV2_REVISION', ['test.json', 'README.md']),
    'hpdv2-images': ('ymhao/HPDv2', 'dataset', 'HPDV2_RAW', 'HPDV2_REVISION', ['train.json', 'test.json', 'train.tar.gz', 'test.tar.gz', 'README.md']),
    'hpdv3': ('MizzenAI/HPDv3', 'dataset', 'HPDV3_RAW', 'HPDV3_REVISION', ['train.json', 'test.json', 'images.tar.gz.*', 'README.md']),
    'sd15': ('stable-diffusion-v1-5/stable-diffusion-v1-5', 'model', 'SD15_MODEL', 'SD15_REVISION', None),
    'sdxl': ('stabilityai/stable-diffusion-xl-base-1.0', 'model', 'SDXL_MODEL', 'SDXL_REVISION', None),
    'sdxl-vae': ('madebyollin/sdxl-vae-fp16-fix', 'model', 'SDXL_VAE', None, ['config.json', 'diffusion_pytorch_model.safetensors']),
    'sd35': ('stabilityai/stable-diffusion-3.5-medium', 'model', 'SD35_MODEL', 'SD35_REVISION', None),
    'pickscore': ('yuvalkirstain/PickScore_v1', 'model', 'PICKSCORE_MODEL', None, ['config.json', 'model.safetensors', 'pytorch_model.bin']),
    'clip-h14': ('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 'model', 'PICKSCORE_PROCESSOR', None,
                 ['*.json', 'vocab.json', 'merges.txt', 'tokenizer.model', 'open_clip_pytorch_model.bin']),
    'clip-l14': ('openai/clip-vit-large-patch14', 'model', 'AESTHETIC_CLIP_MODEL', None,
                 ['*.json', 'vocab.json', 'merges.txt', 'model.safetensors', 'pytorch_model.bin']),
    'hpsv2': ('xswu/HPSv2', 'model', 'HPSV2_MODEL_DIR', None, ['HPS_v2_compressed.pt']),
    'hpsv3': ('MizzenAI/HPSv3', 'model', 'HPSV3_MODEL_DIR', None, ['HPSv3.safetensors','config.json']),
    'hpsv3-base': ('Qwen/Qwen2-VL-7B-Instruct', 'model', 'HPSV3_BASE_MODEL', None, ['*.json','*.safetensors','*.txt','*.model']),
    'image-reward': ('zai-org/ImageReward', 'model', 'IMAGE_REWARD_MODEL_DIR', None, ['ImageReward.pt','med_config.json']),
    'image-reward-tokenizer': ('google-bert/bert-base-uncased', 'model', 'IMAGE_REWARD_TOKENIZER', None, ['*.json','vocab.txt']),
}


def env_path(name):
    value = str(os.environ.get(name) or DEFAULT_PATHS.get(name, ''))
    if not value or 'CHANGE_ME' in value or '$' in value:
        raise ValueError(f'Pass an explicit path or set {name}')
    return Path(value).expanduser().resolve()


def destination(asset):
    return env_path(ASSETS[asset][2])


def select_files(asset, siblings, train_shards=None):
    patterns = ASSETS[asset][4]
    files = {}
    for entry in siblings:
        name = entry.rfilename
        if patterns is None:
            # One full-precision Diffusers representation, no root ckpt/ONNX/FP16 duplicates.
            component = name.split('/')[0]
            keep = name == 'model_index.json' or (component in {
                'unet', 'transformer', 'vae', 'scheduler', 'text_encoder', 'text_encoder_2',
                'text_encoder_3', 'tokenizer', 'tokenizer_2', 'tokenizer_3', 'feature_extractor', 'safety_checker'
            } and (name.endswith(('.json', '.txt', '.model', '.safetensors', '.bin')))
                    and '.fp16.' not in name and '.non_ema.' not in name)
        else:
            keep = any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
        if keep:
            if entry.size is None:
                raise ValueError(f'Upstream did not report a file size: {name}')
            files[name] = entry.size
    # Prefer safetensors when both transformer serialization formats are present.
    if patterns is None:
        safe_components={name.split('/')[0] for name in files if name.endswith('.safetensors')}
        files={name:size for name,size in files.items() if not (name.endswith('.bin') and name.split('/')[0] in safe_components)}
    if asset in ('pickscore', 'clip-l14') and 'model.safetensors' in files:
        files.pop('pytorch_model.bin', None)
    if asset == 'pickapic-v2':
        training = sorted(n for n in files if parquet_split(n) == 'train')
        heldout = [n for n in files if parquet_split(n) == 'test_unique']
        if not training or not heldout:
            raise ValueError('Expected train and test_unique Parquet shards; inspect upstream layout before downloading')
        if train_shards is not None:
            chosen = set(training[:train_shards])
            files = {n: size for n, size in files.items() if parquet_split(n) != 'train' or n in chosen}
    required={
        'hpdv3':['train.json','test.json'], 'hpdv2':['test.json'],
        'hpdv2-images':['train.json','test.json','train.tar.gz','test.tar.gz'],
        'hpsv2':['HPS_v2_compressed.pt'], 'hpsv3':['HPSv3.safetensors','config.json'],
        'image-reward':['ImageReward.pt','med_config.json'],
        'clip-h14':['open_clip_pytorch_model.bin','preprocessor_config.json'],
    }.get(asset,[])
    if any(name not in files for name in required):
        raise ValueError(f'Missing expected upstream files for {asset}: {required}')
    if asset=='hpdv3':
        parts=sorted(name for name in files if name.startswith('images.tar.gz.'))
        if not parts or [int(name.rsplit('.',1)[1]) for name in parts]!=list(range(len(parts))):
            raise ValueError('Expected contiguous HPDv3 image volume numbers beginning with 00')
    if not files:
        raise ValueError(f'No matching files for {asset}')
    return dict(sorted(files.items()))


def parquet_split(name):
    p = Path(name)
    if p.suffix != '.parquet':
        return None
    for split in ('validation_unique', 'test_unique', 'validation', 'test', 'train'):
        if p.name.startswith(split + '-') or split in p.parts[:-1]:
            return split
    return None


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def plan_asset(asset, *, revision=None, train_shards=None, api=None, out=None):
    from huggingface_hub import HfApi
    repo, kind, _, revvar, _ = ASSETS[asset]
    target = Path(out).expanduser().resolve() if out is not None else destination(asset)
    marker = target / '.scd-download.json'
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous and (previous['repo_id'], previous['repo_type']) != (repo, kind):
        raise ValueError(f'Destination belongs to a different repository: {target}')
    requested = revision or (os.environ.get(revvar, 'main') if revvar else 'main')
    # A completed/partial directory remains pinned on rerun even if main moves.
    pinned = previous['revision'] if previous and revision is None else requested
    info = (api or HfApi(endpoint=os.environ['HF_ENDPOINT'])).repo_info(repo, repo_type=kind, revision=pinned, files_metadata=True)
    if previous and previous['revision'] != info.sha:
        raise ValueError('Revision changed; choose a new destination with --out')
    files = select_files(asset, info.siblings, train_shards)
    if target.exists() and not previous and any(target.iterdir()):
        # huggingface-cli --local-dir uses the same cache as snapshot_download.
        # Let the Hub validate/reuse those files and fetch anything still missing.
        if not (target / '.cache/huggingface/download').is_dir():
            raise ValueError(f'Nonempty destination has no Hugging Face download metadata: {target}; use a new --out directory')
    missing = sum(size for name, size in files.items()
                  if not (target / name).is_file() or (target / name).stat().st_size != size)
    ancestor = target
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    return dict(asset=asset, repo_id=repo, repo_type=kind, revision=info.sha,
                destination=str(target), files=files, selected_bytes=sum(files.values()),
                remaining_bytes=missing, free_bytes=free, train_shards_limit=train_shards,
                extraction_bytes_included=False)


def download(plan):
    from huggingface_hub import snapshot_download
    target = Path(plan['destination'])
    if plan['free_bytes'] < plan['remaining_bytes'] + 2 * 1024**3:
        raise RuntimeError('Insufficient download space plus 2 GiB margin; extraction/image export needs additional space')
    with output_lock(target):
        target.mkdir(parents=True, exist_ok=True)
        marker = target / '.scd-download.json'
        previous = json.loads(marker.read_text()) if marker.exists() else {}
        if previous and any(previous[key]!=plan[key] for key in ('repo_id','repo_type','revision')):
            raise ValueError('Download destination identity changed after planning')
        all_files = {**previous.get('files', {}), **plan['files']}
        record = {**plan, 'files': all_files, 'complete': False}
        atomic_json(marker, record)
        snapshot_download(plan['repo_id'], repo_type=plan['repo_type'], revision=plan['revision'],
                          allow_patterns=list(plan['files']), local_dir=str(target), max_workers=4,
                          endpoint=os.environ['HF_ENDPOINT'])
        for name, size in all_files.items():
            if not (target / name).is_file() or (target / name).stat().st_size != size:
                raise ValueError(f'Incomplete downloaded file: {name}')
        atomic_json(marker, {**record, 'complete': True})


def checked_members(tar, prefix):
    members = tar.getmembers()
    for member in members:
        parts = Path(member.name).parts
        if Path(member.name).is_absolute() or '..' in parts or not (member.isfile() or member.isdir()):
            raise ValueError(f'Unsupported archive member: {member.name}')
        if not parts or parts[0] != prefix:
            raise ValueError(f'Expected archive paths under {prefix}/: {member.name}')
    return members


def extract_hpdv2(root=None):
    root = Path(root).expanduser().resolve() if root is not None else env_path('HPDV2_RAW')
    # Never silently replace an earlier extraction or merge into existing images.
    for split in ('train', 'test'):
        archive = root / (split + '.tar.gz')
        if not archive.exists():
            continue
        target = root / split
        if target.exists():
            raise FileExistsError(f'{target}; existing extraction retained')
        with tarfile.open(archive) as tar:
            members = checked_members(tar, split)
            required = sum(m.size for m in members)
            if shutil.disk_usage(root).free < required + 2 * 1024**3:
                raise RuntimeError('Insufficient space for HPDv2 extraction plus 2 GiB margin')
            # Members are checked above, including rejecting links and traversal.
            tar.extractall(root, members=members)
        atomic_json(root / (split + '-extracted.json'), {'archive':archive.name, 'uncompressed_bytes':required})


def extract_hpdv3(root=None):
    """Stream concatenated volumes through gzip; do not create a second 140GB archive."""
    import gzip
    import io
    root = Path(root).expanduser().resolve() if root is not None else env_path('HPDV3_RAW')
    receipt = json.loads((root/'.scd-download.json').read_text())
    if not receipt.get('complete'):
        raise ValueError('Download is incomplete')
    volumes = sorted(root/name for name in receipt['files'] if name.startswith('images.tar.gz.'))
    if not volumes:
        raise ValueError('No image volumes in the download receipt')
    for volume in volumes:
        if not volume.is_file() or volume.stat().st_size != receipt['files'][volume.name]:
            raise ValueError(f'Missing/incomplete volume: {volume.name}')
    if (root/'images').exists():
        raise FileExistsError(root/'images')
    class Volumes(io.RawIOBase):
        def __init__(self):
            self.remaining=iter(volumes); self.current=None
        def readable(self):return True
        def readinto(self, buffer):
            while True:
                if self.current is None:
                    path=next(self.remaining,None)
                    if path is None:return 0
                    self.current=path.open('rb')
                count=self.current.readinto(buffer)
                if count:return count
                self.current.close();self.current=None
        def close(self):
            if self.current:self.current.close()
            super().close()
    written=0
    budget=0
    with Volumes() as stream, io.BufferedReader(stream) as buffered, gzip.GzipFile(fileobj=buffered) as gzip_stream:
        with tarfile.open(fileobj=gzip_stream,mode='r|') as tar:
            for member in tar:
                # Streaming extraction: validate each entry before creating it.
                parts=Path(member.name).parts
                if (Path(member.name).is_absolute() or '..' in parts or not parts or parts[0]!='images'
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unsupported HPDv3 archive member: {member.name}')
                if member.isdir():
                    (root/member.name).mkdir(parents=True,exist_ok=True)
                    continue
                if budget<member.size:
                    free=shutil.disk_usage(root).free
                    if free < member.size + 2*1024**3:
                        raise RuntimeError('Insufficient extraction space plus 2 GiB margin')
                    budget=min(free-2*1024**3,128*1024**2)
                target=root/member.name
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source,dest)
                written+=member.size
                budget-=member.size
    atomic_json(root/'images-extracted.json',{'volumes':[v.name for v in volumes], 'uncompressed_bytes':written})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list', 'plan', 'download', 'extract-hpdv2', 'extract-hpdv3'])
    p.add_argument('assets', nargs='*', help='Asset names from list')
    p.add_argument('--out', help='Download destination for one asset, or the raw directory to extract')
    p.add_argument('--revision', help='Override a single asset revision on first download')
    p.add_argument('--train-shards', type=int, help='Pick-a-Pic train shard subset; recorded as a new dataset version')
    a = p.parse_args()
    if any(name not in ASSETS for name in a.assets):p.error('Unknown asset; use list')
    if a.out and len(a.assets)>1:p.error('--out accepts one asset at a time')
    if a.action == 'list':
        for name, spec in ASSETS.items():
            print(name, spec[0], destination(name))
        return
    if a.action == 'extract-hpdv2':
        extract_hpdv2(a.out)
        return
    if a.action == 'extract-hpdv3':
        extract_hpdv3(a.out)
        return
    if not a.assets or (a.revision and len(a.assets) != 1):
        p.error('Select assets; --revision accepts exactly one asset')
    if a.train_shards is not None and (a.train_shards < 1 or 'pickapic-v2' not in a.assets):
        p.error('--train-shards must be positive and used with pickapic-v2')
    for asset in a.assets:
        plan = plan_asset(asset, revision=a.revision, train_shards=a.train_shards if asset == 'pickapic-v2' else None,out=a.out)
        print(json.dumps({k:v for k,v in plan.items() if k != 'files'}, indent=2), flush=True)
        if a.action == 'download':
            download(plan)


if __name__ == '__main__':
    main()
