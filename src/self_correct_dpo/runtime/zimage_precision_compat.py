"""User-authorized FP32-to-BF16 continuation of the 3148b5b Z-Image curves.

This is a precision change, not a storage-only compatibility exception. Keep
legacy identities and PNGs intact; new image records carry their own precision.
"""
from pathlib import Path

from self_correct_dpo.common import digest, read_json, sha256, source_file
from self_correct_dpo.runtime.checkpoints import atomic_json

LEGACY_RENDER_SHA256 = '2aff52c1c87568da75601fec348a20267577cbae6e358778118a0c6ba8e8da4a'
BF16_RENDER_SHA256 = '29b67d645cbbf44e55add800b1d7b44f12fe7e24c9d79c3bb00b1355058b3122'


def compatible_zimage_bf16_resume(run):
    return (run.get('config', {}).get('model', {}).get('family') == 'zimage'
            and run.get('render_sha256') == LEGACY_RENDER_SHA256
            and sha256(source_file('render.py')) == BF16_RENDER_SHA256
            and 'transformer_autocast_dtype' not in run['config']['sampling']
            and 'transformer_autocast_dtype' not in run['sampling'])


def record_zimage_bf16_resume(root, run):
    if not compatible_zimage_bf16_resume(run):
        raise ValueError('Not the approved legacy Z-Image BF16 continuation')
    report = {'change': 'zimage-transformer-bf16-autocast-v1', 'run_id': run['run_id'],
              'original_render_sha256': LEGACY_RENDER_SHA256,
              'current_render_sha256': BF16_RENDER_SHA256,
              'original_transformer_autocast_dtype': 'none',
              'new_transformer_autocast_dtype': 'bfloat16',
              'policy_sha256': sha256(__file__),
              'existing_images': 'Kept byte-for-byte, including recovered pending transactions',
              'new_images': 'Precision and renderer recorded per image; this run may contain mixed precision'}
    path = Path(root) / ('zimage-precision-transition-' + digest(report)[:16] + '.json')
    if path.exists():
        if read_json(path) != report:
            raise ValueError('Z-Image precision transition record changed')
    else:
        atomic_json(path, report)
    print(f'Keeping existing images; missing Z-Image images use BF16 autocast; audit: {path}', flush=True)
