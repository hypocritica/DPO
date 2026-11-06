"""Validate a local flow model once and write a reusable JSON receipt.

Checks config provenance, shard indexes, safetensors headers and file sizes.
It does not hash tensor payloads or verify GPU compatibility/numerical quality.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from self_correct_dpo.common import artifact_identity, digest, read_json
from self_correct_dpo.flow_receipts import file_binding
from self_correct_dpo.runtime.checkpoints import atomic_json


def tensor_header(path):
    size = path.stat().st_size
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f'Truncated safetensors header: {path}')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(100_000_000, size-8):
            raise ValueError(f'Invalid safetensors header length: {path}')
        header = json.loads(handle.read(length))
    intervals = []
    widths = {'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'I16':2,'U16':2,
              'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        start, end = tensor['data_offsets']
        shape = tensor['shape']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= size-8-length:
            raise ValueError(f'Invalid tensor offsets: {path}:{name}')
        if tensor['dtype'] not in widths or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f'Invalid tensor shape/dtype: {path}:{name}')
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end-start != elements*widths[tensor['dtype']]:
            raise ValueError(f'Tensor size differs from shape/dtype: {path}:{name}')
        if end > start:
            intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f'Non-contiguous or overlapping tensor payload: {path}')
        cursor = end
    if cursor != size-8-length:
        raise ValueError(f'Tensor payload size mismatch: {path}')
    return set(header)-{'__metadata__'}


def validate_model(model_path, family, revision):
    root = Path(model_path).expanduser().resolve(strict=True)
    if revision and any(word in revision.upper() for word in ('FILL_', 'TODO', 'UNVERIFIED')):
        raise ValueError('Provide a traceable model revision or omit it for a local inventory identity')
    expected = 'ZImagePipeline' if family == 'zimage' else 'StableDiffusion3Pipeline'
    if read_json(root/'model_index.json').get('_class_name') != expected:
        raise ValueError(f'Expected {expected}')
    config = read_json(root/'transformer/config.json')
    if family == 'sd35_medium':
        # CPU-only import; no model construction or tensor loads.
        from self_correct_dpo.evaluation.render import validate_sd35_medium_config
        validate_sd35_medium_config(config)
    elif config.get('_class_name') != 'ZImageTransformer2DModel':
        raise ValueError('Expected ZImageTransformer2DModel')
    component_keys = []
    text_components = ['text_encoder'] + (['text_encoder_2','text_encoder_3'] if family == 'sd35_medium' else [])
    for component in text_components:
        relative = component+'/config.json'
        encoder = read_json(root/relative)
        expected_type = ('qwen3' if family == 'zimage' else 't5' if component == 'text_encoder_3' else 'clip_text_model')
        if encoder.get('model_type') != expected_type:
            raise ValueError(f'Unexpected {component} model_type: {encoder.get("model_type")}')
        component_keys.append(relative)
    for tokenizer in (['tokenizer','tokenizer_2','tokenizer_3'] if family == 'sd35_medium' else ['tokenizer']):
        relative = tokenizer+'/tokenizer_config.json'
        tokenizer_config = read_json(root/relative)
        if not tokenizer_config.get('tokenizer_class'):
            raise ValueError(f'Missing tokenizer class: {relative}')
        component_keys.append(relative)
        required_files = ['spiece.model'] if tokenizer == 'tokenizer_3' else ['vocab.json','merges.txt']
        for name in required_files:
            asset = root/tokenizer/name
            if not asset.is_file() or asset.stat().st_size == 0:
                raise ValueError(f'Missing tokenizer asset: {asset}')
        component_keys += [tokenizer+'/'+name for name in required_files]
    vae = read_json(root/'vae/config.json')
    if vae.get('_class_name') != 'AutoencoderKL':
        raise ValueError('Expected AutoencoderKL config')
    scheduler = read_json(root/'scheduler/scheduler_config.json')
    if scheduler.get('use_dynamic_shifting', False) or scheduler.get('_class_name') != 'FlowMatchEulerDiscreteScheduler':
        raise ValueError('Expected static FlowMatchEulerDiscreteScheduler')
    receipt = root/'.scd-download.json'
    provenance = None
    if receipt.is_file():
        provenance = read_json(receipt)
        if provenance.get('complete') is not True or provenance.get('revision') != revision:
            raise ValueError('Download receipt incomplete or revision mismatch')
    elif (root/'.download-manifest.json').is_file():
        receipt = root/'.download-manifest.json'
        provenance = read_json(receipt)
        if provenance.get('status') != 'complete' or provenance.get('revision') != revision:
            raise ValueError('Download manifest incomplete or revision mismatch')
    inventory = artifact_identity(root)
    # Validation artifacts do not belong to the immutable upstream model.
    inventory['files'] = [row for row in inventory['files']
                          if row['path'] != 'flow-validation-report.json'
                          and not row['path'].endswith('/flow-validation-report.json')]
    inventory['inventory_sha256'] = digest(inventory['files'])
    upstream_verified = provenance is not None
    if revision is None:
        revision = 'local-inventory:' + inventory['inventory_sha256']
    headers = {}
    for relative in inventory['files']:
        path = root/relative['path']
        if path.suffix == '.safetensors':
            headers[relative['path']] = tensor_header(path)
    if not headers:
        raise ValueError('No safetensors weights found')
    required = ['transformer', 'vae', 'text_encoder']
    if family == 'sd35_medium':
        required += ['text_encoder_2', 'text_encoder_3']
    for component in required:
        if not any(name.startswith(component+'/') for name in headers):
            raise ValueError(f'Missing safetensors component: {component}')
    index_count = 0
    for item in inventory['files']:
        if not item['path'].endswith('.safetensors.index.json'):
            continue
        index_path = root/item['path']
        weight_map = read_json(index_path).get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f'Invalid weight index: {index_path}')
        for name, shard in weight_map.items():
            candidate = (index_path.parent/shard).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError('Weight shard escaped model root')
            relative = str(candidate.relative_to(root))
            if name not in headers.get(relative, set()):
                raise ValueError(f'Indexed tensor missing: {relative}:{name}')
        index_count += 1
    # Only a few critical configuration bindings are read on later startups.
    keys = ['model_index.json', 'transformer/config.json', 'vae/config.json',
            'scheduler/scheduler_config.json'] + component_keys
    if receipt.is_file():
        keys.append(receipt.name)
    return {'schema_version':1, 'kind':'flow_model_validation', 'status':'passed', 'passed':True,
            'created_at':datetime.now(timezone.utc).isoformat(), 'family':family,
            'revision':revision, 'upstream_revision_verified':upstream_verified, 'model_path':str(root), 'key_files':[file_binding(root/k) for k in keys],
            'artifact_identity':inventory,
            'checks':{'safetensors_files':len(headers), 'shard_indexes':index_count,
                      'tensor_headers_and_sizes':True, 'component_configs':True, 'tokenizer_assets':True},
            'scope':{'weight_payload_hashed':False, 'weight_values_validated':False,
                     'gpu_compatibility_validated':False,
                     'upstream_revision_verified':upstream_verified,
                     'inventory_metadata_rechecked_at_startup':True,
                     'startup_checks':'key configuration hashes and receipt bindings; existing inventory size/mtime checks without payload reads'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--family', choices=['sd35_medium','zimage'], required=True)
    parser.add_argument('--revision')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = args.out or args.model/'flow-validation-report.json'
    revision = args.revision
    try:
        receipt = args.model/'.scd-download.json'
        if revision is None and receipt.is_file():
            revision = read_json(receipt).get('revision')
        if revision is None and (args.model/'.download-manifest.json').is_file():
            revision = read_json(args.model/'.download-manifest.json').get('revision')
        result = validate_model(args.model,args.family,revision)
    except Exception as error:
        # An unsuccessful revalidation must revoke any previous passed receipt.
        result = {'schema_version':1, 'kind':'flow_model_validation',
                  'status':'failed', 'passed':False,
                  'created_at':datetime.now(timezone.utc).isoformat(),
                  'family':args.family, 'revision':revision,
                  'model_path':str(args.model.expanduser().resolve()),
                  'error':str(error),
                  'scope':{'gpu_compatibility_validated':False}}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output,result)
    print(json.dumps({'status':result['status'], 'report':str(output),
                      'checks':result.get('checks'), 'error':result.get('error')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
