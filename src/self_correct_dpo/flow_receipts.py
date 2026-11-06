"""Reusable validation receipts; startup checks small bindings, never model payloads.

Receipts are local trusted evidence, not cryptographic protection against changes
that deliberately preserve file metadata. Images retain consume-time SHA checks.
"""
from pathlib import Path
import json
import os

from self_correct_dpo.common import read_json, sha256


def file_binding(path, *, checksum=True):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    result = {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if checksum:
        result['sha256'] = sha256(path)
    return result


def check_binding(binding, *, path=None, checksum=False):
    if not isinstance(binding, dict):
        raise ValueError('Missing validation receipt file binding')
    bound = Path(binding['path']).expanduser().resolve(strict=True)
    if path is not None and bound != Path(path).expanduser().resolve(strict=True):
        raise ValueError('Validation receipt is bound to another path')
    stat = bound.stat()
    if (stat.st_size, stat.st_mtime_ns) != (binding['size_bytes'], binding['mtime_ns']):
        raise ValueError(f'Validated file changed; regenerate its report: {bound}')
    if checksum and sha256(bound) != binding.get('sha256'):
        raise ValueError(f'Validated configuration changed: {bound}')
    return bound


def _passed(path, kind):
    hint = ('After preparation completes, run the one-time audit; see scripts/audit_flow_data.py --help.'
            if kind == 'flow_data_validation' else
            'Run the one-time model validation; see scripts/validate_flow_model.py --help.')
    try:
        report = read_json(path)
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read required {kind} report: {path}. {hint}') from error
    if (not isinstance(report, dict) or report.get('schema_version') != 1 or report.get('kind') != kind
            or report.get('status') != 'passed' or report.get('passed') is not True):
        raise ValueError(f'A passed {kind} report is required: {path}. {hint}')
    return report


def data_receipt(path, pairs):
    report = _passed(path, 'flow_data_validation')
    scope = report.get('scope', {})
    if scope.get('manifest_checks') is not True or scope.get('image_integrity') != 'verified_during_preparation':
        raise ValueError('Dataset report does not certify preparation and manifest checks')
    identity = report['dataset_identity']
    check_binding(identity['pairs'], path=pairs)
    ready_path = check_binding(identity['ready'], path=Path(pairs).parent/'ready.json', checksum=True)
    check_binding(identity['audit'], path=Path(pairs).parent/'audit.json')
    ready = read_json(ready_path)
    if (identity['pairs'].get('sha256') != ready.get('pairs_sha256')
            or report.get('source_revision') != ready.get('source_revision')):
        raise ValueError('Dataset report and preparation identity disagree')
    return report


def model_receipt(path, model):
    report = _passed(path, 'flow_model_validation')
    if report.get('family') != model['family'] or report.get('revision') != model['revision']:
        raise ValueError('Model report family/revision differs from configuration')
    if Path(report['model_path']).resolve() != Path(model['path']).expanduser().resolve():
        raise ValueError('Model report belongs to another local model')
    if not report.get('key_files') or not report.get('artifact_identity'):
        raise ValueError('Model report lacks configuration/inventory evidence')
    root = Path(model['path']).expanduser().resolve()
    for item in report['key_files']:
        candidate = Path(item['path']).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError('Model report key-file escaped the model root')
        check_binding(item, checksum=True)
    # Reuse the certified inventory rather than walking the model directory or
    # reading tensor payloads. Detect ordinary replacement/truncation since the
    # one-time validation, including non-key tokenizer and shard-index files.
    for item in report['artifact_identity']['files']:
        relative = Path(item['path'])
        candidate = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not candidate.is_relative_to(root):
            raise ValueError('Model report inventory escaped the model root')
        stat = candidate.stat()
        if (not candidate.is_file() or
                (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns'])):
            raise ValueError(f'Validated model artifact changed; regenerate its report: {candidate}')
    return report


def load_certified_pairs(path, report_path):
    """Parse required rows after a receipt check; skip repeated global audits."""
    path = Path(path).expanduser().resolve()
    report = data_receipt(report_path, path)
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    check_binding(report['dataset_identity']['pairs'], path=path)
    if not rows:
        raise ValueError('Certified manifest is empty')
    # Preparation owns schema/split/leakage checks. Resolve relative image paths
    # without touching every image on NFS; image_source validates consumed bytes.
    for row in rows:
        for side in ('winner', 'loser'):
            value = Path(row[side]).expanduser()
            row[side] = os.path.abspath(value if value.is_absolute() else path.parent/value)
    print(json.dumps({'phase': 'data_ready', 'mode': 'receipt', 'rows': len(rows),
                      'startup_image_reads': 0, 'report': str(report_path)}), flush=True)
    return rows
