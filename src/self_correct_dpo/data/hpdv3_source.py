"""Offline source identity for native downloads and plain Hugging Face folders.

A local inventory identifies the available files. It never certifies an upstream
revision, a completed download, or archive payload checksums.
"""
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
import time

from self_correct_dpo.common import digest, output_lock
from self_correct_dpo.data.preparation import atomic_json

INVENTORY = '.flow-source-inventory.json'


def file_stat(path):
    s = Path(path).stat()
    return {'size_bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def annotation_hash(path):
    print(f'[source] Hashing {path.name} ({path.stat().st_size} bytes)', file=sys.stderr, flush=True)
    before = file_stat(path)
    h = hashlib.sha256(); total = 0; last = time.monotonic()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(chunk); total += len(chunk)
            if time.monotonic() - last >= 15:
                print(f'[source] {path.name}: {total}/{before["size_bytes"]} bytes', file=sys.stderr, flush=True)
                last = time.monotonic()
    if file_stat(path) != before:
        raise ValueError(f'Annotation changed while hashing: {path}')
    return {**before, 'sha256': h.hexdigest()}


def source_context(raw):
    """Identify the exact runtime/mount view without reading archive payloads."""
    raw = Path(raw).resolve()
    return {'hostname': socket.gethostname(), 'python': sys.executable,
            'python_version': sys.version.split()[0], 'cwd': str(Path.cwd()),
            'raw': str(raw), 'module': str(Path(__file__).resolve()),
            'inventory_path': str(raw / INVENTORY)}


class SourceStatMismatch(ValueError):
    def __init__(self, raw, name, expected, observed):
        self.diagnostic = {**source_context(raw), 'kind': 'hpdv3_source_stat_mismatch',
                           'file': name, 'expected': expected, 'observed': observed,
                           'changed_fields': [key for key in expected if expected[key] != observed[key]]}
        super().__init__(f'Source file changed since inventory: {Path(raw) / name}. '
                         f'Expected {expected}; observed {observed}. '
                         'Do not reset the inventory before checking this task container view.')


def validate_inventory(raw, record):
    if record.get('kind') != 'hpdv3_local_source_inventory' or record.get('status') != 'inventoried':
        raise ValueError('Invalid HPDv3 source inventory')
    if record.get('revision') != 'local-inventory:' + digest(record['identity']):
        raise ValueError('HPDv3 inventory identity checksum differs')
    expected_names = set(record['annotations']) | set(record['archives'])
    if not {'train.json', 'test.json'} <= expected_names or not record['archives']:
        raise ValueError('Source inventory lacks annotations/archives')
    for name in sorted(expected_names):
        if name not in ('train.json', 'test.json') and not re.fullmatch(r'images\.tar\.gz\.\d{2}', name):
            raise ValueError(f'Unsafe source inventory path: {name}')
        entry = record['annotations'].get(name, record['archives'].get(name))
        expected = {k: entry[k] for k in ('size_bytes', 'mtime_ns')}
        observed = file_stat(raw / name)
        if observed != expected:
            raise SourceStatMismatch(raw, name, expected, observed)
    if {p.name for p in raw.glob('images.tar.gz.[0-9][0-9]')} != set(record['archives']):
        raise ValueError('Archive set changed since inventory; finish downloading before preparing data')
    expected_identity = {
        'annotations': {name: row['sha256'] for name, row in record['annotations'].items()},
        'archives': {name: row['size_bytes'] for name, row in record['archives'].items()}}
    if record['identity'] != expected_identity or record.get('repo_id') != 'local/HPDv3':
        raise ValueError('Source inventory contents disagree with identity')


def source_record(raw, *, create_inventory=False):
    raw = Path(raw).resolve()
    for name in ('.download-manifest.json', '.scd-download.json'):
        path = raw / name
        if path.is_file():
            record = json.loads(path.read_text())
            if not (record.get('status') == 'complete' or record.get('complete') is True):
                raise ValueError(f'Download receipt reports incomplete data: {path}')
            if not all(isinstance(record.get(k), str) and record[k] for k in ('repo_id', 'revision')):
                raise ValueError(f'Download receipt lacks repo_id/revision: {path}')
            # The project HF downloader records expected sizes; check those cheaply.
            for filename, size in record.get('files', {}).items():
                if Path(filename).is_absolute() or '..' in Path(filename).parts:
                    raise ValueError('Unsafe path in download receipt')
                if not (raw / filename).is_file() or (raw / filename).stat().st_size != size:
                    raise ValueError(f'Missing/incomplete source file: {raw / filename}')
            return path, record
    path = raw / INVENTORY
    if path.is_file():
        record = json.loads(path.read_text()); validate_inventory(raw, record)
        return path, record
    if not create_inventory:
        raise ValueError('No source record. Run scripts/inspect_hpdv3_source.py --raw PATH once.')
    with output_lock(raw / INVENTORY):
        if path.is_file():
            record = json.loads(path.read_text()); validate_inventory(raw, record)
            return path, record
        archives = sorted(raw.glob('images.tar.gz.[0-9][0-9]'))
        if not archives:
            raise FileNotFoundError(f'No final images.tar.gz.00-style volumes under {raw}')
        if [int(p.name[-2:]) for p in archives] != list(range(len(archives))):
            raise ValueError('Archive volumes must be contiguous from .00; finish missing volumes first')
        archive_stats = {p.name: file_stat(p) for p in archives}
        if any(not s['size_bytes'] for s in archive_stats.values()):
            raise ValueError('An archive volume is empty')
        leftovers = sorted(p.name for p in raw.glob('images.tar.gz.*.incomplete'))
        if leftovers:
            print(f'[source] Ignoring temporary files: {leftovers}; using final numbered volumes only', file=sys.stderr, flush=True)
        annotations = {name: annotation_hash(raw / name) for name in ('train.json', 'test.json')}
        identity = {'annotations': {n: s['sha256'] for n, s in annotations.items()},
                    'archives': {n: s['size_bytes'] for n, s in archive_stats.items()}}
        # HF metadata is an observation, not proof of complete archive payloads.
        commits = set()
        for name in [*annotations, *archive_stats]:
            metadata = raw / '.cache/huggingface/download' / (name + '.metadata')
            if metadata.is_file():
                first = metadata.read_text().splitlines()[:1]
                if first and re.fullmatch('[0-9a-f]{40}', first[0]): commits.add(first[0])
        record = {'schema_version': 1, 'kind': 'hpdv3_local_source_inventory',
                  'status': 'inventoried', 'repo_id': 'local/HPDv3',
                  'revision': 'local-inventory:' + digest(identity), 'identity': identity,
                  'annotations': annotations, 'archives': archive_stats,
                  'upstream_revision_verified': False, 'archive_payload_checksums_verified': False,
                  'download_complete_verified': False, 'hf_cache_commits_observed': sorted(commits)}
        validate_inventory(raw, record)
        atomic_json(path, record)
        print(f'[source] Local inventory written: {path}; download completion/upstream revision NOT certified', file=sys.stderr, flush=True)
        return path, record
