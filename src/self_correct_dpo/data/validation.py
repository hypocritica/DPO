"""Fast prepared-data preflight, bounded parallel audits and reusable receipts.

Quick startup checks every manifest row but samples image bytes. Training must
verify the bytes it actually decodes; quick mode is not a full corruption scan.
"""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

from self_correct_dpo.common import digest, sha256
from self_correct_dpo.data.pairs import validate_pair_rows, validate_split_disjointness


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _parallel_check(images, workers, progress):
    """Keep at most 2*workers futures, even for a million unique files."""
    def check(item):
        path, expected = item
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Image checksum mismatch: {path}')
        return path, actual
    iterator = iter(images.items())
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(workers * 2):
            item = next(iterator, None)
            if item is not None:
                pending.add(executor.submit(check, item))
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            if not done:
                progress('images', completed, len(images))
            for future in done:
                path, actual = future.result()
                images[path] = actual
                completed += 1
                progress('images', completed, len(images))
                item = next(iterator, None)
                if item is not None:
                    pending.add(executor.submit(check, item))


def _atomic_receipt(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_training_pairs(path, *, mode='auto', workers=8, samples=64):
    """Reuse preparation/full-audit evidence; never skip schema/split checks.

    Receipts bind manifest bytes and location, not all current image contents.
    Auto hashes a deterministic sample; image_source(..., verify_files=True)
    checks every image actually consumed by training. Full always audits all.
    """
    if mode not in ('auto', 'full'):
        raise ValueError('data.validation must be auto or full')
    if type(workers) is not int or workers < 1 or type(samples) is not int or samples < 1:
        raise ValueError('validation_workers and validation_samples must be positive integers')
    path = Path(path).expanduser().resolve()
    if (path.name == 'pairs.jsonl' and (path.parent / 'preparing.json').is_file()
            and not (path.parent / 'ready.json').is_file()):
        raise ValueError('Data preparation is incomplete: ready.json is missing; finish preparation first')
    started = time.monotonic()
    last_phase, last_log = None, 0.
    def progress(phase, count, total, *, force=False):
        nonlocal last_phase, last_log
        now = time.monotonic()
        if force or phase != last_phase or now - last_log >= 10:
            print(json.dumps({'phase': 'data_' + phase, 'checked': count, 'total': total,
                              'rank': os.environ.get('RANK', 'coordinator'),
                              'elapsed_seconds': round(now-started, 1)}), flush=True)
            last_phase, last_log = phase, now

    before = path.stat()
    progress('manifest', 0, None, force=True)
    checksum, rows = hashlib.sha256(), []
    with path.open('rb') as handle:
        for line in handle:
            checksum.update(line)
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) % 10000 == 0:
                    progress('manifest', len(rows), None)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Pair manifest changed while reading')
    manifest_sha = checksum.hexdigest()
    progress('manifest', len(rows), len(rows), force=True)
    # Avoid a second full copy and per-image resolve()/stat() RPCs on AOSS.
    rows = validate_pair_rows(rows, path.parent, verify_files=False,
                              resolve_paths=False, in_place=True, progress=progress)
    progress('splits', len(rows), len(rows), force=True)
    counts = dict(Counter(row['split'] for row in rows))
    images, shards = {}, {}
    has_hashes = True
    progress('inventory', 0, len(rows), force=True)
    for index, row in enumerate(rows, 1):
        storage = row.get('storage', {})
        if storage.get('kind') == 'arrow':
            shard = storage['shard']
            identity = (storage['shard_size'], storage['shard_mtime_ns'])
            if shard in shards and shards[shard] != identity:
                raise ValueError(f'Conflicting Arrow shard inventory: {shard}')
            shards[shard] = identity
            continue
        for side in ('winner', 'loser'):
            image, expected = row[side], row.get(side + '_sha256')
            if expected is not None and not _valid_sha(expected):
                raise ValueError(f'Invalid {side} checksum: {row["pair_id"]}')
            if expected is None:
                has_hashes = False
            if image in images and images[image] is not None and expected is not None and images[image] != expected:
                raise ValueError(f'Conflicting image checksums: {image}')
            images[image] = expected or images.get(image)
        if index % 10000 == 0:
            progress('inventory', index, len(rows))
    for shard, expected in shards.items():
        stat = Path(shard).stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError(f'Arrow shard inventory changed: {shard}')

    ready = None
    ready_path = path.parent / 'ready.json'
    if path.name == 'pairs.jsonl' and ready_path.is_file():
        ready = json.loads(ready_path.read_text())
        if (ready.get('pairs_sha256') != manifest_sha or ready.get('splits') != counts
                or not ready.get('source_revision')):
            raise ValueError('ready.json does not match manifest bytes, splits or source revision')
        for row in rows:
            evidence = row.get('human_preference')
            if evidence and evidence.get('source_revision') != ready['source_revision']:
                raise ValueError('Human preference source differs from ready.json')
    identity = {'format': 1, 'manifest': str(path), 'sha256': manifest_sha,
                'rows': len(rows), 'splits': counts, 'images': len(images),
                'shards': digest(shards), 'ready': digest(ready)}
    cache_path = path.with_name('.' + path.name + '.validation.json')
    cached = None
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except (ValueError, OSError):
            pass
    reusable = isinstance(cached, dict) and cached.get('identity') == identity and cached.get('complete') is True
    quick = mode == 'auto' and has_hashes and (ready is not None or reusable)
    evidence = 'prepared' if ready is not None else 'full-audit-cache'
    print(json.dumps({'phase': 'data_validation', 'mode': 'sampled' if quick else 'full',
                      'rank': os.environ.get('RANK', 'coordinator'),
                      'evidence': evidence if quick else None, 'unique_images': len(images),
                      'workers': workers, 'sample_limit': samples if quick else None}), flush=True)
    if quick:
        names = list(images)
        indices = random.Random(manifest_sha).sample(range(len(names)), min(samples, len(names)))
        selected = {names[i]: images[names[i]] for i in indices}
        # Include winner/loser from every split, even for a tiny sample.
        seen_splits = set()
        for row in rows:
            if row['split'] not in seen_splits and row.get('storage', {}).get('kind') != 'arrow':
                seen_splits.add(row['split'])
                for side in ('winner', 'loser'):
                    selected[row[side]] = images[row[side]]
        _parallel_check(selected, workers, progress)
        progress('images', len(selected), len(selected), force=True)
    else:
        # A failed explicit re-audit must not leave old success evidence behind.
        if cache_path.is_file():
            cache_path.unlink()
        _parallel_check(images, workers, progress)
        progress('images', len(images), len(images), force=True)
        for row in rows:
            if row.get('storage', {}).get('kind') != 'arrow':
                for side in ('winner', 'loser'):
                    row[side + '_sha256'] = images[row[side]]
                if row['winner_sha256'] == row['loser_sha256']:
                    raise ValueError('Winner and loser must be distinct images')
        if not has_hashes:
            # Newly computed hashes may reveal aliases/duplicates across splits.
            validate_split_disjointness(rows, progress=progress)
    if has_hashes:
        try:
            _atomic_receipt(cache_path, {'identity': identity, 'complete': True,
                                        'audit': 'sampled' if quick else 'full',
                                        'verified_images': len(selected) if quick else len(images)})
        except OSError as error:
            print(f'[data_validation] Cache not writable; validation still passed: {error}', flush=True)
    progress('ready', len(rows), len(rows), force=True)
    return rows
