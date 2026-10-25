"""Small parallel image samples at startup; full audits are explicit."""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import random
import time

from self_correct_dpo.common import sha256
from self_correct_dpo.data.pairs import validate_pair_rows, validate_split_disjointness


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _parallel_check(images, workers, progress):
    """Bound pending reads to 2*workers, including an explicit full audit."""
    def check(item):
        path, (row, side) = item
        expected = row.get(side + '_sha256')
        if row.get('storage', {}).get('kind') == 'arrow':
            from self_correct_dpo.data.arrow_images import image_source
            # Validate real bytes and row bounds, not only shard metadata.
            with image_source(row, side, verify_files=True):
                pass
            actual = expected
        else:
            actual = sha256(path)
            if expected is not None and actual != expected:
                raise ValueError(f'Image checksum mismatch: {path}')
        return path, actual

    iterator, verified = iter(images.items()), {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(workers * 2):
            item = next(iterator, None)
            if item is not None:
                pending.add(executor.submit(check, item))
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            if not done:
                progress('images', len(verified), len(images))
            for future in done:
                path, actual = future.result()
                verified[path] = actual
                progress('images', len(verified), len(images))
                item = next(iterator, None)
                if item is not None:
                    pending.add(executor.submit(check, item))
    return verified


def load_training_pairs(path, *, mode='auto', workers=8, samples=8):
    """Read/check the manifest once, then sample images in parallel.

    Auto never upgrades to a full image scan or writes validation receipts.
    Training verifies every image when consumed. Full is an explicit audit.
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
    rows = validate_pair_rows(rows, path.parent, verify_files=False,
                              resolve_paths=False, in_place=True, progress=progress)
    ready = None
    ready_path = path.parent / 'ready.json'
    if path.name == 'pairs.jsonl' and ready_path.is_file():
        ready = json.loads(ready_path.read_text())
        if (ready.get('pairs_sha256') != manifest_sha
                or ready.get('splits') != dict(Counter(row['split'] for row in rows))
                or not ready.get('source_revision')):
            raise ValueError('ready.json does not match manifest bytes, splits or source revision')

    # Sample positions without constructing an inventory of every image/shard.
    indices = set(random.Random(manifest_sha).sample(range(2 * len(rows)), min(samples, 2 * len(rows))))
    selected, seen_splits, has_hashes = {}, set(), True
    for index, row in enumerate(rows):
        evidence = row.get('human_preference')
        if ready and evidence and evidence.get('source_revision') != ready['source_revision']:
            raise ValueError('Human preference source differs from ready.json')
        first_in_split = row['split'] not in seen_splits
        seen_splits.add(row['split'])
        for offset, side in enumerate(('winner', 'loser')):
            expected = row.get(side + '_sha256')
            if expected is None:
                has_hashes = False
            elif not _valid_sha(expected):
                raise ValueError(f'Invalid {side} checksum: {row["pair_id"]}')
            if mode == 'full' or first_in_split or 2 * index + offset in indices:
                image = row[side]
                if image in selected:
                    previous, previous_side = selected[image]
                    previous_hash = previous.get(previous_side + '_sha256')
                    if previous_hash is not None and expected is not None and previous_hash != expected:
                        raise ValueError(f'Conflicting image checksums: {image}')
                    if previous_hash is not None and expected is None:
                        continue
                selected[image] = row, side
    if mode == 'auto' and not has_hashes:
        raise ValueError('Image checksums are required for sampled startup; prepare the manifest with SHA256 '
                         'or explicitly use --data-validation full. No automatic full audit is performed.')
    print(json.dumps({'phase': 'data_validation', 'mode': 'sampled' if mode == 'auto' else 'full',
                      'rank': os.environ.get('RANK', 'coordinator'), 'selected_images': len(selected),
                      'workers': workers, 'sample_limit': samples if mode == 'auto' else None}), flush=True)
    verified = _parallel_check(selected, workers, progress)
    progress('images', len(verified), len(verified), force=True)
    if mode == 'full' and not has_hashes:
        for row in rows:
            for side in ('winner', 'loser'):
                row[side + '_sha256'] = verified[row[side]]
            if row['winner_sha256'] == row['loser_sha256']:
                raise ValueError('Winner and loser must be distinct images')
        validate_split_disjointness(rows, progress=progress)
    progress('ready', len(rows), len(rows), force=True)
    return rows
