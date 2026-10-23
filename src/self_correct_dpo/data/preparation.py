"""Bounded parallel image export and atomic Pick-a-Pic batch checkpoints."""
from hashlib import sha256 as hash_bytes
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from threading import Lock

from self_correct_dpo.common import sha256


class InvalidImageError(Exception):
    """An undecodable source image, distinct from storage/read/write failures."""


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


class ImagePool:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verified = set()
        self.bytes_written = 0
        self.reused = 0
        self.budget = 0
        self.budget_lock = Lock()
        # Same hash/shard serializes, independent shards can perform I/O concurrently.
        self.locks = [Lock() for _ in range(256)]
        self.directories = set()

    def store(self, payload):
        from PIL import Image
        checksum = hash_bytes(payload).hexdigest()
        target = self.root / checksum[:2] / (checksum + '.img')
        with self.locks[int(checksum[:2], 16)]:
            if checksum in self.verified:
                return str(target), checksum
            if target.exists():
                if sha256(target) != checksum:
                    raise ValueError(f'Existing image is corrupt: {target}')
                with self.budget_lock:
                    self.reused += 1
            else:
                try:
                    with Image.open(BytesIO(payload)) as image:
                        image.verify()
                except OSError as error:
                    raise InvalidImageError('Undecodable source image') from error
                with self.budget_lock:
                    if self.budget < len(payload):
                        free = shutil.disk_usage(self.root).free
                        if free < len(payload) + 1024**3:
                            raise RuntimeError('Image export needs more disk space plus 1 GiB margin')
                        self.budget = min(free - 1024**3, 128 * 1024**2)
                    self.budget -= len(payload)
                if checksum[:2] not in self.directories:
                    target.parent.mkdir(exist_ok=True)
                    self.directories.add(checksum[:2])
                partial = target.with_suffix('.partial')
                with partial.open('wb') as handle:
                    handle.write(payload)
                partial.replace(target)
                with self.budget_lock:
                    self.bytes_written += len(payload)
            self.verified.add(checksum)
        return str(target), checksum


class BatchJournal:
    """One atomic file commits row metadata and the next Parquet cursor together.

    Uncommitted images are harmless and reused after an interrupted batch. No
    SQLite database/WAL or repeated whole-dataset checkpoint rewrite is needed.
    """
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        marker = self.root / 'identity.json'
        if marker.exists():
            if json.loads(marker.read_text()) != identity:
                raise ValueError('Preparation inputs/code changed; restore the original inputs/code or use a new --out')
        else:
            if any(self.root.glob('batch-*.json')):
                raise ValueError('Batch checkpoints have no identity; retain this directory for inspection')
            atomic_json(marker, identity)
        self.count = 0
        self.cursor = {'file': 0, 'row_group': 0, 'row': 0}

    def replay(self):
        for index, path in enumerate(sorted(self.root.glob('batch-*.json'))):
            if path.name != f'batch-{index:08d}.json':
                raise ValueError(f'Noncontiguous preparation checkpoints: {path}')
            batch = json.loads(path.read_text())
            self.count = index + 1
            self.cursor = batch['next']
            yield batch

    def commit(self, batch):
        path = self.root / f'batch-{self.count:08d}.json'
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, batch)
        self.count += 1
        self.cursor = batch['next']


def parquet_batches(files, *, batch_size, start):
    """Seek past completed shards/row groups; decode only the active row group.

    Parquet cannot seek to an arbitrary row inside a compressed row group. On
    resume, earlier batches in that row group are decoded but never reprocessed.
    """
    import pyarrow.parquet as pq
    for file_index in range(start['file'], len(files)):
        parquet = pq.ParquetFile(files[file_index])
        columns = [c for c in parquet.schema_arrow.names if c in {
            'ranking_id', 'caption', 'label_0', 'label_1', 'has_label',
            'image_0_uid', 'image_1_uid', 'jpg_0', 'jpg_1'
        }]
        first_group = start['row_group'] if file_index == start['file'] else 0
        for group in range(first_group, parquet.num_row_groups):
            skip = start['row'] if file_index == start['file'] and group == first_group else 0
            offset = 0
            size = parquet.metadata.row_group(group).num_rows
            for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group], columns=columns):
                end = offset + batch.num_rows
                if end <= skip:
                    offset = end
                    continue
                if offset < skip:
                    batch = batch.slice(skip - offset)
                cursor = {'file': file_index, 'row_group': group, 'row': end}
                if end == size:
                    cursor = {'file': file_index, 'row_group': group + 1, 'row': 0}
                    if group + 1 == parquet.num_row_groups:
                        cursor = {'file': file_index + 1, 'row_group': 0, 'row': 0}
                yield batch.to_pylist(), cursor
                offset = end
