"""Read user-owned local Arrow image bytes without exporting or downloading."""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path


@lru_cache(maxsize=128)
def arrow_dataset(path):
    from datasets import Dataset
    return Dataset.from_file(str(path)).select_columns(['jpg_0','jpg_1'])


def image_source(row, side):
    storage=row.get('storage',{})
    if storage.get('kind') != 'arrow': return row[side]
    path=Path(storage['shard'])
    stat=path.stat()
    if stat.st_size != storage['shard_size'] or stat.st_mtime_ns != storage['shard_mtime_ns']:
        raise ValueError('Arrow shard changed since preparation')
    data=arrow_dataset(str(path))[storage['row']][storage[side+'_column']]
    if sha256(data).hexdigest() != row[side+'_sha256']:
        raise ValueError('Arrow image checksum mismatch')
    return BytesIO(data)
