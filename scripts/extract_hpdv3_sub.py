"""Extract only HPDv3-sub and official-test images from downloaded split archives.

Run once on a CPU compute allocation. Original archives and annotations stay intact.
The output is a raw-image view for prepare_hpdv3_sub.py, not a training manifest.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from prepare_hpdv3_sub import iter_selected
from self_correct_dpo.data.json_stream import object_records
from self_correct_dpo.common import sha256, output_lock
from self_correct_dpo.data.preparation import atomic_json
from self_correct_dpo.data.hpdv3_source import source_record


def _write_image(target, payload):
    # The extractor owns the output lock; only one task writes each image.
    if target.is_file() and target.stat().st_size == len(payload):
        return False
    temporary = target.with_suffix(target.suffix+'.partial')
    with temporary.open('wb') as dest:
        dest.write(payload)
    temporary.replace(target)
    return True


def extract(raw, out, max_gib=80, reserve_gib=30, workers=4):
    if workers < 1:
        raise ValueError('workers must be positive')
    with output_lock(out):
        return _extract(raw, out, max_gib, reserve_gib, workers)


def _extract(raw, out, max_gib, reserve_gib, workers):
    print('[extract] Checking source records and final archive volumes', flush=True)
    source_path, source = source_record(raw, create_inventory=True)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'extraction.json').is_file():
        previous = json.loads((out/'extraction.json').read_text())
        if previous.get('revision') != source['revision']:
            raise ValueError('Completed extraction has a different source identity; choose a new output directory')
        print(f'Reusing completed extraction: {out}', flush=True)
        return
    print('[extract] Reading test.json', flush=True)
    tests = json.loads((raw/'test.json').read_text())
    print('[extract] Streaming train.json; keeping only selected image paths', flush=True)
    exclusions = Counter()
    selected_count = 0
    needed = {r[key] for r in tests for key in ('path1', 'path2')}
    for item in iter_selected(object_records(raw/'train.json'), tests, exclusions):
        selected_count += 1
        needed.update(item['record'][key] for key in ('path1', 'path2'))
    for name in needed:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'images':
            raise ValueError(f'Unexpected image path: {name}')
    needed_count = len(needed)
    print(json.dumps({'selected_pairs_before_image_exclusions': selected_count,
                      'required_images_including_official_test': needed_count,
                      'exclusions': exclusions}), flush=True)
    del tests
    for name in ('train.json', 'test.json'):
        destination = out/name
        if not destination.exists(): destination.symlink_to((raw/name).resolve())
    archives = sorted(raw.glob('images.tar.gz.[0-9][0-9]'))
    if not archives: raise FileNotFoundError('No downloaded split image archives')
    total = count = written = pending_bytes = 0
    pending = deque()
    directories = set()
    started = last_report = time.monotonic()
    last_count = 0
    def finish_one():
        nonlocal count, written, pending_bytes, last_report, last_count
        future, size = pending.popleft()
        written += int(future.result())
        pending_bytes -= size
        count += 1
        now = time.monotonic()
        if count % 1000 == 0 or now-last_report >= 30:
            print(json.dumps({'extracted_images': count, 'written_images': written,
                              'reused_images': count-written, 'selected_gib': total/1024**3,
                              'elapsed_seconds': round(now-started, 1),
                              'images_per_second': round((count-last_count)/(now-last_report), 2),
                              'workers': workers}), flush=True)
            last_report, last_count = now, count
    print(json.dumps({'phase': 'extract', 'workers': workers,
                      'max_pending_mib': 64, 'tar_buffer_mib': 1}), flush=True)
    stream = subprocess.Popen(['cat', *map(str, archives)], stdout=subprocess.PIPE)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor, \
                tarfile.open(fileobj=stream.stdout, mode='r|gz', bufsize=1024*1024) as archive:
            for member in archive:
                # Sequential streaming does not need tarfile's growing member cache.
                archive.members.clear()
                name = member.name.removeprefix('./')
                if name not in needed: continue
                if not member.isfile(): raise ValueError(f'Expected image file: {name}')
                total += member.size
                if total > max_gib * 1024**3:
                    raise RuntimeError(f'Selected images exceed the {max_gib} GiB extraction budget')
                if (count+len(pending)) % 1000 == 0 and shutil.disk_usage(out).free < reserve_gib * 1024**3 + pending_bytes + member.size:
                    raise RuntimeError(f'Less than {reserve_gib} GiB remains; extraction stopped')
                target = out/name
                if target.parent not in directories:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.parent.resolve().is_relative_to(out.resolve()):
                        raise ValueError(f'Output directory escapes extraction root: {name}')
                    directories.add(target.parent)
                # Only the reader touches the gzip stream. Workers receive owned bytes.
                while pending and (len(pending) >= workers*4 or pending_bytes+member.size > 64*1024**2):
                    finish_one()
                if member.size > 64*1024**2:
                    raise ValueError(f'Image exceeds the 64 MiB streaming buffer: {name}')
                with archive.extractfile(member) as source:
                    payload = source.read()
                if len(payload) != member.size:
                    raise ValueError(f'Truncated archive member: {name}')
                pending.append((executor.submit(_write_image, target, payload), member.size))
                pending_bytes += member.size
                needed.remove(name)
            while pending:
                finish_one()
        if stream.wait() != 0: raise RuntimeError('Reading split image archives failed')
    finally:
        stream.stdout.close()
        if stream.poll() is None: stream.terminate()
        stream.wait()
    if needed: raise FileNotFoundError(f'{len(needed)} selected images missing from downloaded archives')
    # Recheck the existing source record after the stream; inventory paths must
    # must still match, allowing explicitly logged whole-second truncation.
    source_path, source = source_record(raw)
    annotation_hashes = source.get('identity', {}).get('annotations', {})
    report = {'source': str(raw), 'revision': source['revision'], 'images': count,
              'selected_pairs_before_image_exclusions': selected_count, 'bytes': total,
              'source_stat_validation': source.get('stat_validation'),
              'source_record': str(source_path), 'source_record_kind': source.get('kind', 'download_receipt'),
              'upstream_revision_verified': source.get('kind') != 'hpdv3_local_source_inventory',
              'train_json_sha256': annotation_hashes.get('train.json') or sha256(raw/'train.json'),
              'test_json_sha256': annotation_hashes.get('test.json') or sha256(raw/'test.json')}
    report.update(workers=workers, written_images=written, reused_images=count-written,
                  elapsed_seconds=round(time.monotonic()-started, 1))
    atomic_json(out/'extraction.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--max-gib', type=int, default=80)
    p.add_argument('--reserve-gib', type=int, default=30)
    p.add_argument('--workers', type=int, default=4, help='Bounded image I/O threads; gzip is read once')
    a = p.parse_args()
    extract(a.raw.resolve(), a.out.resolve(), a.max_gib, a.reserve_gib, a.workers)
