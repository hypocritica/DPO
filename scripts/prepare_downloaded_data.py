"""Offline SLAI Pick-a-Pic/HPDv3 -> original images + audited pair JSONL.

Use a CPU worker/container with enough RAM and disk, not a cluster login node.
No GPU, network, image resize/re-encoding, or reward-based relabeling is used.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256 as hash_bytes
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from download_assets import env_path, parquet_split
from self_correct_dpo.data.benchmarks import prepare_panels, prompt_key
from self_correct_dpo.common import digest, sha256, write_json, write_jsonl, output_lock
from self_correct_dpo.data.pickapic import adapt_rows
from self_correct_dpo.data.preparation import ImagePool, InvalidImageError, BatchJournal, parquet_batches, atomic_json, atomic_jsonl


def parquet_records(files, *, metadata_only=False):
    import pyarrow.parquet as pq
    for file in files:
        parquet = pq.ParquetFile(file)
        columns = [c for c in parquet.schema_arrow.names if c in {
            'ranking_id', 'caption', 'label_0', 'label_1', 'has_label',
            'image_0_uid', 'image_1_uid', 'jpg_0', 'jpg_1'
        }]
        if metadata_only:
            columns = [c for c in columns if c not in ('jpg_0', 'jpg_1')]
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            yield from batch.to_pylist()


def image_bytes(value):
    # Official Pick-a-Pic columns are binary; HF Image struct exports are also accepted.
    if isinstance(value, dict):
        value = value.get('bytes')
    if not isinstance(value, bytes) or not value:
        raise ValueError('Expected embedded jpg bytes, not remote image URLs')
    return value


def tokenizer_predicate(model, family):
    from transformers import AutoTokenizer
    names = ['tokenizer'] if family == 'sd15' else ['tokenizer', 'tokenizer_2']
    tokenizers = [AutoTokenizer.from_pretrained(str(model / name), local_files_only=True) for name in names]
    cache = {}
    def allowed(prompt):
        if prompt not in cache:
            cache[prompt] = all(len(t(prompt, truncation=False, padding=False)['input_ids']) <= t.model_max_length
                                for t in tokenizers)
        return cache[prompt]
    return allowed


def pickapic_source(raw, revision=None):
    """Inspect Parquet inventory; a data/ subdirectory may use its parent's receipt."""
    raw = Path(raw).resolve()
    if not raw.is_dir():
        raise FileNotFoundError(f'Pick-a-Pic raw directory does not exist: {raw}')
    receipt = provenance = None
    files = None
    for root in (raw, raw.parent):
        marker = root / '.scd-download.json'
        if not marker.is_file():
            continue
        candidate = json.loads(marker.read_text())
        selected = []
        for name, size in candidate.get('files', {}).items():
            if not name.endswith('.parquet'):
                continue
            file = (root / name).resolve()
            if not file.is_relative_to(root):
                raise ValueError(f'Download receipt path escapes dataset root: {name}')
            if file.is_relative_to(raw):
                selected.append((file, size))
        if root != raw and not selected:
            continue
        if not candidate.get('complete'):
            raise ValueError(f'Download is incomplete: {marker}')
        for file, size in selected:
            if not file.is_file() or file.stat().st_size != size:
                raise ValueError(f'Download inventory changed: {file}')
        receipt, provenance = candidate, marker
        files = sorted(file for file, _ in selected)
        break
    if files is None:
        files = sorted(raw.rglob('*.parquet'))
    splits = {parquet_split(str(file.relative_to(raw))) for file in files}
    if not {'train', 'test_unique'}.issubset(splits):
        raise ValueError(f'Need train and official test_unique Parquet files under {raw}')
    revision = revision or (receipt['repo_id'] + '@' + receipt['revision'] if receipt else None)
    if not revision or any(word in revision.upper() for word in ('REPLACE_WITH', 'FILL_', 'TODO')):
        raise ValueError('No verified dataset revision found. Check the original download record and '
                         'supply --revision with its real repository and revision; do not use a placeholder.')
    return files, revision, receipt, provenance


def prepare_pickapic(raw, out, image_root, **options):
    # The legacy serial exporter has no locks: stop it before adopting its output.
    with output_lock(out), output_lock(image_root):
        return _prepare_pickapic(raw, out, image_root, **options)


def _prepare_pickapic(raw, out, image_root, *, family, model, hpdv2_test=None, hpdv3_test=None,
                      revision=None, allowed=None, workers=8, batch_size=256,
                      resume=False, adopt_legacy=False):
    if workers < 1 or batch_size < 1:
        raise ValueError('workers and batch_size must be positive')
    raw, out, image_root = Path(raw).resolve(), Path(out).resolve(), Path(image_root).resolve()
    files, revision, receipt, provenance = pickapic_source(raw, revision)
    inventory = [{'file':str(f.relative_to(raw)), 'bytes':f.stat().st_size, 'mtime_ns':f.stat().st_mtime_ns} for f in files]
    external_tests = [Path(p) for p in (hpdv2_test, hpdv3_test) if p]
    token_dirs = ['tokenizer'] if family == 'sd15' else ['tokenizer', 'tokenizer_2']
    token_files = sorted(p for name in token_dirs for p in (Path(model)/name).rglob('*') if p.is_file())
    code_files = [Path(__file__)] + [ROOT/'src/self_correct_dpo/data'/name
                                    for name in ('preparation.py', 'pickapic.py', 'pairs.py')]
    identity = {'format':2, 'source':str(raw), 'revision':revision, 'family':family,
                'images':str(image_root), 'inventory':inventory, 'model':str(Path(model).resolve()),
                'tokenizers':{str(p.relative_to(model)):sha256(p) for p in token_files},
                'external_tests':{name: {'path':str(Path(p).resolve()), 'sha256':sha256(p)}
                                  for name, p in (('hpdv2', hpdv2_test), ('hpdv3', hpdv3_test)) if p},
                'code':{p.name:sha256(p) for p in code_files}}
    marker = out/'preparing.json'
    if out.exists():
        if (out/'ready.json').exists():
            raise FileExistsError(f'{out} is already ready; reuse it for training')
        if not marker.is_file():
            raise ValueError('Existing output has no preparing.json; use a new --out')
        previous = json.loads(marker.read_text())
        if any(previous.get(k) != identity[k] for k in ('source', 'revision', 'family')):
            raise ValueError('Existing preparation source/revision/family differs')
        if (out/'.preparation/identity.json').exists() or previous.get('format') == 2:
            if not resume:
                raise ValueError('Interrupted preparation exists; pass --resume')
        elif not adopt_legacy:
            raise ValueError('Legacy preparation has no cursor; stop its process and pass --adopt-legacy')
        else:
            if any((out/name).exists() for name in ('pairs.jsonl', 'audit.json')):
                raise ValueError('Legacy output already reached finalization; retain it and use a new --out')
            print('Adopting legacy images: reconstructing row metadata in parallel; no images are overwritten.', flush=True)
    else:
        if resume or adopt_legacy:
            raise FileNotFoundError(f'No interrupted preparation at {out}')
        out.mkdir(parents=True)
        atomic_json(marker, {'source':str(raw), 'revision':revision, 'family':family, 'format':2})
    journal = BatchJournal(out/'.preparation', identity)
    training = [f for f in files if parquet_split(str(f.relative_to(raw))) == 'train']
    heldout = [f for f in files if parquet_split(str(f.relative_to(raw))) in ('test', 'test_unique', 'validation', 'validation_unique')]
    unique_test = [f for f in files if parquet_split(str(f.relative_to(raw))) == 'test_unique']
    blocked_cache = journal.root/'heldout.json'
    if blocked_cache.exists():
        blocked = json.loads(blocked_cache.read_text())
        blocked_prompts = set(blocked['prompts'])
        blocked_images = set(map(tuple, blocked['images']))
    else:
        # Protect all official held-out captions, image UIDs and byte hashes before splitting train.
        blocked_prompts, blocked_images = set(), set()
        for record in parquet_records(heldout):
            if record.get('caption'):
                blocked_prompts.add(prompt_key(record['caption']))
            for side in (0, 1):
                if record.get(f'image_{side}_uid'):
                    blocked_images.add(('uid', str(record[f'image_{side}_uid'])))
                blocked_images.add(('sha', hash_bytes(image_bytes(record[f'jpg_{side}'])).hexdigest()))
        external_tests = [Path(p) for p in (hpdv2_test,hpdv3_test) if p]
        for test in external_tests:
            hpd_rows = json.loads(test.read_text())
            if not isinstance(hpd_rows, list) or not all(isinstance(r.get('prompt'), str) for r in hpd_rows):
                raise ValueError('Expected official HPD test.json prompt records')
            blocked_prompts.update(prompt_key(r['prompt']) for r in hpd_rows)
            if hpdv3_test and test==Path(hpdv3_test):
                for record in hpd_rows:
                    for side in ('path1','path2'):
                        image=(test.resolve().parent/record[side]).resolve()
                        if not image.is_relative_to(test.resolve().parent):raise ValueError('HPDv3 test image escapes dataset root')
                        if not image.is_file():raise FileNotFoundError('Extract HPDv3 images before preparing Pick-a-Pic')
                        blocked_images.add(('sha',sha256(image)))
        atomic_json(blocked_cache, {'prompts':sorted(blocked_prompts), 'images':sorted(blocked_images)})
    allowed = allowed or tokenizer_predicate(Path(model), family)
    pool = ImagePool(image_root)
    original, groups, exclusions, seen_ids = [], {}, [], set()
    raw_count = written_bytes = reused_images = 0

    def restore(batch):
        nonlocal raw_count, written_bytes, reused_images
        for pair_id in batch['ids']:
            if not pair_id or pair_id in seen_ids:
                raise ValueError(f'Missing or duplicate source ranking_id: {pair_id}')
            seen_ids.add(pair_id)
        original.extend(batch['rows'])
        exclusions.extend(batch['excluded'])
        for row in batch['rows']:
            groups[row['caption']] = 'caption:' + digest(prompt_key(row['caption']))[:24]
            pool.verified.update(row['hashes'])
        raw_count += len(batch['ids'])
        written_bytes += batch['written_bytes']
        reused_images += batch['reused_images']

    for batch in journal.replay():
        restore(batch)
    pool.bytes_written = written_bytes
    pool.reused = reused_images
    import pyarrow.parquet as pq
    total = sum(pq.ParquetFile(file).metadata.num_rows for file in training)
    print(json.dumps({'phase':'scan', 'resumed_scanned':raw_count, 'total':total,
                      'workers':workers, 'batch_size':batch_size, 'cursor':journal.cursor}), flush=True)

    def export(record):
        pair_id = str(record['ranking_id'])
        payloads = [image_bytes(record[f'jpg_{side}']) for side in (0, 1)]
        hashes = [hash_bytes(payload).hexdigest() for payload in payloads]
        uids = [str(record.get(f'image_{side}_uid') or hashes[side]) for side in (0, 1)]
        if any(('uid', uid) in blocked_images for uid in uids) or any(('sha', h) in blocked_images for h in hashes):
            return None, {'pair_id':pair_id, 'reason':'official_heldout_image'}
        try:
            paths = [pool.store(payload)[0] for payload in payloads]
        except InvalidImageError:
            return None, {'pair_id':pair_id, 'reason':'invalid_image'}
        # ValueError (including a corrupt preexisting image) is fatal, never skipped.
        return {'pair_id':pair_id, 'caption':record['caption'],
                'label_0':record['label_0'], 'label_1':record['label_1'],
                'image_0':paths[0], 'image_1':paths[1], 'image_0_uid':uids[0], 'image_1_uid':uids[1],
                'split':'train', 'hashes':hashes}, None

    started = last_log = time.monotonic()
    initial_count = last_count = raw_count
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for records, cursor in parquet_batches(training, batch_size=batch_size, start=journal.cursor):
            ids, eligible, rejected = [], [], []
            current_ids = set()
            for record in records:
                pair_id = str(record.get('ranking_id', ''))
                if not pair_id or pair_id in seen_ids or pair_id in current_ids:
                    raise ValueError(f'Missing or duplicate source ranking_id: {pair_id}')
                ids.append(pair_id)
                current_ids.add(pair_id)
                caption = record.get('caption')
                labels = [record.get('label_0'), record.get('label_1')]
                reason = None
                if not isinstance(caption, str) or not caption.strip(): reason = 'empty_caption'
                elif not record.get('has_label', True): reason = 'unlabelled'
                elif labels not in ([1, 0], [0, 1], [.5, .5]): reason = 'invalid_labels'
                elif prompt_key(caption) in blocked_prompts: reason = 'official_heldout_caption'
                elif not allowed(caption): reason = 'model_token_length'
                if reason:
                    rejected.append({'pair_id':pair_id, 'reason':reason})
                else:
                    eligible.append(record)
            rows = []
            for row, exclusion in executor.map(export, eligible):
                if row is not None: rows.append(row)
                else: rejected.append(exclusion)
            positions = {pair_id:index for index, pair_id in enumerate(ids)}
            rejected.sort(key=lambda item: positions[item['pair_id']])
            batch = {'ids':ids, 'rows':rows, 'excluded':rejected, 'next':cursor,
                     'written_bytes':pool.bytes_written-written_bytes, 'reused_images':pool.reused-reused_images}
            journal.commit(batch)
            restore(batch)
            now = time.monotonic()
            if raw_count-last_count >= 10000 or now-last_log >= 30 or raw_count == total:
                rate = (raw_count-initial_count)/max(now-started, .001)
                print(json.dumps({'phase':'scan', 'scanned':raw_count, 'total':total,
                                  'eligible':len(original), 'unique_images':len(pool.verified),
                                  'reused_images':reused_images, 'rows_per_second':round(rate, 2),
                                  'remaining_scan_hours':round((total-raw_count)/max(rate, .001)/3600, 2)}), flush=True)
                last_count, last_log = raw_count, now
    print(json.dumps({'phase':'finalize', 'scanned':raw_count, 'eligible':len(original)}), flush=True)
    def resolve_images(row):
        return ([row['image_0'],row['image_1']], row['hashes'], [row['image_0_uid'],row['image_1_uid']])
    pairs, rejected, report = adapt_rows(original, out, groups, source_revision=revision,
                                         seed=17, grouping_audited=False, image_resolver=resolve_images)
    exclusions.extend(rejected)
    counts = Counter(row['split'] for row in pairs)
    if any(counts[split] == 0 for split in ('train', 'validation', 'test')):
        raise ValueError('Empty split after preparation; inspect subset size and audit before training')
    for row in pairs:
        for side in ('winner', 'loser'):
            row[side] = os.path.relpath(row[side], out)
    atomic_jsonl(out / 'pairs.jsonl', pairs)
    atomic_jsonl(out / 'excluded.jsonl', exclusions)
    # Captions only: official test_unique is distinct from the local test rows above.
    atomic_jsonl(out / 'official-test-unique.jsonl',
                ({'caption':r['caption']} for r in parquet_records(unique_test, metadata_only=True) if r.get('caption')))
    atomic_json(out / 'audit.json', {**report, 'raw_train_rows':raw_count, 'family':family,
        'exclusions':dict(Counter(r['reason'] for r in exclusions)), 'source_inventory':inventory,
        'download_receipt':receipt, 'download_receipt_file':str(provenance) if provenance else None,
        'external_test_sha256':{str(p):sha256(p) for p in external_tests},
        'model':str(model), 'images_written_bytes':pool.bytes_written,
        'preparation_code_sha256':sha256(__file__), 'preparation_identity':identity,
        'reused_images':reused_images, 'workers':workers,
        'model_download_receipt':json.loads((Path(model)/'.scd-download.json').read_text()) if (Path(model)/'.scd-download.json').is_file() else None,
        'split_policy':'Official validation/test excluded by normalized caption, image UID and SHA256; remaining train components split 90/5/5 with seed17. Local test is NOT official test_unique.',
        'semantic_isolation_limitation':'Exact normalized captions and shared images are isolated; unknown paraphrases are not manually audited.',
        'image_policy':'Original bytes, content-addressed shared files; RGB square resize happens only in the trainer.'})
    atomic_json(out / 'ready.json', {'pairs_sha256':sha256(out / 'pairs.jsonl'), 'splits':dict(counts), 'source_revision':revision})
    print(json.dumps({'out':str(out), 'splits':dict(counts)}), flush=True)


def prepare_hpdv3(raw, out, *, family, model=None, pickapic_raw=None, revision=None, allowed=None, min_confidence=None):
    from PIL import Image
    raw,out=Path(raw).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    receipt_file=raw/'.scd-download.json'
    receipt=json.loads(receipt_file.read_text()) if receipt_file.exists() else None
    if receipt and not receipt.get('complete'):raise ValueError('HPDv3 download is incomplete')
    revision=revision or (receipt['repo_id']+'@'+receipt['revision'] if receipt else None)
    if not revision:raise ValueError('Supply --revision for manually downloaded HPDv3')
    if not (raw/'images').is_dir():raise FileNotFoundError('Extract HPDv3 images before preparing train.json')
    tests=json.loads((raw/'test.json').read_text())
    blocked_prompts={prompt_key(r['prompt']) for r in tests}
    blocked_hashes=set()
    cached={}
    def local_image(name):
        path=(raw/name).resolve()
        if not path.is_relative_to(raw):raise ValueError(f'Image path escapes HPDv3 root: {name}')
        if path not in cached:
            if not path.is_file():
                raise FileNotFoundError(f'{path}; do not mix pickapic.json/imagereward.json without their external images')
            with Image.open(path) as image:image.verify()
            cached[path]=sha256(path)
        return str(path),cached[path]
    for record in tests:
        for side in ('path1','path2'):
            blocked_hashes.add(local_image(record[side])[1])
    pickapic_inventory=[]
    if pickapic_raw:
        heldout=[f for f in Path(pickapic_raw).rglob('*.parquet') if parquet_split(str(f.relative_to(pickapic_raw))) in ('test','test_unique','validation','validation_unique')]
        if not heldout:raise ValueError('Download Pick-a-Pic held-out splits before preparing HPDv3')
        for record in parquet_records(heldout):
            if record.get('caption'):blocked_prompts.add(prompt_key(record['caption']))
            blocked_hashes.update(hash_bytes(image_bytes(record[f'jpg_{side}'])).hexdigest() for side in (0,1))
        pickapic_inventory=[{'path':str(f),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in heldout]
    allowed=allowed or ((lambda caption:True) if family=='unfiltered' else tokenizer_predicate(Path(model),family))
    annotations=json.loads((raw/'train.json').read_text())
    if not isinstance(annotations,list):raise ValueError('Expected official HPDv3 train.json list')
    original,groups,excluded,records=[],{},[],{}
    out.mkdir(parents=True)
    write_json(out/'preparing.json',{'revision':revision,'family':family})
    for index,record in enumerate(annotations):
        pair_id='hpdv3:'+str(index)
        caption=record.get('prompt')
        reason=None
        confidence=record.get('confidence')
        if confidence is not None and (isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1):
            raise ValueError(f'Invalid official HPDv3 confidence: row {index}')
        if not isinstance(caption,str) or not caption.strip():reason='empty_caption'
        elif prompt_key(caption) in blocked_prompts:reason='official_heldout_caption'
        elif not allowed(caption):reason='model_token_length'
        elif min_confidence is not None:
            if confidence is None or confidence<min_confidence:reason='confidence_threshold'
        if reason:
            excluded.append({'pair_id':pair_id,'reason':reason});continue
        paths,hashes=zip(*(local_image(record[side]) for side in ('path1','path2')))
        if blocked_hashes.intersection(hashes):
            excluded.append({'pair_id':pair_id,'reason':'official_heldout_image'});continue
        groups[caption]='caption:'+digest(prompt_key(caption))[:24]
        original.append({'pair_id':pair_id,'caption':caption,'label_0':1,'label_1':0,
                         'image_0':paths[0],'image_1':paths[1],'hashes':list(hashes),
                         'image_0_uid':record['path1'],'image_1_uid':record['path2'],'split':'train'})
        records[pair_id]=record
        if index and index%10000==0:print(json.dumps({'scanned':index,'eligible':len(original)}),flush=True)
    def resolve_images(row):
        return ([row['image_0'],row['image_1']],row['hashes'],[row['image_0_uid'],row['image_1_uid']])
    pairs,rejected,report=adapt_rows(original,out,groups,source_revision=revision,seed=17,
                                    grouping_audited=False,image_resolver=resolve_images,preference_source='hpdv3')
    excluded.extend(rejected)
    counts=Counter(row['split'] for row in pairs)
    if any(counts[split]==0 for split in ('train','validation','test')):raise ValueError('Empty split after HPDv3 preparation')
    for row in pairs:
        row['human_preference'].update(preferred_field='path1',original_record=records[row['pair_id']])
        for side in ('winner','loser'):row[side]=os.path.relpath(row[side],out)
    write_jsonl(out/'pairs.jsonl',pairs)
    write_jsonl(out/'excluded.jsonl',excluded)
    write_json(out/'audit.json',{**report,'family':family,'raw_train_rows':len(annotations),
        'exclusions':dict(Counter(r['reason'] for r in excluded)),
        'train_json_sha256':sha256(raw/'train.json'),'test_json_sha256':sha256(raw/'test.json'),
        'minimum_confidence':min_confidence,'download_receipt':receipt,
        'pickapic_heldout_inventory':pickapic_inventory,'preparation_code_sha256':sha256(__file__),
        'model_download_receipt':json.loads((Path(model)/'.scd-download.json').read_text()) if model and (Path(model)/'.scd-download.json').is_file() else None,
        'split_policy':'Official test captions/images and Pick-a-Pic held-out captions/images excluded; remaining train components split 90/5/5. Local test is NOT official test.json.',
        'limitation':'Exact normalized captions and image hashes isolated; unknown semantic paraphrases are not certified. Unfiltered is data-only and requires the target trainer token policy.'})
    write_json(out/'ready.json',{'pairs_sha256':sha256(out/'pairs.jsonl'),'splits':dict(counts),'source_revision':revision})
    print(json.dumps({'out':str(out),'splits':dict(counts)}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pickapic', 'hpdv3', 'benchmarks'])
    p.add_argument('--family', choices=['sd15', 'sdxl', 'unfiltered'], default='sd15')
    p.add_argument('--dataset', choices=['pickapic','hpdv3'],default='pickapic',help='Training manifest used for benchmark overlap exclusion')
    p.add_argument('--min-confidence',type=float,help='HPDv3 only: explicitly exclude null or lower confidence; default preserves all official preferred pairs')
    p.add_argument('--raw')
    p.add_argument('--model')
    p.add_argument('--images')
    p.add_argument('--out')
    p.add_argument('--hpdv3', help='Optional HPDv3 directory for cross-dataset held-out exclusion')
    p.add_argument('--pickapic', help='Optional raw Pick-a-Pic directory for cross-dataset held-out exclusion')
    p.add_argument('--train-pairs', help='Training pair manifest for benchmark overlap exclusion')
    p.add_argument('--pickapic-test', help='Exported official-test-unique.jsonl for benchmarks')
    p.add_argument('--hpdv2', help='Optional official HPDv2 test.json for benchmarks')
    p.add_argument('--revision', help='Required provenance for manually downloaded Parquet files')
    p.add_argument('--workers', type=int, default=8, help='Pick-a-Pic image I/O threads (default: 8)')
    p.add_argument('--batch-size', type=int, default=256, help='Pick-a-Pic rows per atomic checkpoint (default: 256)')
    p.add_argument('--resume', action='store_true', help='Resume Pick-a-Pic from committed batches')
    p.add_argument('--adopt-legacy', action='store_true', help='After stopping the old exporter, rebuild metadata using its existing images')
    a = p.parse_args()
    if a.workers < 1 or a.batch_size < 1:p.error('--workers and --batch-size must be positive')
    if a.action != 'pickapic' and (a.resume or a.adopt_legacy or a.workers != 8 or a.batch_size != 256):
        p.error('--workers/--batch-size/--resume/--adopt-legacy apply only to pickapic')
    if a.family=='unfiltered' and a.action!='hpdv3':p.error('unfiltered is a data-only HPDv3 export for a separate SD3 trainer')
    if a.min_confidence is not None and (a.action!='hpdv3' or not 0<=a.min_confidence<=1):p.error('--min-confidence must be in [0,1] and used with hpdv3')
    dataset='hpdv3' if a.action=='hpdv3' else a.dataset
    variable=('HPDV3' if dataset=='hpdv3' else 'PICKAPIC')+'_'+a.family.upper()+'_PREPARED'
    hpdv3_raw=Path(a.hpdv3) if a.hpdv3 else env_path('HPDV3_RAW')
    hpd_test=hpdv3_raw/'test.json'
    if not hpd_test.is_file():
        if a.hpdv3 or a.action=='hpdv3':
            raise FileNotFoundError(hpd_test)
        # A planned resource path is not evidence that this optional dataset exists.
        hpdv3_raw=hpd_test=None
    if a.action == 'pickapic':
        prepare_pickapic(a.raw or env_path('PICKAPIC_RAW'), a.out or env_path(variable),
                        a.images or env_path('PICKAPIC_IMAGES'), family=a.family,
                        model=a.model or env_path('SD15_MODEL' if a.family == 'sd15' else 'SDXL_MODEL'),
                        hpdv3_test=hpd_test, revision=a.revision, workers=a.workers,
                        batch_size=a.batch_size, resume=a.resume, adopt_legacy=a.adopt_legacy)
    elif a.action=='hpdv3':
        pickapic_raw=a.pickapic or os.environ.get('PICKAPIC_RAW')
        prepare_hpdv3(a.raw or hpdv3_raw or env_path('HPDV3_RAW'),a.out or env_path(variable),family=a.family,
                     model=a.model or (None if a.family=='unfiltered' else env_path('SD15_MODEL' if a.family=='sd15' else 'SDXL_MODEL')),
                     pickapic_raw=pickapic_raw,revision=a.revision,min_confidence=a.min_confidence)
    else:
        train_pairs=Path(a.train_pairs) if a.train_pairs else env_path(variable)/'pairs.jsonl'
        pick_test=Path(a.pickapic_test) if a.pickapic_test else env_path('PICKAPIC_SD15_PREPARED' if a.family=='sd15' else 'PICKAPIC_SDXL_PREPARED')/'official-test-unique.jsonl'
        old_test=Path(a.hpdv2) if a.hpdv2 else Path(os.environ['HPDV2_RAW'])/'test.json' if os.environ.get('HPDV2_RAW') else None
        prepare_panels(pick_test, old_test if old_test and old_test.is_file() else None,
                       a.out or env_path('SCD_EVALUATION_ROOT'), train_pairs=train_pairs,hpdv3=hpd_test)


if __name__ == '__main__':
    main()
