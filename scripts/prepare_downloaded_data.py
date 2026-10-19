"""Offline SLAI Pick-a-Pic/HPDv3 -> original images + audited pair JSONL.

Use a CPU worker/container with enough RAM and disk, not a cluster login node.
No GPU, network, image resize/re-encoding, or reward-based relabeling is used.
"""
import argparse
from collections import Counter
from hashlib import sha256 as hash_bytes
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from download_assets import env_path, parquet_split
from self_correct_dpo.benchmarks import prepare_panels, prompt_key
from self_correct_dpo.common import digest, sha256, write_json, write_jsonl
from self_correct_dpo.pickapic import adapt_rows


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


class ImagePool:
    def __init__(self, root):
        self.root = Path(root)
        self.verified = set()
        self.bytes_written = 0
        self.budget = 0

    def store(self, payload):
        from PIL import Image
        checksum = hash_bytes(payload).hexdigest()
        target = self.root / checksum[:2] / (checksum + '.img')
        if checksum in self.verified:
            return str(target), checksum
        if target.exists():
            if sha256(target) != checksum:
                raise ValueError(f'Existing image is corrupt: {target}')
        else:
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            self.root.mkdir(parents=True, exist_ok=True)
            if self.budget < len(payload):
                free = shutil.disk_usage(self.root).free
                if free < len(payload) + 1024**3:
                    raise RuntimeError('Image export needs more disk space plus 1 GiB margin')
                self.budget = min(free - 1024**3, 128 * 1024**2)
            target.parent.mkdir(exist_ok=True)
            # A crashed write leaves a .partial file, never a committed image identity.
            partial = target.with_suffix('.partial')
            with partial.open('wb') as f:
                f.write(payload)
            partial.replace(target)
            self.budget -= len(payload)
            self.bytes_written += len(payload)
        self.verified.add(checksum)
        return str(target), checksum


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


def prepare_pickapic(raw, out, image_root, *, family, model, hpdv2_test=None, hpdv3_test=None, revision=None, allowed=None):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    provenance = raw / '.scd-download.json'
    receipt = json.loads(provenance.read_text()) if provenance.exists() else None
    if receipt and not receipt.get('complete'):
        raise ValueError('Download is incomplete; resume it before preparation')
    revision = revision or (receipt['repo_id'] + '@' + receipt['revision'] if receipt else None)
    if not revision:
        raise ValueError('Supply --revision for manually downloaded data, or use download_assets.py')
    # Ignore unrelated leftover files: the download receipt is the exact sample inventory.
    files = sorted(raw / name for name in receipt['files'] if name.endswith('.parquet')) if receipt else sorted(raw.rglob('*.parquet'))
    if receipt:
        for file in files:
            if not file.is_file() or file.stat().st_size != receipt['files'][str(file.relative_to(raw))]:
                raise ValueError(f'Download inventory changed: {file}')
    training = [f for f in files if parquet_split(str(f.relative_to(raw))) == 'train']
    heldout = [f for f in files if parquet_split(str(f.relative_to(raw))) in ('test', 'test_unique', 'validation', 'validation_unique')]
    unique_test = [f for f in files if parquet_split(str(f.relative_to(raw))) == 'test_unique']
    if not training or not unique_test:
        raise ValueError('Need downloaded train and official test_unique Parquet files')
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
    allowed = allowed or tokenizer_predicate(Path(model), family)
    pool = ImagePool(Path(image_root).resolve())
    original, groups, exclusions = [], {}, []
    seen_ids = set()
    raw_count = 0
    # Out is created early so interrupted preparation remains visibly incomplete.
    out.mkdir(parents=True)
    write_json(out / 'preparing.json', {'source':str(raw), 'revision':revision, 'family':family})
    for record in parquet_records(training):
        raw_count += 1
        pair_id = str(record.get('ranking_id', ''))
        if not pair_id or pair_id in seen_ids:
            raise ValueError(f'Missing or duplicate source ranking_id: {pair_id}')
        seen_ids.add(pair_id)
        caption = record.get('caption')
        labels = [record.get('label_0'), record.get('label_1')]
        reason = None
        if not isinstance(caption, str) or not caption.strip(): reason = 'empty_caption'
        elif not record.get('has_label', True): reason = 'unlabelled'
        elif labels not in ([1, 0], [0, 1], [.5, .5]): reason = 'invalid_labels'
        elif prompt_key(caption) in blocked_prompts: reason = 'official_heldout_caption'
        elif not allowed(caption): reason = 'model_token_length'
        if reason:
            exclusions.append({'pair_id':pair_id, 'reason':reason})
            continue
        payloads = [image_bytes(record[f'jpg_{side}']) for side in (0, 1)]
        hashes = [hash_bytes(payload).hexdigest() for payload in payloads]
        uids = [str(record.get(f'image_{side}_uid') or hashes[side]) for side in (0, 1)]
        if any(('uid', uid) in blocked_images for uid in uids) or any(('sha', h) in blocked_images for h in hashes):
            exclusions.append({'pair_id':pair_id, 'reason':'official_heldout_image'})
            continue
        try:
            paths = [pool.store(payload)[0] for payload in payloads]
        except (OSError, ValueError) as exc:
            # A corrupt preexisting content-addressed image is an integrity failure.
            if isinstance(exc, ValueError): raise
            exclusions.append({'pair_id':pair_id, 'reason':'invalid_image'})
            continue
        group = 'caption:' + digest(prompt_key(caption))[:24]
        groups[caption] = group
        original.append({'pair_id':pair_id, 'caption':caption, 'label_0':labels[0], 'label_1':labels[1],
                         'image_0':paths[0], 'image_1':paths[1], 'image_0_uid':uids[0], 'image_1_uid':uids[1],
                         'split':'train', 'hashes':hashes})
        if raw_count % 10000 == 0:
            print(json.dumps({'scanned':raw_count, 'eligible':len(original), 'unique_images':len(pool.verified)}), flush=True)
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
    write_jsonl(out / 'pairs.jsonl', pairs)
    write_jsonl(out / 'excluded.jsonl', exclusions)
    # Captions only: official test_unique is distinct from the local test rows above.
    write_jsonl(out / 'official-test-unique.jsonl',
                ({'caption':r['caption']} for r in parquet_records(unique_test, metadata_only=True) if r.get('caption')))
    inventory = [{'file':str(f.relative_to(raw)), 'bytes':f.stat().st_size, 'mtime_ns':f.stat().st_mtime_ns} for f in files]
    write_json(out / 'audit.json', {**report, 'raw_train_rows':raw_count, 'family':family,
        'exclusions':dict(Counter(r['reason'] for r in exclusions)), 'source_inventory':inventory,
        'download_receipt':receipt, 'external_test_sha256':{str(p):sha256(p) for p in external_tests},
        'model':str(model), 'images_written_bytes':pool.bytes_written,
        'preparation_code_sha256':sha256(__file__),
        'model_download_receipt':json.loads((Path(model)/'.scd-download.json').read_text()) if (Path(model)/'.scd-download.json').is_file() else None,
        'split_policy':'Official validation/test excluded by normalized caption, image UID and SHA256; remaining train components split 90/5/5 with seed17. Local test is NOT official test_unique.',
        'semantic_isolation_limitation':'Exact normalized captions and shared images are isolated; unknown paraphrases are not manually audited.',
        'image_policy':'Original bytes, content-addressed shared files; RGB square resize happens only in the trainer.'})
    write_json(out / 'ready.json', {'pairs_sha256':sha256(out / 'pairs.jsonl'), 'splits':dict(counts), 'source_revision':revision})
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
    p.add_argument('--revision', help='Required provenance for manually downloaded Parquet files')
    a = p.parse_args()
    if a.family=='unfiltered' and a.action!='hpdv3':p.error('unfiltered is a data-only HPDv3 export for a separate SD3 trainer')
    if a.min_confidence is not None and (a.action!='hpdv3' or not 0<=a.min_confidence<=1):p.error('--min-confidence must be in [0,1] and used with hpdv3')
    dataset='hpdv3' if a.action=='hpdv3' else a.dataset
    variable=('HPDV3' if dataset=='hpdv3' else 'PICKAPIC')+'_'+a.family.upper()+'_PREPARED'
    prepared = env_path(variable)
    hpdv3_raw=env_path('HPDV3_RAW')
    hpd_test=hpdv3_raw/'test.json'
    if not hpd_test.is_file():raise FileNotFoundError('Download HPDv3 test.json before data preparation')
    if a.action == 'pickapic':
        prepare_pickapic(a.raw or env_path('PICKAPIC_RAW'), a.out or prepared,
                        a.images or env_path('PICKAPIC_IMAGES'), family=a.family,
                        model=a.model or env_path('SD15_MODEL' if a.family == 'sd15' else 'SDXL_MODEL'),
                        hpdv3_test=hpd_test, revision=a.revision)
    elif a.action=='hpdv3':
        prepare_hpdv3(a.raw or hpdv3_raw,a.out or prepared,family=a.family,
                     model=a.model or (None if a.family=='unfiltered' else env_path('SD15_MODEL' if a.family=='sd15' else 'SDXL_MODEL')),
                     pickapic_raw=env_path('PICKAPIC_RAW'),revision=a.revision,min_confidence=a.min_confidence)
    else:
        if not (prepared / 'ready.json').is_file():
            raise ValueError('Prepare Pick-a-Pic successfully first')
        pick_prepared=env_path('PICKAPIC_SD15_PREPARED' if a.family=='sd15' else 'PICKAPIC_SDXL_PREPARED')
        old_test=Path(os.environ['HPDV2_RAW'])/'test.json' if os.environ.get('HPDV2_RAW') else None
        prepare_panels(pick_prepared / 'official-test-unique.jsonl', old_test if old_test and old_test.is_file() else None,
                       a.out or env_path('SCD_EVALUATION_ROOT'), train_pairs=prepared / 'pairs.jsonl',hpdv3=hpd_test)


if __name__ == '__main__':
    main()
