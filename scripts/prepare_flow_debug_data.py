"""Snapshot a few already-extracted human pairs while full HPDv3 preparation runs.

Debug data only: excludes official test captions/UIDs, but does not claim the
full official-test image-hash audit. Never reads archives or writes source data.
"""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from self_correct_dpo.common import digest, output_lock, sha256
from self_correct_dpo.data.json_stream import object_records
from self_correct_dpo.data.pickapic import prompt_key
from self_correct_dpo.data.preparation import atomic_json, atomic_jsonl
from self_correct_dpo.data.validation import load_training_pairs
from prepare_hpdv3_sub import ALLOWED_MODELS
from prepare_downloaded_data import prepare_hpdv3


def prepare(raw, images, out, *, count=128, max_mib=256, max_scan=200000):
    from PIL import Image
    raw, images, out = (Path(p).resolve() for p in (raw, images, out))
    if not 32 <= count <= 512 or not 1 <= max_mib <= 512 or max_scan < count:
        raise ValueError('Use 32..512 pairs, 1..512 MiB and max-scan >= count')
    if out == raw or out.is_relative_to(images) or out in raw.parents or out in images.parents:
        raise ValueError('Debug output must not replace/contain source directories')
    with output_lock(out):
        if out.exists():
            report = json.loads((out/'debug-data.json').read_text())
            if (report['kind'] != 'flow_debug_snapshot' or report['raw'] != str(raw)
                    or report['images_source'] != str(images) or report['requested_pairs'] != count
                    or report['pairs_sha256'] != sha256(out/'prepared/pairs.jsonl')):
                raise ValueError('Existing debug snapshot differs; choose a new --out')
            load_training_pairs(out/'prepared/pairs.jsonl')
            print(f'[debug] Reusing immutable snapshot: {out}', flush=True)
            return report
        tests = json.loads((raw/'test.json').read_text())
        blocked = {prompt_key(r['prompt']) for r in tests}
        blocked_uids = {r[k] for r in tests for k in ('path1','path2')}
        selected, seen_prompts, seen_uids, skips = [], set(), set(), Counter()
        total = 0
        with tempfile.TemporaryDirectory(prefix='.'+out.name+'.building-', dir=out.parent) as temporary:
            stage = Path(temporary); snapshot = stage/'snapshot'; snapshot.mkdir()
            for index, row in enumerate(object_records(raw/'train.json')):
                if index >= max_scan:
                    break
                if row.get('model1') not in ALLOWED_MODELS or row.get('model2') not in ALLOWED_MODELS:
                    skips['model_subset'] += 1; continue
                caption = row.get('prompt')
                if not isinstance(caption,str) or not caption.strip():
                    skips['empty_prompt'] += 1; continue
                key = prompt_key(caption); names = [row['path1'],row['path2']]
                if key in blocked or blocked_uids.intersection(names):
                    skips['official_test_caption_or_uid'] += 1; continue
                if key in seen_prompts or seen_uids.intersection(names):
                    skips['repeated_prompt_or_uid'] += 1; continue
                sources = []
                for name in names:
                    relative = Path(name)
                    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'images':
                        raise ValueError(f'Unsafe image path: {name}')
                    source = (images/relative).resolve()
                    if not source.is_relative_to(images):
                        raise ValueError(f'Image escapes extraction root: {name}')
                    sources.append(source)
                if not all(p.is_file() for p in sources):
                    skips['not_extracted_yet'] += 1; continue
                sizes = [p.stat().st_size for p in sources]
                if max(sizes) > 16*1024**2 or total+sum(sizes) > max_mib*1024**2:
                    raise RuntimeError('Debug snapshot exceeds byte budget; use fewer pairs or larger --max-mib (max 512)')
                for name, source in zip(names,sources):
                    before = source.stat(); payload = source.read_bytes(); after = source.stat()
                    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                        raise ValueError(f'Image changed during snapshot: {source}')
                    with Image.open(BytesIO(payload)) as image:
                        image.verify()
                    target = snapshot/name; target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(payload); total += len(payload)
                selected.append({'source_index':index,'record':row})
                seen_prompts.add(key); seen_uids.update(names)
                if len(selected)%16 == 0:
                    print(json.dumps({'phase':'debug_snapshot','selected':len(selected),
                                      'target':count,'scanned_rows':index+1}), flush=True)
                if len(selected) == count:
                    break
            if len(selected) < count:
                raise RuntimeError(f'Only {len(selected)}/{count} complete pairs available in scanned rows; let extraction progress and retry')
            atomic_json(snapshot/'train.json',[r['record'] for r in selected])
            # Held-out prompt/UID exclusion occurred above. This deliberately
            # omits the full official image-hash audit and cannot certify paper data.
            atomic_json(snapshot/'test.json',[])
            atomic_jsonl(stage/'selection.jsonl',selected)
            revision = 'debug-snapshot:'+digest(selected)
            prepare_hpdv3(snapshot,stage/'prepared',family='unfiltered',revision=revision,
                          allowed_models=ALLOWED_MODELS,workers=2)
            report = {'kind':'flow_debug_snapshot','ready_for_debug':True,'ready_for_paper':False,
                      'raw':str(raw),'images_source':str(images),'requested_pairs':count,
                      'scanned_rows':index+1,'copied_bytes':total,'skips':dict(skips),
                      'pairs_sha256':sha256(stage/'prepared/pairs.jsonl'),'revision':revision,
                      'official_test_caption_and_uid_excluded':True,
                      'official_test_image_hash_audited':False,
                      'note':'Independent snapshot from available images; never resume this run on the full dataset.'}
            atomic_json(stage/'debug-data.json',report)
            stage.rename(out)
        print(f'[debug] Ready: {out}/prepared/pairs.jsonl (debug only)', flush=True)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--max-mib',type=int,default=256)
    parser.add_argument('--max-scan',type=int,default=200000)
    args=parser.parse_args()
    prepare(args.raw,args.images,args.out,count=args.count,max_mib=args.max_mib,max_scan=args.max_scan)


if __name__=='__main__':main()
