"""Snapshot a few already-extracted human pairs while full HPDv3 preparation runs.

Debug data only: excludes official test captions/UIDs, but does not claim the
full official-test image-hash audit. Never reads archives or writes source data.
"""
import argparse
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from self_correct_dpo.common import digest, output_lock, sha256
from self_correct_dpo.data.json_stream import object_records
from self_correct_dpo.data.pickapic import prompt_key
from self_correct_dpo.data.preparation import atomic_json, atomic_jsonl
from self_correct_dpo.data.validation import load_training_pairs
from prepare_hpdv3_sub import ALLOWED_MODELS
from prepare_downloaded_data import prepare_hpdv3


def completed_image_names(images):
    """List names once, avoiding per-row remote stat calls during a full scan."""
    root = images/'images'
    if not root.is_dir():
        raise FileNotFoundError(f'Expected extracted image directory: {root}; --images must be its parent')
    names = set()
    for directory, subdirs, files in os.walk(root, followlinks=False):
        for name in files:
            if not name.endswith(('.partial','.incomplete')):
                names.add((Path(directory)/name).relative_to(images).as_posix())
    return names


def prepare(raw, images, out, *, count=128, max_mib=256, max_scan=0):
    from PIL import Image
    raw, images, out = (Path(p).resolve() for p in (raw, images, out))
    if not 32 <= count <= 512 or not 1 <= max_mib <= 512 or max_scan < 0 or (max_scan and max_scan < count):
        raise ValueError('Use 32..512 pairs, 1..512 MiB and max-scan=0 (unlimited) or >= count')
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
        print(f'[debug] Indexing completed image names under {images}/images', flush=True)
        available = completed_image_names(images)
        print(json.dumps({'phase':'debug_available_images','images_root':str(images),
                          'completed_files':len(available),'example_files':sorted(available)[:3]}), flush=True)
        tests = json.loads((raw/'test.json').read_text())
        blocked = {prompt_key(r['prompt']) for r in tests}
        blocked_uids = {r[k] for r in tests for k in ('path1','path2')}
        selected, seen_prompts, seen_uids, skips = [], set(), set(), Counter()
        total = scanned = 0
        availability = Counter(); examples = []
        stop_reason = 'end_of_annotations'
        last_log = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='.'+out.name+'.building-', dir=out.parent) as temporary:
            stage = Path(temporary); snapshot = stage/'snapshot'; snapshot.mkdir()
            for index, row in enumerate(object_records(raw/'train.json')):
                if max_scan and index >= max_scan:
                    stop_reason = 'scan_limit'; break
                scanned = index+1
                now = time.monotonic()
                if scanned % 10000 == 0 or now-last_log >= 15:
                    print(json.dumps({'phase':'debug_scan','scanned_rows':scanned,'selected':len(selected),
                                      'skips':dict(skips),'availability':dict(availability)}), flush=True)
                    last_log = now
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
                    sources.append(images/relative)
                present = [Path(name).as_posix() in available for name in names]
                state = ('both_images' if all(present) else 'winner_only' if present[0]
                         else 'loser_only' if present[1] else 'neither_image')
                availability[state] += 1
                if len(examples) < 3:
                    examples.append({'source_index':index,'paths':names,'available':present})
                if not all(present):
                    skips['not_extracted_yet'] += 1; continue
                sources = [path.resolve() for path in sources]
                if any(not path.is_relative_to(images) for path in sources):
                    raise ValueError('Image escapes extraction root')
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
                    stop_reason = 'target_reached'; break
            scan = {'kind':'flow_debug_availability','raw':str(raw),'images_root':str(images),
                    'completed_files_at_start':len(available),'scanned_rows':scanned,
                    'max_scan':max_scan,'stop_reason':stop_reason,'selected_pairs':len(selected),
                    'requested_pairs':count,'skips':dict(skips),'availability':dict(availability),
                    'examples':examples,'ready_for_training':False}
            scan_path = out.with_name(out.name+'.availability.json')
            atomic_json(scan_path,scan)
            print('[debug-availability] '+json.dumps(scan), flush=True)
            if len(selected) < count:
                raise RuntimeError(f'Only {len(selected)}/{count} complete pairs in {scanned} scanned rows '
                                   f'(stop={stop_reason}); inspect {scan_path}. '
                                   'No training snapshot was created; extraction may not have both sides yet.')
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
                      'scanned_rows':scanned,'availability_report':str(scan_path),'copied_bytes':total,'skips':dict(skips),
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
    parser.add_argument('--max-scan',type=int,default=0,help='0 streams until enough pairs or EOF; positive value limits scanned rows')
    args=parser.parse_args()
    prepare(args.raw,args.images,args.out,count=args.count,max_mib=args.max_mib,max_scan=args.max_scan)


if __name__=='__main__':main()
