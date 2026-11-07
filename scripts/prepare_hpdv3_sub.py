"""Prepare HPDv3-sub without altering labels, paths or source metadata.

Metadata-only mode does not decode/hash images or mark data ready for training.
Full preparation verifies bytes and uses existing prompt/image component splits.
Run full preparation on allocated compute resources.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from self_correct_dpo.common import sha256, write_json, write_jsonl
from self_correct_dpo.data.pickapic import prompt_key
from self_correct_dpo.data.json_stream import object_records
from self_correct_dpo.data.preparation import atomic_jsonl

ALLOWED_MODELS = frozenset({'flux', 'kolors', 'sd3', 'hunyuan', 'real_images'})


def iter_selected(records, tests, counts):
    heldout = {prompt_key(r['prompt']) for r in tests}
    for index, row in enumerate(records):
        if row.get('model1') not in ALLOWED_MODELS or row.get('model2') not in ALLOWED_MODELS:
            counts['model_subset'] += 1
        elif prompt_key(row['prompt']) in heldout:
            counts['official_heldout_prompt'] += 1
        else:
            # Keep path1=preferred/path2=rejected and exact original records.
            yield {'source_index': index, 'record': row}


def select(records, tests):
    counts = Counter()
    selected = list(iter_selected(records, tests, counts))
    return selected, dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--revision', required=True)
    p.add_argument('--metadata-only', action='store_true')
    p.add_argument('--workers', type=int, default=4, help='Concurrent image verification reads')
    a = p.parse_args()
    if a.workers < 1:p.error('--workers must be positive')
    if a.out.exists(): raise FileExistsError(a.out)
    if a.metadata_only:
        records = object_records(a.raw / 'train.json')
        tests = json.loads((a.raw / 'test.json').read_text())
        exclusions = Counter()
        source_rows = selected_rows = 0
        def counted():
            nonlocal source_rows
            for record in records:
                source_rows += 1
                yield record
        def selection():
            nonlocal selected_rows
            for row in iter_selected(counted(), tests, exclusions):
                selected_rows += 1
                yield row
        a.out.mkdir(parents=True)
        atomic_jsonl(a.out / 'selection.jsonl', selection())
        write_json(a.out / 'filter.json', {'revision': a.revision, 'allowed_models': sorted(ALLOWED_MODELS),
            'source_rows': source_rows, 'selected_rows': selected_rows, 'exclusions': exclusions,
            'train_json_sha256': sha256(a.raw/'train.json'), 'test_json_sha256': sha256(a.raw/'test.json'),
            'selection_sha256': sha256(a.out/'selection.jsonl'), 'ready_for_training': False,
            'image_resolution_filter': None, 'label_policy': 'path1 preferred, path2 rejected',
            'note': 'Metadata selection only; full preparation may further remove heldout image overlaps and invalid rows.'})
    else:
        from prepare_downloaded_data import prepare_hpdv3
        prepare_hpdv3(a.raw, a.out, family='unfiltered', revision=a.revision,
                      allowed_models=ALLOWED_MODELS, workers=a.workers)


if __name__ == '__main__': main()
