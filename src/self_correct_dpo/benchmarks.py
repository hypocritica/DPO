"""Offline benchmark inputs and frozen prompt panels; no torch dependency."""
import json
import csv
from pathlib import Path
import random
import unicodedata

from self_correct_dpo.common import digest, read_jsonl, sha256, write_json, write_jsonl

BENCHMARKS=('hpdv2','pickapic','partiprompts','hpdv3','imagereward')


def prompt_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def source_rows(path, *, split='test'):
    path = Path(path).resolve()
    if path.is_dir():
        from datasets import load_from_disk
        data = load_from_disk(str(path))
        if hasattr(data, 'keys'):
            key = 'test_unique' if 'test_unique' in data else split
            data = data[key]
        column = 'caption' if 'caption' in data.column_names else 'prompt'
        return [{'prompt':p} for p in data[column]]
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        names = pq.read_schema(path).names
        columns = [k for k in ('caption', 'prompt', 'category', 'split') if k in names]
        if not any(k in columns for k in ('caption', 'prompt')):
            raise ValueError('Parquet must contain caption or prompt')
        return pq.read_table(path, columns=columns).to_pylist()
    if path.suffix == '.arrow':
        from datasets import Dataset
        data = Dataset.from_file(str(path))
        column = 'caption' if 'caption' in data.column_names else 'prompt'
        return [{'prompt':p} for p in data[column]]
    if path.suffix == '.jsonl':
        return read_jsonl(path)
    if path.suffix.lower() in ('.tsv','.csv'):
        with path.open(newline='') as f:
            rows=[{k.strip().casefold():v for k,v in row.items() if k is not None}
                  for row in csv.DictReader(f,delimiter='\t' if path.suffix.lower()=='.tsv' else ',')]
        if rows and not any(key in rows[0] for key in ('prompt','caption')):
            raise ValueError('CSV/TSV must have a Prompt or caption column')
        return rows
    if path.suffix == '.json':
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if split in data:
                data = data[split]
            elif 'prompts' in data:
                data = data['prompts']
            else:
                raise ValueError('JSON must be a prompt list or contain the requested split/prompts')
        return [{'prompt':v} if isinstance(v, str) else v for v in data]
    return [{'prompt':line.strip()} for line in path.read_text().splitlines() if line.strip()]


def normalize_panel(rows, benchmark, *, exclude=()):
    blocked = {prompt_key(p) for p in exclude}
    seen, suite = set(), []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            row = {'prompt':row}
        if row.get('split') not in (None, 'test', 'test_unique', 'validation', 'validation_unique'):
            continue
        text = row.get('prompt', row.get('caption'))
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        key = prompt_key(text)
        if key in blocked or key in seen:
            continue
        seen.add(key)
        tag = row.get('category', row.get('tag', benchmark))
        tag = tag if isinstance(tag, str) and tag else benchmark
        suite.append({'scene_id':benchmark + '-' + digest(key)[:20],
            'variant_id':'canonical', 'prompt':text, 'tag':tag,
            'metadata':{'prompt':text, 'tag':tag}, 'suite_mode':'canonical-only-v1',
            'template_family':'canonical', 'transform_type':'canonical', 'source_index':index})
    if not suite:
        raise ValueError('No eligible unique held-out prompts')
    return suite


def select_monitor(suite, count=128, seed=17):
    if type(count) is not int or count < 1 or len(suite) < count:
        raise ValueError(f'Need {count} distinct held-out captions; found {len(suite)}')
    ordered = sorted(suite, key=lambda r:r['scene_id'])
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda r:r['scene_id'])


def prepare_panels(pickapic, hpdv2, out, *, train_pairs, parti=None, hpdv3=None,
                   imagereward=None, count=128, seed=17):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    # Caption-only scan: no training image reads or image hashes.
    train = [p['prompt'] for row in read_jsonl(train_pairs) if row.get('split') == 'train'
             for p in row['prompts']]
    sources={'pickapic':pickapic,'hpdv2':hpdv2,'partiprompts':parti,
             'hpdv3':hpdv3,'imagereward':imagereward}
    if not any(sources.values()):
        raise ValueError('Provide at least one benchmark input')
    panels={}
    for name,source in sources.items():
        if not source:continue
        output_name=name if name=='partiprompts' else name+'-test'
        panels[output_name]=normalize_panel(source_rows(source),output_name,exclude=train)
    if hpdv2 and len(panels['hpdv2-test']) != 400:
        raise ValueError(f"HPDv2 panel must have 400 unique held-out prompts; got {len(panels['hpdv2-test'])}. Verify test.json/test400 input and training overlap.")
    if pickapic:
        pick=panels['pickapic-test']
        monitor=select_monitor(pick,count,seed)
        monitor_ids={r['scene_id'] for r in monitor}
        panels['pickapic-monitor']=monitor
        panels['pickapic-test-remaining']=[r for r in pick if r['scene_id'] not in monitor_ids]
    out.mkdir(parents=True)
    for name, suite in panels.items():
        write_jsonl(out / (name + '.jsonl'), suite)
    write_json(out / 'selection.json', {'seed':seed, 'monitor_count':count if pickapic else None,
        'source_paths':{name:str(Path(p).resolve()) for name,p in sources.items() if p},
        'source_sha256':{str(p):sha256(p) for p in (*sources.values(),train_pairs) if p and Path(p).is_file()},
        'panels':{name:{'prompts':len(suite), 'sha256':sha256(out / (name + '.jsonl'))}
                  for name, suite in panels.items()},
        'limitation':'Inputs must be official test metadata/caption exports, never training data. Normalized exact captions are excluded from train. Unknown paraphrases need a separate semantic audit. Report actual post-filter counts and panel hashes; HPDv3/ImageReward have no assumed fixed prompt count.'})
