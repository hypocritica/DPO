"""Score a verified image panel with one local model at a time."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys

from self_correct_dpo.common import digest, output_lock, read_json, read_jsonl, require_gpu_allocation, sha256, verify_images, write_json
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.config import load_config
from self_correct_dpo.evaluation.scorers import METRICS, load_scorer, scorer_identity


def aggregate(rows):
    if not rows or any(not math.isfinite(float(r['score'])) for r in rows):
        raise ValueError('Scores must be nonempty and finite')
    by_prompt, by_tag = defaultdict(list), defaultdict(list)
    for row in rows:
        by_prompt[(row['scene_id'],row['variant_id'])].append(float(row['score']))
        by_tag[row['tag']].append(float(row['score']))
    means = [statistics.mean(values) for values in by_prompt.values()]
    result = {'mean':statistics.mean(float(r['score']) for r in rows),
            'prompt_mean':statistics.mean(means), 'images':len(rows), 'prompts':len(means),
            'prompt_standard_error':statistics.stdev(means)/math.sqrt(len(means)) if len(means)>1 else 0.,
            'by_category':{tag:{'mean':statistics.mean(values),'images':len(values)} for tag,values in by_tag.items()}}
    if any('components' in row for row in rows):
        keys = set(rows[0].get('components', {}))
        if not keys or any(set(row.get('components', {})) != keys for row in rows):
            raise ValueError('Inconsistent structured reward components')
        result['components'] = {key: statistics.mean(float(row['components'][key]) for row in rows)
                                for key in sorted(keys)}
        result['aggregation'] = rows[0]['aggregation']
    return result


def score_value(value, metric):
    """Preserve UnifiedReward judgments; invalid model output is never a zero."""
    if not isinstance(value, dict):
        if metric == 'unifiedreward':
            raise ValueError('UnifiedReward must retain its three component judgments')
        score = float(value)
        if not math.isfinite(score):
            raise FloatingPointError('Scorer returned a nonfinite value')
        return score, {}
    if metric != 'unifiedreward':
        raise ValueError('Unexpected structured reward')
    components = value.get('components', {})
    if set(components) != {'alignment', 'coherence', 'style'}:
        raise ValueError('UnifiedReward requires alignment, coherence and style')
    if any(not math.isfinite(float(v)) or not 1 <= float(v) <= 5 for v in components.values()):
        raise ValueError('UnifiedReward component outside [1, 5]')
    score = float(value['score'])
    if not math.isfinite(score) or not math.isclose(score, statistics.mean(components.values()), rel_tol=1e-7):
        raise ValueError('UnifiedReward score differs from the declared component mean')
    if value.get('aggregation') != 'arithmetic_mean_of_alignment_coherence_style' or not isinstance(value.get('raw_output'), str):
        raise ValueError('Missing UnifiedReward aggregation or raw judgment')
    return score, {k: value[k] for k in ('components', 'raw_output', 'aggregation')}


def _score_panel(run_dir, out, metric, spec, *, batch_size=8, execution=None, scorer_factory=None, asset_id=None):
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    run, panel = verify_images(run_dir)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    assets = asset_id if asset_id is not None else scorer_identity(metric, spec)
    versions = {}
    for name in ('torch','transformers','Pillow','torchvision','timm',*({'hpsv3':('hpsv3','trl','qwen-vl-utils'),
                 'image_reward':('image-reward','fairscale'),
                 'unifiedreward':('qwen-vl-utils','accelerate')}.get(metric,()))):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'missing'
    identity = {'run_id':run['run_id'], 'manifest_sha256':sha256(run_dir/'manifest.jsonl'),
                'scorer':assets, 'versions':versions, 'python':sys.executable, 'code_sha256':sha256(__file__)}
    meta = {'identity':identity, 'evaluation_id':digest(identity)}
    if out.exists():
        if read_json(out/'identity.json') != meta:
            raise ValueError('Cannot resume scoring with changed images, scorer or implementation')
        if (out/'complete.json').exists():
            complete = read_json(out/'complete.json')
            if complete['scores_sha256'] != sha256(out/'scores.jsonl') or complete['summary_sha256'] != sha256(out/'summary.json'):
                raise ValueError('Completed scoring output checksum mismatch')
            return read_json(out/'summary.json')
    else:
        out.mkdir(parents=True)
        write_json(out/'identity.json',meta)
    journal = out/'scores.jsonl'
    # Repair only an uncommitted trailing line; committed values are checked below.
    if journal.exists():
        payload = journal.read_bytes()
        if payload and not payload.endswith(b'\n'):
            journal.write_bytes(payload[:payload.rfind(b'\n')+1])
    rows = read_jsonl(journal) if journal.exists() else []
    expected = {(r['scene_id'],r['variant_id'],r['seed']):r for r in panel}
    seen = set()
    for row in rows:
        key = (row['scene_id'],row['variant_id'],row['seed'])
        record_id = row.get('record_id')
        if (key not in expected or key in seen or record_id != digest({k:v for k,v in row.items() if k!='record_id'})
                or row.get('metric') != metric or row.get('evaluation_id') != meta['evaluation_id']):
            raise ValueError('Invalid or duplicate committed score')
        source = expected[key]
        if row['prompt'] != source['prompt'] or row['image_sha256'] != source['image_sha256'] or not math.isfinite(float(row['score'])):
            raise ValueError('Score/image identity changed')
        if metric == 'unifiedreward':
            score_value(row, metric)
        seen.add(key)
    remaining = [r for r in panel if (r['scene_id'],r['variant_id'],r['seed']) not in seen]
    if remaining:
        if scorer_factory is None:
            require_gpu_allocation(execution)
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('Scoring requires an allocated CUDA GPU')
        scorer = (scorer_factory or load_scorer)(metric,spec)
        from PIL import Image, ImageOps
        with open(journal,'a') as f:
            for start in range(0,len(remaining),batch_size):
                batch = remaining[start:start+batch_size]
                images = []
                for row in batch:
                    with Image.open(run_dir/row['image']) as im:
                        images.append(ImageOps.exif_transpose(im).convert('RGB'))
                values = scorer(images,[r['prompt'] for r in batch])
                if len(values) != len(batch):
                    raise FloatingPointError('Scorer returned missing or nonfinite values')
                for row,value in zip(batch,values):
                    numeric, details = score_value(value, metric)
                    result = {k:row[k] for k in ('scene_id','variant_id','seed','prompt','tag','image_sha256')}
                    result.update(score=numeric,metric=metric,evaluation_id=meta['evaluation_id'], **details)
                    result['record_id'] = digest(result)
                    f.write(json.dumps(result,allow_nan=False)+'\n')
                    rows.append(result)
                f.flush()
    summary = {'metric':metric,'evaluation_id':meta['evaluation_id'],
               'convention':assets.get('scale'),'version':assets.get('version'),**aggregate(rows)}
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'complete.json',{'scores_sha256':sha256(journal),'summary_sha256':sha256(out/'summary.json')})
    return summary


def score_panel(run_dir,out,metric,spec,**kwargs):
    with output_lock(str(out)+'.score'):
        return _score_panel(run_dir,out,metric,spec,**kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--scorers-config',required=True)
    p.add_argument('--metric',choices=METRICS,required=True)
    p.add_argument('--batch-size',type=int,help='Override scorer batch_size (default 8; optional large scorers use 1)')
    p.add_argument('--execution',choices=['slurm','standalone'],default='slurm')
    p.add_argument('--run-label')
    p.add_argument('--check-only', action='store_true', help='Validate images and local assets without loading the scorer')
    a=p.parse_args()
    spec=load_config(a.scorers_config,scorers=[a.metric])['scorers'][a.metric]
    if a.check_only:
        run, panel = verify_images(a.run)
        identity = scorer_identity(a.metric, spec)
        print(json.dumps({'metric': a.metric, 'images': len(panel), 'run_id': run['run_id'],
                          'scorer': identity, 'gpu_inference_checked': False}))
        return
    print(json.dumps(score_panel(a.run,a.out,a.metric,spec,batch_size=a.batch_size if a.batch_size is not None else spec.get('batch_size',8),
        execution={'mode':a.execution,'run_label':a.run_label})))


if __name__=='__main__': main()
