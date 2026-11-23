"""Validate native evaluation journals and reuse individual prompt/seed slots."""
from functools import lru_cache
import copy
import math
import shutil
from pathlib import Path

from self_correct_dpo.common import digest, read_json, read_jsonl, sha256
from self_correct_dpo.evaluation import metrics as metrics_module
from self_correct_dpo.evaluation.appendix_generation import noise_runtime, payload_jsonl, prompt_key, image_key
from self_correct_dpo.evaluation.render import effective_sampling
from self_correct_dpo.evaluation.suite import safe_target, freeze_json
from self_correct_dpo.runtime.checkpoints import atomic_json


def protocol(config):
    # Per-image generators are reset to the seed, independent of packing/shards.
    return {k:v for k,v in effective_sampling(config).items() if k not in ('seeds','generation_batch_size','cpu_offload')}


def key_string(row):
    return digest(list(image_key(row)))


def read_source(render, scorers, score_versions=None):
    render=Path(render);run=read_json(render/'run.json')
    if run['run_id']!=digest({k:v for k,v in run.items() if k!='run_id'}) or run['suite_id']!=digest(run['suite']):
        raise ValueError('historical render identity checksum differs')
    if run['sampling']!=effective_sampling(run['config']):raise ValueError('sampling/config mismatch')
    expected={(*prompt_key(p),s):p for p in run['suite'] for s in run['sampling']['seeds']}
    images={};files={str(render/'run.json'):sha256(render/'run.json'),str(render/'manifest.jsonl'):sha256(render/'manifest.jsonl')}
    for row in read_jsonl(render/'manifest.jsonl'):
        key=image_key(row); p=expected.get(key)
        if (not p or key in images or row['image_id']!=digest({'run_id':run['run_id'],**{k:v for k,v in row.items() if k!='image_id'}})
                or any(row[k]!=p[k] for k in ('prompt','tag'))):raise ValueError('invalid historical image record')
        safe_target(render,row['image']);images[key]=row
    if (render/'complete.json').exists():
        receipt=read_json(render/'complete.json');files[str(render/'complete.json')]=sha256(render/'complete.json')
        if receipt!={'images':len(expected),'manifest_sha256':files[str(render/'manifest.jsonl')]} or len(images)!=len(expected):
            raise ValueError('historical completion receipt differs')
    scores={};rejected=[]
    for metric,scorer in scorers.items():
        target=render.parent/'scores'/metric
        if not (target/'identity.json').exists() or not (target/'scores.jsonl').exists():continue
        try:
            meta=read_json(target/'identity.json');identity=meta['identity']
            if (meta['evaluation_id']!=digest(identity) or identity['run_id']!=run['run_id']
                    or identity['manifest_sha256']!=files[str(render/'manifest.jsonl')]
                    or identity['scorer']!=scorer or identity['code_sha256']!=sha256(metrics_module.__file__)
                    or (score_versions is not None and identity['versions']!=score_versions[metric])):
                raise ValueError('scorer/version/manifest mismatch')
            valid={}
            for row in read_jsonl(target/'scores.jsonl'):
                key=image_key(row);image=images.get(key)
                if (not image or key in valid or row['metric']!=metric or row['evaluation_id']!=meta['evaluation_id']
                        or row['record_id']!=digest({k:v for k,v in row.items() if k!='record_id'})
                        or any(row[k]!=image[k] for k in ('prompt','tag','image_sha256'))
                        or not math.isfinite(float(row['score']))):raise ValueError('invalid historical score record')
                valid[key]=row
            scores[metric]=valid
            for name in ('identity.json','scores.jsonl'):files[str(target/name)]=sha256(target/name)
        except (OSError,ValueError,KeyError,TypeError) as error:
            rejected.append({'path':str(target),'reason':str(error)})
    return dict(render=str(render),run=run,images=images,scores=scores,files=files,rejected=rejected,runtime=noise_runtime(run))


@lru_cache(maxsize=32768)
def pixels_present(path, checksum):
    path=Path(path)
    return path.is_file() and sha256(path)==checksum


def choose_slots(rows, seed, metrics, candidates):
    """Scores are never mixed across different pixels; missing pixels trigger regeneration only if needed."""
    chosen=[];counts={'reuse_scores':0,'reuse_images':0,'generate':0}
    for p in rows:
        key=(*prompt_key(p),seed);options=[]
        for sid,source in candidates.items():
            row=source['images'].get(key)
            if not row or any(row[k]!=p[k] for k in ('prompt','tag')):continue
            available=[m for m in metrics if key in source['scores'].get(m,{})]
            image_ok=len(available)==len(metrics) or pixels_present(str(Path(source['render'])/row['image']),row['image_sha256'])
            if image_ok:options.append((len(available),sid,available))
        if options:
            _,sid,available=sorted(options,key=lambda v:(-v[0],v[1]))[0]
            state='reuse_scores' if len(available)==len(metrics) else 'reuse_images'
            chosen.append(dict(key=list(key),source=sid,metrics=available,state=state))
        else:
            state='generate';chosen.append(dict(key=list(key),source=None,metrics=[],state=state))
        counts[state]+=1
    return chosen,counts


def import_pixels(source, keys, destination):
    """Create a traceable read-only pixel subset for missing scores; no old files are changed."""
    destination=Path(destination);old=source['run'];keys=list(keys)
    if not keys or len({k[2] for k in keys})!=1:raise ValueError('import requires one seed')
    seed=keys[0][2]; lookup={prompt_key(p):p for p in old['suite']}
    suite=[lookup[k[:2]] for k in keys]
    run=copy.deepcopy(old);run.pop('run_id')
    run.update(suite=suite,suite_id=digest(suite),full_suite_id=digest(suite),suite_indices=list(range(len(suite))),
               shard={'index':0,'count':1,'assignment':'sorted-scene-round-robin-v1'},expected_images=len(keys),
               imported_from={'render':source['render'],'run_id':old['run_id'],'files':source['files']})
    run['sampling']['seeds']=[seed];run['config']['sampling']['seeds']=[seed];run['run_id']=digest(run)
    destination.mkdir(parents=True,exist_ok=True);freeze_json(destination/'run.json',run)
    records=[]
    for i,key in enumerate(keys):
        row=copy.deepcopy(source['images'][key]);original=Path(source['render'])/row['image']
        # Check live pixels again; the preparation cache is not a runtime integrity check.
        if not original.is_file() or sha256(original)!=row['image_sha256']:raise ValueError(f'Historical pixels changed: {original}')
        row.pop('image_id');row['image']=f'images/{i:05d}.png'
        path=destination/row['image'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.is_symlink() or sha256(path)!=row['image_sha256']:raise ValueError('Imported image changed')
        else:
            temporary=path.with_suffix('.tmp')
            shutil.copyfile(original,temporary)
            if sha256(temporary)!=row['image_sha256']:raise ValueError('Historical image changed during copy')
            temporary.replace(path)
        row['image_id']=digest({'run_id':run['run_id'],**row});records.append(row)
    payload_jsonl(destination/'manifest.jsonl',records)
    atomic_json(destination/'complete.json',{'images':len(records),'manifest_sha256':sha256(destination/'manifest.jsonl')})
    return run,records
