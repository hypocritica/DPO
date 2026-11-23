"""Frozen four-backbone appendix generation with per-slot historical reuse."""
import argparse
from collections import defaultdict
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from self_correct_dpo.common import digest, load_suite, model_artifact_identity, output_lock, read_json, read_jsonl, sha256
from self_correct_dpo.config import load_config
from self_correct_dpo.resources import CONFIG_DIR, load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.evaluation.appendix_generation import (aligned_config, checked_render, checked_scores, noise_runtime,
    payload_jsonl, prompt_key, image_key, resolve_adapter, preflight_noise_runtime)
from self_correct_dpo.evaluation.experiment_catalog import validate_method, model_spec
from self_correct_dpo.evaluation.generation_inventory import discover, export_inventory
from self_correct_dpo.evaluation.multiseed_history import protocol, read_source, choose_slots, import_pixels
from self_correct_dpo.evaluation.render import effective_sampling
from self_correct_dpo.evaluation.scorers import scorer_identity
from self_correct_dpo.evaluation.suite import freeze_json, render_header, safe_target

PACKS={'A1':'sd15','A2':'sd15','F1':'sd15','E1':'sd15','E2':'sd15','E3':'sd15',
       'B':'sd35_medium','E4':'sd35_medium','C':'zimage','D1':'sdxl','D2':'sdxl','F2':'sdxl'}
ROOT=CONFIG_DIR.parent


def sources():
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'src/self_correct_dpo').rglob('*.py'))}


def checked_json(path, key):
    value=read_json(path)
    if value[key]!=digest({k:v for k,v in value.items() if k!=key}):raise ValueError(f'Checksum differs: {path}')
    return value


def configuration(path):
    config=read_json(path)
    if config['version']!=1 or config['full_count']!=1632 or config['curve_count']!=512:raise ValueError('Unexpected appendix design')
    names=set()
    for exp in config['experiments']:
        key=(exp['family'],exp['method'])
        if key in names or PACKS[exp['pack']]!=exp['family']:raise ValueError('Duplicate/mismatched experiment')
        names.add(key)
        steps=set([exp['main_step'],*exp['curve_steps']])
        if set(map(int,exp['checkpoints']))!=steps or any(s<=0 for s in steps):raise ValueError('Checkpoint grid differs')
        if exp['curve_steps'] and PACKS[exp['curve_pack']]!=exp['family']:raise ValueError('Curve pack differs')
    if len(names)!=18 or config['shards'] not in (2,4,8):raise ValueError('Expected 18 configurations and 2/4/8 shards')
    for spec in config['families'].values():
        if spec.get('shards',config['shards']) not in (2,4,8):raise ValueError('Invalid family shard count')
        if len(spec['seeds'])!=3 or len(set(spec['seeds']))!=3 or any(type(s)!=int or s<0 for s in spec['seeds']):raise ValueError('Require three distinct seeds')
    return config


def jobs_for(config,family,seeds):
    jobs=[]
    for exp in config['experiments']:
        if exp['family']!=family:continue
        for seed in seeds:
            jobs.append(dict(name=f"{exp['method']}-{exp['main_step']}-s{seed}",method=exp['method'],step=exp['main_step'],
                             seed=seed,panel='full',pack=exp['pack'],metrics=config['families'][family]['metrics']))
        for step in exp['curve_steps']:
            if step==exp['main_step']:continue
            jobs.append(dict(name=f"{exp['method']}-{step}-s{seeds[0]}",method=exp['method'],step=step,seed=seeds[0],
                             panel='curve',pack=exp['curve_pack'],metrics=['pickscore','hpsv2' if family=='sd15' else 'hpsv3','image_reward']))
    return jobs


def static_plan(config):
    result=[]
    for family,spec in config['families'].items():
        for job in jobs_for(config,family,spec['seeds']):
            result.append({**job,'family':family,'images':config['full_count'] if job['panel']=='full' else config['curve_count']})
    totals={pack:sum(j['images'] for j in result if j['pack']==pack) for pack in PACKS}
    return dict(main_configurations=18,main_images=18*1632*3,unique_images=sum(totals.values()),packs=totals,jobs=result)


def versions(python,metric):
    names=['torch','transformers','Pillow','torchvision','timm']+{'hpsv3':['hpsv3','trl','qwen-vl-utils'],
        'image_reward':['image-reward','fairscale']}.get(metric,[])
    return package_versions(python,names)


def package_versions(python,names):
    code='import json,importlib.metadata as m\nnames='+repr(names)+'\nr={}\nfor n in names:\n try:r[n]=m.version(n)\n except m.PackageNotFoundError:r[n]="missing"\nprint(json.dumps(r))'
    return json.loads(subprocess.check_output([python,'-c',code],text=True))


def prepare(args,config,registry):
    family=args.family;out=args.output/family
    if (out/'plan.json').exists():
        plan,_=load_plan(out)
        options=dict(protocol_source=str(args.protocol_source) if args.protocol_source else None,batch=args.batch_size,
                     fresh=args.fresh,generation_gpu=getattr(args,'generation_gpu',None),history_roots=[str(p) for p in args.history_root],prompts=str(args.prompts) if args.prompts else None)
        if plan['config']!=config or plan['options']!=options:raise ValueError('Frozen preparation differs; use a new output root')
        return dict(plan_id=plan['plan_id'],status='already prepared',counts=plan['counts'])
    cfg=load_config(CONFIG_DIR/config['protocol_config'],benchmarks=['partiprompts'],resources=args.resources)
    spec=config['families'][family];sample=copy.deepcopy(cfg['protocols'][family]);seeds=list(spec['seeds'])
    reference=None
    if args.protocol_source:
        reference=checked_json(args.protocol_source,'run_id')
        if reference['model']['family']!=family or reference['arm']!='aligned':raise ValueError('Protocol source family/arm differs')
        noise_runtime(reference)  # Refuse unreviewed historical renderer semantics.
        sample=copy.deepcopy(reference['sampling']);s0=sample['seeds'][0]
        seeds=list(dict.fromkeys([s0,23,42,17,101]))[:3]
    elif family=='sd15':
        raise ValueError('SD1.5 paper/code protocol conflict: supply --protocol-source /actual/evaluation/render/run.json from the main-table evaluation')
    sample.update(seeds=seeds,generation_batch_size=args.batch_size or spec['generation_batch_size'],cpu_offload=spec['cpu_offload'])
    if type(sample['generation_batch_size'])!=int or sample['generation_batch_size']<1:raise ValueError('Invalid generation batch')
    prompt_path=Path(args.prompts or cfg['benchmarks']['partiprompts'])
    full=load_suite(prompt_path)
    if len(full)!=config['full_count'] or any(p['variant_id']!='canonical' for p in full):raise ValueError('Expected 1632 canonical PartiPrompts')
    selected=set(prompt_key(p) for p in sorted(full,key=lambda p:digest([config['subset_salt'],*prompt_key(p)]))[:config['curve_count']])
    panels={'full':full,'curve':[p for p in full if prompt_key(p) in selected]}
    py=registry['SCD_SD15_PY' if family in ('sd15','sdxl') else 'SCD_FLOW_PY']
    interpreters={'render':py,'legacy':registry['SCD_SD15_PY'],'hpsv3':registry['SCD_HPSV3_PY']}
    render_versions=package_versions(py,['torch','diffusers','transformers','safetensors','peft']+
        (['sentencepiece','accelerate','huggingface-hub','tokenizers'] if family in ('sd35_medium','zimage') else []))
    if reference and not args.fresh and reference['versions']!=render_versions:
        raise ValueError('Protocol source renderer package versions differ from selected interpreter; use matching environment or explicit --fresh')
    scorer_cfg=load_config(cfg['scorers_config'],scorers=spec['metrics'],resources=args.resources)
    scorers={m:scorer_identity(m,scorer_cfg['scorers'][m]) for m in spec['metrics']}
    score_versions={m:versions(interpreters['hpsv3' if m=='hpsv3' else 'legacy'],m) for m in spec['metrics']}
    model=base=None;adapters={};missing=[];training={}
    for exp in config['experiments']:
        if exp['family']!=family:continue
        run=Path(registry[exp['run_resource']])/exp['run']
        try:
            info=read_json(run/'training.json');validate_method(info['config'],exp['method'],str(run))
            current=model_spec(info['config']['model'])
            if model is None:model=current;base=model_artifact_identity(model)
            if current!=model or info['base_identity']!=base:raise ValueError('Model/base identity differs across methods')
            training[exp['method']]=dict(run=str(run),run_id=info['run_id'],pairs_id=info['pairs_id'])
            for step,cpid in exp['checkpoints'].items():
                try:adapters[(exp['method'],int(step))]=resolve_adapter(run,int(step),exp['run_id'],cpid)
                except (OSError,ValueError,KeyError) as error:missing.append(dict(method=exp['method'],step=int(step),reason=str(error)))
        except (OSError,ValueError,KeyError) as error:
            missing.extend(dict(method=exp['method'],step=int(s),reason=str(error)) for s in exp['checkpoints'])
    if model is None:raise ValueError('No usable training metadata/base model for this family')
    if reference and (model_spec(reference['model'])!=model or reference['base_identity']!=base):raise ValueError('Protocol source base differs')
    if reference:
        main_ids={e['checkpoints'][str(e['main_step'])] for e in config['experiments'] if e['family']==family}
        if reference['config']['aligned']['revision'] not in main_ids or reference['full_suite_id']!=digest(full):
            raise ValueError('Protocol source must be a main-table checkpoint on this full PartiPrompts panel (or its recorded shard)')
    sample=effective_sampling({'model':model,'sampling':sample})
    roots=args.history_root or [Path(registry['EVALUATIONS_ROOT']),Path(registry['RUNS_ROOT']),Path(registry['FLOW_RUNS_ROOT'])]
    files,issues=discover(roots,exclude=args.output)
    available={};audit=list(issues)
    wanted={prompt_key(p):(p['prompt'],p['tag']) for p in full}
    def relevant(source):
        return sum(key[2] in seeds and wanted.get(key[:2])==(row['prompt'],row['tag']) for key,row in source['images'].items())
    if not args.fresh:
        ids={a['checkpoint_id']:a for a in adapters.values()}
        for path in files:
            if path.name!='run.json':continue
            try:
                header=read_json(path);aligned=header.get('config',{}).get('aligned',{});adapter=ids.get(aligned.get('revision'))
                if not adapter or header.get('model',{}).get('family')!=family:continue
                if (header.get('arm')!='aligned' or model_spec(header['model'])!=model or header['base_identity']!=base
                        or header['model_identity']!=adapter['inventory'] or aligned!=aligned_config(model,sample,adapter)['aligned']
                        or protocol(header['config'])!=protocol({'model':model,'sampling':sample})):
                    raise ValueError('checkpoint/model/sampling protocol mismatch')
                source=read_source(path.parent,scorers,score_versions)
                if source['run']['versions']!=render_versions:raise ValueError('Renderer package versions differ')
                if not relevant(source):continue
                sid=digest(str(path.parent));available[sid]=source
                audit.extend(source['rejected'])
            except (OSError,ValueError,KeyError,TypeError) as error:audit.append(dict(path=str(path),reason=str(error)))
    runtimes={digest(s['runtime']):s['runtime'] for s in available.values()}
    if reference and not args.fresh:runtime=noise_runtime(reference)
    else:
        # Select purely by available slots, never by reward values.
        runtime=max(runtimes.values(),key=lambda rt:sum(relevant(s) for s in available.values() if s['runtime']==rt),default=None)
    available={sid:s for sid,s in available.items() if s['runtime']==runtime}
    history_runtime=runtime
    generation_gpu=getattr(args,'generation_gpu',None)
    if generation_gpu:
        runtime={'gpu':generation_gpu,**{k:render_versions[k] for k in ('torch','diffusers')}}
    cross_gpu=bool(history_runtime and runtime and history_runtime['gpu']!=runtime['gpu'] and available)
    out.mkdir(parents=True,exist_ok=True)
    jobs=jobs_for(config,family,seeds);counts={'reuse_scores':0,'reuse_images':0,'generate':0,'blocked':0};used=set()
    for job in jobs:
        adapter=adapters.get((job['method'],job['step']));job['adapter']=adapter
        job['shards']=spec.get('shards',config['shards']);job['slots']=f"slots/{job['name']}.jsonl"
        if not adapter:
            job['blocked']=next((m['reason'] for m in missing if m['method']==job['method'] and m['step']==job['step']),'checkpoint unavailable')
            slots=[dict(key=[*prompt_key(p),job['seed']],state='blocked',source=None,metrics=[]) for p in panels[job['panel']]]
            counts['blocked']+=len(slots)
        else:
            candidates={sid:s for sid,s in available.items() if s['run']['config']['aligned']['revision']==adapter['checkpoint_id']}
            slots,n=choose_slots(panels[job['panel']],job['seed'],job['metrics'],candidates)
            for k,v in n.items():counts[k]+=v
            used.update(s['source'] for s in slots if s['source'])
        target=out/job['slots'];target.parent.mkdir(exist_ok=True);payload_jsonl(target,slots);job['slots_sha256']=sha256(target)
    for name,rows in panels.items():payload_jsonl(out/f'{name}.jsonl',rows)
    freeze_json(out/'scorers.resolved.json',scorer_cfg)
    plan=dict(version=1,family=family,config=config,model=model,base_identity=base,sampling=sample,seeds=seeds,
        protocol_source={'path':str(args.protocol_source),'sha256':sha256(args.protocol_source)} if reference else {'path':str(CONFIG_DIR/config['protocol_config'])},
        options=dict(protocol_source=str(args.protocol_source) if args.protocol_source else None,batch=args.batch_size,fresh=args.fresh,
                     history_roots=[str(p) for p in args.history_root],prompts=str(args.prompts) if args.prompts else None,
                     generation_gpu=getattr(args,'generation_gpu',None)),
        panels={k:sha256(out/f'{k}.jsonl') for k in panels},jobs=jobs,counts=counts,missing=missing,training=training,
        sources={sid:dict(render=available[sid]['render'],files=available[sid]['files'],runtime=available[sid]['runtime']) for sid in sorted(used)},
        code=sources(),interpreters=interpreters,render_versions=render_versions,scorers=scorers,score_versions=score_versions,
        scorer_config_sha256=sha256(out/'scorers.resolved.json'),noise_runtime=runtime,history_noise_runtime=history_runtime,
        cross_gpu_reuse=cross_gpu,history_audit=audit)
    plan['plan_id']=digest(plan);freeze_json(out/'plan.json',plan)
    with (out/'reuse-summary.csv').open('w',newline='') as handle:
        fields=['pack','method','step','seed','prompts','reuse_scores','reuse_images','generate','blocked']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for job in jobs:
            slots=read_jsonl(out/job['slots'])
            writer.writerow({**{k:job[k] for k in fields[:4]},'prompts':len(slots),
                **{k:sum(slot['state']==k for slot in slots) for k in fields[5:]}})
    return dict(plan_id=plan['plan_id'],family=family,counts=counts,blocked_checkpoints=missing,noise_runtime=runtime,history_noise_runtime=history_runtime,cross_gpu_reuse=cross_gpu,seeds=seeds)


def load_plan(out,check_code=True):
    plan=checked_json(out/'plan.json','plan_id')
    if check_code and plan['code']!=sources():raise ValueError('Collector/renderer/scorer source changed; use pinned code or a new output')
    if sha256(out/'scorers.resolved.json')!=plan['scorer_config_sha256']:raise ValueError('Scorer configuration changed')
    panels={k:load_suite(out/f'{k}.jsonl') for k in plan['panels']}
    for name,checksum in plan['panels'].items():
        if sha256(out/f'{name}.jsonl')!=checksum:raise ValueError('Frozen prompts changed')
    for job in plan['jobs']:
        if sha256(out/job['slots'])!=job['slots_sha256']:raise ValueError('Frozen reuse slots changed')
    return plan,panels


def load_frozen_source(plan,sid):
    info=plan['sources'][sid]
    for filename,checksum in info['files'].items():
        if sha256(filename)!=checksum:raise ValueError(f'Historical metadata changed: {filename}; use a stable snapshot')
    source=read_source(Path(info['render']),plan['scorers'],plan['score_versions'])
    if source['files']!=info['files']:raise ValueError('Historical metadata changed during read')
    return source


def result_path(out,job,index):return out/'jobs'/job['name']/f'shard-{index}'


def selected_slots(out,job,index):
    slots=read_jsonl(out/job['slots'])
    # Sort independent of prompt source order, with a common assignment across methods.
    slots.sort(key=lambda s:tuple(map(str,s['key'])))
    return slots[index::job['shards']]


def completed(out,plan,job,index):
    dest=result_path(out,job,index)
    if not (dest/'complete.json').exists():return False
    receipt=read_json(dest/'complete.json')
    if receipt!={'plan_id':plan['plan_id'],'scores_sha256':sha256(dest/'scores.jsonl')}:
        raise ValueError(f'Completed shard changed: {dest}')
    rows=read_jsonl(dest/'scores.jsonl');wanted={tuple(s['key']) for s in selected_slots(out,job,index)}
    if len(rows)!=len(wanted) or {tuple(r['key']) for r in rows}!=wanted:raise ValueError('Result coverage mismatch')
    if any(set(r['scores'])!=set(job['metrics']) or any(not __import__('math').isfinite(v) for v in r['scores'].values()) for r in rows):
        raise ValueError('Result has incomplete/nonfinite scores')
    return True


def worker(out,plan,panels,job,index):
    if job.get('blocked'):raise ValueError(job['blocked'])
    dest=result_path(out,job,index);dest.mkdir(parents=True,exist_ok=True)
    with output_lock(dest):
        if completed(out,plan,job,index):return
        slots=selected_slots(out,job,index);lookup={prompt_key(p):p for p in panels[job['panel']]}
        loaded={sid:load_frozen_source(plan,sid) for sid in {s['source'] for s in slots if s['source']}}
        scores={tuple(s['key']):{} for s in slots};provenance={tuple(s['key']):{} for s in slots};groups=defaultdict(list)
        for slot in slots:
            key=tuple(slot['key']);sid=slot['source']
            if sid:
                for metric in slot['metrics']:
                    row=loaded[sid]['scores'][metric][key];scores[key][metric]=float(row['score'])
                    provenance[key][metric]=dict(source=loaded[sid]['render'],record_id=row['record_id'],image_sha256=row['image_sha256'])
            missing=tuple(m for m in job['metrics'] if m not in scores[key])
            if missing:groups[(sid,missing)].append(key)
        for (sid,metrics),keys in groups.items():
            group=dest/('reuse-'+digest([sid,metrics])[:16] if sid else 'new')
            render=group/'render'
            if sid:import_pixels(loaded[sid],keys,render)
            else:
                adapter=job['adapter']
                if resolve_adapter(Path(adapter['run']),job['step'],adapter['run_id'],adapter['checkpoint_id'])!=adapter:
                    raise ValueError('Checkpoint changed since prepare')
                config=aligned_config(plan['model'],dict(plan['sampling'],seeds=[job['seed']]),adapter)
                group.mkdir(parents=True,exist_ok=True);freeze_json(group/'config.json',config)
                prompts=[lookup[k[:2]] for k in keys];payload_jsonl(group/'prompts.jsonl',prompts)
                if not (render/'complete.json').exists():
                    command=[plan['interpreters']['render'],'-m','self_correct_dpo.evaluation.render','--config',str(group/'config.json'),
                        '--prompts',str(group/'prompts.jsonl'),'--out',str(render),'--arm','aligned','--execution','standalone','--run-label','mvpo-multiseed']
                    if render.exists():command.append('--resume')
                    subprocess.run(command,check=True)
                run,_,_=checked_render(render,config,plan['base_identity'],adapter,[prompts])
                runtime=plan['noise_runtime'] or read_json(out/'noise-runtime.json')['runtime']
                if noise_runtime(run)!=runtime or run['versions']!=plan['render_versions']:raise ValueError('Generated noise/package runtime differs')
            run,receipt=render_header(group);images={image_key(r):r for r in read_jsonl(render/'manifest.jsonl')}
            if set(images)!=set(keys):raise ValueError('Render does not cover requested slots')
            for metric in metrics:
                score_out=group/'scores'/metric
                if not (score_out/'complete.json').exists():
                    python=plan['interpreters']['hpsv3' if metric=='hpsv3' else 'legacy']
                    subprocess.run([python,'-m','self_correct_dpo.evaluation.metrics','--metric',metric,'--run',str(render),
                        '--out',str(score_out),'--scorers-config',str(out/'scorers.resolved.json'),
                        '--execution','standalone','--run-label','mvpo-multiseed-score'],check=True)
                rows,identity,_=checked_scores(group,metric,run,receipt,images,plan['scorers'][metric])
                if identity['identity']['versions']!=plan['score_versions'][metric]:raise ValueError('Scoring package versions changed')
                for row in rows:
                    key=image_key(row);scores[key][metric]=float(row['score'])
                    provenance[key][metric]=dict(source=str(group),record_id=row['record_id'],image_sha256=row['image_sha256'])
        rows=[dict(key=list(key),scores=scores[key],provenance=provenance[key]) for key in scores]
        if any(set(r['scores'])!=set(job['metrics']) for r in rows):raise ValueError('Missing requested score')
        payload_jsonl(dest/'scores.jsonl',rows)
        atomic_json(dest/'complete.json',dict(plan_id=plan['plan_id'],scores_sha256=sha256(dest/'scores.jsonl')))


def run_pack(args):
    sys.path.insert(0,str(ROOT/'scripts'))
    from appendix_jobs import gpu_tokens,run_probes
    out=args.output/PACKS[args.pack];plan,panels=load_plan(out)
    jobs=[j for j in plan['jobs'] if j['pack']==args.pack]
    blocked=[j for j in jobs if j.get('blocked')]
    if blocked:raise ValueError('Frozen missing checkpoints: '+json.dumps([(j['name'],j['blocked']) for j in blocked]))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    with output_lock(out/('pack-'+args.pack)):
        todo=[(j,i) for j in jobs for i in range(j['shards']) if not completed(out,plan,j,i)]
        if not todo:return {'pack':args.pack,'status':'complete'}
        devices=gpu_tokens(args.gpus or env.get('SCD_EVAL_GPUS'),env,plan['interpreters']['render'])
        if len(devices) not in (2,4,8):raise ValueError('Use 2, 4 or 8 allocated GPUs')
        # Preflight noise only when new pixels are needed; score-only reuse can use another GPU model.
        new=any(s['state']=='generate' for j,i in todo for s in selected_slots(out,j,i))
        if new:
            if package_versions(plan['interpreters']['render'],list(plan['render_versions']))!=plan['render_versions']:
                raise ValueError('Rendering environment changed after prepare')
            preflight_noise_runtime(plan['interpreters']['render'],devices,out,plan['noise_runtime'])
        for start in range(0,len(todo),len(devices)):
            commands=[]
            for gpu,(job,index) in zip(devices,todo[start:start+len(devices)]):
                command=[sys.executable,str(ROOT/'scripts/appendix_multiseed.py'),'worker','--output',str(args.output),
                         '--family',plan['family'],'--job',job['name'],'--index',str(index)]
                commands.append((gpu,job['name']+'-'+str(index),command))
            run_probes(commands,out/('pack-'+args.pack),env)
    return {'pack':args.pack,'status':'complete'}


def status(output):
    result=[]
    for family in ('sd15','sdxl','sd35_medium','zimage'):
        out=output/family
        if not (out/'plan.json').exists():result.append(dict(family=family,status='unprepared'));continue
        plan,_=load_plan(out,check_code=False)
        for job in plan['jobs']:
            result.append(dict(family=family,pack=job['pack'],job=job['name'],blocked=job.get('blocked'),
                complete_shards=sum(completed(out,plan,job,i) for i in range(job['shards'])),shards=job['shards']))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','plan','prepare','run','worker','status','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=CONFIG_DIR/'appendix-multiseed.json')
    parser.add_argument('--resources',type=Path)
    parser.add_argument('--family',choices=['sd15','sdxl','sd35_medium','zimage'])
    parser.add_argument('--pack',choices=list(PACKS));parser.add_argument('--job');parser.add_argument('--index',type=int)
    parser.add_argument('--history-root',type=Path,action='append',default=[])
    parser.add_argument('--protocol-source',type=Path);parser.add_argument('--prompts',type=Path)
    parser.add_argument('--generation-gpu',choices=['NVIDIA GeForce RTX 5090','NVIDIA H100 80GB HBM3'],
                        help='Explicit new-render GPU; preserves historical source GPU and reports cross-device reuse')
    parser.add_argument('--batch-size',type=int);parser.add_argument('--fresh',action='store_true');parser.add_argument('--gpus')
    parser.add_argument('--check-images',action='store_true');parser.add_argument('--max-file-mb',type=int,default=128)
    args=parser.parse_args(argv);args.output=args.output.expanduser().resolve()
    if args.resources:os.environ['SCD_RESOURCES']=str(args.resources.resolve())
    try:
        config=configuration(args.config);registry=load_resources(args.resources)
        if args.action=='inventory':
            roots=args.history_root or [Path(registry[k]) for k in ('EVALUATIONS_ROOT','RUNS_ROOT','FLOW_RUNS_ROOT')]
            result=export_inventory(roots,args.output,args.max_file_mb,args.check_images)
        elif args.action=='plan':result=static_plan(config)
        elif args.action=='prepare':
            if not args.family:parser.error('prepare requires --family')
            with output_lock(str(args.output)+'-'+args.family+'-prepare'):result=prepare(args,config,registry)
        elif args.action=='run':
            if not args.pack:parser.error('run requires --pack')
            result=run_pack(args)
        elif args.action=='worker':
            if not args.family or args.job is None or args.index is None:parser.error('worker requires family/job/index')
            out=args.output/args.family;plan,panels=load_plan(out);job=next(j for j in plan['jobs'] if j['name']==args.job)
            if not 0<=args.index<job['shards']:raise ValueError('Invalid shard index')
            worker(out,plan,panels,job,args.index);result={'job':job['name'],'shard':args.index,'status':'complete'}
        elif args.action=='status':result=status(args.output)
        else:
            from self_correct_dpo.evaluation.multiseed_report import report
            result=report(args.output,args.family)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,RuntimeError,StopIteration,subprocess.CalledProcessError) as error:parser.exit(1,f'{error}\n')
    return 0
