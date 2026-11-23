"""Paired qualitative panels, checkpoint montages and a shallow portable figure bundle."""
import argparse
from collections import defaultdict
import copy
import html
import os
from pathlib import Path
import shutil
import sys
import textwrap

from self_correct_dpo.common import digest,read_json,read_jsonl,sha256,load_suite,verify_images,output_lock
from self_correct_dpo.resources import CONFIG_DIR,load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.evaluation.appendix_generation import resolve_adapter,noise_runtime,payload_jsonl,preflight_noise_runtime
from self_correct_dpo.evaluation.multiseed_generation import load_plan,checked_json,ROOT
from self_correct_dpo.evaluation.multiseed_history import read_source,protocol,pixels_present
from self_correct_dpo.evaluation.experiment_catalog import validate_method,model_spec
from self_correct_dpo.evaluation.suite import freeze_json
from self_correct_dpo.evaluation.appendix_progress import run_stage,run_workers

LABELS={'base':'Base','sft':'SFT','kto':'KTO','dpo':'DPO','linear_dpo':'LinearDPO','dspo':'DSPO','dmpo':'DMPO',
        'stagewise_k6':'DPO+MVPO','stagewise_dspo_k6':'DSPO+MVPO','stagewise_dmpo_k6':'DMPO+MVPO',
        'stagewise_k4':'Stage MVPO K4','global_k4':'Global MVPO K4','stagewise_k2':'Stage MVPO K2',
        'global_k2':'Global MVPO K2','lossavg_k2':'LossAvg K2','lossavg_k4':'LossAvg K4'}
PROMPTS=CONFIG_DIR/'appendix-prompts'


def requests(spec,family):
    panels={'parti':load_suite(PROMPTS/'parti-display-12.jsonl'),'art':load_suite(PROMPTS/'art-100.jsonl'),
            'complex':load_suite(PROMPTS/'complex-30.jsonl')}
    jobs={}
    def add(method,step,seed,panel):
        key=(method,step,seed);j=jobs.setdefault(key,dict(name=f'{method}-{step}-s{seed}',method=method,step=step,seed=seed,prompts={},panels=[]))
        j['panels'].append(panel)
        for p in panels[panel]:j['prompts'][(p['scene_id'],p['variant_id'])]=p
    if family in ('sd15','sdxl'):
        seed=spec['art_seeds'][family]
        for m,e in spec['methods'][family].items():
            for panel in ('parti','art'):add(m,e['main_step'],seed,panel)
    else:
        for seed in spec['complex_seeds']:
            for m,e in spec['methods'][family].items():add(m,e['main_step'],seed,'complex')
    if family in spec['progress_methods']:
        seed=spec['art_seeds'].get(family,17)
        for m in spec['progress_methods'][family]:
            for step in spec['progress_steps'][family]:add(m if step else 'base',step,seed,'parti')
    return [dict(j,prompts=list(j['prompts'].values()),panels=sorted(set(j['panels']))) for j in jobs.values()],panels


def prepare(output,family):
    core,panels=load_plan(output/family);spec=read_json(CONFIG_DIR/'appendix-figures.json');jobs,display=requests(spec,family)
    curve={(p['scene_id'],p['variant_id']) for p in panels['curve']}
    if any((p['scene_id'],p['variant_id']) not in curve for p in display['parti']):raise ValueError('Display panel must be inside frozen curve panel')
    registry=load_resources();out=output/'work-figures'/family;out.mkdir(parents=True,exist_ok=True)
    exp={e['method']:e for e in core['config']['experiments'] if e['family']==family};exp.update(spec['methods'][family])
    for job in jobs:
        job['shards']=2;job['adapter']=None;job['blocked']=None
        if job['method']=='base':continue
        try:
            e=exp[job['method']];run=Path(registry[e['run_resource']])/e['run'];info=read_json(run/'training.json')
            validate_method(info['config'],job['method'],str(run))
            if model_spec(info['config']['model'])!=core['model'] or info['base_identity']!=core['base_identity']:raise ValueError('Figure model/base differs')
            job['adapter']=resolve_adapter(run,job['step'],e['run_id'],e['checkpoints'].get(str(job['step'])))
        except (OSError,ValueError,KeyError) as error:job['blocked']=str(error)
    from string import Template
    plan=dict(core_plan_id=core['plan_id'],family=family,spec=spec,jobs=jobs,display=display,
              history=[Template(p).substitute(registry) for p in spec['history_sources'][family]],
              input_sha256={str(p.relative_to(CONFIG_DIR)):sha256(p) for p in [CONFIG_DIR/'appendix-figures.json',*sorted(PROMPTS.glob('*.jsonl'))]})
    plan['plan_id']=digest(plan);freeze_json(out/'plan.json',plan)
    return dict(family=family,figure_slots=sum(len(j['prompts']) for j in jobs),blocked=[(j['name'],j['blocked']) for j in jobs if j['blocked']])


def figure_plan(output,family):
    core,_=load_plan(output/family);plan=checked_json(output/'work-figures'/family/'plan.json','plan_id')
    if plan['core_plan_id']!=core['plan_id']:raise ValueError('Figure/core plan differs')
    for name,h in plan['input_sha256'].items():
        if sha256(CONFIG_DIR/name)!=h:raise ValueError('Figure prompt/config inputs changed')
    return plan,core


def source_index(output,plan,core):
    paths={Path(p) for p in plan['history']};paths.update((output/plan['family']/'jobs').glob('*/shard-*/*/render/run.json'))
    needed={'base':None}
    for j in plan['jobs']:
        if j['adapter']:needed[j['adapter']['checkpoint_id']]=j['adapter']
    sources={};rejected=[]
    for path in sorted(paths):
        try:
            run=read_json(path);cp=(run['config'].get('aligned') or {}).get('revision') if run['arm']=='aligned' else 'base'
            if cp not in needed or run['model']!=core['model'] or run['base_identity']!=core['base_identity']:continue
            if protocol(run['config'])!=protocol({'model':core['model'],'sampling':core['sampling']}) or run['versions']!=core['render_versions']:continue
            if cp!='base' and (run['model_identity']!=needed[cp]['inventory'] or run['config']['aligned']['path']!=needed[cp]['path']):continue
            source=read_source(path.parent,{})
            sid=digest(str(path));sources[sid]=dict(render=str(path.parent),checkpoint=cp,images=list(source['images'].values()),
                files=source['files'],runtime=source['runtime'],truncated_prompts=run.get('truncated_prompts',[]))
        except (OSError,ValueError,KeyError) as error:rejected.append(dict(path=str(path),reason=str(error)))
    cache=dict(plan_id=plan['plan_id'],sources=sources,rejected=rejected);cache['cache_id']=digest(cache)
    atomic_json(output/'work-figures'/plan['family']/'sources.json',cache)


def task_root(output,family,job,index):return output/'work-figures'/family/'jobs'/job['name']/f'shard-{index}'
def selected(job,index):return sorted(job['prompts'],key=lambda p:(p['scene_id'],p['variant_id']))[index::2]


def results(output,plan,job,index):
    root=task_root(output,plan['family'],job,index)
    if not (root/'complete.json').exists():return None
    receipt=read_json(root/'complete.json')
    if receipt!={'plan_id':plan['plan_id'],'images_sha256':sha256(root/'images.jsonl')}:raise ValueError('Figure result receipt differs')
    rows=read_jsonl(root/'images.jsonl');expected={(p['scene_id'],p['variant_id'],job['seed']) for p in selected(job,index)}
    if len(rows)!=len(expected) or {(r['scene_id'],r['variant_id'],r['seed']) for r in rows}!=expected:raise ValueError('Figure coverage differs')
    return rows


def worker(output,family,name,index):
    plan,core=figure_plan(output,family);job=next(j for j in plan['jobs'] if j['name']==name)
    if job['blocked']:raise ValueError(job['blocked'])
    root=task_root(output,family,job,index);root.mkdir(parents=True,exist_ok=True)
    with output_lock(root):
        if results(output,plan,job,index) is not None:return
        cache=checked_json(output/'work-figures'/family/'sources.json','cache_id')
        if cache['plan_id']!=plan['plan_id']:raise ValueError('Figure source cache plan differs')
        cp=job['adapter']['checkpoint_id'] if job['adapter'] else 'base';wanted=selected(job,index);found={};verified=set()
        for sid,src in cache['sources'].items():
            if src['checkpoint']!=cp:continue
            lookup={(p['scene_id'],p['variant_id']):p for p in wanted}
            for row in src['images']:
                key=(row['scene_id'],row['variant_id']);p=lookup.get(key)
                if not p or key in found or row['seed']!=job['seed'] or row['prompt']!=p['prompt'] or row['tag']!=p['tag']:continue
                path=Path(src['render'])/row['image']
                if not pixels_present(str(path),row['image_sha256']):continue
                if sid not in verified:
                    if any(sha256(Path(f))!=h for f,h in src['files'].items()):raise ValueError('Historical figure source metadata changed')
                    verified.add(sid)
                found[key]=dict(row,image=str(path),generation_gpu=src['runtime']['gpu'],origin='reused',
                    source_render=src['render'],truncated_prompts=[t for t in src['truncated_prompts'] if t['prompt'] in (f"{row['scene_id']} / {row['variant_id']}",'negative_prompt')])
        missing=[p for p in wanted if (p['scene_id'],p['variant_id']) not in found]
        freeze_json(root/'selection.json',dict(reused=list(found.values()),missing=missing))
        if missing:
            sample=dict(core['sampling'],seeds=[job['seed']],prompt_truncation='truncate');config={'model':core['model'],'sampling':sample}
            arm='base'
            if job['adapter']:
                a=job['adapter']
                if resolve_adapter(Path(a['run']),job['step'],a['run_id'],a['checkpoint_id'])!=a:raise ValueError('Figure checkpoint changed')
                config['aligned']=dict(kind='peft',path=a['path'],revision=a['checkpoint_id']);arm='aligned'
            freeze_json(root/'render-config.json',config);payload_jsonl(root/'prompts.jsonl',missing);render=root/'render'
            if not (render/'complete.json').exists():
                cmd=[core['interpreters']['render'],'-m','self_correct_dpo.evaluation.render','--config',str(root/'render-config.json'),
                     '--prompts',str(root/'prompts.jsonl'),'--out',str(render),'--arm',arm,'--execution','standalone','--run-label','appendix-figures']
                if render.exists():cmd.append('--resume')
                run_stage(cmd,log=root/'generation.log',journal=render/'manifest.jsonl',total=len(missing),label='gallery generate')
            run,rows=verify_images(render)
            if (run['config']!=config or run['suite']!=missing or run['arm']!=arm or run['base_identity']!=core['base_identity']
                    or run['versions']!=core['render_versions'] or noise_runtime(run)!=core['noise_runtime']):raise ValueError('Figure render identity/runtime differs')
            if job['adapter'] and run['model_identity']!=job['adapter']['inventory']:raise ValueError('Figure adapter inventory differs')
            for row in rows:
                found[(row['scene_id'],row['variant_id'])]=dict(row,image=str(render/row['image']),generation_gpu=run['gpu'],origin='generated',
                    source_render=str(render),truncated_prompts=[t for t in run.get('truncated_prompts',[]) if t['prompt'] in (f"{row['scene_id']} / {row['variant_id']}",'negative_prompt')])
        rows=[dict(found[(p['scene_id'],p['variant_id'])],family=family,method=job['method'],step=job['step'],checkpoint_id=cp) for p in wanted]
        payload_jsonl(root/'images.jsonl',rows);atomic_json(root/'complete.json',dict(plan_id=plan['plan_id'],images_sha256=sha256(root/'images.jsonl')))


def run(output,family,gpus=None):
    plan,core=figure_plan(output,family);out=output/'work-figures'/family
    sys.path.insert(0,str(ROOT/'scripts'));from appendix_jobs import gpu_tokens
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    devices=gpu_tokens(gpus or env.get('SCD_EVAL_GPUS'),env,core['interpreters']['render'])
    if len(devices)!=2:raise ValueError('Figure task requires two allocated H100 GPUs')
    with output_lock(out/'run'):
        source_index(output,plan,core)
        todo=[(j,i) for j in plan['jobs'] if not j['blocked'] for i in range(2) if results(output,plan,j,i) is None]
        if todo:preflight_noise_runtime(core['interpreters']['render'],devices,out,core['noise_runtime'])
        print(f'{family}: {len(todo)} figure shards remaining; {sum(bool(j["blocked"]) for j in plan["jobs"])} blocked jobs',flush=True)
        for start in range(0,len(todo),2):
            commands=[];files={}
            for gpu,(j,i) in zip(devices,todo[start:start+2]):
                label=j['name']+'-'+str(i);files[label]=task_root(output,family,j,i)/'progress.json'
                commands.append((gpu,label,[sys.executable,str(ROOT/'scripts/appendix_figures.py'),'worker','--output',str(output),'--family',family,'--job',j['name'],'--index',str(i)]))
            run_workers(commands,out,env,files)
        return export(output,family)


def copy_pixel(row,dest):
    name=f"{row['family']}-{row['method']}-{row['step']}-s{row['seed']}-{digest([row['scene_id'],row['variant_id']])[:16]}.png"
    target=dest/'images'/name;target.parent.mkdir(exist_ok=True)
    if not target.exists():
        if sha256(row['image'])!=row['image_sha256']:raise ValueError('Source PNG changed')
        tmp=target.with_suffix('.tmp');shutil.copyfile(row['image'],tmp)
        if sha256(tmp)!=row['image_sha256']:raise ValueError('PNG changed while copying')
        tmp.replace(target)
    elif sha256(target)!=row['image_sha256']:raise ValueError('Exported PNG changed')
    return dict(row,export_image='images/'+name)


def montage(dest,name,columns,rows):
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    try:
        from matplotlib.font_manager import findfont
        font=ImageFont.truetype(findfont('DejaVu Sans'),16);title=ImageFont.truetype(findfont('DejaVu Sans'),17)
    except (OSError,ImportError):font=title=ImageFont.load_default()
    cell=248;left=350;height=300
    canvas=Image.new('RGB',(left+cell*len(columns),65+height*len(rows)), 'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,10),'Prompt / method',fill='black',font=title)
    for i,col in enumerate(columns):draw.multiline_text((left+i*cell+5,8),str(col),fill='black',font=title,spacing=2)
    for y,(caption,images) in enumerate(rows):
        top=65+y*height;draw.multiline_text((12,top+4),'\n'.join(textwrap.wrap(caption,37)),fill='black',font=font,spacing=4)
        for x,row in enumerate(images):
            pos=(left+x*cell,top)
            if row:
                with Image.open(dest/row['export_image']) as image:
                    tile=ImageOps.contain(image.convert('RGB'),(cell-8,cell-8));canvas.paste(tile,(pos[0]+(cell-tile.width)//2,pos[1]+(cell-tile.height)//2))
                draw.text((pos[0]+4,top+cell),f"step {row['step']} / seed {row['seed']}",fill='black',font=font)
                gpu='H100' if 'H100' in row['generation_gpu'] else row['generation_gpu'].replace('NVIDIA GeForce RTX ','')
                draw.text((pos[0]+4,top+cell+22),gpu+(' / token truncation' if row.get('truncated_prompts') else ''),fill='#555555',font=font)
            else:
                draw.rectangle((pos[0]+4,top+4,pos[0]+cell-4,top+cell-4),fill='#eeeeee');draw.text((pos[0]+20,top+100),'Missing / blocked',fill='black',font=font)
    canvas.save(dest/(name+'.png'));canvas.save(dest/(name+'.pdf'),resolution=140)
    return name+'.png'


def export(output,family):
    dest=output/'figures';dest.mkdir(exist_ok=True)
    with output_lock(dest/(family+'-export')):return _export(output,family)


def _export(output,family):
    plan,core=figure_plan(output,family);dest=output/'figures';dest.mkdir(exist_ok=True);data=[];status=[]
    for j in plan['jobs']:
        for i in range(2):
            rows=results(output,plan,j,i) if not j['blocked'] else None
            status.append(dict(job=j['name'],shard=i,status='complete' if rows is not None else 'missing',reason=j['blocked']))
            if rows is not None:data.extend(copy_pixel(r,dest) for r in rows)
    payload_jsonl(dest/(family+'-image-data.jsonl'),data);atomic_json(dest/(family+'-status.json'),status)
    atomic_json(dest/(family+'-figure-protocol.json'),plan)
    lookup={(r['method'],r['step'],r['scene_id'],r['seed']):r for r in data};boards=[];spec=plan['spec'];steps={m:e['main_step'] for m,e in spec['methods'][family].items()}
    panels=['parti','art'] if family in ('sd15','sdxl') else ['complex']
    for panel in panels:
        seeds=[spec['art_seeds'][family]] if panel!='complex' else spec['complex_seeds']
        prompts=plan['display'][panel]
        for seed in seeds:
            for start in range(0,len(prompts),3):
                rows=[(p['prompt'],[lookup.get((m,step,p['scene_id'],seed)) for m,step in steps.items()]) for p in prompts[start:start+3]]
                name=f'{family}-{panel}-s{seed}-{start//3+1:02d}'
                boards.append(dict(panel=panel,image=montage(dest,name,[LABELS[m] for m in steps],rows),seed=seed))
    if family in spec['progress_methods']:
        panels=plan['display']['parti'];seed=spec['art_seeds'].get(family,17);grid=[]
        for p in panels:
            for m in spec['progress_methods'][family]:
                grid.append((LABELS[m]+' — '+p['prompt'],[lookup.get((m if step else 'base',step,p['scene_id'],seed)) for step in spec['progress_steps'][family]]))
        for start in range(0,len(grid),3):
            name=f'{family}-progress-{start//3+1:02d}';boards.append(dict(panel='progress',image=montage(dest,name,spec['progress_steps'][family],grid[start:start+3]),seed=seed))
    # Quantitative files stay flat next to image montages; no model tensors or all benchmark PNGs copied.
    for path in (output/family/'report').glob('*'):
        if path.is_file() and path.suffix in ('.csv','.json','.jsonl','.png','.pdf'):shutil.copyfile(path,dest/(family+'-quant-'+path.name))
    for p in PROMPTS.iterdir():
        if p.is_file():
            tmp=dest/(family+'-'+p.name+'.tmp');shutil.copyfile(p,tmp);tmp.replace(dest/p.name)
    atomic_json(dest/(family+'-boards.json'),boards)
    with output_lock(dest/'index'):
        parts=['<!doctype html><meta charset="utf-8"><title>MVPO appendix figures</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}li{margin:8px}summary{font-weight:bold;margin:18px}</style>',
               '<h1>附录数据、曲线与拼图</h1><p>固定prompt与seed，完整展示所有方法；缺失保持空格。定性样例不替代全量统计。每图数据含checkpoint、原图哈希、生成GPU与截断记录。</p>']
        for f in ('sd15','sdxl','sd35_medium','zimage'):
            path=dest/(f+'-boards.json')
            if not path.exists():continue
            parts.append('<h2>'+f+'</h2><ul>')
            for p in sorted(dest.glob(f+'-*')):
                if p.suffix in ('.csv','.json','.jsonl'):parts.append(f'<li><a href="{p.name}">{html.escape(p.name)}</a></li>')
            parts.append('</ul>')
            charts=sorted(dest.glob(f+'-quant-*.png'))
            if charts:
                parts.append('<details><summary>Quantitative curves and win ratios</summary>')
                for chart in charts:
                    parts.append(f'<p><a href="{chart.name}">{chart.name}</a></p><img loading="lazy" src="{chart.name}">')
                parts.append('</details>')
            for panel in ('parti','art','complex','progress'):
                items=[r for r in read_json(path) if r['panel']==panel]
                if not items:continue
                parts.append(f'<details><summary>{panel}: {len(items)} pages</summary>')
                for r in items:parts.append(f'<p><a href="{r["image"]}">{r["image"]}</a></p><img loading="lazy" src="{r["image"]}">')
                parts.append('</details>')
        tmp=dest/'index.html.tmp';tmp.write_text('\n'.join(parts));tmp.replace(dest/'index.html')
    return dict(family=family,exported_images=len(data),boards=len(boards),index=str(dest/'index.html'),missing_shards=sum(r['status']!='complete' for r in status))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker','export','plan']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=['sd15','sdxl','sd35_medium','zimage']);p.add_argument('--gpus');p.add_argument('--job');p.add_argument('--index',type=int,choices=[0,1])
    a=p.parse_args();a.output=a.output.resolve()
    if a.action=='plan':
        spec=read_json(CONFIG_DIR/'appendix-figures.json');result={f:sum(len(j['prompts']) for j in requests(spec,f)[0]) for f in spec['methods']}
    else:
        if not a.family:p.error('--family required')
        if a.action=='worker':result=worker(a.output,a.family,a.job,a.index)
        elif a.action=='run':result=run(a.output,a.family,a.gpus)
        elif a.action=='prepare':
            with output_lock(a.output/'work-figures'/(a.family+'-prepare')):result=prepare(a.output,a.family)
        else:result=export(a.output,a.family)
    print(__import__('json').dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
