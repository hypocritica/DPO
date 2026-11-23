#!/usr/bin/env python3
"""Tasks 4/5: native SD1.5 DSPO/DMPO and their stagewise MVPO checkpoints."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from self_correct_dpo.common import digest, output_lock, read_json, sha256
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.runtime.distributed import subprocess_environment
from self_correct_dpo.evaluation.appendix_probe import load_checkpoint
from self_correct_dpo.evaluation.appendix_scan import fixed_conditions, validate_job
from gradient_experiment import checked_json, inspect
from gradient_scan import FIELDS, source_identity, shared_frozen, run_checkpoint, assert_checks, check_panel


def configuration(path,batch=None):
    config=read_json(path)
    if config['schema_version']!=2 or config['research_phase']!='exploration' or [t['task'] for t in config['tasks']] != [4,5]:
        raise ValueError('expected the frozen task 4/5 exploration definition')
    if batch is not None:
        config['view_batch_sizes']['sd15']=batch
    if config['view_batch_sizes']['sd15'] not in (1,2,4):
        raise ValueError('view batch must be 1, 2 or 4')
    fixed_conditions('sd15',config['indices'])
    if (any(type(config[k]) is not int or config[k]<1 for k in FIELDS) or config['source_blocks']<2
            or config['audit_blocks']<2 or config['audit_blocks']%2
            or any(config['source_views_per_block']%k for k in (1,2,4,6,8))):
        raise ValueError('invalid sampling')
    for task in config['tasks']:
        obj=task['objective']
        if obj != {4:'dspo',5:'dmpo'}[task['task']] or [m['method'] for m in task['methods']] != [obj,f'stagewise_{obj}_k6']:
            raise ValueError('task objective/methods differ')
        steps=[[c['step'] for c in m['checkpoints']] for m in task['methods']]
        if steps[0]!=steps[1] or steps[0]!=sorted(set(steps[0])) or any(type(i) is not int or i<1 for i in steps[0]):
            raise ValueError('both methods must have the same distinct positive checkpoint grid')
    return config


def entries(config,task_id=None):
    for task in config['tasks']:
        if task_id is not None and task['task']!=task_id:
            continue
        for method in task['methods']:
            for checkpoint in method['checkpoints']:
                yield task,method,checkpoint


def location(output,task,method,step,phase='formal'):
    return output/f"task-{task}"/'sd15'/method/f'step-{step:07d}'/phase


def plan(config,task):
    jobs=[dict(method=m['method'],step=c['step'],indices=config['indices']) for _,m,c in entries(config,task)]
    unit=config['pair_count']*(config['source_blocks']*config['source_views_per_block']+config['audit_blocks']*config['audit_views_per_block'])
    conditions=len(jobs)*len(config['indices'])
    components=3 if task==4 else 1
    return dict(task=task,checkpoints=jobs,conditions=conditions,pairs_per_condition=config['pair_count'],
                view_batch_size=config['view_batch_sizes']['sd15'],allocated_gpus=2,
                pair_view_observations=conditions*unit,raw_component_VJPs=conditions*unit*components,
                note='plus pilot/numerical checks; VJP counts are not GPU-hours or generated image counts')


def freeze(args,config,registry):
    sources=source_identity()
    sources['scripts/objective_scan.py']=sha256(ROOT/'scripts/objective_scan.py')
    signature=dict(config=config,sources=sources,resources=str(args.resources),
                   python=registry['MVPO_SD15_PROBE_PY'],
                   runs={m['resource']:registry[m['resource']] for _,m,_ in entries(config)})
    shared_frozen(args.output/'scan.json',signature)
    target=args.output/f'task-{args.task}'/'selection.json'
    if target.exists():
        selected=checked_json(target,'selection_id')
        if selected['signature']!=signature:
            raise ValueError('frozen native scan changed; preserve original code/settings or use new output')
        return selected
    jobs=[]
    for task,method,cp in entries(config,args.task):
        expected={**cp,'run_id':method['expected_run_id']}
        path=Path(registry[method['resource']])/'checkpoints'/f"step-{cp['step']:07d}"
        _,_,training,identity=load_checkpoint('sd15',path,args.resources,expected,method=method['method'])
        job=dict(schema_version=1,family='sd15',method=method['method'],research_phase='exploration',checkpoint=identity,
                 conditions=fixed_conditions('sd15',config['indices']),
                 rng_scope=dict(phase='exploration',family='sd15',pairs_id=training['pairs_id'],panel_seed=config['panel_seed']),
                 latent_seed=config['latent_seed'],excluded_pair_ids=[],view_batch_size=config['view_batch_sizes']['sd15'])
        job['job_id']=digest(job);jobs.append(validate_job(job,'sd15'))
    selected=dict(signature=signature,task=args.task,jobs=jobs)
    selected['selection_id']=digest(selected);atomic_json(target,selected)
    return selected


def status(output,config,task_id=None):
    result=[]
    for task,method,cp in entries(config,task_id):
        for phase in ('pilot','formal'):
            root=location(output,task['task'],method['method'],cp['step'],phase)
            state=inspect(root)
            result.append(dict(task=task['task'],method=method['method'],step=cp['step'],phase=phase,
                               ready=state.get('ready_to_collect',False),complete=len(state['complete']),
                               missing=len(state['missing']),failed=state['failed'],checks=state['checks']))
    return result


def report(args,config):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_rows=[]
    for task,method,cp in entries(config,args.task):
        name=method['method'];root=location(args.output,task['task'],name,cp['step'])
        if not inspect(root).get('ready_to_collect'):
            all_rows.extend(dict(task=task['task'],method=name,step=cp['step'],index=i,status='missing_or_failed',
                                 k=None,metric=None,estimate=None,low=None,high=None,audit_half_cosine=None)
                            for i in config['indices'])
            continue
        protocol,_=assert_checks(root)
        selection=checked_json(args.output/f"task-{task['task']}"/'selection.json','selection_id')
        job=next(j for j in selection['jobs'] if j['method']==name and j['checkpoint']['step']==cp['step'])
        expected={k:config[k] for k in FIELDS if k!='pair_count'}
        expected.update(pair_seed=config['panel_seed'],view_seed=config['view_seed'])
        if (selection['signature']['config']!=config or protocol['scan']!=job or len(protocol['pairs'])!=config['pair_count']
                or any(protocol['sampling'][k]!=v for k,v in expected.items())):
            raise ValueError('native result differs from frozen catalog/sampling')
        check_panel(args.output,'sd15',checked_json(root/'prepared.json','prepared_id'))
        summary_path=root/'report/summary.json'
        if not summary_path.exists() or read_json(summary_path)['protocol_id']!=protocol['protocol_id']:
            subprocess.run([sys.executable,str(ROOT/'scripts/appendix_gradient_report.py'),'--input',str(root),
                            '--repeats',str(args.repeats)],check=True)
        summary=read_json(summary_path)
        definitions={c['condition']:c for c in job['conditions']}
        for condition in summary['conditions']:
            audits=[p['audit_split']['direction_cosine'] for p in summary['pairs']
                    if p['condition']==condition['condition'] and p['audit_split']['direction_cosine'] is not None]
            for diff in condition['paired_differences']:
                for metric,value in diff['metrics'].items():
                    lo,hi=value.get('pair_bootstrap_95') or (None,None)
                    all_rows.append(dict(task=task['task'],method=name,step=cp['step'],
                        index=definitions[condition['condition']]['index_start'],status='measured',k=diff['k'],
                        metric=metric,estimate=value['estimate'],low=lo,high=hi,
                        audit_half_cosine=float(np.mean(audits)) if audits else None))
    dest=args.output/('report' if args.task is None else f'task-{args.task}/report')
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'scan.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    atomic_json(dest/'status.json',status(args.output,config,args.task))
    for task in config['tasks']:
        if args.task is not None and args.task!=task['task']:continue
        for metric in ('source_corrected_distance_squared','descent_cosine','signed_projection','stochastic_variance','audit_half_cosine'):
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            panels=[]
            for method in task['methods']:
                steps=[c['step'] for c in method['checkpoints']]
                data=np.full((len(steps),len(config['indices'])),np.nan)
                for row in all_rows:
                    if row['method']==method['method'] and row['k']==6 and (metric=='audit_half_cosine' or row['metric']==metric):
                        v=row['audit_half_cosine'] if metric=='audit_half_cosine' else row['estimate']
                        if v is not None:data[steps.index(row['step']),config['indices'].index(row['index'])]=v
                panels.append(data)
            finite=np.concatenate([d[np.isfinite(d)] for d in panels]);scale=max(float(np.max(np.abs(finite))) if finite.size else 1,1e-12)
            for ax,method,data in zip(axes,task['methods'],panels):
                art=ax.imshow(np.ma.masked_invalid(data),cmap='coolwarm',vmin=-scale,vmax=scale,aspect='auto')
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):ax.text(j,i,'N/A' if not np.isfinite(data[i,j]) else f'{data[i,j]:.3g}',ha='center',va='center',fontsize=8)
                ax.set(xticks=range(len(config['indices'])),xticklabels=config['indices'],
                       yticks=range(len(method['checkpoints'])),yticklabels=[c['step'] for c in method['checkpoints']],
                       xlabel='Scheduler index',ylabel='Training step',title=method['method'])
                fig.colorbar(art,ax=ax,shrink=.8)
            label='mean audit loss-gradient split-half cosine' if metric=='audit_half_cosine' else 'native MVPO minus native LossAvg, K=6'
            fig.suptitle(f"{task['objective'].upper()}: {metric.replace('_',' ')}\n{label}; finite audit reference; N/A = missing/undefined",fontsize=11)
            for ext in ('png','pdf'):fig.savefig(dest/f"{task['objective']}-{metric}.{ext}",dpi=160)
            plt.close(fig)
    print(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','run','status','report'))
    parser.add_argument('--task',type=int,choices=(4,5))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/appendix-objective-scan.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resources',type=Path,default=Path(os.environ.get('SCD_RESOURCES',ROOT/'configs/resources.json')))
    parser.add_argument('--sd15-view-batch-size',type=int)
    parser.add_argument('--gpus')
    parser.add_argument('--scratch-dir',type=Path,default=Path(os.environ.get('TMPDIR','/tmp')))
    parser.add_argument('--pilot-only',action='store_true')
    parser.add_argument('--repeats',type=int,default=2000)
    args=parser.parse_args();args.output=args.output.expanduser().resolve();args.resources=args.resources.expanduser().resolve()
    try:
        config=configuration(args.config,args.sd15_view_batch_size)
        if args.action=='plan':
            print(json.dumps([plan(config,t) for t in ([args.task] if args.task else [4,5])],indent=2));return
        if args.action in ('status','report'):
            config=read_json(args.output/'scan.json')['config']
            if args.action=='status':print(json.dumps(status(args.output,config,args.task),indent=2))
            else:report(args,config)
            return
        if args.task is None:parser.error('run requires --task 4 or 5')
        registry=load_resources(args.resources)
        env=dict(subprocess_environment(os.environ),PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        env.setdefault('OMP_NUM_THREADS','2')
        env['PYTHONPATH']=os.pathsep.join(filter(None,[str(ROOT/'src'),env.get('PYTHONPATH')]))
        with output_lock(args.output/f'task-{args.task}'/'execution'):
            selected=freeze(args,config,registry)
            for job in selected['jobs']:run_checkpoint(args,config,registry,job,env)
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as error:
        parser.exit(1,f'{error}\n')


if __name__=='__main__':main()
