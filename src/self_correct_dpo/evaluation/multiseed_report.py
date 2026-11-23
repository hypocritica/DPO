"""Complete-design prompt-cluster summaries; missing cells are never silently dropped."""
import csv
import fcntl
from pathlib import Path
import numpy as np

from self_correct_dpo.common import read_json,read_jsonl
from self_correct_dpo.evaluation.appendix_generation import prompt_key,payload_jsonl
from self_correct_dpo.evaluation.multiseed_generation import load_plan,completed,result_path
from self_correct_dpo.runtime.checkpoints import atomic_json


def interval(values,repeats):
    """A row is one prompt with every seed/method/timepoint kept paired."""
    values=np.asarray(values,dtype=float)
    if values.ndim==1:values=values[:,None]
    rng=np.random.default_rng(20261123);draws=[]
    for start in range(0,repeats,256):
        indices=rng.integers(len(values),size=(min(256,repeats-start),len(values)))
        draws.append(values[indices].mean(axis=1))
    lo,hi=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    return values.mean(axis=0),lo,hi


def compare(a,b,repeats):
    diff=a-b
    mean,lo,hi=interval(np.column_stack([diff.mean(axis=1),((diff>0)+.5*(diff==0)).mean(axis=1)]),repeats)
    return dict(mean_delta=float(mean[0]),low=float(lo[0]),high=float(hi[0]),scorer_win_rate=float(mean[1]),
        win_low=float(lo[1]),win_high=float(hi[1]),per_seed_deltas=diff.mean(axis=0).tolist(),
        seed_delta_sd=float(diff.mean(axis=0).std(ddof=1)) if diff.shape[1]>1 else None,
        wins=int((diff>0).sum()),ties=int((diff==0).sum()),losses=int((diff<0).sum()),
        comparison_count=int(diff.size),prompt_count=int(diff.shape[0]),generation_seed_count=int(diff.shape[1]),
        strict_win_rate=float((diff>0).mean()),tie_rate=float((diff==0).mean()),loss_rate=float((diff<0).mean()),
        per_seed_win_rates=((diff>0)+.5*(diff==0)).mean(axis=0).tolist())


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or ['status']);writer.writeheader();writer.writerows(rows)


def hardware_index(out,plan):
    """Describe the GPU used to generate each planned pixel, not the scorer GPU."""
    runtime=plan.get('noise_runtime')
    if runtime is None and (out/'noise-runtime.json').exists():runtime=read_json(out/'noise-runtime.json')['runtime']
    new_gpu=(runtime or {}).get('gpu','unfrozen')
    history=plan.get('history_noise_runtime') or runtime or {}
    index={};rows=[]
    for job in plan['jobs']:
        counts={}
        for slot in read_jsonl(out/job['slots']):
            sid=slot.get('source')
            gpu=(plan['sources'][sid].get('runtime') or history).get('gpu','unknown') if sid else new_gpu
            if slot['state']=='blocked':gpu='blocked'
            index[(job['method'],job['step'],*slot['key'])]=gpu
            key=(slot['state'],gpu);counts[key]=counts.get(key,0)+1
        rows.extend(dict(job=job['name'],method=job['method'],step=job['step'],seed=job['seed'],state=state,
                         generation_gpu=gpu,slots=n) for (state,gpu),n in counts.items())
    return index,rows


def hardware_fields(a,b=None):
    models=sorted(set(a+(b or [])))
    result={'generation_gpu_models':' | '.join(models),'mixed_generation_hardware':len(models)>1}
    if b is not None:result['paired_same_generation_gpu']=len(a)==len(b) and all(x==y for x,y in zip(a,b))
    return result


def family_report(out):
    # Multiple completed packs may request a report concurrently. Serialize publication.
    with (out/'.report.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return _family_report(out)


def _family_report(out):
    plan,panels=load_plan(out,check_code=False);family=plan['family'];dest=out/'report';dest.mkdir(exist_ok=True)
    hardware,hardware_rows=hardware_index(out,plan)
    csv_write(dest/'hardware-provenance.csv',hardware_rows)
    data={};portable=[];status=[]
    for job in plan['jobs']:
        ready=not job.get('blocked') and all(completed(out,plan,job,i) for i in range(job['shards']))
        status.append(dict(job=job['name'],pack=job['pack'],status='complete' if ready else 'missing_or_blocked',reason=job.get('blocked')))
        if ready:
            for i in range(job['shards']):
                for row in read_jsonl(result_path(out,job,i)/'scores.jsonl'):
                    key=(job['method'],job['step'],*row['key'])
                    if key in data:raise ValueError('Duplicate consolidated slot')
                    data[key]=row['scores'];portable.append(dict(family=family,method=job['method'],step=job['step'],**row))
    payload_jsonl(dest/'scores.jsonl',portable);atomic_json(dest/'status.json',status)
    atomic_json(dest/'protocol.json',dict(plan_id=plan['plan_id'],family=family,seeds=plan['seeds'],sampling=plan['sampling'],
        panels=panels,scorers=plan['scorers'],score_versions=plan['score_versions'],experiments=plan['config']['experiments'],
        bootstrap=dict(unit='whole prompt, retaining all seeds/methods/checkpoints',replicates=plan['config']['bootstrap_repeats'],
        seed=20261123,coverage='pointwise; fixed trained checkpoints; not independent training repetitions'),
        noise_runtime=plan['noise_runtime'],history_noise_runtime=plan.get('history_noise_runtime'),
        cross_gpu_reuse=plan.get('cross_gpu_reuse',False),
        hardware_note='Cross-GPU historical reuse: seed variation and some checkpoint differences include hardware effects.' if plan.get('cross_gpu_reuse') else None,
        checkpoint_adapters={j['name']:j['adapter'] for j in plan['jobs']}))
    repeats=plan['config']['bootstrap_repeats'];seeds=plan['seeds'];exps=[e for e in plan['config']['experiments'] if e['family']==family]
    by_method={e['method']:e for e in exps};means=[];paired=[];adjacent=[]
    def matrix(method,step,chosen_seeds,panel,metric):
        try:
            return np.array([[data[(method,step,*prompt_key(p),seed)][metric] for seed in chosen_seeds] for p in panels[panel]])
        except KeyError:return None
    def gpu_values(method,step,chosen_seeds,panel):
        return [hardware.get((method,step,*prompt_key(p),seed),'unknown') for p in panels[panel] for seed in chosen_seeds]
    def add_mean(method,step,chosen_seeds,panel,metric,scope):
        row=dict(scope=scope,method=method,step=step,seeds=','.join(map(str,chosen_seeds)),metric=metric,prompts=len(panels[panel]))
        row.update(hardware_fields(gpu_values(method,step,chosen_seeds,panel)))
        a=matrix(method,step,chosen_seeds,panel,metric)
        if a is None:row.update(status='missing',mean=None,low=None,high=None)
        else:
            mean,lo,hi=interval(a.mean(axis=1),repeats)
            row.update(status='complete',mean=float(mean[0]),low=float(lo[0]),high=float(hi[0]),
                       seed_mean_sd=float(a.mean(axis=0).std(ddof=1)) if len(chosen_seeds)>1 else None)
        means.append(row)
    def add_pair(method,baseline,step,base_step,chosen_seeds,panel,metric,scope,destination):
        row=dict(scope=scope,method=method,baseline=baseline,step=step,baseline_step=base_step,
                 seeds=','.join(map(str,chosen_seeds)),metric=metric,prompts=len(panels[panel]))
        row.update(hardware_fields(gpu_values(method,step,chosen_seeds,panel),gpu_values(baseline,base_step,chosen_seeds,panel)))
        a=matrix(method,step,chosen_seeds,panel,metric);b=matrix(baseline,base_step,chosen_seeds,panel,metric)
        if a is None or b is None:row.update(status='missing',mean_delta=None,low=None,high=None)
        else:row.update(status='complete',**compare(a,b,repeats))
        destination.append(row)
    comparisons=([('stagewise_k6','dpo'),('stagewise_dmpo_k6','dmpo'),('stagewise_dspo_k6','dspo')]
        if family in ('sd15','sdxl') else [('stagewise_k4','dpo')]+([('stagewise_k4','linear_dpo'),('stagewise_k4','global_k4'),('global_k4','dpo')] if family=='sd35_medium' else []))
    if family=='sd35_medium':
        comparisons += [(a,b) for a,b in [('global_k2','dpo'),('stagewise_k2','dpo'),('stagewise_k2','global_k2'),('global_k2','lossavg_k2'),('stagewise_k2','lossavg_k2'),('global_k4','lossavg_k4'),('stagewise_k4','lossavg_k4'),('lossavg_k2','dpo'),('lossavg_k4','dpo')] if a in by_method and b in by_method]
    for exp in exps:
        for metric in plan['config']['families'][family]['metrics']:
            for ss in [[s] for s in seeds]+[seeds]:add_mean(exp['method'],exp['main_step'],ss,'full',metric,'main')
        curve_metrics=['pickscore','hpsv2' if family=='sd15' else 'hpsv3','image_reward']
        for metric in curve_metrics:
            for step in exp['curve_steps']:add_mean(exp['method'],step,[seeds[0]],'curve',metric,'curve')
            for before,after in zip(exp['curve_steps'],exp['curve_steps'][1:]):
                add_pair(exp['method'],exp['method'],after,before,[seeds[0]],'curve',metric,'adjacent_checkpoint',adjacent)
    for method,baseline in comparisons:
        a,b=by_method[method],by_method[baseline]
        for metric in plan['config']['families'][family]['metrics']:
            for ss in [[s] for s in seeds]+[seeds]:add_pair(method,baseline,a['main_step'],b['main_step'],ss,'full',metric,'main',paired)
        for step in sorted(set(a['curve_steps'])&set(b['curve_steps'])):
            for metric in ['pickscore','hpsv2' if family=='sd15' else 'hpsv3','image_reward']:
                add_pair(method,baseline,step,step,[seeds[0]],'curve',metric,'curve',paired)
    csv_write(dest/'means.csv',means);csv_write(dest/'paired-differences.csv',paired);csv_write(dest/'adjacent-checkpoints.csv',adjacent)
    win_rows=[dict(benchmark='partiprompts',family=family,tie_rule='half credit; exact score equality',**r)
              for r in paired if r['scope']=='main']
    csv_write(dest/'win-ratios.csv',win_rows)
    note='Mixed generation GPUs; hardware effects not separated' if plan.get('cross_gpu_reuse') else ''
    plot_wins(dest,family,win_rows,','.join(map(str,seeds)),note)
    plot(dest,family,means,paired,comparisons,len(panels['curve']),note)
    if family=='sd35_medium' and 'lossavg_k2' in by_method:plot_lossavg(dest,means,len(panels['curve']),note)
    return dict(family=family,report=str(dest),complete_jobs=sum(s['status']=='complete' for s in status),total_jobs=len(status))


def plot(dest,family,means,paired,comparisons,curve_count,hardware_note=""):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=list(dict.fromkeys(r['metric'] for r in means if r['scope']=='curve'))
    if not metrics:return
    for method,baseline in comparisons:
        if not any(r['scope']=='curve' and r['method']==method for r in means):continue
        fig,axes=plt.subplots(2,len(metrics),figsize=(4.3*len(metrics),6),squeeze=False,layout='constrained')
        for col,metric in enumerate(metrics):
            for name in (baseline,method):
                rows=[r for r in means if r['scope']=='curve' and r['method']==name and r['metric']==metric]
                if not rows:continue
                x=[r['step'] for r in rows];y=[np.nan if r['mean'] is None else r['mean'] for r in rows]
                axes[0,col].plot(x,y,'o-',label=name)
                axes[0,col].fill_between(x,[np.nan if r['low'] is None else r['low'] for r in rows],
                    [np.nan if r['high'] is None else r['high'] for r in rows],alpha=.15)
                axes[0,col].set_xticks(x,[f'{v/1000:g}k' for v in x])
            rows=[r for r in paired if r['scope']=='curve' and r['method']==method and r['baseline']==baseline and r['metric']==metric]
            x=[r['step'] for r in rows];y=[np.nan if r['mean_delta'] is None else r['mean_delta'] for r in rows]
            axes[1,col].axhline(0,color='.5',lw=.8);axes[1,col].plot(x,y,'o-')
            axes[1,col].vlines(x,[np.nan if r['low'] is None else r['low'] for r in rows],[np.nan if r['high'] is None else r['high'] for r in rows])
            for r in rows:
                if r['status']=='missing':axes[1,col].text(r['step'],.05,'N/A',transform=axes[1,col].get_xaxis_transform(),ha='center')
            axes[0,col].set_title(metric);axes[0,col].legend(fontsize=8)
            axes[1,col].set(xlabel='Optimizer update',ylabel='Paired score difference')
            axes[1,col].set_xticks(x,[f'{v/1000:g}k' for v in x])
        fig.suptitle(f'{family}: {method} vs {baseline}\n{curve_count} prompts; pointwise 95% prompt-bootstrap intervals; gaps = missing'+('\n'+hardware_note if hardware_note else ''))
        for ext in ('png','pdf'):fig.savefig(dest/f'{method}-vs-{baseline}.{ext}',dpi=160)
        plt.close(fig)



LABELS={'dpo':'DPO','stagewise_k6':'DPO + MVPO K6','dmpo':'DMPO','stagewise_dmpo_k6':'DMPO + MVPO K6',
        'dspo':'DSPO','stagewise_dspo_k6':'DSPO + MVPO K6','linear_dpo':'LinearDPO',
        'lossavg_k2':'LossAvg K2','lossavg_k4':'LossAvg K4','global_k2':'Global MVPO K2','stagewise_k2':'Stage-wise MVPO K2','global_k4':'Global MVPO K4','stagewise_k4':'Stage-wise MVPO K4'}
METRIC_LABELS={'pickscore':'PickScore','hpsv2':'HPSv2','hpsv3':'HPSv3','image_reward':'ImageReward','clip':'CLIP','aesthetic':'Aesthetic'}


def plot_wins(dest,family,rows,all_seeds,hardware_note=""):
    """One percentage chart per scorer, matching the full three-seed paired design."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    selected=[r for r in rows if r['seeds']==all_seeds]
    for metric in dict.fromkeys(r['metric'] for r in selected):
        items=[r for r in selected if r['metric']==metric]
        fig,ax=plt.subplots(figsize=(12,max(3.1,1.0*len(items)+1.6)),layout='constrained')
        labels=[]
        for y,row in enumerate(items):
            label=f"{LABELS.get(row['method'],row['method'])} ({row['step']:,})"
            label+=f"\nvs {LABELS.get(row['baseline'],row['baseline'])} ({row['baseline_step']:,})"
            labels.append(label)
            if row['status']!='complete':
                ax.barh(y,100,color='#eeeeee',edgecolor='#bbbbbb',height=.68)
                ax.text(50,y,'N/A — paired scores incomplete',ha='center',va='center',color='#555555')
                continue
            win=100*row['scorer_win_rate'];other=100-win
            ax.barh(y,win,color='#3065f5',edgecolor='black',linewidth=.6,height=.68)
            ax.barh(y,other,left=win,color='#f4a51c',edgecolor='black',linewidth=.6,height=.68)
            ax.errorbar(win,y,xerr=[[100*(row['scorer_win_rate']-row['win_low'])],
                                   [100*(row['win_high']-row['scorer_win_rate'])]],fmt='none',ecolor='black',capsize=4,lw=1.2)
            ax.text(max(3,win/2),y,f'{win:.1f}%',ha='center',va='center',weight='bold',
                    color='white' if win>=12 else 'black',fontsize=11)
            if other>=12:ax.text(win+other/2,y,f'{other:.1f}%',ha='center',va='center',weight='bold',fontsize=11)
        ax.axvline(50,color='black',linestyle='--',linewidth=1.1)
        ax.set(xlim=(0,100),xticks=range(0,101,20),yticks=range(len(items)),yticklabels=labels,
               xlabel='Scorer-based paired win ratio (%) — ties split equally')
        ax.invert_yaxis();ax.set_axisbelow(True);ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(handles=[Patch(color='#3065f5',label='Compared method'),Patch(color='#f4a51c',label='Baseline')],
                  loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False)
        fig.suptitle(f"{family} • PartiPrompts • {METRIC_LABELS.get(metric,metric)}\n"
                     f"{items[0]['prompts']:,} prompts × {len(all_seeds.split(','))} generation seeds; 95% prompt-bootstrap intervals"+("\n"+hardware_note if hardware_note else ""),fontsize=12)
        for ext in ('png','pdf'):fig.savefig(dest/f'win-ratios-{metric}.{ext}',dpi=180)
        plt.close(fig)

def report(output,family=None):
    families=[family] if family else ['sd15','sdxl','sd35_medium','zimage']
    return [family_report(output/f) if (output/f/'plan.json').exists() else dict(family=f,status='unprepared') for f in families]


def plot_lossavg(dest,means,curve_count,hardware_note=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['dpo','lossavg_k2','global_k2','stagewise_k2','lossavg_k4','global_k4','stagewise_k4']
    metrics=['pickscore','hpsv3','image_reward'];fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,metric in zip(axes,metrics):
        for name in names:
            rows=[r for r in means if r['scope']=='curve' and r['method']==name and r['metric']==metric]
            if not rows:continue
            x=[r['step'] for r in rows];mean=[np.nan if r['mean'] is None else r['mean'] for r in rows]
            line,=ax.plot(x,mean,marker='o',markersize=3,linestyle='--' if name.startswith('lossavg') else '-',label=LABELS[name])
            ax.fill_between(x,[np.nan if r['low'] is None else r['low'] for r in rows],[np.nan if r['high'] is None else r['high'] for r in rows],color=line.get_color(),alpha=.09)
        ax.set(xlabel='Optimizer update',title=METRIC_LABELS[metric]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'SD3.5-M: DPO, LossAvg and MVPO controls\n{curve_count} fixed prompts; one generation seed; pointwise 95% prompt-bootstrap intervals'+('\n'+hardware_note if hardware_note else ''))
    for ext in ('png','pdf'):fig.savefig(dest/f'lossavg-controls.{ext}',dpi=160)
    plt.close(fig)
