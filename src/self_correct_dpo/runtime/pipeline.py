"""Pause training at fixed steps; evaluate after the trainer releases its GPU memory."""
from self_correct_dpo.common import source_file
import csv
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import uuid

from self_correct_dpo.data.benchmarks import prompt_key
from self_correct_dpo.runtime.checkpoints import atomic_json, resolve_checkpoint
from self_correct_dpo.common import digest, load_suite, read_json, read_jsonl, sha256, write_json
from self_correct_dpo.evaluation.runner import checkpoint_adapter, evaluate_checkpoint, run_command, validate_evaluation, verify_complete


def evaluation_steps(max_steps, settings):
    first, every = settings.get('first_step',200),settings.get('every',500)
    if any(type(v) is not int or v<1 for v in (max_steps,first,every)):
        raise ValueError('Evaluation schedule requires positive integer steps')
    steps=set(range(every,max_steps+1,every)) | {max_steps}
    if first<=max_steps: steps.add(first)
    if settings.get('include_base',True): steps.add(0)
    return sorted(steps)


def keep_images(step, max_steps, settings):
    every=settings.get('keep_images_every',1000)
    if type(every) is not int or every<1:
        raise ValueError('keep_images_every must be positive')
    return step==max_steps or step % every==0


def monitor_preflight(config, train_rows):
    settings=config['evaluation']
    evaluation_steps(config['training']['max_steps'],settings)
    keep_images(0,config['training']['max_steps'],settings)
    from self_correct_dpo.evaluation.render import validate_config as validate_render
    validate_render({'model':config['model'],'sampling':settings['sampling']},'base')
    suite=load_suite(settings['prompts'])
    train_keys={prompt_key(p['prompt']) for row in train_rows for p in row['prompts']}
    if any(prompt_key(row['prompt']) in train_keys for row in suite):
        raise ValueError('Monitoring panel overlaps a training caption')
    if len({prompt_key(r['prompt']) for r in suite})!=len(suite):
        raise ValueError('Monitoring requires distinct held-out captions')
    expected=settings.get('expected_prompts',128)
    result=validate_evaluation(settings['prompts'],settings['scorers_config'],['pickscore'],expected_prompts=expected)
    return {'settings':settings,'inputs':result,'code_sha256':{
        name:sha256(source_file(name)) for name in
        ('pipeline.py','evaluation.py','metrics.py','scorers.py','render.py','benchmarks.py')}}


def training_command(config,config_path,out,*,execution,stop_after_step=None,resume=None,extend_from=None):
    world=config['training'].get('world_size',1)
    command=[sys.executable,'-m']
    if world>1:
        if config['training'].get('launcher', 'torchrun') == 'accelerate':
            precision = 'bf16' if config['training'].get('dtype') == 'bfloat16' else 'no'
            command+=['accelerate.commands.launch','--multi_gpu','--num_machines=1', '--machine_rank=0', '--gpu_ids=all',
                      f'--num_processes={world}', f'--mixed_precision={precision}',
                      '--dynamo_backend=no', '--max_restarts=0', '--module']
        else:
            command+=['torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={world}',
                      '--max-restarts=0','--module']
    command+=['self_correct_dpo.train','--config',str(Path(config_path).resolve()),
              '--out',str(Path(out).resolve()),'--internal-stage','--stage-id',uuid.uuid4().hex,'--execution',execution['mode']]
    if execution.get('run_label'):command+=['--run-label',execution['run_label']]
    if stop_after_step is not None:command+=['--stop-after-step',str(stop_after_step)]
    if resume:command+=['--resume',str(Path(resume).resolve())]
    if extend_from:command+=['--extend-from',str(Path(extend_from).resolve())]
    return command


def stage_interrupted(command,out):
    path=Path(out)/'stage-result.json'
    if not command or '--stage-id' not in command or not path.is_file():return False
    result=read_json(path)
    if result.get('stage_id')!=command[command.index('--stage-id')+1] or not result.get('interrupted'):return False
    checkpoint=read_json(resolve_checkpoint(Path(out)/'checkpoints/latest.json')/'checkpoint.json')
    if result['run_id']!=checkpoint['run_id'] or result['step']!=checkpoint['step']:
        raise ValueError('Interrupted stage result does not match its committed checkpoint')
    return True


def paired_delta(current, baseline, repeats=1000):
    def mapping(rows):
        result={(r['scene_id'],r['variant_id'],r['seed']):r for r in rows}
        if len(result)!=len(rows): raise ValueError('Duplicate paired score keys')
        return result
    a,b=mapping(current),mapping(baseline)
    if a.keys()!=b.keys() or any(a[k]['prompt']!=b[k]['prompt'] for k in a):
        raise ValueError('Cannot compare different prompt/seed panels')
    groups={}
    for key in sorted(a):
        groups.setdefault(key[:2],[]).append(a[key]['score']-b[key]['score'])
    values=[statistics.mean(v) for v in groups.values()]
    rng=random.Random(17)
    boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(repeats))
    return {'delta_base':statistics.mean(values),'delta_ci_low':boot[int(.025*(repeats-1))],
            'delta_ci_high':boot[int(.975*(repeats-1))]}


def update_curve(out):
    out=Path(out)
    panels=[]
    for path in sorted((out/'evaluations').glob('step-*')):
        if (path/'complete.json').is_file():
            summary=verify_complete(path)
            score=summary['metrics']['pickscore']
            panels.append((summary['step'],path,score))
    if not panels:return
    baseline=next((p for step,p,_ in panels if step==0),None)
    base_rows=read_jsonl(baseline/'scores/pickscore/scores.jsonl') if baseline else None
    rows=[]
    for step,path,score in sorted(panels):
        row={'step':step,'pickscore':score['mean'],'prompts':score['prompts'],'images':score['images']}
        if base_rows is not None:
            row.update(paired_delta(read_jsonl(path/'scores/pickscore/scores.jsonl'),base_rows))
        rows.append(row)
    temporary=out/'.pickscore-curve.csv.tmp'
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary.replace(out/'pickscore-curve.csv')
    atomic_json(out/'pickscore-curve.json',{'scores':rows,'convention':'PickScore raw cosine',
        'interval':'Paired bootstrap over prompts, seeds averaged; fixed training seed, not training-seed uncertainty'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([r['step'] for r in rows],[r['pickscore'] for r in rows],marker='o')
    ax.set(xlabel='Optimizer step',ylabel='PickScore (raw cosine)')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'pickscore-curve.png',dpi=160);plt.close(fig)


def run_pipeline(config, config_path, out, *, execution, resume=None, extend_from=None,
                 stop_after_step=None, command_runner=run_command, evaluator=evaluate_checkpoint,
                 monitor_identity=None):
    settings=config['evaluation'];out=Path(out).resolve()
    budget=config['training']['max_steps']
    end=stop_after_step if stop_after_step is not None else budget
    planned=evaluation_steps(budget,settings)
    if out.exists() and not resume:
        raise FileExistsError('Existing training run requires --resume')
    if resume:
        pointer=resolve_checkpoint(resume)
        current=read_json(pointer/'checkpoint.json')['step']
        if current>end: raise ValueError('Requested endpoint precedes current checkpoint')
    elif extend_from:
        current=read_json(resolve_checkpoint(extend_from)/'checkpoint.json')['step']
    else: current=0
    frozen=monitor_identity or {'settings':settings,'inputs':validate_evaluation(settings['prompts'],
        settings['scorers_config'],['pickscore'],expected_prompts=settings.get('expected_prompts',128)),
        'code_sha256':{name:sha256(source_file(name)) for name in
                      ('pipeline.py','evaluation.py','metrics.py','scorers.py','render.py','benchmarks.py')}}
    input_file=out/'monitor-inputs.json'
    if input_file.exists() and read_json(input_file)!=frozen:
        raise ValueError('Monitoring inputs changed; start a fresh experiment')
    if (out/'training.json').exists():
        recorded=read_json(out/'training.json').get('monitor_inputs')
        if recorded is not None and recorded!=frozen['inputs']:
            raise ValueError('Scorer assets differ from the saved training identity')
    def evaluate(step):
        if not input_file.exists():write_json(input_file,frozen)
        path=out/'evaluations'/f'step-{step:07d}'
        if (path/'complete.json').exists():
            verify_complete(path)
            return
        model,adapter=checkpoint_adapter(out,step)
        evaluator(model,adapter,settings['prompts'],settings['scorers_config'],['pickscore'],
            settings['sampling'],path,step=step,keep_images=keep_images(step,budget,settings),
            execution=execution,expected_prompts=settings.get('expected_prompts',128))
        if config['training'].get('logging', 'jsonl') not in ('none', 'jsonl'):
            from self_correct_dpo.runtime.tracking import log_evaluation
            log_evaluation(config, out, step, verify_complete(path))
    targets=sorted(set(s for s in planned if current<s<=end)|({end} if end>current else set()))
    if not targets and not out.exists(): raise ValueError('No training updates requested')
    command=None
    try:
        if out.exists():
            if settings.get('include_base',True):evaluate(0)
            if current in planned and current>0:evaluate(current)
            update_curve(out)
        for target in targets:
            command=training_command(config,config_path,out,execution=execution,stop_after_step=target,
                resume=out/'checkpoints/latest.json' if out.exists() else None,
                extend_from=extend_from if not out.exists() else None)
            command_runner(command)
            pointer=resolve_checkpoint(out/'checkpoints/latest.json')
            current=read_json(pointer/'checkpoint.json')['step']
            if stage_interrupted(command,out):
                return {'complete':False,'planned_pause':False,'step':current,'paused':True}
            if current!=target:raise RuntimeError('Trainer did not reach the requested evaluation step')
            # Base is evaluated once the first stage has created the run directory.
            # It uses the frozen base, independent of the trained adapter.
            if settings.get('include_base',True):evaluate(0)
            if current in planned:evaluate(current)
            update_curve(out)
    except subprocess.CalledProcessError as error:
        if stage_interrupted(command,out) or error.returncode in (75,-15,-2,130,143):
            pointer=out/'checkpoints/latest.json'
            if pointer.exists(): current=read_json(resolve_checkpoint(pointer)/'checkpoint.json')['step']
            return {'complete':False,'planned_pause':False,'step':current,'paused':True}
        raise
    result={'complete':current==budget,'planned_pause':current==end and end<budget,'step':current,
            'evaluation_steps':[s for s in planned if s<=current]}
    if result['complete']:
        atomic_json(out/'evaluation-complete.json',result)
    return result
