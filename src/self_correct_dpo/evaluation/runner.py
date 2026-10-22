"""Checkpoint evaluation using separate generation and scorer processes."""
from self_correct_dpo.common import source_file
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from self_correct_dpo.common import artifact_identity, model_artifact_identity, digest, load_suite, output_lock, read_json, require_gpu_allocation, sha256, write_json
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.config import load_config
from self_correct_dpo.data.benchmarks import BENCHMARKS
from self_correct_dpo.evaluation.scorers import DEFAULT_METRICS, METRICS, scorer_identity


def run_command(command):
    """Forward TERM/INT and leave complete checkpoints and journals for a rerun."""
    from self_correct_dpo.runtime.distributed import subprocess_environment
    environment = subprocess_environment(os.environ)
    src = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = src + (os.pathsep+environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    previous, stopped = {}, [False]
    child = subprocess.Popen(command,env=environment,start_new_session=True)
    def forward(signum, frame):
        stopped[0] = True
        if child.poll() is None:
            try:os.killpg(child.pid,signum)
            except ProcessLookupError:pass
    for signum in (signal.SIGTERM,signal.SIGINT):
        previous[signum]=signal.signal(signum,forward)
    try:
        code=child.wait()
    finally:
        for signum,handler in previous.items(): signal.signal(signum,handler)
    if code or stopped[0]:
        raise subprocess.CalledProcessError(code or 75,command)


def checkpoint_adapter(run_dir, step):
    from self_correct_dpo.training import checkpoint_manifest
    run_dir=Path(run_dir).resolve()
    info=read_json(run_dir/'training.json')
    if step == 0:
        return info['config']['model'],None
    checkpoint=run_dir/'checkpoints'/f'step-{step:07d}'
    if not checkpoint.is_dir():
        from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
        current=resolve_checkpoint(run_dir/'checkpoints/latest.json')
        if checkpoint_manifest(current)['step'] != step:
            raise ValueError(f'Step {step} was not retained. Retained adapters are saved every 1000 steps and at the final step.')
        checkpoint=current
    saved=checkpoint_manifest(checkpoint)
    if saved['run_id'] != info['run_id'] or saved['step'] != step:
        raise ValueError('Checkpoint run/step identity mismatch')
    return info['config']['model'],checkpoint/'adapter'


def validate_evaluation(prompts, scorers_config, metrics, *, expected_prompts=None):
    suite=load_suite(prompts)
    if expected_prompts is not None and len(suite)!=expected_prompts:
        raise ValueError(f'Expected {expected_prompts} prompts, found {len(suite)}')
    if not metrics or len(set(metrics))!=len(metrics) or any(m not in METRICS for m in metrics):
        raise ValueError('Use distinct supported metrics')
    specs=load_config(scorers_config,scorers=metrics)['scorers']
    return {'prompts_sha256':sha256(prompts),'scorers_config_sha256':sha256(scorers_config),
            'scorers':{m:scorer_identity(m,specs[m]) for m in metrics},'prompts':len(suite)}


def verify_complete(out):
    out=Path(out)
    complete=read_json(out/'complete.json')
    if complete['summary_sha256']!=sha256(out/'summary.json'):
        raise ValueError('Evaluation summary checksum mismatch')
    for name,checksum in complete['scores_sha256'].items():
        path=(out/name).resolve()
        if not path.is_relative_to(out.resolve()) or sha256(path)!=checksum:
            raise ValueError('Evaluation score checksum mismatch')
    return read_json(out/'summary.json')


def prune_images(out):
    """Only delete image paths registered by this completed evaluation."""
    from self_correct_dpo.common import read_jsonl
    out=Path(out).resolve()
    render=out/'render'
    for row in read_jsonl(render/'manifest.jsonl'):
        path=render/row['image']
        if path.is_symlink() or not path.resolve().is_relative_to(render/'images'):
            raise ValueError('Unsafe evaluation image path')
        if path.exists():
            if sha256(path)!=row['image_sha256']:
                raise ValueError('Refusing to prune an image whose checksum changed')
            path.unlink()
    atomic_json(out/'images-pruned.json',{'reason':'Non-retention monitoring step; scores and manifests retained'})


def _evaluate_checkpoint(model, adapter, prompts, scorers_config, metrics, sampling, out, *, step=None,
                        keep_images=True, execution=None, expected_prompts=None, check_only=False,
                        command_runner=run_command):
    inputs=validate_evaluation(prompts,scorers_config,metrics,expected_prompts=expected_prompts)
    model_id=model_artifact_identity(model)
    config={'model':model,'sampling':sampling}
    arm='base'
    if adapter:
        adapter=Path(adapter).resolve()
        arm='aligned'
        config['aligned']={'kind':'peft','path':str(adapter),'revision':digest(artifact_identity(adapter))}
    identity={'inputs':inputs,'config':config,'base_identity':model_id,'step':step,
              'keep_images':keep_images,'code_sha256':{name:sha256(source_file(name))
              for name in ('evaluation.py','render.py','metrics.py','scorers.py','common.py')}}
    meta={'identity':identity,'evaluation_id':digest(identity)}
    out=Path(out).resolve()
    if out.exists():
        if read_json(out/'evaluation.json')!=meta:
            raise ValueError('Cannot resume evaluation with changed checkpoint, prompts or scorers')
        if (out/'complete.json').is_file():
            summary=verify_complete(out)
            if not keep_images and not (out/'images-pruned.json').exists(): prune_images(out)
            return summary
    if check_only:
        return {'evaluation_id':meta['evaluation_id'],'step':step,'prompts':inputs['prompts'],
                'images':inputs['prompts']*len(sampling['seeds']),'metrics':metrics}
    require_gpu_allocation(execution)
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out/'evaluation.json',meta)
        write_json(out/'render-config.json',config)
    execution=execution or {'mode':'slurm'}
    extra=['--execution',execution['mode']]
    if execution.get('run_label'): extra+=['--run-label',execution['run_label']]
    render=out/'render'
    command=[sys.executable,'-m','self_correct_dpo.evaluation.render','--config',str(out/'render-config.json'),
             '--prompts',str(Path(prompts).resolve()),'--arm',arm,'--out',str(render),*extra]
    if render.exists(): command+=['--resume']
    command_runner(command)
    scores={}
    for metric in metrics:
        target=out/'scores'/metric
        spec=inputs.get('scorers',{}).get(metric,{}).get('spec',{})
        command_runner([spec.get('python') or sys.executable,'-m','self_correct_dpo.evaluation.metrics','--run',str(render),
            '--out',str(target),'--scorers-config',str(Path(scorers_config).resolve()),'--metric',metric,*extra])
        scores[metric]=read_json(target/'summary.json')
    summary={'evaluation_id':meta['evaluation_id'],'step':step,'prompts':inputs['prompts'],
             'images':inputs['prompts']*len(sampling['seeds']),'metrics':scores,'images_retained':keep_images}
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'complete.json',{'summary_sha256':sha256(out/'summary.json'),
        'scores_sha256':{str(p.relative_to(out)):sha256(p) for p in (out/'scores').rglob('*.json*')}})
    if not keep_images: prune_images(out)
    return summary


def evaluate_checkpoint(model, adapter, prompts, scorers_config, metrics, sampling, out, **kwargs):
    if kwargs.get('check_only'):
        return _evaluate_checkpoint(model,adapter,prompts,scorers_config,metrics,sampling,out,**kwargs)
    with output_lock(str(out)+'.evaluation'):
        return _evaluate_checkpoint(model,adapter,prompts,scorers_config,metrics,sampling,out,**kwargs)


def evaluate_benchmarks(config,model,adapter,benchmarks,metrics,out,**kwargs):
    if config.get('model_family') and config['model_family']!=model.get('family'):
        raise ValueError('Evaluation sampling config model_family differs from the training run')
    def evaluate(name,target):
        return evaluate_checkpoint(model,adapter,config['benchmarks'][name],config['scorers_config'],
            metrics,config['sampling'],target,expected_prompts=400 if name=='hpdv2' else None,**kwargs)
    if len(benchmarks)==1:
        return evaluate(benchmarks[0],out)
    # Validate every selected panel/model before starting any generation.
    for name in benchmarks:
        validate_evaluation(config['benchmarks'][name],config['scorers_config'],metrics,
                            expected_prompts=400 if name=='hpdv2' else None)
    out=Path(out).resolve()
    def run():
        result={'step':kwargs.get('step'),'benchmarks':{name:evaluate(name,out/name) for name in benchmarks}}
        if not kwargs.get('check_only'): atomic_json(out/'summary.json',result)
        return result
    if kwargs.get('check_only'):return run()
    with output_lock(str(out)+'.benchmarks'):
        request={'config':config,'model':model,'adapter':str(adapter) if adapter else None,
                 'benchmarks':benchmarks,'metrics':metrics,'step':kwargs.get('step')}
        if out.exists():
            if not (out/'benchmarks.json').is_file() or read_json(out/'benchmarks.json')!=request:
                raise ValueError('Use a fresh multi-benchmark output directory or resume the same request')
        else:
            out.mkdir(parents=True)
            write_json(out/'benchmarks.json',request)
        return run()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='Benchmark/scorer/sampling config; defaults use HPDv2 and four metrics')
    p.add_argument('--run',required=True,help='Training run containing training.json and checkpoints')
    p.add_argument('--step',type=int,required=True,help='0 for base, or any retained training step')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--benchmark',choices=BENCHMARKS,help='One panel; default: hpdv2')
    group.add_argument('--benchmarks',nargs='+',choices=BENCHMARKS,help='Evaluate several panels into separate subdirectories')
    p.add_argument('--metrics',nargs='+',choices=METRICS,default=list(DEFAULT_METRICS))
    p.add_argument('--out',required=True)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--execution',choices=['slurm','standalone'],default='slurm')
    p.add_argument('--run-label')
    a=p.parse_args(argv)
    if a.step<0: p.error('--step must be nonnegative')
    a.benchmarks=a.benchmarks or [a.benchmark or 'hpdv2']
    if len(set(a.benchmarks))!=len(a.benchmarks):p.error('Select each benchmark once')
    if len(set(a.metrics))!=len(a.metrics):p.error('Select each metric once')
    return a


def main():
    a=parse_args()
    config=load_config(a.config,benchmarks=a.benchmarks)
    model,adapter=checkpoint_adapter(a.run,a.step)
    result=evaluate_benchmarks(config,model,adapter,a.benchmarks,a.metrics,a.out,step=a.step,
        execution={'mode':a.execution,'run_label':a.run_label},check_only=a.check_only)
    print(json.dumps(result))


if __name__=='__main__': main()
