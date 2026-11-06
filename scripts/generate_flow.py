"""Generate base/LoRA images from a frozen flow training config and heldout suite."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from self_correct_dpo.common import read_json
from self_correct_dpo.config import load_config
from self_correct_dpo.runtime.checkpoints import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-config',required=True,type=Path)
    p.add_argument('--prompts',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--adapter',type=Path,help='LoRA adapter directory, not a complete checkpoint parent')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--steps',type=int)
    p.add_argument('--guidance-scale',type=float)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--execution',default='slurm',choices=['slurm','standalone'])
    p.add_argument('--run-label')
    a=p.parse_args()
    training=load_config(a.training_config);family=training['model']['family']
    if family not in ('sd35_medium','zimage'):p.error('Expected a flow training configuration')
    template=load_config(ROOT/'configs/flow'/f'{family}-dpo.json')
    sampling={**template['evaluation']['sampling'],**training.get('evaluation',{}).get('sampling',{})}
    sampling['prompt_truncation']=training['model'].get('prompt_truncation','error')
    sampling['seeds']=[a.seed]
    if a.steps is not None:sampling['num_inference_steps']=a.steps
    if a.guidance_scale is not None:sampling['guidance_scale']=a.guidance_scale
    config={'model':training['model'],'sampling':sampling}
    if a.adapter:
        from self_correct_dpo.common import digest,sha256
        files={p.name:sha256(p) for p in a.adapter.iterdir() if p.is_file()}
        config['aligned']={'kind':'peft','path':str(a.adapter.resolve()),'revision':digest(files)}
    frozen=a.out.resolve().with_name('.'+a.out.name+'.render-config.json')
    frozen.parent.mkdir(parents=True,exist_ok=True)
    if frozen.exists():
        if read_json(frozen)!=config:raise ValueError('Generation configuration differs; use a fresh output')
    else:atomic_json(frozen,config)
    command=[sys.executable,'-m','self_correct_dpo.evaluation.render','--config',str(frozen),
        '--prompts',str(a.prompts.resolve()),'--out',str(a.out.resolve()),
        '--arm','aligned' if a.adapter else 'base','--execution',a.execution]
    if a.run_label:command+=['--run-label',a.run_label]
    if a.check_only:command+=['--check-only']
    if a.resume:command+=['--resume']
    from self_correct_dpo.evaluation.runner import run_command
    run_command(command)


if __name__=='__main__':main()
