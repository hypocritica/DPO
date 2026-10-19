"""Small CPU/Gloo worker exercising the production detached-rollout trainer."""
import argparse
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import random
import signal
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from self_correct_dpo.checkpoints import atomic_json
from self_correct_dpo.common import read_json
from self_correct_dpo.distributed import capture_rng, cleanup
from self_correct_dpo.independent import RolloutConfig
from self_correct_dpo.joint import JointRolloutBackend
from self_correct_dpo.training import train


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2));self.adapter=True

    @contextmanager
    def disable_adapter(self):
        previous=self.adapter;self.adapter=False
        try:yield
        finally:self.adapter=previous

    def forward(self,noisy,times,encoder_hidden_states=None,added_cond_kwargs=None,return_dict=False):
        scale=self.weight if self.adapter else .1
        return (scale*noisy+.03,)


class Backend(JointRolloutBackend):
    def __init__(self,model,training,device):
        self.family=model['family'];self.device=torch.device(device);self.resolution=64
        self.dtype=torch.float32;self.training_config=training;self.noise_views=training['noise_views']
        self.model=Denoiser();self.frozen_modules=[];self.microbatches=[];self.sent=False
        self.rollout_config=RolloutConfig(**training['independent_rollout'])
        self.scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000,prediction_type='epsilon'),
                                       alphas_cumprod=torch.linspace(.999,.001,1000))

    def _encode_images(self,images,generator):
        value=images[:,:,:1,:2,:2]
        return value+.01*torch.randn(value.shape,generator=generator)

    def _condition(self,texts):
        return {'encoder_hidden_states':torch.ones(len(texts),1),
                'added_cond_kwargs':{'text_embeds':torch.ones(len(texts),2),'time_ids':torch.ones(len(texts),6)}}

    def _autocast(self):return nullcontext()

    def save_adapter(self,path):
        Path(path).mkdir();torch.save(self.model.state_dict(),Path(path)/'weight.pt')

    def load_adapter(self,path):
        self.model.load_state_dict(torch.load(Path(path)/'weight.pt',weights_only=True))

    def score_errors(self,*args,**kwargs):
        rank=int(os.environ['RANK'])
        if os.environ.get('SCD_TEST_FAIL_RANK')==str(rank):raise RuntimeError('intentional worker failure')
        if self.training_config.get('test_signal_rank')==rank and not self.sent:
            self.sent=True;signal.raise_signal(signal.SIGTERM)
        values=super().score_errors(*args,**kwargs)
        self.microbatches.append(values['prepared'])
        return values


def images(rows,resolution,device,workers=0):
    return torch.stack([torch.full((2,3,64,64),-.6+.1*int(row['pair_id'])) for row in rows])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--stop-after-step',type=int);p.add_argument('--resume');p.add_argument('--extend-from')
    p.add_argument('--internal-stage',action='store_true');p.add_argument('--stage-id')
    p.add_argument('--execution');p.add_argument('--run-label')
    a=p.parse_args();config=read_json(a.config)
    rows=[{'pair_id':str(i),'split':'train','reviewed':True,
           'prompts':[{'variant_id':'canonical','prompt':f'caption {i}'}]} for i in range(7)]
    holder={}
    def backend(*args):holder['backend']=Backend(*args);return holder['backend']
    try:
        with patch('self_correct_dpo.training.image_batch',side_effect=images):
            result=train(config,rows,a.out,resume=a.resume,extend_from=a.extend_from,
                backend_factory=backend,device='cpu',stop_after_step=a.stop_after_step)
        b=holder['backend'];rank=int(os.environ['RANK'])
        probe={'weight':b.model.weight.detach(),'rng':capture_rng(torch.Generator().manual_seed(19)),
               'random_probe':[random.random(),float(np.random.rand()),float(torch.rand(()))],
               'microbatches':b.microbatches,'result':{k:v for k,v in result.items() if k!='cost'}}
        torch.save(probe,Path(a.out)/f'probe-step-{result["step"]}-rank-{rank}.pt')
        if rank==0 and a.stage_id:
            atomic_json(Path(a.out)/'stage-result.json',{'stage_id':a.stage_id,
                **{k:result[k] for k in ('run_id','step','complete','planned_pause','interrupted')}})
    finally:cleanup()


if __name__=='__main__':main()
