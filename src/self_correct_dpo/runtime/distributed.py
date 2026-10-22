"""Small single-node DDP helpers; checkpoints retain every rank's RNG stream."""
import os
import random
from datetime import timedelta


def capture_rng(generator):
    import numpy as np
    import torch
    state=np.random.get_state()
    return {'generator':generator.get_state(),'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state() if generator.device.type=='cuda' else [],
            'python_rng':random.getstate(),'numpy_rng':(state[0],state[1].tolist(),*state[2:]),
            'rank':int(os.environ.get('RANK','0')),'world_size':int(os.environ.get('WORLD_SIZE','1'))}


def restore_rng(state,generator):
    import numpy as np
    import torch
    generator.set_state(state['generator'])
    torch.set_rng_state(state['cpu_rng'])
    random.setstate(state['python_rng'])
    name,keys,position,gauss,cached=state['numpy_rng']
    np.random.set_state((name,np.asarray(keys,dtype=np.uint32),position,gauss,cached))
    cuda=state['cuda_rng']
    if isinstance(cuda,torch.Tensor):
        if generator.device.type!='cuda':raise ValueError('A CUDA checkpoint requires a CUDA worker')
        torch.cuda.set_rng_state(cuda)
    elif cuda:torch.cuda.set_rng_state_all(cuda)


def initialize(device):
    import torch
    import torch.distributed as dist
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if int(os.environ.get('GROUP_WORLD_SIZE','1'))!=1:
        raise ValueError('This launcher supports one node with one process per GPU')
    if world>1:
        if str(device).startswith('cuda'):
            local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=f'cuda:{local}'
        if not dist.is_initialized():dist.init_process_group(backend='nccl' if str(device).startswith('cuda') else 'gloo',
            timeout=timedelta(seconds=int(os.environ.get('SCD_DISTRIBUTED_TIMEOUT_SECONDS','600'))))
    return world,rank,device


def average_metrics(metrics, device):
    import torch
    import torch.distributed as dist
    keys=sorted(metrics);values=torch.tensor([metrics[k] for k in keys],dtype=torch.float64,device=device)
    dist.all_reduce(values);values/=dist.get_world_size()
    return dict(zip(keys,values.cpu().tolist()))


def gather_rng(generator):
    import torch.distributed as dist
    states=[None]*dist.get_world_size();dist.all_gather_object(states,capture_rng(generator));return states


def cleanup():
    import torch.distributed as dist
    if dist.is_initialized():dist.destroy_process_group()


def subprocess_environment(environment):
    """A coordinator's children must not inherit another torchrun rank identity."""
    return {key:value for key,value in environment.items() if key not in
        {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','GROUP_WORLD_SIZE',
         'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'} and not key.startswith('TORCHELASTIC_')}
