"""Two-rank Gloo integration of production LoRA/EMA checkpointing."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
from test_flow import TinyFlow, build_config, train
import torch
from self_correct_dpo.runtime.distributed import cleanup

root=Path(sys.argv[1]);base=root/'base'
c=build_config('sd35_medium','linear_dpo')
c['model'].update(path=str(base),revision='tiny-test')
c['training'].update(world_size=2,max_steps=4,lora_rank=2,lora_alpha=2,checkpoint_every=2,
    gradient_accumulation_steps=1,logging='jsonl',resolution=64,ema_decay=.9,beta=2.,learning_rate=.01,warmup_steps=0)
c['training'].pop('comparison_budget');c['data']={'require_reviewed':True};c['evaluation']={'enabled':False}
rows=[{'pair_id':str(i),'split':'train','reviewed':True,
       'prompts':[{'variant_id':'canonical','prompt':f'test {i}'}]} for i in range(3)]
images=torch.linspace(-.8,.8,2*3*64*64).reshape(1,2,3,64,64)
try:
    with patch('self_correct_dpo.training.image_batch',return_value=images),redirect_stdout(io.StringIO()):
        train(c,rows,root/'full',backend_factory=TinyFlow,device='cpu')
        train(c,rows,root/'split',backend_factory=TinyFlow,device='cpu',stop_after_step=2)
        train(c,rows,root/'split',backend_factory=TinyFlow,device='cpu',resume=root/'split/checkpoints/latest.json')
finally:cleanup()
