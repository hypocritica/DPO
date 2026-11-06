"""Flow math, independent-view sampling and real tiny PEFT EMA resume checks."""
import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import torch
from peft import LoraConfig, get_peft_model
from self_correct_dpo.algorithms.adapter_ema import AdapterEMA
from self_correct_dpo.algorithms.formal import MultiViewBackend, formal_loss
from self_correct_dpo.algorithms.objectives import baseline_loss
from self_correct_dpo.models import PreferenceBackend
from self_correct_dpo.training import train, validate_config
from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
from run_flow import build_config
from test_joint import backend


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__();self.proj=torch.nn.Conv2d(1,1,1,bias=False)
    def forward(self, hidden_states, timestep=None, return_dict=False, **kwargs):
        return (self.proj(hidden_states),)


class TinyFlow(MultiViewBackend):
    def __init__(self, model, training, device):
        self.family,self.model_config,self.training_config=model['family'],model,training
        self.device,self.dtype=torch.device('cpu'),torch.float32
        self.resolution,self.frozen_modules,self.component_name=64,[],'transformer'
        self.error_parameterization='flow_velocity_mse_uniform_schedule_index'
        self.model=get_peft_model(TinyDenoiser(),LoraConfig(r=2,lora_alpha=2,target_modules=['proj'],init_lora_weights='gaussian'))
        self.ema=AdapterEMA(self.model,training.get('ema_decay',.9)) if training.get('reference_mode')=='ema' else None
        self.scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),
            timesteps=torch.arange(1000,0,-1),sigmas=torch.linspace(1,.001,1000))
        self._autocast=nullcontext
    def _encode_images(self,images,generator):return images[:,:,:1,:2,:2].detach()
    def _condition(self,prompts):return {'encoder_hidden_states':torch.zeros(len(prompts),1)}


class FlowTests(unittest.TestCase):
    def test_paper_pilot_preserves_pairs_and_evaluates_after_warmup(self):
        from self_correct_dpo.runtime.pipeline import evaluation_steps
        for family in ('sd35_medium', 'zimage'):
            for algorithm in ('dpo', 'mvpo', 'linear_dpo'):
                c = build_config(family, algorithm, recipe='paper-pilot', world=4)
                t = c['training']
                self.assertEqual(t['gradient_accumulation_steps'] * t['batch_size'] * 4, 128)
                self.assertEqual(t['batch_size'], 4 if family == 'sd35_medium' else 2)
                self.assertEqual(t['comparison_budget']['unit'], 'pairs')
                self.assertEqual(t['max_steps'], 3200)
                self.assertFalse(t['offload_frozen'])
                self.assertEqual(evaluation_steps(3200, c['evaluation']),
                                 [0, 200, 400, 800, 1300, 1800, 2300, 2800, 3200])
                from self_correct_dpo.runtime.checkpoints import progress_steps
                self.assertEqual(progress_steps(t),
                                 [200, 400, 800, 1200, 1300, 1600, 1800, 2000, 2300, 2400, 2800, 3200])
                self.assertEqual(t['full_checkpoint_every'], 400)
                self.assertEqual(c['evaluation']['expected_prompts'], 64)
        small_gpu = build_config('sd35_medium', 'mvpo', recipe='paper-pilot', profile='5090')
        self.assertFalse(small_gpu['training']['offload_frozen'])
        self.assertEqual(small_gpu['training']['batch_size'], 1)
        equal_compute = build_config('sd35_medium', 'mvpo', recipe='paper-pilot', budget_unit='pair_views')
        self.assertEqual(equal_compute['training']['gradient_accumulation_steps'] * equal_compute['training']['batch_size'] * 2 * 2, 128)
        with self.assertRaises(ValueError):
            build_config('sd35_medium', 'dpo', recipe='paper-pilot', smoke=True)

    def test_flow_condition_encodes_unique_captions_and_restores_pair_order(self):
        from self_correct_dpo.models import check_prompt_lengths
        for family in ('sd35_medium', 'zimage'):
            b = object.__new__(PreferenceBackend)
            b.family, b.frozen_modules, b.pipe = family, [], SimpleNamespace(vae=object())
            calls = []
            def encode(prompts, prompt_weights=None):
                calls.append((prompts, prompt_weights))
                if family == 'zimage':
                    return {'cap_feats': [torch.full((i+1, 2), float(i)) for i in range(len(prompts))]}
                return {'encoder_hidden_states': torch.tensor([[1.], [2.]]),
                        'pooled_projections': torch.tensor([[3.], [4.]])}
            b._condition_on_device = encode
            result = b._condition(['a', 'a', 'b', 'b', 'a', 'a'])
            self.assertEqual(calls, [(['a', 'b'], [4, 2])])
            if family == 'zimage':
                self.assertEqual([len(x) for x in result['cap_feats']], [1, 1, 2, 2, 1, 1])
            else:
                self.assertEqual(result['encoder_hidden_states'].flatten().tolist(), [1., 1., 2., 2., 1., 1.])
        tokenizer = lambda *args, **kw: {'input_ids': [[1, 2, 3], [1]]}
        self.assertEqual(check_prompt_lengths(tokenizer, ['a', 'b'], 2, 'test', 'truncate', [4, 2]), 4)

    def test_2_4_gpu_budgets_configs_and_stagewise(self):
        for family in ('sd35_medium','zimage'):
            for algorithm in ('dpo','mvpo','linear_dpo'):
                for world in (2,4):
                    for unit in ('pairs','pair_views'):
                        c=build_config(family,algorithm,world=world,budget_unit=unit)
                        t=c['training'];k=t['noise_views']
                        self.assertEqual(t['batch_size']*t['gradient_accumulation_steps']*world*(k if unit=='pair_views' else 1),128)
                        validate_config(c,files=False)
            c=build_config(family,'mvpo',stagewise=True)
            self.assertEqual(c['training']['stagewise']['objective'],'dpo')
        with self.assertRaises(ValueError):build_config('zimage','dpo',profile='5090')
        with self.assertRaises(ValueError):build_config('sd35_medium','mvpo',views=3)

    def test_flow_sampling_pair_sharing_independent_views_and_joint_gradients(self):
        images=torch.linspace(-.8,.8,2*2*3*64*64).reshape(2,2,3,64,64)
        for family in ('sd35_medium','zimage'):
            for k in (1,2,4):
                for stagewise in (False,True):
                    results=[]
                    for packed in (False,True):
                        b=backend(k=k,rollout=False);b.family=family
                        b.training_config['batch_views']=packed
                        if stagewise:b.training_config['stagewise']={'boundary':437}
                        b.scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),
                            timesteps=torch.arange(1000,0,-1),sigmas=torch.linspace(1,.001,1000))
                        g=torch.Generator().manual_seed(11)
                        values=b.score_errors(images,[['p']*k]*2,g)
                        for view in values['prepared']['views']:
                            self.assertTrue(torch.equal(view['times'][::2],view['times'][1::2]))
                            flat=images[:,:,:1,:2,:2].flatten(0,1)
                            noise=view['target']+flat
                            torch.testing.assert_close(noise[::2],noise[1::2])
                            sigma=view['times'].float()/1000
                            torch.testing.assert_close(view['noisy'],flat*(1-sigma[:,None,None,None])+sigma[:,None,None,None]*noise)
                            if stagewise:
                                self.assertTrue(torch.equal((1000-view['times'][::2])<437,values['prepared']['low_stage']))
                        if k>1:self.assertFalse(torch.equal(values['prepared']['views'][0]['target'],values['prepared']['views'][1]['target']))
                        loss,_=formal_loss(values,'C',2.)
                        if k==1:
                            other,_=formal_loss(values,'B',2.);torch.testing.assert_close(loss,other)
                        loss.backward();results.append((loss.detach(),b.weight.grad,g.get_state()))
                    torch.testing.assert_close(results[0][0],results[1][0]);torch.testing.assert_close(results[0][1],results[1][1])
                    self.assertTrue(torch.equal(results[0][2],results[1][2]))

    def test_margin_sign_linear_detached_weight_and_no_batch_aggregation(self):
        p=torch.tensor([[[.1,.6]],[[.5,.3]]],requires_grad=True)
        r=torch.tensor([[[.4,.4]],[[.4,.4]]],requires_grad=True)
        values={'policy_errors':p,'reference_errors':r}
        loss,metrics=formal_loss(values,'C',2.)
        self.assertLess(loss.item(),torch.nn.functional.softplus(torch.tensor(.2)).item())
        loss.backward();self.assertGreater(p.grad[0,0,0].item(),0);self.assertLess(p.grad[0,0,1].item(),0)
        self.assertIsNone(r.grad)
        p.grad=None
        loss,metrics=baseline_loss(values,'linear_dpo',10.,linear_eta=.1)
        loss.backward()
        torch.testing.assert_close(p.grad,torch.tensor([[[.05,-.05]],[[.45,-.45]]]))
        self.assertIsNone(r.grad)

    def test_real_peft_ema_no_policy_mutation_and_strict_state(self):
        torch.manual_seed(4)
        model=get_peft_model(TinyDenoiser(),LoraConfig(r=2,lora_alpha=2,target_modules=['proj']))
        ema=AdapterEMA(model,.5)
        ids=[id(p) for p in model.parameters()]
        x=torch.randn(2,1,2,2)
        out=model(x)[0]
        expected=torch.autograd.grad(out.square().mean(),[p for p in model.parameters() if p.requires_grad],retain_graph=True)
        ref=ema.forward(model,(x,),{})[0]
        self.assertFalse(ref.requires_grad);torch.testing.assert_close(out,ref)
        actual=torch.autograd.grad(out.square().mean(),[p for p in model.parameters() if p.requires_grad])
        for left,right in zip(expected,actual):torch.testing.assert_close(left,right)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        for p,g in zip([p for p in model.parameters() if p.requires_grad],actual):p.grad=g
        before=copy.deepcopy(ema.state_dict());opt.step();ema.update(model)
        for name,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(ema.shadow[name],.5*before['shadow'][name]+.5*p)
        self.assertEqual(ids,[id(p) for p in model.parameters()])
        state=ema.state_dict();restored=AdapterEMA(model,.5);restored.load_state_dict(state,model)
        self.assertEqual(restored.updates,1)
        for key in ema.shadow:torch.testing.assert_close(restored.shadow[key],ema.shadow[key])
        for bad in (None,{**state,'updates':-1},{**state,'shadow':{}}):
            with self.assertRaises(ValueError):restored.load_state_dict(bad,model)

    def test_production_checkpoint_and_ema_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir();(base/'model_index.json').write_text('{}')
            c=build_config('sd35_medium','linear_dpo',smoke=True)
            c['model'].update(path=str(base),revision='tiny-test')
            c['training'].update(world_size=1,max_steps=4,lora_rank=2,lora_alpha=2,checkpoint_every=2,
                gradient_accumulation_steps=1,logging='jsonl',resolution=64,ema_decay=.9,beta=2.,learning_rate=.01)
            c['training'].pop('comparison_budget');c['data']={'require_reviewed':True}
            rows=[{'pair_id':str(i),'split':'train','reviewed':True,
                   'prompts':[{'variant_id':'canonical','prompt':f'test {i}'}]} for i in range(3)]
            images=torch.linspace(-.8,.8,2*3*64*64).reshape(1,2,3,64,64)
            with patch('self_correct_dpo.training.image_batch',return_value=images),redirect_stdout(io.StringIO()):
                train(c,rows,root/'full',backend_factory=TinyFlow,device='cpu')
                train(c,rows,root/'split',backend_factory=TinyFlow,device='cpu',stop_after_step=2)
                mid=resolve_checkpoint(root/'split/checkpoints/latest.json')
                midstate=torch.load(mid/'state.pt',weights_only=True)
                self.assertEqual(midstate['reference_state']['updates'],2)
                train(c,rows,root/'split',backend_factory=TinyFlow,device='cpu',resume=root/'split/checkpoints/latest.json')
            from safetensors.torch import load_file
            states=[]
            for name in ('full','split'):
                checkpoint=resolve_checkpoint(root/name/'checkpoints/latest.json')
                self.assertIn('state.pt',json.loads((checkpoint/'checkpoint.json').read_text())['files'])
                states.append(torch.load(checkpoint/'state.pt',weights_only=True))
                adapter=load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
                if name=='full':expected=adapter
                else:
                    for key in adapter:torch.testing.assert_close(adapter[key],expected[key],rtol=0,atol=0)
            for key in states[0]['reference_state']['shadow']:
                torch.testing.assert_close(states[0]['reference_state']['shadow'][key],states[1]['reference_state']['shadow'][key],rtol=0,atol=0)
            self.assertEqual(states[1]['reference_state']['updates'],4)
            for key in ('generator','torch_rng'):
                if key in states[0]:self.assertTrue(torch.equal(states[0][key],states[1][key]))
            for i,v in states[0]['optimizer']['state'].items():
                for key in v:torch.testing.assert_close(v[key],states[1]['optimizer']['state'][i][key],rtol=0,atol=0)

    def test_two_rank_gloo_policy_ema_optimizer_and_rng_resume(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'base').mkdir();(root/'base/model_index.json').write_text('{}')
            result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc-per-node=2',str(ROOT/'tests/flow_ddp_worker.py'),str(root)],
                check=False,timeout=120,env={**os.environ,'OMP_NUM_THREADS':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertEqual(result.returncode,0,result.stdout)
            states=[]
            for name in ('full','split'):
                checkpoint=resolve_checkpoint(root/name/'checkpoints/latest.json')
                states.append(torch.load(checkpoint/'state.pt',weights_only=True))
            for key,value in states[0]['reference_state']['shadow'].items():
                torch.testing.assert_close(value,states[1]['reference_state']['shadow'][key],rtol=0,atol=0)
            self.assertEqual(states[1]['reference_state']['updates'],4)
            self.assertEqual(len(states[1]['rank_states']),2)
            for rank in range(2):
                self.assertEqual(states[1]['rank_states'][rank]['rank'],rank)
                for key in ('generator','cpu_rng'):
                    self.assertTrue(torch.equal(states[0]['rank_states'][rank][key],states[1]['rank_states'][rank][key]))
            self.assertFalse(torch.equal(states[1]['rank_states'][0]['generator'],states[1]['rank_states'][1]['generator']))
            for i,v in states[0]['optimizer']['state'].items():
                for key in v:torch.testing.assert_close(v[key],states[1]['optimizer']['state'][i][key],rtol=0,atol=0)

    def test_qwen_truncation_mask_and_generation_negative_audit(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from self_correct_dpo.evaluation.render import check_prompt_lengths as render_lengths
        tokens=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'a':2,'b':3,'c':4,'d':5,'chat':6},unk_token='[UNK]'))
        tokens.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokens,pad_token='[PAD]',unk_token='[UNK]')
        tokenizer.chat_template="{{ messages[0]['content'] }} chat"
        b=object.__new__(PreferenceBackend)
        b.family,b.device,b.dtype,b.max_sequence_length='zimage',torch.device('cpu'),torch.float32,3
        b.model_config={'prompt_truncation':'truncate'}
        def encoder(ids,**kwargs):
            hidden=ids.float().unsqueeze(-1).repeat(1,1,4)
            return SimpleNamespace(hidden_states=[hidden,hidden])
        b.pipe=SimpleNamespace(tokenizer=tokenizer,text_encoder=encoder)
        condition=b._condition_on_device(['a b c d','a'])
        self.assertEqual([len(row) for row in condition['cap_feats']],[3,2])
        self.assertEqual(b.last_prompt_truncations,{'Qwen3_chat':1})
        b.model_config['prompt_truncation']='error'
        with self.assertRaisesRegex(ValueError,'truncation'):b._condition_on_device(['a b c d'])
        audit=render_lengths(b.pipe,[{'scene_id':'s','variant_id':'v','prompt':'a b c d'}],
            {'negative_prompt':'a b c d','max_sequence_length':3,'prompt_truncation':'truncate'},'zimage')
        self.assertEqual([row['prompt'] for row in audit],['s / v','negative_prompt'])

    def test_fp32_decode_and_zimage_native_forward(self):
        from self_correct_dpo.evaluation.render import cast_vae_decode_input
        seen=[]
        vae=SimpleNamespace(dtype=torch.float32,decode=lambda x,**kw:seen.append(x.dtype) or x)
        cast_vae_decode_input(vae);vae.decode(torch.ones(1,dtype=torch.bfloat16))
        self.assertEqual(seen,[torch.float32])
        b=object.__new__(PreferenceBackend);b.family='zimage'
        def forward(**kwargs):
            self.assertEqual([tuple(x.shape) for x in kwargs['x']],[(2,1,3,3)]*2)
            torch.testing.assert_close(kwargs['t'],torch.tensor([.2,.8]))
            return ([-x for x in kwargs['x']],)
        b.model=forward
        value=torch.randn(2,2,3,3)
        prediction=b._predict(value,torch.tensor([800,200]),{'cap_feats':[torch.ones(3,4),torch.ones(5,4)]})
        torch.testing.assert_close(prediction,value)


if __name__=='__main__':unittest.main()
