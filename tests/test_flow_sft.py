"""Winner-only flow SFT: parameterization, budget and a tiny CPU resume."""
from contextlib import nullcontext,redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from self_correct_dpo.algorithms.joint import BaselineGaussianBackend
from self_correct_dpo.algorithms.objectives import baseline_loss
from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
from self_correct_dpo.runtime.pipeline import evaluation_steps
from self_correct_dpo.training import train,training_identity,validate_config
from run_flow import build_config


class WinnerVAE:
    def __init__(self):
        self.config=SimpleNamespace(scaling_factor=.3611,shift_factor=.1159)
        self.encoded=[];self.placements=[]
    def to(self,device):self.placements.append(str(device));return self
    def encode(self,images):
        self.encoded.append(images.clone())
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator:images[:,:1,:2,:2]))


def tiny_sft(family):
    b=object.__new__(BaselineGaussianBackend)
    b.family,b.device,b.dtype=family,torch.device('cpu'),torch.float32
    b.training_config={'objective':'sft','noise_views':1,'vae_batch_size':4,'preflight':'once'}
    b.pipe=SimpleNamespace(vae=WinnerVAE());b.offload_frozen=True;b.ema=None
    b.weight=torch.nn.Parameter(torch.tensor(.2))
    sigma=torch.linspace(1,.001,1000);shift=3 if family=='sd35_medium' else 6
    sigma=shift*sigma/(1+(shift-1)*sigma)
    b.scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),sigmas=sigma,timesteps=sigma*1000)
    b._autocast=nullcontext;b.texts=[];b.predictions=[]
    b._condition=lambda texts:b.texts.append(texts) or {'encoder_hidden_states':torch.ones(len(texts),1)}
    def predict(x,t,c):
        b.predictions.append((x.detach().clone(),t.clone()));return b.weight*x
    b._predict=predict
    def forbidden():raise AssertionError('SFT must not create/forward a reference')
    b.reference_context=forbidden
    return b


class TrainSFT(BaselineGaussianBackend):
    def __init__(self,model,settings,device):
        self.__dict__.update(tiny_sft(model['family']).__dict__)
        self.training_config=settings
    def train(self):return self
    def trainable_parameters(self):return [self.weight]
    def save_adapter(self,path):
        Path(path).mkdir();torch.save(self.weight.detach(),Path(path)/'weight.pt')
    def load_adapter(self,path):
        with torch.no_grad():self.weight.copy_(torch.load(Path(path)/'weight.pt',weights_only=True))


class FlowSFTTests(unittest.TestCase):
    def test_both_backbones_encode_only_winner_and_regress_flow_velocity(self):
        for family in ('sd35_medium','zimage'):
            b=tiny_sft(family)
            images=torch.linspace(-.8,.8,2*2*3*4*4).reshape(2,2,3,4,4)
            images[:,1]=float('nan')  # Any loser access would poison the loss.
            generator=torch.Generator().manual_seed(7)
            values=b.score_errors(images,[['a'],['b']],generator)
            replay=torch.Generator().manual_seed(7)
            latent=(images[:,0,:1,:2,:2]-.1159)*.3611
            noise=torch.randn(latent.shape,generator=replay)
            indices=torch.randint(1000,(2,),generator=replay)
            sigma=b.scheduler.sigmas[indices,None,None,None]
            target=noise-latent;noisy=(1-sigma)*latent+sigma*noise
            torch.testing.assert_close(b.predictions[0][0],noisy)
            torch.testing.assert_close(b.predictions[0][1],b.scheduler.timesteps[indices])
            torch.testing.assert_close(b.pipe.vae.encoded[0],images[:,0])
            self.assertEqual(b.pipe.vae.placements,['cpu','cpu'])
            self.assertEqual(b.texts,[['a','b']]);self.assertEqual(len(b.predictions),1)
            self.assertEqual(values['policy_errors'].shape,(2,1,1))
            self.assertEqual(values['policy_errors'].dtype,torch.float32);self.assertIsNone(values['reference_errors'])
            self.assertTrue(torch.equal(generator.get_state(),replay.get_state()))
            loss,_=baseline_loss(values,'sft',500)
            weight=torch.tensor(.2,requires_grad=True)
            expected=(weight*noisy-target).square().mean()
            loss.backward();expected.backward()
            torch.testing.assert_close(loss,expected);torch.testing.assert_close(b.weight.grad,weight.grad)

    def test_sft_config_winner_budget_and_existing_schedule(self):
        points=[0,200,400,800,1300,1800,2300,2800,3200]
        for family in ('sd35_medium','zimage'):
            config=build_config(family,'sft',world=2,recipe='paper-pilot')
            t=config['training'];validate_config(config,files=False)
            self.assertEqual((t['objective'],t['reference_mode'],t['noise_views']),('sft','none',1))
            self.assertNotIn('ema_decay',t);self.assertNotIn('linear_eta',t)
            self.assertEqual(t['batch_size']*t['world_size']*t['gradient_accumulation_steps'],128)
            self.assertEqual((t['max_steps'],t['learning_rate'],t['warmup_steps']),(3200,5e-6,200))
            self.assertEqual(t['preflight'],'once')
            self.assertEqual(evaluation_steps(3200,config['evaluation']),points)
            smoke=build_config(family,'sft',world=1,smoke=True)
            self.assertEqual(smoke['training']['lora_checkpoint_steps'],[2])
        c=build_config('sd35_medium','sft',world=4,profile='5090',recipe='paper-pilot')
        self.assertEqual((c['training']['batch_size'],c['training']['gradient_accumulation_steps']),(1,32))

    def test_training_never_reads_loser_and_records_winner_budget_on_resume(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir();(base/'model_index.json').write_text('{}')
            winner=root/'winner.png';Image.new('RGB',(64,64),(42,63,84)).save(winner)
            rows=[{'pair_id':str(i),'split':'train','reviewed':True,'winner':str(winner),
                   'loser':str(root/'DO_NOT_READ_LOSER.png'),
                   'prompts':[{'variant_id':'canonical','prompt':f'test {i}'}]} for i in range(3)]
            c=build_config('zimage','sft',world=1,smoke=True)
            c['model'].update(path=str(base),revision='tiny-test');c['data']={'require_reviewed':True,'num_workers':0}
            c['training'].update(resolution=64,learning_rate=.01);c['training'].pop('comparison_budget')
            saved={'config':c,'rows':rows,'validation_rows':[],'identity':training_identity(c,rows,[])}
            with redirect_stdout(io.StringIO()):
                train(c,rows,root/'full',backend_factory=TrainSFT,device='cpu',preflight=saved)
                train(c,rows,root/'split',backend_factory=TrainSFT,device='cpu',preflight=saved,stop_after_step=1)
                train(c,rows,root/'split',backend_factory=TrainSFT,device='cpu',preflight=saved,
                    resume=root/'split/checkpoints/latest.json')
            states=[]
            for name in ('full','split'):
                checkpoint=resolve_checkpoint(root/name/'checkpoints/latest.json')
                states.append(torch.load(checkpoint/'state.pt',weights_only=True))
            self.assertEqual(states[-1]['cost']['policy_image_evaluations'],2)
            self.assertEqual(states[-1]['cost']['reference_image_evaluations'],0)
            self.assertIsNone(states[-1]['reference_state'])
            for key in ('exp_avg','exp_avg_sq','step'):
                torch.testing.assert_close(states[0]['optimizer']['state'][0][key],states[1]['optimizer']['state'][0][key],rtol=0,atol=0)
            info=json.loads((root/'split/training.json').read_text())
            self.assertEqual(info['supervised_winner_images_per_update'],1)
            self.assertEqual(info['sampled_pair_rows_per_update'],1)
            self.assertEqual(info['effective_batch_size_pairs'],0)
            self.assertEqual(info['effective_batch_size_pair_views'],0)
            records=[json.loads(line) for log in (root/'split').glob('metrics-*.jsonl')
                     for line in log.read_text().splitlines() if line.strip()]
            record=max(records,key=lambda row:row['step'])
            self.assertEqual(record['winner_images_seen'],2)
            self.assertEqual(record['pair_views_seen'],0)
            self.assertGreater(record['winner_images_per_second'],0)


if __name__=='__main__':unittest.main()
