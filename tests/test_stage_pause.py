import copy
from contextlib import redirect_stdout
import io
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from self_correct_dpo.common import read_json, sha256
from self_correct_dpo.training import train
from self_correct_dpo.checkpoints import resolve_checkpoint


class ToyBackend:
    interrupt = False

    def __init__(self, model, training, device):
        self.weight = torch.nn.Parameter(torch.tensor(.2))

    def train(self):
        pass

    def trainable_parameters(self):
        return [self.weight]

    def save_adapter(self, path):
        Path(path).mkdir()
        torch.save(self.weight.detach(), Path(path) / 'weight.pt')

    def load_adapter(self, path):
        with torch.no_grad():
            self.weight.copy_(torch.load(Path(path) / 'weight.pt', weights_only=True))

    def view_errors(self, prepared, view):
        return (prepared[:, view:view+1] + self.weight).square()

    def score_errors(self, images, prompts, generator):
        if self.interrupt: signal.raise_signal(signal.SIGTERM)
        prepared = torch.randn(len(prompts), 2, 2, generator=generator)
        return {'prepared': prepared, 'policy_errors': (prepared + self.weight).square(),
                'reference_errors': prepared.square()}

    def pair_errors(self, images, prompts, generator, reference=True):
        prepared = torch.randn(len(prompts), 1, 2, generator=generator)
        return {'policy_errors': (prepared + self.weight).square(), 'reference_errors': prepared.square()}


class StagePause(unittest.TestCase):
    def test_200_step_recovery_and_1000_step_permanent_lora_survive_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'base';base.mkdir()
            (base/'model_index.json').write_text('{}')
            rows=[{'pair_id':str(i),'split':'train','reviewed':True,
                   'prompts':[{'variant_id':'canonical','prompt':f'test {i}'}]} for i in range(3)]
            config={'seed':17,'model':{'family':'sd15','path':str(base)},'data':{},
                'training':{'formal_arm':'C','noise_views':2,'objective':'dpo','batch_size':1,
                    'max_steps':1200,'gradient_accumulation_steps':1,'checkpoint_every':200,'resolution':64,
                    'beta':2.,'temperature':.1,'learning_rate':.01,'max_grad_norm':1.,
                    'checkpoint_policy':'bounded','lora_checkpoint_every':1000,'max_lora_checkpoints':None,
                    'prompt_mode':'canonical','sampling':'balanced'}}
            from self_correct_dpo.checkpoints import ManagedCheckpoints
            original_save=ManagedCheckpoints.save_resume
            committed=[]
            def save(manager,*args,**kwargs):
                path=original_save(manager,*args,**kwargs)
                committed.append(read_json(path/'checkpoint.json')['step'])
                self.assertEqual(resolve_checkpoint(manager.root/'latest.json'),path)
                self.assertEqual(len(list(manager.root.glob('resume-*'))),1)
                self.assertTrue((path/'adapter/weight.pt').is_file())
                self.assertTrue((path/'state.pt').is_file())
                return path
            calls=0
            def interrupted_images(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==1051:raise RuntimeError('intentional preemption without a graceful save')
                return torch.zeros(1,2,3,1,1)
            out=root/'staged'
            with redirect_stdout(io.StringIO()),patch.object(ManagedCheckpoints,'save_resume',save),patch(
                    'self_correct_dpo.training.image_batch',side_effect=interrupted_images):
                with self.assertRaisesRegex(RuntimeError,'intentional preemption'):
                    train(config,rows,out,backend_factory=ToyBackend,device='cpu')
            self.assertEqual(committed,[200,400,600,800,1000])
            self.assertFalse((out/'complete.json').exists())
            rolling=resolve_checkpoint(out/'checkpoints/latest.json')
            self.assertEqual(read_json(rolling/'checkpoint.json')['step'],1000)
            permanent=out/'checkpoints/step-0001000'
            permanent_hash=sha256(permanent/'adapter/weight.pt')
            self.assertFalse((permanent/'state.pt').exists())
            with redirect_stdout(io.StringIO()),patch.object(ManagedCheckpoints,'save_resume',save),patch(
                    'self_correct_dpo.training.image_batch',return_value=torch.zeros(1,2,3,1,1)):
                resumed=train(config,rows,out,resume=out/'checkpoints/latest.json',backend_factory=ToyBackend,device='cpu')
            self.assertTrue(resumed['complete'])
            self.assertEqual(committed,[200,400,600,800,1000,1200])
            self.assertFalse(rolling.exists())
            self.assertEqual(sha256(permanent/'adapter/weight.pt'),permanent_hash)
            self.assertEqual(sorted(p.name for p in (out/'checkpoints').glob('step-*')),
                             ['step-0001000','step-0001200'])
            # Recovery replays the 50 unsaved updates from the last committed model/state.
            with redirect_stdout(io.StringIO()),patch('self_correct_dpo.training.image_batch',return_value=torch.zeros(1,2,3,1,1)):
                train(config,rows,root/'full',backend_factory=ToyBackend,device='cpu')
            a=resolve_checkpoint(out/'checkpoints/latest.json')
            b=resolve_checkpoint(root/'full/checkpoints/latest.json')
            torch.testing.assert_close(torch.load(a/'adapter/weight.pt',weights_only=True),
                                       torch.load(b/'adapter/weight.pt',weights_only=True),rtol=0,atol=0)
            a,b=(torch.load(p/'state.pt',weights_only=True) for p in (a,b))
            self.assertEqual(a['scheduler'],b['scheduler'])
            self.assertEqual(a['cost']['exposure'],b['cost']['exposure'])
            torch.testing.assert_close(a['generator'],b['generator'],rtol=0,atol=0)
            for key in ('exp_avg','exp_avg_sq','step'):
                torch.testing.assert_close(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key],rtol=0,atol=0)

    def test_interval_loras_are_retained_without_a_count_cap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'base';base.mkdir()
            (base/'model_index.json').write_text('{}')
            rows=[{'pair_id':str(i),'split':'train','reviewed':True,
                   'prompts':[{'variant_id':'canonical','prompt':f'test {i}'}]} for i in range(3)]
            config={'seed':17,'model':{'family':'sd15','path':str(base)},'data':{},
                'training':{'formal_arm':'C','noise_views':2,'objective':'dpo','batch_size':1,
                    'max_steps':12,'gradient_accumulation_steps':1,'checkpoint_every':1,'resolution':64,
                    'beta':2.,'temperature':.1,'learning_rate':.01,'max_grad_norm':1.,
                    'checkpoint_policy':'bounded','lora_checkpoint_every':2,'max_lora_checkpoints':None,
                    'prompt_mode':'canonical','sampling':'balanced'}}
            with patch('self_correct_dpo.training.image_batch',return_value=torch.zeros(1,2,3,1,1)):
                train(config,rows,root/'out',backend_factory=ToyBackend,device='cpu')
            checkpoint=root/'out/checkpoints'
            self.assertEqual(sorted(p.name for p in checkpoint.glob('step-*')),[f'step-{step:07d}' for step in (2,4,6,8,10,12)])
            self.assertEqual(len(list(checkpoint.glob('resume-*'))),1)

    def test_continuous_and_paused_resume_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            base.mkdir()
            (base / 'model_index.json').write_text('{}')
            rows = [{'pair_id': str(i), 'split': 'train', 'reviewed': True,
                     'prompts': [{'variant_id': 'canonical', 'prompt': 'test'}]} for i in range(5)]
            config = {'seed': 17, 'model': {'family': 'sd15', 'path': str(base)}, 'data': {},
                      'training': {'objective': 'dpo', 'batch_size': 2, 'max_steps': 4,
                                   'gradient_accumulation_steps': 2, 'checkpoint_every': 3,
                                   'resolution': 64, 'beta': 2., 'temperature': .1,
                                   'learning_rate': .01, 'max_grad_norm': 1., 'lr_schedule': 'cosine',
                                   'checkpoint_policy': 'bounded', 'checkpoint_fractions': [1.],
                                   'prompt_mode': 'canonical', 'sampling': 'balanced', 'noise_views': 2}}
            def run(conf, out, **kwargs):
                with patch('self_correct_dpo.training.image_batch', return_value=torch.zeros(2, 2, 3, 1, 1)):
                    return train(conf, rows, out, backend_factory=ToyBackend, device='cpu', **kwargs)
            for arm in ('B', 'C'):
                conf = copy.deepcopy(config)
                conf['training']['formal_arm'] = arm
                full = root / (arm + '-full')
                staged = root / (arm + '-staged')
                continuous = run(conf, full)
                self.assertEqual(continuous['cost']['pairs_seen'], 16)
                self.assertEqual(continuous['cost']['policy_image_evaluations'], 64)
                self.assertEqual(continuous['cost']['reference_image_evaluations'], 64)
                paused = run(conf, staged, stop_after_step=2)
                self.assertTrue(paused['planned_pause'])
                self.assertFalse(paused['complete'])
                self.assertFalse((staged / 'complete.json').exists())
                checkpoint = resolve_checkpoint(staged / 'checkpoints/latest.json')
                self.assertEqual(read_json(checkpoint / 'checkpoint.json')['step'], 2)
                resumed = run(conf, staged, resume=staged / 'checkpoints/latest.json')
                self.assertTrue(resumed['complete'])
                self.assertFalse(resumed['planned_pause'])
                self.assertEqual(continuous['run_id'], paused['run_id'])
                self.assertEqual(continuous['run_id'], resumed['run_id'])
                end_full = resolve_checkpoint(full / 'checkpoints/latest.json')
                end_staged = resolve_checkpoint(staged / 'checkpoints/latest.json')
                torch.testing.assert_close(torch.load(end_full / 'adapter/weight.pt', weights_only=True),
                                           torch.load(end_staged / 'adapter/weight.pt', weights_only=True), rtol=0, atol=0)
                a = torch.load(end_full / 'state.pt', weights_only=True)
                b = torch.load(end_staged / 'state.pt', weights_only=True)
                self.assertEqual(a['scheduler'], b['scheduler'])
                for key in ('exp_avg', 'exp_avg_sq', 'step'):
                    torch.testing.assert_close(a['optimizer']['state'][0][key], b['optimizer']['state'][0][key], rtol=0, atol=0)
                self.assertTrue(torch.equal(a['generator'], b['generator']))
                self.assertEqual(a['cost']['exposure'], b['cost']['exposure'])
            conf['training'].update(formal_arm='C', max_steps=10000)
            budget_pause = run(conf, root / 'budget', stop_after_step=2)
            self.assertTrue(budget_pause['planned_pause'])
            self.assertEqual(read_json(root / 'budget/training.json')['config']['training']['max_steps'], 10000)
            with patch.object(ToyBackend, 'interrupt', True):
                interrupted = run(conf, root / 'signal', stop_after_step=1)
            self.assertFalse(interrupted['planned_pause'])
            self.assertFalse(interrupted['complete'])
            self.assertFalse((root / 'signal/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
