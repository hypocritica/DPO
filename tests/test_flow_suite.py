"""Actual launcher budgets and the Flow Stagewise LossAvg control."""
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import torch
import torch.nn.functional as F
import run_flow
import run_flow_suite as suite
from self_correct_dpo.algorithms.formal import formal_loss, stagewise_metrics
from self_correct_dpo.algorithms.losses import preference_logits
from self_correct_dpo.training import train, validate_config
from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
from test_flow import TinyFlow


class FlowSuiteTests(unittest.TestCase):
    def test_all_cases_real_shell_and_config_builder_have_exact_budget_and_unique_outputs(self):
        # Record the final Python argv, then use the real run_flow config builder
        # in this CPU process. Only lock/file writes are replaced, not parsing.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shim = root / 'python'
            shim.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            shim.chmod(0o755)
            env = dict(os.environ, SCD_FLOW_PYTHON=str(shim), SCD_FLOW_CONDA_SH='',
                       SCD_RESOURCES=str(ROOT / 'configs/resources-slai-flow.json'))
            for family, world in (('sd35', 4), ('zimage', 2)):
                paths = set()
                for name, case in suite.CASES.items():
                    cmd = suite.command(family, name, forwarded=['--config-only', '--config-dir',
                        str(root / 'configs'), '--runs-root', str(root / 'runs')])
                    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, f'{family}/{name}: {result.stderr}')
                    argv = json.loads(result.stdout)
                    self.assertEqual(argv[:2], ['-u', 'scripts/run_flow.py'])
                    frozen = []
                    with patch.object(sys, 'argv', ['run_flow.py', *argv[2:]]), \
                         patch.object(run_flow, 'training_lock', side_effect=lambda *a: nullcontext()), \
                         patch.object(run_flow, 'atomic_json', side_effect=lambda p,c: frozen.append((p,c))), \
                         redirect_stdout(io.StringIO()):
                        run_flow.main()
                    path, config = frozen[0]
                    self.assertNotIn(path.name, paths, f'duplicate identity: {family}/{name}')
                    paths.add(path.name)
                    t = config['training']
                    self.assertEqual((t['world_size'], t['batch_size'], t['noise_views']), (world, 4, case['k']))
                    self.assertEqual(t['comparison_budget'], dict(unit=case['unit'], per_update=128))
                    self.assertEqual(world * 4 * t['gradient_accumulation_steps'] *
                                     (case['k'] if case['unit']=='pair_views' else 1), 128)
                    self.assertEqual(t['formal_arm'], case['arm'])
                    self.assertEqual('stagewise' in t, case['stage'])
                    if case['stage']:
                        self.assertEqual(t['stagewise']['boundary'], case['boundary'])
                        self.assertEqual(config['method'], f'stagewise-{case["arm"]}-gaussian')
                    self.assertEqual(t['max_steps'], 3200)
                    self.assertEqual(t['lora_checkpoint_steps'], [200,400,800,1300,1800,2300,2800,3200])
                    self.assertFalse(t['offload_frozen'])
                    self.assertFalse(config['evaluation']['enabled'])
                    self.assertIn('sampling', config['evaluation'])
                    self.assertEqual(config['model'].get('transformer_dtype', t['dtype']),
                                     'float32' if family=='zimage' else 'bfloat16')
                    overrides = dict(zip(case['overrides'][::2], case['overrides'][1::2]))
                    self.assertEqual(t['beta'], int(overrides.get('--beta', 500)))
                    self.assertEqual(config['seed'], int(overrides.get('--seed', 17)))
                    if case['algorithm']=='linear_dpo':
                        self.assertEqual((t['reference_mode'],t['ema_decay'],t['linear_eta']),('ema',.995,.01))

    def test_numbered_entry_and_batch_fallback_use_real_config_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, SCD_FLOW_PYTHON=sys.executable, SCD_FLOW_CONDA_SH='',
                       SCD_RESOURCES=str(ROOT / 'configs/resources-slai-flow.json'))
            for family, script, k, arm, batch in (
                ('sd35','01_dpo_k1.sh',1,'C',4),
                ('zimage','08_stagewise_mvpo_k4_views128.sh',4,'C',4),
                ('zimage','16_stagewise_lossavg_k2_views128.sh',2,'B',4),
                ('sd35','18_stagewise_lossavg_k8_views128.sh',8,'B',2)):
                folder = root / f'{family}-{script}'
                result = subprocess.run(['bash',str(ROOT/'launchers'/family/script), '--config-only',
                    '--config-dir',str(folder),'--runs-root',str(root/'runs'),
                    '--batch-size',str(batch)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                c=json.loads(next(folder.glob('*.json')).read_text());t=c['training']
                self.assertEqual((t['batch_size'],t['noise_views'],t['formal_arm']),(batch,k,arm))
                self.assertEqual(t['world_size']*batch*k*t['gradient_accumulation_steps'],128)

    def test_stage_lossavg_matches_same_view_loss_and_gradient_with_k1_initial_checks(self):
        prepared={'low_stage':torch.tensor([True,False]),'stage_axis':'descending_flow_schedule_index'}
        settings=dict(boundary=500,objective='dpo',target_margin=1.,huber_delta=1.,huber_weight=1.)
        for k in (1,2,4,8):
            p=torch.linspace(-.4,.8,4*k).reshape(2,k,2).requires_grad_()
            r=torch.zeros_like(p)
            values=dict(policy_errors=p,reference_errors=r,prepared=prepared)
            loss,metrics=formal_loss(values,'B',3.,settings)
            expected=F.softplus(-preference_logits(p,r,3.)).mean()
            torch.testing.assert_close(loss,expected,rtol=0,atol=0)
            g,=torch.autograd.grad(loss,p,retain_graph=True)
            other,=torch.autograd.grad(expected,p)
            torch.testing.assert_close(g,other,rtol=0,atol=0)
            self.assertIn('stage_high_noise_loss_weighted',metrics)
            for logits in (torch.zeros(2,k),torch.tensor([[.4],[-.8]]).expand(2,k)):
                logits=logits.clone().requires_grad_()
                b,_=stagewise_metrics(logits,prepared,settings,'B')
                c,_=stagewise_metrics(logits,prepared,settings,'C')
                torch.testing.assert_close(b,c)
                torch.testing.assert_close(torch.autograd.grad(b,logits,retain_graph=True)[0],
                                           torch.autograd.grad(c,logits)[0])
        bad=run_flow.build_config('zimage','mvpo',stagewise=True,aggregation='B')
        bad['model']['family']='sdxl'
        with self.assertRaisesRegex(ValueError,'Flow DPO'):validate_config(bad,files=False)

    def test_stage_lossavg_real_peft_production_save_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/'base';base.mkdir();(base/'model_index.json').write_text('{}')
            c=run_flow.build_config('sd35_medium','mvpo',world=1,smoke=True,smoke_steps=4,
                                    stagewise=True,aggregation='B')
            c['model'].update(path=str(base),revision='tiny-stage-b')
            c['training'].update(resolution=64,checkpoint_every=2,logging='jsonl',beta=2.,learning_rate=.01)
            c['data']={'require_reviewed':True}
            rows=[{'pair_id':str(i),'split':'train','reviewed':True,
                   'prompts':[{'variant_id':'canonical','prompt':f'caption {i}'}]} for i in range(3)]
            images=torch.linspace(-.8,.8,2*3*64*64).reshape(1,2,3,64,64)
            with patch('self_correct_dpo.training.image_batch',return_value=images),redirect_stdout(io.StringIO()):
                train(c,rows,root/'full',backend_factory=TinyFlow,device='cpu')
                train(c,rows,root/'split',backend_factory=TinyFlow,device='cpu',stop_after_step=2)
                train(c,rows,root/'split',backend_factory=TinyFlow,device='cpu',resume=root/'split/checkpoints/latest.json')
            from safetensors.torch import load_file
            checkpoints=[resolve_checkpoint(root/name/'checkpoints/latest.json') for name in ('full','split')]
            a,b=[load_file(str(path/'adapter/adapter_model.safetensors')) for path in checkpoints]
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            a,b=[torch.load(path/'state.pt',weights_only=True) for path in checkpoints]
            for i,values in a['optimizer']['state'].items():
                for key in values:torch.testing.assert_close(values[key],b['optimizer']['state'][i][key],rtol=0,atol=0)
            metrics=[json.loads(line) for p in (root/'full').glob('metrics-*.jsonl') for line in p.read_text().splitlines()]
            self.assertTrue(all(row['loss'] > 0 for row in metrics))
            for row in metrics:
                self.assertAlmostEqual(row['loss'],sum(row[f'loss_view_{j}'] for j in range(2))/2,places=6)

    def test_plan_is_readonly_deduplicated_and_failure_stops_the_group(self):
        with patch('self_correct_dpo.evaluation.runner.run_command') as run,redirect_stdout(io.StringIO()) as output:
            self.assertEqual(suite.main(['zimage','main,dpo','--plan-only','--batch-size','2']),0)
            self.assertEqual(len(json.loads(output.getvalue())),4)
            run.assert_not_called()
        fail=subprocess.CalledProcessError(75,['paused'])
        with patch('self_correct_dpo.evaluation.runner.run_command',side_effect=fail) as run,redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main(['zimage','main']),75)
            self.assertEqual(run.call_count,1)


if __name__=='__main__':unittest.main()
