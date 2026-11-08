"""Continuous training and independent evaluation preserve snapshot identities."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import torch
from peft import LoraConfig, get_peft_model
from run_flow import build_config
from evaluate_flow import EvaluationSession, committed_adapter
from self_correct_dpo.common import read_json
from self_correct_dpo.training import train
from test_stage_pause import ToyBackend


class DetachedEvaluationTests(unittest.TestCase):
    def test_real_numbered_launchers_default_to_trainonly_and_keep_inline_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = dict(os.environ, SCD_FLOW_PYTHON=sys.executable,
                               SCD_RESOURCES=str(ROOT / 'configs/resources-slai-flow.json'))
            for family, launcher in (('sd35_medium', 'sd35/01_dpo_k1.sh'), ('zimage', 'zimage/01_dpo_k1.sh')):
                for inline in (False, True):
                    folder = root / f'{family}-{inline}'
                    command = ['bash', str(ROOT / 'launchers' / launcher), '--config-only',
                               '--config-dir', str(folder), '--runs-root', str(root / 'runs')]
                    if inline:
                        command += ['--inline-evaluation']
                    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    path = next(folder.glob('local-*.json'))
                    config = read_json(path)
                    self.assertEqual(config['evaluation']['enabled'], inline)
                    self.assertEqual('-trainonly-' in path.name, not inline)
                    self.assertEqual(config['evaluation']['expected_prompts'], 64)
                    self.assertIn('sampling', config['evaluation'])
                    self.assertEqual(config['training']['lora_checkpoint_steps'], [200,400,800,1300,1800,2300,2800,3200])
                    self.assertEqual(config['training']['checkpoint_every'], 200)
                    self.assertEqual(config['training']['full_checkpoint_every'], 400)

    def test_default_trainer_launches_one_worker_group_without_evaluation_stops(self):
        from self_correct_dpo import train as entry
        config = build_config('zimage', 'dpo', recipe='paper-v2')
        rows = [{'pair_id': 'train'}]
        saved = {'rows': rows, 'validation_rows': [{'pair_id': 'heldout'}]}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch.object(sys, 'argv', ['train.py', '--out', str(Path(directory) / 'run'), '--execution', 'standalone', '--run-label', 'fixture']), \
             patch.object(entry, 'training_config', return_value=config), \
             patch('self_correct_dpo.training.load_preflight', return_value=saved), \
             patch.object(entry, 'require_gpu_allocation'), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=2), \
             patch('self_correct_dpo.runtime.pipeline.run_pipeline', side_effect=AssertionError('Inline evaluator started')), \
             patch('self_correct_dpo.evaluation.runner.run_command') as launch:
            entry.main()
            self.assertEqual(launch.call_count, 1)
            self.assertNotIn('--stop-after-step', launch.call_args.args[0])
            self.assertIn('--nproc-per-node=2', launch.call_args.args[0])

    def test_permanent_snapshots_are_published_without_reconstructing_the_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'; base.mkdir(); (base / 'model_index.json').write_text('{}')
            rows = [{'pair_id': str(i), 'split': 'train', 'reviewed': True,
                     'prompts': [{'variant_id': 'canonical', 'prompt': f'caption {i}'}]} for i in range(3)]
            config = {'seed': 17, 'model': {'family': 'zimage', 'path': str(base), 'revision': 'fixture'}, 'data': {},
                'training': {'formal_arm': 'C', 'noise_views': 2, 'objective': 'dpo', 'batch_size': 1,
                    'max_steps': 4, 'gradient_accumulation_steps': 1, 'checkpoint_every': 1, 'resolution': 64,
                    'beta': 2., 'temperature': .1, 'learning_rate': .01, 'max_grad_norm': 1.,
                    'checkpoint_policy': 'bounded', 'full_checkpoint_every': 2,
                    'lora_checkpoint_steps': [1,2,4], 'max_lora_checkpoints': None,
                    'prompt_mode': 'canonical', 'sampling': 'balanced', 'logging': 'jsonl'},
                'evaluation': {'enabled': False, 'steps': [1,2,4], 'include_base': True,
                    'prompts': str(root / 'panel.jsonl'), 'scorers_config': str(root / 'scorers.json'), 'sampling': {}}}
            run = root / 'run'
            with patch('self_correct_dpo.training.image_batch', return_value=torch.zeros(1,2,3,1,1)), redirect_stdout(io.StringIO()):
                factory = unittest.mock.Mock(side_effect=ToyBackend)
                train(config, rows, run, backend_factory=factory, device='cpu')
            self.assertEqual(factory.call_count, 1)
            run_id = read_json(run / 'training.json')['run_id']
            for step in (1,2,4):
                self.assertTrue(committed_adapter(run, step, run_id)[0].is_dir())
            (run / 'checkpoints/.pending-step-0000003').mkdir()
            self.assertIsNone(committed_adapter(run, 3, run_id))
            with self.assertRaisesRegex(ValueError, 'run/step'):
                committed_adapter(run, 1, 'wrong-run')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_flow.py'),
                '--run', str(run), '--check-only'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['ready_steps'], [0,1,2,4])
            self.assertFalse((run / 'offline-eval-v1').exists())
            adapter = committed_adapter(run, 1, run_id)[0]
            (adapter / 'weight.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                committed_adapter(run, 1, run_id)

    def test_real_peft_checkpoint_switching_and_base_replay_loads_pipeline_once(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.proj = torch.nn.Linear(1, 1, bias=False)
                self.proj.weight.data.fill_(1.)
            def forward(self, x):
                return self.proj(x)
        class Scheduler:
            config = {}
            @classmethod
            def from_config(cls, config):
                return cls()
        class Pipeline:
            def __init__(self):
                self.transformer = Model(); self.scheduler = Scheduler(); self.hooks = []
            def __call__(self, **kwargs):
                return self.transformer(kwargs['x'])
            def remove_all_hooks(self): self.hooks.append('remove')
            def enable_model_cpu_offload(self, **kwargs): self.hooks.append('offload')
            def to(self, device): self.hooks.append(device)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for value in (.1, .3):
                model = get_peft_model(Model(), LoraConfig(r=1, lora_alpha=1, target_modules=['proj']))
                for name, parameter in model.named_parameters():
                    if 'lora_A' in name: parameter.data.fill_(1.)
                    if 'lora_B' in name: parameter.data.fill_(value)
                path = root / str(value); model.save_pretrained(path); paths.append(path)
            config = {'model': {'family': 'zimage', 'path': str(root), 'revision': 'fixture'},
                      'sampling': {'cpu_offload': True}}
            pipeline = Pipeline(); session = EvaluationSession()
            with patch('self_correct_dpo.evaluation.render.load_pipeline', return_value=pipeline) as load:
                base = session.load_pipeline(config, 'base', torch)
                torch.testing.assert_close(base(x=torch.ones(1,1)), torch.ones(1,1))
                for path, expected in zip(paths, (1.1, 1.3)):
                    aligned = session.load_pipeline({**config, 'aligned': {'path': str(path)}}, 'aligned', torch)
                    torch.testing.assert_close(aligned(x=torch.ones(1,1)), torch.tensor([[expected]]))
                    baseline = session.load_pipeline(config, 'base', torch)
                    torch.testing.assert_close(baseline(x=torch.ones(1,1)), torch.ones(1,1))
                    self.assertEqual(len(pipeline.transformer.peft_config), 1)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(session.pipeline_loads, 1)
                changed = copy.deepcopy(config); changed['sampling']['cpu_offload'] = False
                with self.assertRaisesRegex(ValueError, 'cannot change'):
                    session.load_pipeline(changed, 'base', torch)
            with patch('self_correct_dpo.evaluation.scorers.load_scorer', return_value=lambda images, prompts: [0.]) as load:
                self.assertIs(session.load_scorer('pickscore', {}), session.load_scorer('pickscore', {}))
                self.assertEqual(load.call_count, 1)

    def test_in_process_render_and_score_keep_real_transaction_and_resume_checks(self):
        from PIL import Image
        from self_correct_dpo.common import sha256, source_file, digest, model_artifact_identity
        import evaluate_flow
        from self_correct_dpo.evaluation import render
        from self_correct_dpo.evaluation.runner import evaluate_checkpoint, verify_complete
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / 'panel.jsonl'
            row = {'scene_id': 'heldout', 'variant_id': 'canonical', 'prompt': 'heldout caption',
                   'tag': 'test', 'metadata': {'tag': 'test', 'prompt': 'heldout caption'},
                   'transform_type': 'canonical', 'template_family': 'canonical', 'suite_mode': 'canonical-only-v1'}
            panel.write_text(json.dumps(row) + '\n')
            config = build_config('zimage', 'dpo', recipe='paper-v2')
            base = root / 'base'; base.mkdir(); (base / 'model_index.json').write_text('{}')
            model = {'family': 'zimage', 'path': str(base), 'revision': 'fixture'}
            preflight = {'inputs': {'prompts': 1, 'scorers': {'pickscore': {'spec': {}}}},
                         'base_identity': {'fixture': 'base'},
                         'code_sha256': {name: sha256(source_file(name)) for name in
                                         ('evaluation.py','render.py','metrics.py','scorers.py','common.py','zimage_compat.py')}}
            session = EvaluationSession()
            pipe = SimpleNamespace(scheduler=SimpleNamespace(config={}))
            class FakePipeline:
                scheduler = pipe.scheduler
                def __call__(self, **kwargs):
                    return SimpleNamespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']), 'red')])
            generator = torch.Generator
            with redirect_stdout(io.StringIO()), \
                 patch.object(render, 'require_gpu_allocation'), \
                 patch('self_correct_dpo.common.require_gpu_allocation'), \
                 patch('self_correct_dpo.evaluation.runner.require_gpu_allocation'), \
                 patch.object(render, 'check_prompt_lengths', return_value={}), \
                 patch.object(render, 'check_dependencies', return_value={'fixture': 'cpu'}), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'synchronize'), \
                 patch.object(torch.cuda, 'get_device_name', return_value='cpu-fixture'), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                 patch.object(torch.cuda, 'max_memory_reserved', return_value=0), \
                 patch.object(torch, 'Generator', side_effect=lambda device: generator('cpu')), \
                 patch.object(session, 'load_pipeline', return_value=FakePipeline()) as load, \
                 patch('self_correct_dpo.evaluation.scorers.load_scorer', return_value=lambda images, prompts: [.5] * len(images)) as score:
                for step in (0, 2):
                    output = root / f'eval-{step}'
                    for repeat in range(2):
                        result = evaluate_checkpoint(model, root / 'adapter' if step else None,
                            panel, ROOT / 'configs/scorers.json', ['pickscore'], config['evaluation']['sampling'], output,
                            step=step, keep_images=True, execution={'mode':'standalone','run_label':'fixture'},
                            expected_prompts=1, command_runner=session.run_command, preflight=preflight)
                        self.assertEqual(result['metrics']['pickscore']['mean'], .5)
                        self.assertEqual(verify_complete(output)['images'], 1)
                self.assertEqual(load.call_count, 2)  # Completed panels do not render again.
                self.assertEqual(score.call_count, 1)  # Both panels reuse the scorer.
                # Exercise the standalone job around the same real renderer/scorer.
                run = root / 'training'; run.mkdir()
                job_config = copy.deepcopy(config)
                job_config['model'] = model
                job_config['evaluation'].update(prompts=str(panel), steps=[2], expected_prompts=1,
                                                scorers_config=str(ROOT / 'configs/scorers.json'))
                record = {'run_id': 'e' * 64, 'config': job_config, 'base_identity': model_artifact_identity(model)}
                (run / 'training.json').write_text(json.dumps(record))
                snapshot = run / 'checkpoints/step-0000002'; (snapshot / 'adapter').mkdir(parents=True)
                weights = snapshot / 'adapter/fixture.bin'; weights.write_bytes(b'fixture adapter')
                manifest = {'run_id': record['run_id'], 'step': 2, 'files': {'adapter/fixture.bin': sha256(weights)}}
                manifest['checkpoint_id'] = digest(manifest)
                (snapshot / 'checkpoint.json').write_text(json.dumps(manifest))
                run.with_name('.training.preflight.pkl').touch()
                command = ['evaluate_flow.py', '--run', str(run), '--steps', '0', '2']
                with patch.object(sys, 'argv', command), \
                     patch('self_correct_dpo.training.load_preflight', return_value={'rows': [{'prompts': [{'prompt': 'training caption'}]}]}), \
                     patch('self_correct_dpo.evaluation.runner.validate_evaluation', return_value=preflight['inputs']), \
                     patch.object(evaluate_flow, 'EvaluationSession', return_value=session):
                    self.assertEqual(evaluate_flow.main(), 0)
                    self.assertEqual(evaluate_flow.main(), 0)
                self.assertTrue(read_json(run / 'offline-eval-v1/status.json')['complete'])
                self.assertEqual(load.call_count, 4)  # Reissuing the whole job also skips completed panels.
                self.assertEqual(score.call_count, 1)


if __name__ == '__main__':
    unittest.main()
