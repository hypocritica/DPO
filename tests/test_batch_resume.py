import copy
from contextlib import redirect_stdout
import importlib.util
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
sys.path.insert(0, str(ROOT / 'src'))

import torch
from self_correct_dpo.common import read_json, sha256
from self_correct_dpo.config import load_config
from self_correct_dpo.runtime.batch_resume import (
    ENVIRONMENT_FLAG, compatible_batch_sources, validate_batch_change)
from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
from self_correct_dpo.training import sample_batch, train
from tests.test_stage_pause import ToyBackend


class SFTToy(ToyBackend):
    def __init__(self, model, training, device):
        super().__init__(model, training, device)
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def score_errors(self, images, prompts, generator):
        # Uniform draws preserve the CPU stream when the same sample sequence
        # is partitioned differently; real CUDA diffusion need not be bitwise equal.
        noise = torch.rand(len(prompts), generator=generator)
        targets = torch.tensor([int(p[0]) / 10 for p in prompts]) + noise
        return {'policy_errors': (self.weight - targets).square()[:, None, None],
                'reference_errors': None}


class DistributedSFTToy(SFTToy):
    def __init__(self, model, training, device):
        super().__init__(model, training, device)
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(.2)
        self.weight = self.model.weight

    def enable_distributed(self):
        self.distributed_model = torch.nn.parallel.DistributedDataParallel(self.model)

    def score_errors(self, images, prompts, generator):
        targets = torch.tensor([int(p[0]) / 10 for p in prompts]) + torch.rand(len(prompts), generator=generator)
        prediction = self.distributed_model(torch.ones(len(prompts), 1)).flatten()
        return {'policy_errors': (prediction - targets).square()[:, None, None], 'reference_errors': None}


class BatchResume(unittest.TestCase):
    def fixture(self, root):
        base = root / 'base'
        base.mkdir()
        (base / 'model_index.json').write_text('{}')
        rows = [{'pair_id': str(i), 'split': 'train', 'reviewed': True,
                 'prompts': [{'variant_id': 'canonical', 'prompt': str(i)}]} for i in range(13)]
        config = {'seed': 17, 'model': {'family': 'sd15', 'path': str(base)}, 'data': {},
                  'training': {'objective': 'sft', 'batch_size': 2, 'max_steps': 6,
                      'gradient_accumulation_steps': 4, 'checkpoint_every': 2,
                      'resolution': 64, 'beta': 2., 'temperature': .1, 'learning_rate': .01,
                      'max_grad_norm': 1., 'checkpoint_policy': 'bounded',
                      'lora_checkpoint_every': 6, 'sampling': 'balanced', 'noise_views': 1}}
        return rows, config

    def run_training(self, config, rows, out, **kwargs):
        with redirect_stdout(io.StringIO()), patch('self_correct_dpo.training.image_batch',
                return_value=torch.zeros(1, 1, 3, 1, 1)):
            return train(config, rows, out, backend_factory=SFTToy, device='cpu', **kwargs)

    def test_resize_restores_state_progress_and_original_identity_across_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, old = self.fixture(root)
            new = copy.deepcopy(old)
            new['training'].update(batch_size=4, gradient_accumulation_steps=2)
            out = root / 'resized'
            # Simulate the published source identity before this training hook.
            real_sha256 = sha256
            def original_source(path):
                return 'published-training-source' if Path(path).name == 'training.py' else real_sha256(path)
            with patch('self_correct_dpo.training.sha256', side_effect=original_source):
                self.run_training(old, rows, out, stop_after_step=2)
            original_metadata = (out / 'training.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                self.run_training(new, rows, out, resume=out / 'checkpoints/latest.json')
            with patch.dict(os.environ, {ENVIRONMENT_FLAG: '1'}):
                with self.assertRaisesRegex(ValueError, 'model, data, validation'):
                    self.run_training(new, list(reversed(rows)), out, resume=out / 'checkpoints/latest.json')
                first = self.run_training(new, rows, out, resume=out / 'checkpoints/latest.json', stop_after_step=4)
                resumed = self.run_training(new, rows, out, resume=out / 'checkpoints/latest.json')
            reference = self.run_training(old, rows, root / 'reference')
            self.assertEqual((first['step'], resumed['step']), (4, 6))
            self.assertEqual(resumed['run_id'], read_json(out / 'training.json')['run_id'])
            self.assertEqual(original_metadata, (out / 'training.json').read_bytes())
            self.assertEqual(resumed['cost']['pairs_seen'], 48)
            self.assertEqual(resumed['cost']['exposure'], reference['cost']['exposure'])
            states = [torch.load(resolve_checkpoint(path / 'checkpoints/latest.json') / 'state.pt',
                                 weights_only=True) for path in (out, root / 'reference')]
            a, b = states
            self.assertEqual(a['scheduler'], b['scheduler'])
            torch.testing.assert_close(a['generator'], b['generator'], rtol=0, atol=0)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(a['optimizer']['state'][0][key],
                                           b['optimizer']['state'][0][key], rtol=1e-5, atol=1e-8)
            audits = [read_json(path) for path in out.glob('batch-resume-*.json')]
            self.assertEqual(sorted(a['start_step'] for a in audits), [2, 4])
            self.assertTrue(all(a['active_batch']['batch_size'] == 4 for a in audits))
            records = [json.loads(line) for path in out.glob('metrics-*.jsonl')
                       for line in path.read_text().splitlines()]
            self.assertEqual(sorted(r['step'] for r in records), list(range(1, 7)))
            self.assertTrue(all(r['effective_batch_size_pairs'] == 8 for r in records))
            self.assertTrue(all(r['per_device_batch_size'] == 4 for r in records if r['step'] > 2))

    def test_only_batch_partition_and_training_hook_are_relaxed(self):
        original = {'model': {}, 'data': {}, 'evaluation': {'every': 500}, 'training': {
            'objective': 'sft', 'sampling': 'balanced', 'noise_views': 1,
            'world_size': 2, 'batch_size': 2, 'gradient_accumulation_steps': 96,
            'learning_rate': 1e-5}}
        current = copy.deepcopy(original)
        current['training'].update(batch_size=8, gradient_accumulation_steps=24)
        validate_batch_change(original, current)
        for section, key, value in [('training', 'world_size', 4),
                ('training', 'learning_rate', 2e-5), ('training', 'gradient_accumulation_steps', 12),
                ('training', 'objective', 'kto'), ('training', 'sampling', 'replacement'),
                ('evaluation', 'every', 1000), ('model', 'path', '/different')]:
            changed = copy.deepcopy(current)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_batch_change(original, changed)
        old_code = {'training.py': 'old', 'models.py': 'model'}
        self.assertTrue(compatible_batch_sources(old_code, dict(old_code, **{'training.py': 'new'})))
        self.assertFalse(compatible_batch_sources(old_code, {'training.py': 'new', 'models.py': 'changed'}))

    def test_balanced_global_samples_match_after_nonzero_step(self):
        rows = [{'pair_id': str(i), 'prompts': [{'variant_id': 'canonical', 'prompt': str(i)}]}
                for i in range(101)]
        for batch, accumulation in [(2, 96), (8, 24), (16, 12)]:
            config = {'seed': 17, 'data': {}, 'training': {'objective': 'sft', 'sampling': 'balanced',
                      'batch_size': batch, 'prompt_mode': 'canonical'}}
            selected = []
            for inner in range(accumulation):
                for rank in range(2):
                    chosen, _, _ = sample_batch(rows, config, 37 * accumulation + inner, world=2, rank=rank)
                    selected.extend(r['pair_id'] for r in chosen)
            if batch == 2:
                baseline = selected
            else:
                self.assertEqual(selected, baseline)

    def test_config_only_reuses_recorded_recipe_and_keeps_original_file(self):
        spec = importlib.util.spec_from_file_location('resume_sft_batch', ROOT / 'scripts/resume_sft_batch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            out = root / 'existing'
            out.mkdir()
            config = load_config(ROOT / 'configs/sft.json')
            config['training'].update(batch_size=2, gradient_accumulation_steps=96, world_size=2, max_steps=10000)
            recorded = {'config': config, 'code': {'training.py': 'original'}}
            (out / 'training.json').write_text(json.dumps(recorded))
            module.ROOT = root
            for batch, accumulation in [(8, 24), (16, 12)]:
                with patch.object(sys, 'argv', ['resume', '--run', str(out), '--batch-size', str(batch),
                                              '--config-only']), redirect_stdout(io.StringIO()):
                    module.main()
                active = read_json(root / 'configs' / f'local-existing-resume-b{batch}.json')
                self.assertEqual(active['training']['gradient_accumulation_steps'], accumulation)
                self.assertEqual(active['evaluation'], config['evaluation'])
            self.assertEqual(read_json(out / 'training.json'), recorded)


    def test_two_rank_gloo_resume_keeps_optimizer_and_both_rng_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = self.fixture(root)
            config['training']['world_size'] = 2
            out = root / 'distributed'
            config_path = root / 'config.json'
            def worker(resume, endpoint):
                config_path.write_text(json.dumps(config))
                environment = dict(os.environ, OMP_NUM_THREADS='1')
                environment.pop(ENVIRONMENT_FLAG, None)
                if resume:
                    environment[ENVIRONMENT_FLAG] = '1'
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                           '--nproc-per-node=2', '--module', 'tests.test_batch_resume',
                           '--ddp-worker', str(config_path), str(out), str(endpoint)]
                if resume:
                    command.append('--resume')
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout)
                return torch.load(resolve_checkpoint(out / 'checkpoints/latest.json') / 'state.pt', weights_only=True)
            before = worker(False, 2)
            metadata = (out / 'training.json').read_bytes()
            config['training'].update(batch_size=4, gradient_accumulation_steps=2)
            worker(True, 4)
            after = worker(True, 6)
            self.assertEqual((out / 'training.json').read_bytes(), metadata)
            self.assertEqual(after['step'], 6)
            self.assertEqual(after['cost']['pairs_seen'], 96)
            self.assertEqual(after['scheduler']['last_epoch'], 6)
            self.assertEqual(int(after['optimizer']['state'][0]['step']), 6)
            self.assertEqual(len(after['rank_states']), 2)
            for previous, current in zip(before['rank_states'], after['rank_states']):
                expected = torch.Generator().set_state(previous['generator'])
                torch.rand(32, generator=expected)
                torch.testing.assert_close(current['generator'], expected.get_state(), rtol=0, atol=0)
            records = [json.loads(line) for path in out.glob('metrics-*.jsonl')
                       for line in path.read_text().splitlines()]
            self.assertEqual(sorted(r['step'] for r in records), list(range(1, 7)))
            self.assertTrue(all(r['effective_batch_size_pairs'] == 16 for r in records))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--ddp-worker':
        from self_correct_dpo.runtime.distributed import cleanup
        config_path, out, endpoint = sys.argv[2:5]
        config = read_json(config_path)
        rows = [{'pair_id': str(i), 'split': 'train', 'reviewed': True,
                 'prompts': [{'variant_id': 'canonical', 'prompt': str(i)}]} for i in range(13)]
        try:
            with patch('self_correct_dpo.training.image_batch', return_value=torch.zeros(1, 1, 3, 1, 1)):
                train(config, rows, out, backend_factory=DistributedSFTToy, device='cpu',
                      stop_after_step=int(endpoint),
                      resume=Path(out) / 'checkpoints/latest.json' if '--resume' in sys.argv else None)
        finally:
            cleanup()
    else:
        unittest.main()
