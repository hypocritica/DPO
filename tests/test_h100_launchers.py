"""CPU checks for fixed two-H100 recipes and interrupted launch recovery."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('h100_runner', ROOT / 'scripts/run_sdxl_h100.py')
h100 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h100)


class H100Launchers(unittest.TestCase):
    def test_three_real_configs_keep_budget_monitoring_and_frozen_settings(self):
        from self_correct_dpo.runtime.pipeline import evaluation_steps
        with tempfile.TemporaryDirectory() as directory:
            configs, runs = Path(directory) / 'configs', Path(directory) / 'runs'
            for recipe, k in (('dmpo', 1), ('dmpo-k6-stage', 6), ('dspo', 1)):
                command = [sys.executable, str(ROOT / 'scripts/run_sdxl_h100.py'), recipe,
                           '--config-only', '--config-dir', str(configs), '--runs-root', str(runs)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                frozen = configs / f'local-sdxl-{recipe}-pv192-2h100-10000-v1.json'
                config = json.loads(frozen.read_text())
                t, e = config['training'], config['evaluation']
                self.assertEqual((config['model']['family'], t['world_size'], t['batch_size']), ('sdxl', 2, 1))
                self.assertEqual((t['resolution'], t['max_steps'], t['noise_views']), (1024, 10000, k))
                self.assertEqual(t['gradient_accumulation_steps'], 96 if k == 1 else 16)
                self.assertEqual(t['batch_size'] * 2 * k * t['gradient_accumulation_steps'], 192)
                self.assertEqual(config['data']['validation'], 'none')
                self.assertTrue(e['enabled'])
                self.assertFalse(e['include_base'])
                self.assertEqual(evaluation_steps(10000, e)[:3], [200, 500, 1000])
                self.assertEqual((t['checkpoint_every'], t['lora_checkpoint_every']), (200, 1000))
                self.assertEqual(t['beta'], 2000 if recipe == 'dspo' else 6000)
                if k == 6:
                    self.assertEqual(config['method'], 'stagewise-C-gaussian')
                    self.assertEqual(t['stagewise']['objective'], 'dpo')
                # Frozen experiments must reject changed settings on a restart.
                config['evaluation']['include_base'] = True
                frozen.write_text(json.dumps(config))
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Frozen config differs', result.stdout + result.stderr)

    def test_committed_state_recovers_and_uncheckpointed_attempt_is_preserved(self):
        import torch
        from self_correct_dpo.training import train
        from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
        from test_stage_pause import ToyBackend
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'; base.mkdir()
            (base / 'model_index.json').write_text('{}')
            rows = [{'pair_id': str(i), 'split': 'train', 'reviewed': True,
                     'prompts': [{'variant_id': 'canonical', 'prompt': 'test'}]} for i in range(3)]
            config = {'seed': 17, 'model': {'family': 'sd15', 'path': str(base)}, 'data': {},
                'training': {'formal_arm': 'C', 'noise_views': 2, 'objective': 'dpo', 'batch_size': 1,
                    'max_steps': 4, 'gradient_accumulation_steps': 1, 'checkpoint_every': 2,
                    'resolution': 64, 'beta': 2., 'temperature': .1, 'learning_rate': .01,
                    'max_grad_norm': 1., 'checkpoint_policy': 'bounded', 'lora_checkpoint_every': 4,
                    'prompt_mode': 'canonical', 'sampling': 'balanced'}}
            output = root / 'out'
            with redirect_stdout(io.StringIO()), patch('self_correct_dpo.training.image_batch',
                    return_value=torch.zeros(1, 2, 3, 1, 1)):
                train(config, rows, output, backend_factory=ToyBackend, device='cpu', stop_after_step=2)
                pointer = output / 'checkpoints/latest.json'
                saved = resolve_checkpoint(pointer)
                pointer.unlink()  # Power loss after managed commit, before publishing latest.
                self.assertFalse(h100.prepare_output(output, config))
                self.assertEqual(resolve_checkpoint(pointer), saved)
                train(config, rows, output, backend_factory=ToyBackend, device='cpu', resume=pointer)
            self.assertEqual(json.loads((output / 'complete.json').read_text())['step'], 4)
            with redirect_stdout(io.StringIO()):
                self.assertFalse(h100.prepare_output(output, config))  # Final evaluation still pending.
                (output / 'evaluation-complete.json').write_text(json.dumps({'complete': True, 'step': 4}))
                self.assertTrue(h100.prepare_output(output, config))
            empty = root / 'no-checkpoint'
            with redirect_stdout(io.StringIO()), patch('self_correct_dpo.training.image_batch',
                    side_effect=RuntimeError('power loss before first save')):
                with self.assertRaisesRegex(RuntimeError, 'power loss'):
                    train(config, rows, empty, backend_factory=ToyBackend, device='cpu')
            (empty / 'diagnostic.txt').write_text('keep this evidence')
            with redirect_stdout(io.StringIO()):
                self.assertFalse(h100.prepare_output(empty, config))
            preserved = list(root.glob('no-checkpoint.interrupted-*'))
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0] / 'diagnostic.txt').read_text(), 'keep this evidence')
            self.assertFalse(empty.exists())

    def test_two_rank_periodic_evaluation_resumes_without_base_evaluation(self):
        from test_distributed import fixture, execute, state
        from self_correct_dpo.common import source_file, write_json, write_jsonl, sha256
        from self_correct_dpo.runtime.pipeline import run_pipeline
        from self_correct_dpo.evaluation.runner import validate_evaluation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root, 2, monitor=True)
            config['evaluation']['include_base'] = False
            write_json(root / 'config.json', config)
            evaluated, failed = [], [False]
            def evaluator(model, adapter, prompts, scorers, metrics, sampling, path, **kw):
                step = kw['step']; evaluated.append(step)
                if step == 1 and not failed[0]:
                    failed[0] = True
                    raise subprocess.CalledProcessError(75, ['interrupted evaluation'])
                target = Path(path) / 'scores/pickscore'; target.mkdir(parents=True)
                write_jsonl(target / 'scores.jsonl', [
                    {'scene_id': str(i), 'variant_id': 'canonical', 'seed': 17,
                     'prompt': str(i), 'score': .2 + .01 * step} for i in range(2)])
                write_json(Path(path) / 'summary.json', {'step': step, 'metrics': {
                    'pickscore': {'mean': .2 + .01 * step, 'prompts': 2, 'images': 2}}})
                write_json(Path(path) / 'complete.json', {
                    'summary_sha256': sha256(Path(path) / 'summary.json'),
                    'scores_sha256': {'scores/pickscore/scores.jsonl': sha256(target / 'scores.jsonl')}})
            settings = config['evaluation']
            identity = {'settings': settings, 'inputs': validate_evaluation(settings['prompts'],
                settings['scorers_config'], ['pickscore'], expected_prompts=2),
                'code_sha256': {name: sha256(source_file(name)) for name in
                    ('pipeline.py', 'evaluation.py', 'metrics.py', 'scorers.py', 'render.py', 'benchmarks.py')}}
            kwargs = {'execution': {'mode': 'standalone'}, 'command_runner': execute,
                      'evaluator': evaluator, 'monitor_identity': identity}
            with patch('self_correct_dpo.runtime.pipeline.update_curve'):
                first = run_pipeline(config, root / 'config.json', root / 'run', **kwargs)
                self.assertFalse(first['complete'])
                self.assertEqual(first['step'], 1)
                second = run_pipeline(config, root / 'config.json', root / 'run',
                    resume=root / 'run/checkpoints/latest.json', **kwargs)
            self.assertTrue(second['complete'])
            self.assertEqual(evaluated, [1, 1, 2, 3])
            self.assertEqual(len(state(root / 'run')['rank_states']), 2)


if __name__ == '__main__':
    unittest.main()
