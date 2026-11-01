"""CPU coverage for exact checkpoint selection and resumable batch provenance."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from self_correct_dpo.common import digest, sha256
from self_correct_dpo.evaluation.batch import METRICS, build_request, exact_checkpoint, parse_args, run_batch
from self_correct_dpo.evaluation.runner import evaluate_checkpoint


class BatchEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'runs' / 'chosen'
        self.checkpoint = self.run / 'checkpoints' / 'step-0005000'
        adapter = self.checkpoint / 'adapter'
        adapter.mkdir(parents=True)
        (adapter / 'weights.bin').write_bytes(b'fixture')
        self.model = {'family': 'sd15', 'path': str(self.root / 'model')}
        self.write(self.run / 'training.json', {'run_id': 'chosen-id', 'config': {'model': self.model}})
        saved = {'run_id': 'chosen-id', 'step': 5000,
                 'files': {'adapter/weights.bin': sha256(adapter / 'weights.bin')}}
        saved['checkpoint_id'] = digest(saved)
        self.write(self.checkpoint / 'checkpoint.json', saved)
        self.request = {'runs_root': str(self.root / 'runs'), 'runs': ['chosen'], 'steps': [5000, 6000],
                        'panels': {'hpsv2': str(self.root / 'prompts.jsonl')},
                        'inputs': {'hpsv2': {'prompts': 2}}, 'sampling': {'seeds': [0]},
                        'scorers_config': str(self.root / 'scorers.json'), 'metrics': METRICS}
        self.out = self.root / 'evaluation'
        validation = patch('self_correct_dpo.evaluation.batch.validate_config')
        self.validation = validation
        validation.start()
        self.addCleanup(validation.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def test_missing_never_substitutes_latest_and_strict_reporting(self):
        self.write(self.run / 'checkpoints/latest.json', {'path': str(self.checkpoint)})
        with self.assertRaises(FileNotFoundError):
            exact_checkpoint(self.run, 6000)
        result, failed = run_batch(self.request, self.out, check_only=True, plan_only=True)
        self.assertFalse(failed)
        self.assertEqual([r['status'] for r in result['rows']], ['planned', 'missing_checkpoint'])
        self.assertIsNone(result['rows'][1]['pickscore'])
        self.assertFalse(self.out.exists())
        _, failed = run_batch(self.request, self.out, check_only=True, plan_only=True, strict_missing=True)
        self.assertTrue(failed)

    def test_checkpoint_manifest_and_step_identity(self):
        path = self.checkpoint / 'checkpoint.json'
        saved = json.loads(path.read_text())
        saved['step'] = 9000
        saved['checkpoint_id'] = digest({k: v for k, v in saved.items() if k != 'checkpoint_id'})
        self.write(path, saved)
        with self.assertRaisesRegex(ValueError, 'run/step'):
            exact_checkpoint(self.run, 5000)
        (self.checkpoint / 'adapter/weights.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            exact_checkpoint(self.run, 5000)

    def test_resume_sequential_five_metrics_summary_and_tamper(self):
        calls = []
        def command(command):
            calls.append(command)
            if '--metric' in command:
                metric = command[command.index('--metric') + 1]
                target = Path(command[command.index('--out') + 1])
                target.mkdir(parents=True)
                self.write(target / 'summary.json', {'mean': float(METRICS.index(metric)), 'images': 2})
        def evaluator(*args, **kwargs):
            return evaluate_checkpoint(*args, **kwargs, command_runner=command)
        inputs = {'prompts': 2, 'scorers': {}}
        with patch('self_correct_dpo.evaluation.runner.validate_evaluation', return_value=inputs), \
             patch('self_correct_dpo.evaluation.runner.model_artifact_identity', return_value={'fixture': True}), \
             patch('self_correct_dpo.evaluation.runner.require_gpu_allocation'):
            result, failed = run_batch(self.request, self.out, evaluator=evaluator)
            self.assertFalse(failed)
            self.assertEqual(len(calls), 6)  # One render followed by five separate scorers.
            self.assertEqual([c[c.index('--metric')+1] for c in calls[1:]], METRICS)
            self.assertEqual(result['rows'][0]['status'], 'complete')
            self.assertEqual(result['rows'][0]['image_reward'], 4.)
            result2, failed = run_batch(self.request, self.out, evaluator=evaluator)
            self.assertEqual(result, result2)
            self.assertEqual(len(calls), 6)
            self.assertTrue((self.out / 'summary.csv').is_file())
            # A score summary may not silently be reused after tampering.
            target = Path(result['rows'][0]['output'])
            (target / 'scores/pickscore/summary.json').write_text('{}')
            result3, failed = run_batch(self.request, self.out, evaluator=evaluator)
            self.assertTrue(failed)
            self.assertEqual(result3['rows'][0]['status'], 'failed')
            self.assertIn('checksum', result3['rows'][0]['error'])
        changed = dict(self.request, sampling={'seeds': [1]})
        with self.assertRaisesRegex(ValueError, 'changed inputs'):
            run_batch(changed, self.out, evaluator=evaluator)

    def test_plan_cli_without_scorer_or_model_assets(self):
        prompt = {'scene_id': 'p', 'variant_id': 'canonical', 'prompt': 'A cat',
                  'tag': 'test', 'metadata': {'tag': 'test', 'prompt': 'A cat', 'canonical_only': True},
                  'transform_type': 'canonical', 'template_family': 'canonical', 'suite_mode': 'canonical-only-v1'}
        panel = self.root / 'prompts.jsonl'
        panel.write_text(json.dumps(prompt) + '\n')
        config = self.root / 'batch-config.json'
        self.write(config, {'runs': ['chosen'], 'steps': [5000, 6000],
                            'benchmarks': {'hpsv2': str(panel)},
                            'scorers_config': 'missing-scorers.json', 'sampling': {'seeds': [0]}})
        script = Path(__file__).resolve().parents[1] / 'evaluate_batch.py'
        completed = subprocess.run([sys.executable, str(script), '--config', str(config),
            '--runs-root', str(self.run.parent), '--out', str(self.out), '--benchmarks', 'hpsv2',
            '--plan-only'], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual([r['status'] for r in result['rows']], ['planned', 'missing_checkpoint'])
        self.assertFalse(self.out.exists())

    def test_actual_two_worker_cli_missing_resume_and_strict_failure(self):
        assets = self.root / 'assets'
        for package in ('open_clip', 'ImageReward'):
            (assets / package).mkdir(parents=True)
            (assets / package / '__init__.py').write_text('')
        checkpoint = assets / 'HPS_v2_compressed.pt'
        checkpoint.write_bytes(b'fixture')
        (assets / 'config.json').write_text('{}')
        spec = {k: str(assets) for k in ('model', 'processor', 'open_clip_source', 'source', 'tokenizer')}
        spec.update(checkpoint='${FIXTURE_ASSETS}/HPS_v2_compressed.pt', config=str(checkpoint))
        scorers = self.root / 'scorers.json'
        self.write(scorers, {'scorers': {m: spec for m in METRICS}})
        panel = self.root / 'prompts.jsonl'
        panel.write_text(json.dumps({'scene_id': 'p', 'variant_id': 'canonical', 'prompt': 'A cat',
            'tag': 'test', 'metadata': {'tag': 'test', 'prompt': 'A cat'},
            'suite_mode': 'canonical-only-v1', 'transform_type': 'canonical', 'template_family': 'canonical'}) + '\n')
        config = self.root / 'config.json'
        self.write(config, {'runs': ['chosen'], 'steps': [6000, 7000],
            'benchmarks': {'hpsv2': str(panel)}, 'scorers_config': str(scorers), 'sampling': {'seeds': [0]}})
        script = Path(__file__).resolve().parents[1] / 'evaluate_batch.py'
        resources = self.root / 'resources.json'
        self.write(resources, {'RUNS_ROOT': str(self.run.parent), 'FIXTURE_ASSETS': str(assets)})
        command = [sys.executable, str(script), '--resources', str(resources), '--config', str(config), '--runs-root', str(self.run.parent),
                   '--out', str(self.out), '--benchmarks', 'hpsv2', '--gpus', 'fixture0,fixture1']
        import os
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='fixture0,fixture1')
        for extra, expected in (([], 0), ([], 0), (['--strict-missing'], 1)):
            completed = subprocess.run(command + extra, capture_output=True, text=True, env=env)
            self.assertEqual(completed.returncode, expected, completed.stderr)
            summary = json.loads((self.out / 'summary.json').read_text())
            self.assertEqual(len(summary['rows']), 2)
            self.assertEqual({r['status'] for r in summary['rows']}, {'missing_checkpoint'})
        self.assertTrue((self.out / 'summary-worker-0.json').exists())
        self.assertTrue((self.out / 'summary-worker-1.json').exists())
        broken = json.loads(config.read_text())
        broken['steps'] = [5000, 6000]
        self.write(config, broken)
        (self.checkpoint / 'adapter/weights.bin').write_bytes(b'corrupt')
        failed_out = self.root / 'failed-evaluation'
        failed_command = list(command)
        failed_command[failed_command.index('--out') + 1] = str(failed_out)
        completed = subprocess.run(failed_command, capture_output=True, text=True, env=env)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        statuses = {r['status'] for r in json.loads((failed_out / 'summary.json').read_text())['rows']}
        self.assertEqual(statuses, {'invalid_checkpoint', 'missing_checkpoint'})

    def test_euler_sd15_sampling_passes_real_render_validation(self):
        from self_correct_dpo.evaluation.render import validate_config
        model = self.root / 'model'
        (model / 'unet').mkdir(parents=True)
        self.write(model / 'model_index.json', {'_class_name': 'StableDiffusionPipeline'})
        self.write(model / 'unet/config.json', {'cross_attention_dim': 768, 'in_channels': 4})
        sampling = json.loads((Path(__file__).resolve().parents[1] /
            'configs/evaluate-selected-sd15.json').read_text())['sampling']
        config = {'model': dict(self.model, revision='fixture'), 'sampling': sampling}
        self.assertEqual(sampling['scheduler'], 'EulerDiscreteScheduler')
        self.assertEqual(validate_config(config, 'base')['family'], 'sd15')
        config['sampling'] = dict(sampling, scheduler='unsupported')
        with self.assertRaisesRegex(ValueError, 'Unsupported scheduler'):
            validate_config(config, 'base')

    def test_prompts_root_loads_only_selected_panel_and_changes_identity(self):
        full = self.root / 'full'
        smoke = self.root / 'smoke'
        prompt = {'scene_id': 'p', 'variant_id': 'canonical', 'prompt': 'A cat', 'tag': 'test',
                  'metadata': {'tag': 'test', 'prompt': 'A cat'}, 'suite_mode': 'canonical-only-v1',
                  'transform_type': 'canonical', 'template_family': 'canonical'}
        for directory in (full, smoke):
            directory.mkdir()
            (directory / 'hpsv2.jsonl').write_text(json.dumps(prompt) + '\n')
        config = self.root / 'config.json'
        self.write(config, {'runs': ['chosen'], 'steps': [5000], 'benchmarks': {
            'hpsv2': str(full / 'hpsv2.jsonl'), 'pickv2': 'not-present.jsonl'},
            'scorers_config': 'not-present.json', 'sampling': {'seeds': [0]}})
        common = ['--config', str(config), '--benchmarks', 'hpsv2', '--out', str(self.out), '--plan-only']
        original, _ = build_request(parse_args(common))
        changed, _ = build_request(parse_args(common + ['--prompts-root', str(smoke)]))
        self.assertEqual(changed['panels'], {'hpsv2': str(smoke / 'hpsv2.jsonl')})
        self.assertNotEqual(digest(original), digest(changed))
        self.assertEqual(original['inputs']['hpsv2']['prompts_sha256'], changed['inputs']['hpsv2']['prompts_sha256'])

    def test_check_only_rejects_invalid_render_before_evaluator(self):
        self.validation.stop()
        model = self.root / 'model'
        (model / 'unet').mkdir(parents=True)
        self.write(model / 'model_index.json', {'_class_name': 'StableDiffusionPipeline'})
        self.write(model / 'unet/config.json', {'cross_attention_dim': 768, 'in_channels': 4})
        info = json.loads((self.run / 'training.json').read_text())
        info['config']['model']['revision'] = 'fixture'
        self.write(self.run / 'training.json', info)
        self.request['sampling'] = json.loads((Path(__file__).resolve().parents[1] /
            'configs/evaluate-selected-sd15.json').read_text())['sampling']
        def forbidden(*args, **kwargs):
            self.fail('Evaluator must not run with an invalid adapter')
        result, failed = run_batch(self.request, self.out, check_only=True, evaluator=forbidden)
        self.assertTrue(failed)
        self.assertEqual(result['rows'][0]['status'], 'invalid_checkpoint')
        self.assertIn('adapter_config.json', result['rows'][0]['error'])
        self.assertFalse(self.out.exists())

    def test_missing_committed_weight_is_invalid_and_child_failure_persists_progress(self):
        (self.checkpoint / 'adapter/weights.bin').unlink()
        result, failed = run_batch(self.request, self.out, check_only=True)
        self.assertTrue(failed)
        self.assertEqual(result['rows'][0]['status'], 'invalid_checkpoint')
        (self.checkpoint / 'adapter/weights.bin').write_bytes(b'fixture')
        def stopped(*args, **kwargs):
            raise subprocess.CalledProcessError(143, ['fixture-scorer'])
        with self.assertRaises(subprocess.CalledProcessError):
            run_batch(self.request, self.out, evaluator=stopped)
        progress = json.loads((self.out / 'summary.json').read_text())
        self.assertEqual(progress['rows'][0]['status'], 'failed')
        self.assertIn('143', progress['rows'][0]['error'])
        self.assertTrue((self.out / 'summary.csv').is_file())

    def test_runtime_failure_survives_parent_summary_and_clears_on_success(self):
        def failed_evaluator(*args, **kwargs):
            raise RuntimeError('No allocated GPU')
        result, failed = run_batch(self.request, self.out, evaluator=failed_evaluator)
        self.assertTrue(failed)
        self.assertEqual(result['rows'][0]['status'], 'failed')
        def incomplete(*args, **kwargs):
            return {'metrics': METRICS}
        summary, failed = run_batch(self.request, self.out, summarize_only=True, evaluator=incomplete)
        self.assertTrue(failed)
        self.assertEqual(summary['rows'][0]['status'], 'failed')
        self.assertIn('No allocated GPU', summary['rows'][0]['error'])
        def complete(*args, **kwargs):
            return {'images': 2, 'metrics': {m: {'mean': 1., 'images': 2} for m in METRICS}}
        result, failed = run_batch(self.request, self.out, evaluator=complete)
        self.assertFalse(failed)
        self.assertEqual(result['rows'][0]['status'], 'complete')
        self.assertFalse((self.out / 'chosen/step-0005000/hpsv2-failure.json').exists())

    def test_changed_training_metadata_refuses_resume(self):
        def evaluator(*args, **kwargs):
            return {'images': 2, 'metrics': {m: {'mean': 1., 'images': 2} for m in METRICS}}
        run_batch(self.request, self.out, evaluator=evaluator)
        info = json.loads((self.run / 'training.json').read_text())
        info['tampered'] = True
        self.write(self.run / 'training.json', info)
        result, failed = run_batch(self.request, self.out, evaluator=evaluator)
        self.assertTrue(failed)
        self.assertIn('changed inputs', result['rows'][0]['error'])


if __name__ == '__main__':
    unittest.main()
