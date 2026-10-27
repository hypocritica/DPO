"""AOSS image commit recovery and exact-version compatibility for existing runs."""
from contextlib import redirect_stdout
import errno
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.common import digest, read_json, sha256, source_file, write_json
from self_correct_dpo.evaluation import render
from self_correct_dpo.runtime.storage_compat import compatible_identity, compatible_sources
from self_correct_dpo.runtime.storage_compat_hashes import SOURCE_HASHES


class ImagePublication(unittest.TestCase):
    def fixture(self, root):
        from PIL import Image
        run = {'run_id': 'original-run', 'suite': [
            {'scene_id': 's', 'variant_id': 'v', 'tag': 'test', 'prompt': 'a test'}],
            'sampling': {'seeds': [17]}}
        record = next(iter(render.expected_records(run).values()))
        (root / record['image']).parent.mkdir(parents=True)
        return Image.new('RGB', (8, 8), 'red'), run, record

    def test_normal_and_unsupported_hardlink_commit(self):
        for unsupported in (False, True):
            with self.subTest(unsupported=unsupported), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image, run, record = self.fixture(root)
                error = OSError(errno.ENOSYS, 'Function not implemented')
                real_link = render.os.link
                with render.render_lock(root), open(root / 'manifest.jsonl', 'a') as manifest, patch(
                        'self_correct_dpo.evaluation.render.os.link', side_effect=error if unsupported else real_link):
                    saved = render.commit_image(image, root, run, record, manifest)
                records, incomplete = render.read_resume_manifest(root, run)
                self.assertFalse(incomplete)
                self.assertEqual(records, [saved])
                self.assertEqual(list((root / 'pending').iterdir()), [])

    def test_resume_legacy_pending_png_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, run, record = self.fixture(root)
            with render.render_lock(root), open(root / 'manifest.jsonl', 'a') as manifest:
                with patch('self_correct_dpo.evaluation.render.publish_image',
                           side_effect=OSError(errno.ENOSYS, 'legacy hardlink failure')):
                    with self.assertRaises(OSError):
                        render.commit_image(image, root, run, record, manifest)
                staged = next((root / 'pending').glob('*.png'))
                checksum = sha256(staged)
                with patch('self_correct_dpo.evaluation.render.os.link',
                           side_effect=OSError(errno.EOPNOTSUPP, 'unsupported')):
                    restored = render.recover_transactions(root, run, [], manifest)
            self.assertEqual(len(restored), 1)
            self.assertEqual(sha256(root / record['image']), checksum)
            self.assertEqual(len(render.read_resume_manifest(root, run)[0]), 1)

    def test_crash_after_rename_recovers_existing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, run, record = self.fixture(root)
            with render.render_lock(root), open(root / 'manifest.jsonl', 'a') as manifest:
                with patch('self_correct_dpo.evaluation.render.os.link',
                           side_effect=OSError(errno.ENOSYS, 'unsupported')), patch(
                           'self_correct_dpo.evaluation.render.append_record', side_effect=RuntimeError('preemption')):
                    with self.assertRaisesRegex(RuntimeError, 'preemption'):
                        render.commit_image(image, root, run, record, manifest)
                self.assertTrue((root / record['image']).is_file())
                self.assertEqual(list((root / 'pending').glob('*.png')), [])
                with patch('self_correct_dpo.evaluation.render.os.link', side_effect=AssertionError('already published')):
                    restored = render.recover_transactions(root, run, [], manifest)
            self.assertEqual(len(restored), 1)
            self.assertEqual(list((root / 'pending').iterdir()), [])

    def test_collision_and_permission_errors_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged, destination = root / 'staged.png', root / 'existing.png'
            staged.write_bytes(b'new')
            destination.write_bytes(b'old')
            with render.render_lock(root), patch('self_correct_dpo.evaluation.render.os.link',
                    side_effect=OSError(errno.ENOSYS, 'unsupported')):
                with self.assertRaises(FileExistsError):
                    render.publish_image(staged, destination)
            self.assertEqual(destination.read_bytes(), b'old')
            self.assertEqual(staged.read_bytes(), b'new')
            with patch('self_correct_dpo.evaluation.render.os.link', side_effect=PermissionError(errno.EACCES, 'denied')):
                with self.assertRaises(PermissionError):
                    render.publish_image(staged, root / 'new.png')
            self.assertFalse((root / 'new.png').exists())


class SourceCompatibility(unittest.TestCase):
    def test_exact_allowlist_and_all_other_identity_fields(self):
        old = {name: value['original'] for name, value in SOURCE_HASHES.items()}
        current = {name: sha256(source_file(name)) for name in old}
        self.assertEqual(current, {name: value['fixed'] for name, value in SOURCE_HASHES.items()})
        self.assertTrue(compatible_sources(old, current))
        self.assertFalse(compatible_sources(current, old))
        self.assertFalse(compatible_sources(old, {**current, 'models.py': 'different'}))
        self.assertFalse(compatible_sources(old, {**current, 'render.py': 'unreviewed'}))
        self.assertTrue(compatible_sources({'runner.py': old['evaluation.py']},
                                           {'runner.py': current['evaluation.py']}))
        a, b = {'settings': {'every': 500}, 'code_sha256': old}, {'settings': {'every': 500}, 'code_sha256': current}
        self.assertTrue(compatible_identity(a, b))
        self.assertFalse(compatible_identity(a, {**b, 'settings': {'every': 200}}))

    def test_legacy_render_identity_and_pending_records_remain_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = {'config': {'sampling': 'unchanged'}, 'arm': 'base', 'suite': [],
                         'full_suite_id': 'suite', 'shard': {}, 'base_identity': {}, 'model_identity': {},
                         'suite_indices': None}
            run = {**arguments, 'suite_id': digest([]), 'render_sha256': SOURCE_HASHES['render.py']['original'],
                   'common_sha256': sha256(source_file('common.py'))}
            run['run_id'] = digest(run)
            write_json(root / 'run.json', run)
            self.assertEqual(render.validate_resume_run(root, **arguments), run)
            with self.assertRaisesRegex(ValueError, 'config changed'):
                render.validate_resume_run(root, **{**arguments, 'config': {'sampling': 'changed'}})

    def test_legacy_evaluation_identity_is_kept_without_rewriting_metadata(self):
        from self_correct_dpo.evaluation import runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {'family': 'sd15', 'path': 'same-model'}
            inputs = {'prompts': 1}
            sampling = {'seeds': [17]}
            names = ('evaluation.py', 'render.py', 'metrics.py', 'scorers.py', 'common.py')
            previous = {name: SOURCE_HASHES[name]['original'] if name in SOURCE_HASHES
                        else sha256(source_file(name)) for name in names}
            identity = {'inputs': inputs, 'config': {'model': model, 'sampling': sampling},
                        'base_identity': {'model': 'same'}, 'step': 0, 'keep_images': True,
                        'code_sha256': previous}
            meta = {'identity': identity, 'evaluation_id': digest(identity)}
            write_json(root / 'evaluation.json', meta)
            before = (root / 'evaluation.json').read_bytes()
            with patch.object(runner, 'validate_evaluation', return_value=inputs), patch.object(
                    runner, 'model_artifact_identity', return_value={'model': 'same'}):
                result = runner._evaluate_checkpoint(model, None, 'prompts', 'scorers', ['pickscore'], sampling,
                                                     root, step=0, check_only=True)
                self.assertEqual(result['evaluation_id'], meta['evaluation_id'])
                with self.assertRaisesRegex(ValueError, 'changed checkpoint'):
                    runner._evaluate_checkpoint(model, None, 'prompts', 'scorers', ['pickscore'],
                                                {'seeds': [18]}, root, step=0, check_only=True)
            self.assertEqual((root / 'evaluation.json').read_bytes(), before)
            self.assertEqual(list(root.iterdir()), [root / 'evaluation.json'])

    def test_legacy_200_step_resume_keeps_identity_and_optimizer_rng(self):
        import torch
        from self_correct_dpo.training import train
        from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
        from test_stage_pause import ToyBackend
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            base.mkdir()
            (base / 'model_index.json').write_text('{}')
            rows = [{'pair_id': str(i), 'split': 'train', 'reviewed': True,
                     'prompts': [{'variant_id': 'canonical', 'prompt': f'test {i}'}]} for i in range(3)]
            config = {'seed': 17, 'model': {'family': 'sd15', 'path': str(base)}, 'data': {},
                      'training': {'formal_arm': 'C', 'noise_views': 2, 'objective': 'dpo', 'batch_size': 1,
                                   'max_steps': 201, 'gradient_accumulation_steps': 1, 'checkpoint_every': 200,
                                   'resolution': 64, 'beta': 2., 'temperature': .1, 'learning_rate': .01,
                                   'max_grad_norm': 1., 'checkpoint_policy': 'bounded', 'lora_checkpoint_every': 1000,
                                   'prompt_mode': 'canonical', 'sampling': 'balanced', 'logging': 'jsonl'}}
            legacy = {source_file(name).resolve(): value['original'] for name, value in SOURCE_HASHES.items()}
            def previous_sha(path):
                return legacy.get(Path(path).resolve()) or sha256(path)
            out = root / 'resumed'
            with redirect_stdout(io.StringIO()), patch('self_correct_dpo.training.image_batch',
                    return_value=torch.zeros(1, 2, 3, 1, 1)), patch('self_correct_dpo.training.sha256', side_effect=previous_sha):
                train(config, rows, out, backend_factory=ToyBackend, device='cpu', stop_after_step=200)
            recorded_bytes = (out / 'training.json').read_bytes()
            original_id = read_json(out / 'training.json')['run_id']
            latest = out / 'checkpoints/latest.json'
            self.assertEqual(read_json(resolve_checkpoint(latest) / 'checkpoint.json')['step'], 200)
            def unreviewed_sha(path):
                return 'unreviewed' if Path(path).resolve() == source_file('models.py').resolve() else sha256(path)
            with patch('self_correct_dpo.training.sha256', side_effect=unreviewed_sha):
                with self.assertRaisesRegex(ValueError, 'training code changed'):
                    train(config, rows, out, resume=latest, backend_factory=ToyBackend, device='cpu')
            with redirect_stdout(io.StringIO()), patch('self_correct_dpo.training.image_batch',
                    return_value=torch.zeros(1, 2, 3, 1, 1)):
                resumed = train(config, rows, out, resume=latest, backend_factory=ToyBackend, device='cpu')
                train(config, rows, root / 'continuous', backend_factory=ToyBackend, device='cpu')
            self.assertEqual(resumed['run_id'], original_id)
            self.assertEqual((out / 'training.json').read_bytes(), recorded_bytes)
            self.assertEqual(len(list(out.glob('storage-compatibility-*.json'))), 1)
            a, b = [resolve_checkpoint(path / 'checkpoints/latest.json') for path in (out, root / 'continuous')]
            torch.testing.assert_close(torch.load(a / 'adapter/weight.pt', weights_only=True),
                                       torch.load(b / 'adapter/weight.pt', weights_only=True), rtol=0, atol=0)
            a, b = [torch.load(path / 'state.pt', weights_only=True) for path in (a, b)]
            self.assertEqual(a['step'], 201)
            self.assertEqual(a['scheduler'], b['scheduler'])
            torch.testing.assert_close(a['generator'], b['generator'], rtol=0, atol=0)
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                torch.testing.assert_close(a['optimizer']['state'][0][key], b['optimizer']['state'][0][key], rtol=0, atol=0)

    def test_legacy_monitor_resumes_200_evaluation_before_training_to_500(self):
        from self_correct_dpo.runtime import pipeline
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            settings = {'enabled': True, 'first_step': 200, 'every': 500, 'include_base': True,
                        'prompts': 'same-prompts', 'scorers_config': 'same-scorers', 'sampling': {}}
            config = {'model': {'family': 'sd15'}, 'evaluation': settings,
                      'training': {'max_steps': 1000, 'logging': 'jsonl'}}
            inputs = {'prompts': 128}
            names = ('pipeline.py', 'evaluation.py', 'metrics.py', 'scorers.py', 'render.py', 'benchmarks.py')
            current = {'settings': settings, 'inputs': inputs,
                       'code_sha256': {name: sha256(source_file(name)) for name in names}}
            previous = {**current, 'code_sha256': {name: SOURCE_HASHES[name]['original'] if name in SOURCE_HASHES
                                                  else current['code_sha256'][name] for name in names}}
            write_json(out / 'monitor-inputs.json', previous)
            before = (out / 'monitor-inputs.json').read_bytes()
            checkpoint = out / 'checkpoints/resume-0000200'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'state.pt').write_bytes(b'fixture-state')
            info = {'run_id': 'original', 'step': 200, 'full_state': True,
                    'files': {'state.pt': sha256(checkpoint / 'state.pt')}}
            info['checkpoint_id'] = digest(info)
            write_json(checkpoint / 'checkpoint.json', info)
            write_json(out / 'checkpoints/latest.json', {'directory': checkpoint.name, 'checkpoint_id': info['checkpoint_id']})
            write_json(out / 'training.json', {'run_id': 'original', 'config': config, 'monitor_inputs': inputs})
            steps = []
            def evaluate(*args, **kwargs):
                steps.append(kwargs['step'])
            def next_training(command):
                self.assertEqual(command[command.index('--stop-after-step') + 1], '500')
                raise RuntimeError('next training reached')
            with redirect_stdout(io.StringIO()), patch.object(pipeline, 'update_curve'):
                with self.assertRaisesRegex(RuntimeError, 'next training reached'):
                    pipeline.run_pipeline(config, out / 'config.json', out, execution={'mode': 'standalone'},
                                          resume=out / 'checkpoints/latest.json', monitor_identity=current,
                                          evaluator=evaluate, command_runner=next_training)
            self.assertEqual(steps, [0, 200])
            self.assertEqual((out / 'monitor-inputs.json').read_bytes(), before)
            self.assertEqual(pipeline.evaluation_steps(10000, settings), [0, 200] + list(range(500, 10001, 500)))


if __name__ == '__main__':
    unittest.main()
