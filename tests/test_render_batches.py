"""CPU pipeline stubs exercise real batching and atomic image commits."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from self_correct_dpo.common import digest, sha256
from self_correct_dpo.evaluation import render


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU PyTorch is required')
class RenderBatchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite = [{'scene_id': f'p{i}', 'variant_id': 'canonical', 'prompt': f'prompt {i}',
                       'tag': 'fixture', 'metadata': {'prompt': f'prompt {i}', 'tag': 'fixture'}} for i in range(3)]
        self.sampling = {'seeds': [17, 29], 'generation_batch_size': 4,
                         'negative_prompt': 'negative', 'height': 64, 'width': 64,
                         'num_inference_steps': 2, 'guidance_scale': 4.5, 'max_sequence_length': 256}
        self.calls = []

    def run_info(self, family='sd35_medium'):
        run = {'suite': self.suite, 'sampling': self.sampling,
               'model': {'family': family}, 'expected_images': 6, 'arm': 'base'}
        run['run_id'] = digest(run)
        return run

    def pipeline(self, **kwargs):
        self.calls.append(kwargs)
        images = []
        for prompt, generator in zip(kwargs['prompt'], kwargs['generator']):
            value = int(self.torch.randint(0, 256, (), generator=generator).item())
            images.append(Image.new('RGB', (64, 64), (value, len(kwargs['prompt']), int(prompt[-1]))))
        return SimpleNamespace(images=images)

    def generate(self, directory, run, completed=None, pipe=None):
        directory.mkdir(exist_ok=True)
        completed = {} if completed is None else completed
        with (directory / 'manifest.jsonl').open('a') as stream, redirect_stdout(io.StringIO()):
            stats = render.render_batches(pipe or self.pipeline, self.torch, directory, run, stream,
                                          completed, device='cpu', synchronize=lambda: None)
        return completed, stats

    def test_real_list_inputs_generators_tail_batch_and_shared_timing(self):
        run = self.run_info()
        with patch.object(render.time, 'perf_counter', side_effect=[1.0, 7.0, 10.0, 12.0]):
            completed, stats = self.generate(self.root / 'full', run)
        self.assertEqual([len(call['prompt']) for call in self.calls], [4, 2])
        self.assertEqual(self.calls[0]['prompt'], ['prompt 0', 'prompt 0', 'prompt 1', 'prompt 1'])
        self.assertEqual(self.calls[0]['negative_prompt'], ['negative'] * 4)
        self.assertEqual([g.initial_seed() for g in self.calls[0]['generator']], [17, 29, 17, 29])
        self.assertEqual(len({id(g) for call in self.calls for g in call['generator']}), 6)
        self.assertTrue(all(call['max_sequence_length'] == 256 for call in self.calls))
        self.assertEqual([r['generation_batch_images'] for r in completed.values()], [4] * 4 + [2] * 2)
        self.assertEqual(sum(r['generation_seconds'] for r in completed.values()), 8.0)
        self.assertEqual(stats, {'generation_batches': 2, 'generation_images_computed': 6,
                                 'generation_forward_seconds': 8.0})

    def test_partial_commit_resume_replays_original_batch_and_preserves_existing_images(self):
        run = self.run_info()
        reference, _ = self.generate(self.root / 'reference', run)
        self.calls.clear()
        target = self.root / 'resume'
        original_commit = render.commit_image
        commits = []
        def interrupted(image, root, run, record, manifest):
            if len(commits) == 2:
                raise RuntimeError('simulated interruption')
            saved = original_commit(image, root, run, record, manifest)
            commits.append(saved)
            return saved
        with patch.object(render, 'commit_image', side_effect=interrupted), self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.generate(target, run)
        before = {row['image']: sha256(target / row['image']) for row in commits}
        records, _ = render.read_resume_manifest(target, run)
        completed = {(r['scene_id'], r['variant_id'], r['seed']): r for r in records}
        self.calls.clear()
        completed, stats = self.generate(target, run, completed)
        self.assertEqual([len(call['prompt']) for call in self.calls], [4, 2])
        self.assertEqual(stats['generation_images_computed'], 6)
        for name, checksum in before.items():
            self.assertEqual(sha256(target / name), checksum)
        self.assertEqual({key: row['image_sha256'] for key, row in completed.items()},
                         {key: row['image_sha256'] for key, row in reference.items()})
        self.calls.clear()
        self.generate(target, run, completed)
        self.assertEqual(self.calls, [])
        final, _ = render.read_resume_manifest(target, run)
        self.assertEqual(len(final), 6)

    def test_completed_batch_is_skipped_without_repacking_missing_slots(self):
        run = self.run_info('sd15')
        completed, _ = self.generate(self.root / 'full', run)
        first_batch = dict(list(completed.items())[:4])
        self.calls.clear()
        target = self.root / 'tail'
        # Test scheduling only: first four slots have existing valid records.
        self.generate(target, run, first_batch)
        self.assertEqual([call['prompt'] for call in self.calls], [['prompt 2', 'prompt 2']])
        self.assertNotIn('max_sequence_length', self.calls[0])

    def test_wrong_output_count_and_oom_are_not_retried_or_committed(self):
        run = self.run_info()
        calls = []
        def bad(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new('RGB', (64, 64))])
        with self.assertRaisesRegex(ValueError, '1 images for a batch of 4'):
            self.generate(self.root / 'bad', run, pipe=bad)
        self.assertEqual(len(calls), 1)
        self.assertEqual((self.root / 'bad/manifest.jsonl').read_text(), '')
        def oom(**kwargs):
            calls.append(kwargs)
            raise self.torch.cuda.OutOfMemoryError('simulated OOM')
        with self.assertRaises(self.torch.cuda.OutOfMemoryError):
            self.generate(self.root / 'oom', run, pipe=oom)
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.root / 'oom/manifest.jsonl').read_text(), '')

    def test_default_one_and_invalid_batch_sizes_are_checked_before_pipeline(self):
        config = {'model': {'family': 'sd15', 'path': '/local', 'revision': 'fixed'}, 'sampling': {}}
        self.assertEqual(render.effective_sampling(config)['generation_batch_size'], 1)
        for size in (0, -1, True, 1.5, '4', None):
            config['sampling']['generation_batch_size'] = size
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                render.effective_sampling(config)
        self.sampling.pop('generation_batch_size')
        self.generate(self.root / 'single', self.run_info())
        self.assertEqual([len(call['prompt']) for call in self.calls], [1] * 6)


if __name__ == '__main__':
    unittest.main()
