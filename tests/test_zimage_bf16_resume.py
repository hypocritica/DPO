"""Continue the exact legacy renderer without rewriting committed FP32 images."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from self_correct_dpo.common import digest, read_json, sha256, source_file, write_json
from self_correct_dpo.evaluation import render
from self_correct_dpo.runtime import zimage_precision_compat as precision
import test_render_batches as fixtures


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU PyTorch is required')
class ZImagePrecisionResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RenderBatchTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        model = {'family': 'zimage', 'path': '/local-model', 'revision': 'fixed'}
        self.arguments = {'config': {'model': model, 'sampling': f.sampling}, 'arm': 'aligned',
                          'suite': f.suite, 'full_suite_id': digest(f.suite),
                          'shard': {'index': 0, 'count': 1}, 'suite_indices': None,
                          'base_identity': {}, 'model_identity': {}}
        self.run = {**self.arguments, 'suite_id': digest(f.suite), 'model': model,
                    'sampling': f.sampling, 'expected_images': 6,
                    'render_sha256': precision.LEGACY_RENDER_SHA256,
                    'common_sha256': sha256(source_file('common.py')),
                    'zimage_compat_sha256': sha256(source_file('zimage_compat.py'))}
        self.run['run_id'] = digest(self.run)
        write_json(f.root / 'run.json', self.run)

    def test_missing_images_use_bf16_metadata_and_existing_bytes_are_unchanged(self):
        f = self.fixture
        before_run = (f.root / 'run.json').read_bytes()
        expected = list(render.expected_records(self.run).values())
        with (f.root / 'manifest.jsonl').open('a') as manifest:
            for spec in expected[:4]:
                (f.root / spec['image']).parent.mkdir(parents=True, exist_ok=True)
                render.commit_image(Image.new('RGB', (64, 64), 'red'), f.root, self.run, spec, manifest)
        before_manifest = (f.root / 'manifest.jsonl').read_bytes()
        images = {s['image']: (f.root / s['image']).read_bytes() for s in expected[:4]}
        resumed = render.validate_resume_run(f.root, **self.arguments)
        self.assertEqual(resumed, self.run)
        records, _ = render.read_resume_manifest(f.root, resumed)
        with render.render_lock(f.root), (f.root / 'manifest.jsonl').open('a') as manifest:
            precision.record_zimage_bf16_resume(f.root, resumed)
            completed = render.recover_transactions(f.root, resumed, records, manifest)
            render.render_batches(f.pipeline, f.torch, f.root, resumed, manifest, completed,
                                  device='cpu', synchronize=lambda: None)
        final, _ = render.read_resume_manifest(f.root, resumed)
        self.assertEqual(len(final), 6)
        self.assertEqual([len(c['prompt']) for c in f.calls], [2])
        self.assertTrue((f.root / 'manifest.jsonl').read_bytes().startswith(before_manifest))
        self.assertEqual((f.root / 'run.json').read_bytes(), before_run)
        for name, original in images.items():
            self.assertEqual((f.root / name).read_bytes(), original)
        self.assertTrue(all('transformer_autocast_dtype' not in r for r in final[:4]))
        self.assertTrue(all(r['transformer_autocast_dtype'] == 'bfloat16' for r in final[4:]))
        self.assertTrue(all(r['render_sha256'] == precision.BF16_RENDER_SHA256 for r in final[4:]))
        from self_correct_dpo.evaluation.curves import _timing
        timing = _timing(final, f.root)
        self.assertEqual(timing['generation_bf16_autocast_images'], 2)
        self.assertEqual(timing['generation_no_autocast_images'], 4)
        self.assertTrue(timing['generation_mixed_precision'])
        precision.record_zimage_bf16_resume(f.root, resumed)
        audits = list(f.root.glob('zimage-precision-transition-*.json'))
        self.assertEqual(len(audits), 1)
        self.assertEqual(read_json(audits[0])['run_id'], resumed['run_id'])

    def test_migration_rejects_other_code_models_and_changed_frozen_inputs(self):
        f = self.fixture
        self.assertTrue(precision.compatible_zimage_bf16_resume(self.run))
        for key, replacement in (('common_sha256', 'changed'), ('zimage_compat_sha256', 'changed'),
                                 ('render_sha256', 'unreviewed')):
            changed = dict(self.run, **{key: replacement})
            changed['run_id'] = digest({k: v for k, v in changed.items() if k != 'run_id'})
            (f.root / 'run.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, key + ' changed'):
                render.validate_resume_run(f.root, **self.arguments)
        (f.root / 'run.json').write_text(json.dumps(self.run))
        changed_args = copy.deepcopy(self.arguments)
        changed_args['config']['sampling']['seeds'] = [123]
        with self.assertRaisesRegex(ValueError, 'config changed'):
            render.validate_resume_run(f.root, **changed_args)
        changed = copy.deepcopy(self.run)
        changed['config']['model']['family'] = 'sd35_medium'
        self.assertFalse(precision.compatible_zimage_bf16_resume(changed))
        changed = copy.deepcopy(self.run)
        changed['config']['sampling']['transformer_autocast_dtype'] = 'none'
        self.assertFalse(precision.compatible_zimage_bf16_resume(changed))
        with patch.object(precision, 'BF16_RENDER_SHA256', 'different-target'):
            self.assertFalse(precision.compatible_zimage_bf16_resume(self.run))


if __name__ == '__main__':
    unittest.main()
