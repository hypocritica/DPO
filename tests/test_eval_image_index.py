"""Lookup generated originals before scoring and without training artifacts."""
import csv
from pathlib import Path
import shutil
import unittest
from self_correct_dpo.common import read_json, sha256
from self_correct_dpo.evaluation.image_index import export_image_index
from self_correct_dpo.evaluation.suite import make_plan, parse_args
import test_evaluation_suite as fixtures


class ImageIndexTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EvaluationSuiteTest(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_index_before_scoring_without_training_and_missing_image_status(self):
        plan, task = self.fixture.generated()
        shutil.rmtree(self.fixture.runs)
        out = self.fixture.out
        args = parse_args(['index', '--out', str(out)])
        result = export_image_index(plan, out, args)
        self.assertTrue(result['complete'])
        self.assertEqual(result['images'], 4)
        with (out / 'image-index.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['checkpoint_id'], task['checkpoint']['checkpoint_id'])
        self.assertEqual(sha256(rows[0]['image_path']), rows[0]['image_sha256'])
        Path(rows[0]['image_path']).unlink()
        result = export_image_index(plan, out, args)
        self.assertFalse(result['complete'])
        self.assertEqual(result['images'], 0)
        self.assertEqual(read_json(out / 'image-index.json')['csv_sha256'], sha256(out / 'image-index.csv'))

    def test_generation_batch_is_frozen_and_invalid_batch_rejected(self):
        args = self.fixture.plan_args
        args.generation_batch_size = 32
        plan = make_plan(args)
        self.assertEqual(plan['tasks'][0]['sampling']['generation_batch_size'], 32)
        args.generation_batch_size = 4
        with self.assertRaisesRegex(ValueError, 'different plan'):
            make_plan(args)
        args.generation_batch_size = 0
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            make_plan(args)


if __name__ == '__main__':
    unittest.main()
