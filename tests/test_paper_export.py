"""Real temporary image/score lifecycle for paper and portable figure exports."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.evaluation.gallery import make_gallery
from self_correct_dpo.evaluation.metrics import score_panel
from self_correct_dpo.evaluation.paper import export_paper
from self_correct_dpo.evaluation.suite import generate_task, make_plan, parse_args
import test_evaluation_suite as fixtures


class PaperExportTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EvaluationSuiteTest(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.out = self.fixture.out

    def args(self, **kwargs):
        return argparse.Namespace(paper_name='table-v1', families=None, methods=None,
            run_glob=None, steps=None, benchmarks=None, metrics=['pickscore'], metric_group=None,
            allow_incomplete=False, **kwargs)

    def scored(self, both=False, different_scorer=False):
        if both:
            self.fixture.plan_args.steps = [0, 1000]
        plan = make_plan(self.fixture.plan_args)
        for i, task in enumerate(plan['tasks']):
            task['method_label'] = 'DPO_1 & control%' if task['step'] else 'Base'
            task['method'] = 'dpo' if task['step'] else 'base'
            # This tests the export's rendering/score contract without modifying
            # the already-tested planner; annotations do not change render config.
            generate_task(task, plan, self.out, parse_args(['generate', '--out', str(self.out)]),
                          command_runner=self.fixture.command_render)
            target = self.out / task['output']
            score_panel(target / 'render', target / 'scores/pickscore', 'pickscore', {},
                scorer_factory=lambda *a: lambda images, prompts: [.25] * len(images),
                asset_id={'metric': 'pickscore', 'scale': 'cosine', 'version': str(i) if different_scorer else 'same'})
        return plan

    def test_export_without_training_assets_retains_sources_and_escapes_tex(self):
        plan = self.scored()
        shutil.rmtree(self.fixture.runs)
        result = export_paper(plan, self.out, self.args())
        self.assertTrue(result['complete'])
        destination = Path(result['output'])
        saved = read_json(destination / 'results.json')
        self.assertEqual(saved['rows'][0]['mean'], .25)
        self.assertEqual(saved['rows'][0]['prompt_standard_error'], 0)
        self.assertEqual(len(saved['provenance']), 1)
        self.assertIn('DPO\\_1 \\& control\\%', next(destination.glob('*.tex')).read_text())
        with (destination / 'results.csv').open() as stream:
            self.assertEqual(next(csv.DictReader(stream))['checkpoint_id'], plan['tasks'][0]['checkpoint']['checkpoint_id'])
        self.assertEqual(export_paper(plan, self.out, self.args()), result)
        for name, checksum in read_json(destination / 'export.json')['files_sha256'].items():
            self.assertEqual(sha256(destination / name), checksum)

    def test_missing_scores_fail_or_export_labelled_empty_draft(self):
        plan, task = self.fixture.generated()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            export_paper(plan, self.out, self.args())
        self.assertFalse((self.out / 'paper').exists())
        args = self.args()
        args.allow_incomplete = True
        result = export_paper(plan, self.out, args)
        saved = read_json(Path(result['output']) / 'results.json')
        self.assertFalse(result['complete'])
        self.assertEqual(saved['status'], 'INCOMPLETE DRAFT')
        self.assertIsNone(saved['rows'][0]['mean'])
        self.assertIn('INCOMPLETE', next(Path(result['output']).glob('*.md')).read_text())

    def test_different_scorer_protocols_are_not_pooled(self):
        plan = self.scored(both=True, different_scorer=True)
        result = export_paper(plan, self.out, self.args())
        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['tables'], 2)
        saved = read_json(Path(result['output']) / 'results.json')
        self.assertEqual({r['step'] for r in saved['rows']}, {0, 1000})

    def test_corrupted_or_failed_results_cannot_be_exported_as_valid(self):
        plan = self.scored()
        target = self.out / plan['tasks'][0]['output']
        path = target / 'scores/pickscore/summary.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            export_paper(plan, self.out, self.args())
        args = self.args()
        args.allow_incomplete = True
        result = export_paper(plan, self.out, args)
        row = read_json(Path(result['output']) / 'results.json')['rows'][0]
        self.assertEqual(row['status'], 'invalid')
        self.assertIsNone(row['mean'])

    def test_portable_gallery_copies_exact_originals_and_preserves_full_panel(self):
        plan, task = self.fixture.generated()
        render = self.out / task['output'] / 'render'
        before = sha256(render / 'manifest.jsonl')
        args = parse_args(['gallery', '--out', str(self.out), '--scene-id', 'p0', '--seed', '0', '--limit', '1'])
        args.copy_images = True
        index = Path(make_gallery(plan, self.out, args))
        selected = read_json(index.parent / 'selection.json')
        self.assertTrue(selected['portable_images'])
        self.assertEqual(len(selected['images']), 1)
        row = selected['images'][0]
        self.assertEqual(sha256(index.parent / row['export_image']), row['image_sha256'])
        self.assertEqual(row['checkpoint_id'], task['checkpoint']['checkpoint_id'])
        self.assertEqual(sha256(render / 'manifest.jsonl'), before)
        self.assertEqual(len(list((render / 'images').glob('*.png'))), 4)
        self.assertEqual(Path(make_gallery(plan, self.out, args)), index)
        moved = self.fixture.root / 'portable'
        shutil.copytree(index.parent, moved)
        shutil.rmtree(render)
        self.assertIn('src="' + row['export_image'] + '"', (moved / 'index.html').read_text())
        self.assertEqual(sha256(moved / row['export_image']), row['image_sha256'])

    def test_portable_gallery_rejects_changed_exported_image(self):
        plan, task = self.fixture.generated()
        args = parse_args(['gallery', '--out', str(self.out), '--limit', '1'])
        args.copy_images = True
        index = Path(make_gallery(plan, self.out, args))
        row = read_json(index.parent / 'selection.json')['images'][0]
        image = index.parent / row['export_image']
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Exported image changed'):
            make_gallery(plan, self.out, args)
        self.assertEqual(image.read_bytes(), b'changed')

    def test_cli_export_keeps_smoke_scope_after_training_and_prompt_sources_removed(self):
        source = self.fixture.prompts
        self.fixture.write(source.parent / 'selection.json', {
            'mode': 'smoke-only', 'panels': {'pickv2': {'sha256': sha256(source), 'prompts': 2}}})
        plan = self.scored()
        shutil.rmtree(self.fixture.runs)
        source.unlink()
        (source.parent / 'selection.json').unlink()
        script = Path(__file__).resolve().parents[1] / 'export_paper.py'
        process = subprocess.run([sys.executable, str(script), '--out', str(self.out),
            '--metrics', 'pickscore'], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        destination = self.out / 'paper/paper-v1'
        saved = read_json(destination / 'results.json')
        self.assertEqual(saved['rows'][0]['scope'], 'smoke-only')
        self.assertEqual(saved['panels']['pickv2']['provenance']['status'], 'selection-verified')
        self.assertIn('smoke-only', next(destination.glob('*.tex')).read_text())


if __name__ == '__main__':
    unittest.main()
