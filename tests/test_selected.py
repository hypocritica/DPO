"""Formal selection reuses real tiny retained checkpoint/curve fixtures; CPU only."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

from self_correct_dpo.common import read_json, write_json
from self_correct_dpo.evaluation.curve_prepare import prepare_curve
from self_correct_dpo.evaluation.selected import check, initialize, plan
import test_curve_prepare as curve_fixture


class SelectedTest(unittest.TestCase):
    def setUp(self):
        fixture = curve_fixture.CurvePrepareTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        prepared = prepare_curve(fixture.args)
        for prompt in fixture.official.values():
            Path(prompt).write_text(json.dumps({'scene_id': 'formal', 'variant_id': 'canonical', 'prompt': 'formal prompt', 'tag': 'formal', 'metadata': {'tag': 'formal', 'prompt': 'formal prompt'}, 'suite_mode': 'canonical-only-v1', 'transform_type': 'canonical', 'template_family': 'canonical'}) + '\n')
        self.path = fixture.root / 'selected.json'
        self.args = SimpleNamespace(family='sd35_medium', from_prepared=prepared['prepare_dir'],
                    catalog=None, runs_root=None, resources=None, config=str(fixture.config),
                    out=str(self.path), generation_batch_size=None)
        initialize(self.args)

    def choose(self):
        selection = read_json(self.path)
        for index, choice in enumerate(selection['choices']):
            choice['selected_step'] = 1000 if index % 2 else 3200
        self.path.write_text(json.dumps(selection))
        return selection

    def test_null_choices_and_identity_mutation_are_rejected(self):
        self.assertEqual(len(check(self.path)[2]), 7)
        args = SimpleNamespace(selection=str(self.path), out=str(self.fixture.root / 'evaluation'),
                               benchmarks=['partiprompts'], resources=None, dry_run=False)
        with self.assertRaisesRegex(ValueError, 'select exactly one'):
            plan(args)
        self.assertFalse(Path(args.out).exists())
        selection = self.choose()
        selection['suite']['protocols']['sd35_medium']['seeds'] = [17]
        self.path.write_text(json.dumps(selection))
        with self.assertRaisesRegex(ValueError, 'Imported identities'):
            check(self.path)

    def test_staged_plans_keep_exact_steps_and_full_seed_protocol(self):
        selection = self.choose()
        for panel in ('partiprompts', 'geneval'):
            args = SimpleNamespace(selection=str(self.path), out=str(self.fixture.root / panel),
                                   benchmarks=[panel], resources=None, dry_run=False)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(plan(args), 0)
                self.assertEqual(plan(args), 0)
            frozen = read_json(Path(args.out) / 'plan.json')
            self.assertEqual(len(frozen['tasks']), 8)
            self.assertEqual(frozen['report_selection']['selection'], selection)
            expected = {c['method']: c['selected_step'] for c in selection['choices']}
            expected['base'] = 0
            self.assertEqual({t['method']: t['step'] for t in frozen['tasks']}, expected)
            self.assertTrue(all(t['sampling']['seeds'] == [17, 29, 43, 71] for t in frozen['tasks']))
            self.assertTrue(all(t['sampling']['cpu_offload'] is False and t['sampling']['generation_batch_size'] == 8 for t in frozen['tasks']))
            receipt = read_json(Path(args.out) / 'selected-checkpoints.json')
            self.assertEqual(receipt['selection']['provenance']['curve_manifest']['preparation_id'], selection['provenance']['curve_manifest']['preparation_id'])

    def test_dry_run_writes_nothing_and_missing_other_panels_do_not_block(self):
        self.choose()
        Path(self.fixture.official['dpg']).unlink()
        args = SimpleNamespace(selection=str(self.path), out=str(self.fixture.root / 'preview'),
                               benchmarks=['partiprompts'], resources=None, dry_run=True)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(plan(args), 0)
        self.assertFalse(Path(args.out).exists())
        self.assertFalse(Path(args.out + '.selected-inputs').exists())

    def test_steps_cannot_change_between_benchmark_stages(self):
        selection = self.choose()
        args = SimpleNamespace(selection=str(self.path), out=str(self.fixture.root / 'parti'),
                               benchmarks=['partiprompts'], resources=None, dry_run=False)
        with redirect_stdout(io.StringIO()):
            plan(args)
        selection['choices'][0]['selected_step'] = 1000
        self.path.write_text(json.dumps(selection))
        args.out = str(self.fixture.root / 'geneval-next')
        args.benchmarks = ['geneval']
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            plan(args)
        self.assertFalse(Path(args.out).exists())

    def test_training_drift_and_unretained_step_rejected(self):
        selection = self.choose()
        choice = selection['choices'][0]
        choice['selected_step'] = 1200
        self.path.write_text(json.dumps(selection))
        self.assertTrue(check(self.path)[2])
        self.choose()
        training_path = Path(choice['run_path']) / 'training.json'
        training = read_json(training_path)
        training['changed'] = True
        training_path.write_text(json.dumps(training))
        with self.assertRaisesRegex(ValueError, 'Training identity changed'):
            check(self.path)


class ZImageSelectedTest(unittest.TestCase):
    def setUp(self):
        import shutil
        fixture = curve_fixture.CurvePrepareTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        for run in fixture.runs.iterdir():
            if run.name not in ('dpo', 'stagewise_k4'):
                shutil.rmtree(run)
                continue
            metadata = read_json(run / 'training.json')
            metadata['config']['model'].update(family='zimage', path='/not-loaded/Z-Image', transformer_dtype='float32')
            metadata['pairs_id'] = 'same-corpus-content-id'
            metadata['config']['data']['validation_report'] = str(run / 'different-report.json')
            (run / 'training.json').write_text(json.dumps(metadata))
        config = read_json(fixture.config)
        config['protocols']['zimage'] = config['protocols'].pop('sd35_medium')
        config['protocols']['zimage'].update(num_inference_steps=50, guidance_scale=4.0)
        fixture.config.write_text(json.dumps(config))
        scorers = read_json(fixture.scorers)
        scorers['scorers']['hpsv3'] = {'model': '/not-loaded/hpsv3'}
        fixture.scorers.write_text(json.dumps(scorers))
        fixture.args.family = 'zimage'
        fixture.args.selection_metric = 'hpsv3'
        fixture.args.generation_batch_size = 1
        fixture.args.cpu_offload = 'false'

    def test_zimage_development_and_formal_identity_protocol(self):
        from self_correct_dpo.evaluation.curves import validate_curve_plan, summarize_curves
        from self_correct_dpo.evaluation import suite
        prepared = prepare_curve(self.fixture.args)
        catalog = read_json(prepared['catalog'])
        self.assertEqual({e['method'] for e in catalog['experiments']}, {'dpo', 'stagewise_k4'})
        curve_config = read_json(prepared['config'])
        self.assertEqual(curve_config['metric_groups']['modern'], ['hpsv3'])
        out = self.fixture.root / 'z-curves'
        command = ['plan', '--catalog', prepared['catalog'], '--config', prepared['config'],
                   '--families', 'zimage', '--methods', 'base', 'dpo', 'stagewise_k4',
                   '--benchmarks', 'pickv2', '--out', str(out)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main(command), 0)
        frozen, _ = suite.read_plan(out)
        validate_curve_plan(frozen, out)
        self.assertEqual(len(frozen['tasks']), 5)
        self.assertTrue(all(t['sampling']['seeds'] == [17] and t['sampling']['cpu_offload'] is False for t in frozen['tasks']))
        # Both metrics summarize the exact same PNG schedule, without loading scorers.
        for metric in ('hpsv3', 'pickscore'):
            result = summarize_curves(out, metric=metric)
            self.assertFalse(result['complete'])
        selected_path = self.fixture.root / 'z-selected.json'
        args = SimpleNamespace(family='zimage', from_prepared=prepared['prepare_dir'],
                               catalog=None, runs_root=None, config=str(self.fixture.config), resources=None,
                               out=str(selected_path), generation_batch_size=None)
        selection = initialize(args)
        self.assertEqual(len(selection['choices']), 2)
        self.assertTrue(all(c['selected_step'] is None for c in selection['choices']))
        self.assertEqual(selection['suite']['protocols']['zimage']['seeds'], [17])
        self.assertEqual(selection['catalog']['models']['zimage']['transformer_dtype'], 'float32')
        self.assertIn('matching recorded metadata', selection['provenance']['dataset_comparability'])
        for c in selection['choices']:
            c['selected_step'] = 1000
        selected_path.write_text(json.dumps(selection))
        self.assertEqual(check(selected_path)[2], [])

    def test_zimage_mismatching_content_identity_and_zero_batch_refused(self):
        args = SimpleNamespace(family='zimage', from_prepared=None, catalog=None,
                               runs_root=[str(self.fixture.runs)], run_glob=None,
                               config=str(self.fixture.config), resources=None,
                               out=str(self.fixture.root / 'z-selected.json'), generation_batch_size=0)
        with self.assertRaisesRegex(ValueError, 'batch size must be positive'):
            initialize(args)
        args.generation_batch_size = 1
        metadata_path = self.fixture.runs / 'dpo/training.json'
        metadata = read_json(metadata_path)
        metadata['pairs_id'] = 'different'
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'dataset identities differ'):
            initialize(args)
