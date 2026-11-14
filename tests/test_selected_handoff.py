"""Confirmed SD3.5 handoff, exact retained identity and report separation; CPU only."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest

from self_correct_dpo.common import read_json, sha256
from self_correct_dpo.evaluation.curve_prepare import prepare_curve
from self_correct_dpo.evaluation import selected
import test_curve_prepare as curve_fixture


class SelectedHandoffTest(unittest.TestCase):
    def setUp(self):
        fixture = curve_fixture.CurvePrepareTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        for run in fixture.runs.iterdir():
            fixture.add_checkpoint(run, run.name, 1600)
            if run.name == 'dpo':
                fixture.add_checkpoint(run, run.name, 800)
        self.prepared = prepare_curve(fixture.args)
        for prompt in fixture.official.values():
            Path(prompt).write_text(json.dumps({'scene_id': 'formal', 'variant_id': 'canonical',
                'prompt': 'formal prompt', 'tag': 'formal', 'metadata': {'tag': 'formal', 'prompt': 'formal prompt'},
                'suite_mode': 'canonical-only-v1', 'transform_type': 'canonical', 'template_family': 'canonical'}) + '\n')
        variants = []
        for method in selected.METHODS['sd35_medium']:
            variants.append(self.variant(method, 0 if method == 'base' else 1600, 'main'))
        variants.append(self.variant('dpo', 800, 'development_selected_baseline'))
        self.handoff = fixture.root / 'handoff.json'
        self.document = {'schema': 'sd35_evaluation_selection_handoff_v1', 'family': 'sd35_medium',
            'model': fixture.model, 'sampling_reference': {'cpu_offload': True}, 'variants': variants,
            'main_selection_ids': [v['selection_id'] for v in variants[:-1]],
            'dpo_step_comparison_ids': [v['selection_id'] for v in variants if v['method'] == 'dpo']}
        self.handoff.write_text(json.dumps(self.document))
        self.path = fixture.root / 'selected-main.json'
        self.args = SimpleNamespace(family='sd35_medium', from_prepared=self.prepared['prepare_dir'],
            catalog=None, runs_root=None, resources=None, config=str(fixture.config), out=str(self.path),
            generation_batch_size=None, handoff=str(self.handoff))

    def variant(self, method, step, role):
        value = {'selection_id': f'sd35-{method}-{step:07d}', 'family': 'sd35_medium', 'method': method,
                 'label': 'DPO (development-selected)' if step == 800 else method.upper(),
                 'step': step, 'report_role': role, 'experiment_id': 'historical-' + method}
        if method in ('sft', 'base'):
            return value
        run = self.fixture.runs / method
        checkpoint = run / f'checkpoints/step-{step:07d}'
        value.update(run_path=str(run), run_name=run.name, run_id=read_json(run / 'training.json')['run_id'],
                     checkpoint_path=str(checkpoint), adapter_path=str(checkpoint / 'adapter'),
                     checkpoint_id=read_json(checkpoint / 'checkpoint.json')['checkpoint_id'])
        return value

    def test_nine_variants_generated_once_and_fixed_exports_after_training_removed(self):
        selection = selected.initialize(self.args)
        self.assertEqual([c['selected_step'] for c in selection['choices']], [1600] * 7)
        self.assertEqual(selection['extras'][0]['selected_step'], 800)
        self.assertEqual(selection['handoff']['sha256'], sha256(self.handoff))
        sft = next(v for v in selection['report_variants'] if v['method'] == 'sft')
        self.assertEqual(sft['run_path'], str(self.fixture.runs / 'sft'))
        self.assertEqual(sft['run_id'], 'sft')
        out = self.fixture.root / 'formal'
        args = SimpleNamespace(selection=str(self.path), out=str(out), benchmarks=None, resources=None, dry_run=False)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(selected.plan(args), 0)
            self.assertEqual(selected.plan(args), 0)
        frozen = read_json(out / 'plan.json')
        self.assertEqual(len(frozen['tasks']), 45)
        self.assertEqual(len({t['output'] for t in frozen['tasks']}), 45)
        self.assertEqual({t['step'] for t in frozen['tasks']}, {0, 800, 1600})
        self.assertTrue(all(t['sampling']['cpu_offload'] is False and t['sampling']['seeds'] == [17, 29, 43, 71]
                            and t['sampling']['generation_batch_size'] == 8 for t in frozen['tasks']))
        main = selected.report_plan(frozen, 'main')
        comparison = selected.report_plan(frozen, 'dpo-step-comparison')
        self.assertEqual(len(main['tasks']), 40)
        self.assertEqual(len(comparison['tasks']), 10)
        self.assertEqual({t['step'] for t in main['tasks'] if t['method'] == 'dpo'}, {1600})
        self.assertEqual({t['step'] for t in comparison['tasks']}, {800, 1600})
        self.assertTrue(all('development-selected' in t['method_label'] for t in comparison['tasks'] if t['step'] == 800))
        self.assertEqual({t['task_id'] for t in main['tasks'] if t['method'] == 'dpo'},
                         {t['task_id'] for t in comparison['tasks'] if t['step'] == 1600})
        shutil.rmtree(self.fixture.runs)
        shutil.rmtree(self.prepared['prepare_dir'])
        self.handoff.unlink()
        export_args = SimpleNamespace(out=str(out), report='all', paper_name='main-v1', comparison_name='dpo-v1',
            benchmarks=None, metrics=['pickscore'], metric_group=None, allow_incomplete=True)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(selected.export(export_args), 1)
        main_rows = read_json(out / 'paper/main-v1/results.json')['rows']
        extra_rows = read_json(out / 'paper/dpo-v1/results.json')['rows']
        self.assertEqual(len(main_rows), 40)
        self.assertEqual(len(extra_rows), 10)
        self.assertNotIn(800, {r['step'] for r in main_rows})
        self.assertEqual({r['step'] for r in extra_rows}, {800, 1600})

    def test_historical_identity_mismatch_is_not_relocated_or_substituted(self):
        for key in ('run_path', 'run_id', 'checkpoint_id'):
            with self.subTest(key=key):
                bad = json.loads(json.dumps(self.document))
                next(v for v in bad['variants'] if v['method'] == 'dpo')[key] = 'wrong'
                self.handoff.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'historical identity mismatch'):
                    selected.initialize(self.args)
                self.assertFalse(self.path.exists())

    def test_missing_checkpoint_and_corrupt_payload_refused(self):
        checkpoint = self.fixture.runs / 'dpo/checkpoints/step-0000800'
        (checkpoint / 'adapter/weights.bin').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            selected.initialize(self.args)
        shutil.rmtree(checkpoint)
        with self.assertRaisesRegex(ValueError, 'missing exact permanent checkpoints'):
            selected.initialize(self.args)

    def test_missing_or_ambiguous_sft_never_falls_back(self):
        catalog_path = Path(self.prepared['catalog'])
        catalog = read_json(catalog_path)
        # Re-freeze deliberately changed fixture catalogs with an updated manifest.
        for ambiguous in (False, True):
            changed = json.loads(json.dumps(catalog))
            sft = next(e for e in changed['experiments'] if e['method'] == 'sft')
            if ambiguous:
                duplicate = dict(sft, id='duplicate-sft', steps=[1600])
                changed['experiments'].append(duplicate)
            else:
                changed['experiments'].remove(sft)
            catalog_path.write_text(json.dumps(changed))
            manifest_path = Path(self.prepared['prepare_dir']) / 'prepare.json'
            manifest = read_json(manifest_path)
            manifest['files']['catalog.json'] = sha256(catalog_path)
            from self_correct_dpo.common import digest
            manifest.pop('preparation_id')
            manifest['preparation_id'] = digest(manifest)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'missing from catalog|Ambiguous'):
                selected.initialize(self.args)

    def test_report_ids_cannot_drift_from_frozen_identity(self):
        selection = selected.initialize(self.args)
        out = self.fixture.root / 'formal'
        args = SimpleNamespace(selection=str(self.path), out=str(out), benchmarks=['partiprompts'], resources=None, dry_run=False)
        with redirect_stdout(io.StringIO()):
            selected.plan(args)
        frozen = read_json(out / 'plan.json')
        frozen['report_selection']['selection']['reports']['main'][-1] = selection['reports']['dpo-step-comparison'][-1]
        with self.assertRaisesRegex(ValueError, 'selection identity mismatch'):
            selected.report_plan(frozen, 'main')

    def test_confirmed_steps_cannot_change_before_first_plan(self):
        selection = selected.initialize(self.args)
        for collection in ('choices', 'extras'):
            with self.subTest(collection=collection):
                changed = json.loads(json.dumps(selection))
                entry = next(c for c in changed[collection] if c['method'] == 'dpo')
                entry['selected_step'] = 800 if collection == 'choices' else 1600
                self.path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'identities/protocol changed'):
                    selected.check(self.path)

    def test_old_one_step_selection_remains_supported(self):
        self.args.handoff = None
        selection = selected.initialize(self.args)
        for choice in selection['choices']:
            choice['selected_step'] = 1600
        self.path.write_text(json.dumps(selection))
        _, catalog, gaps = selected.check(self.path)
        self.assertFalse(gaps)
        self.assertTrue(all(e['steps'] == [1600] for e in catalog['experiments']))
        self.assertNotIn('extras', selection)
