"""Real metadata fixtures for explicit method/step selection and independent Base."""
import json
from pathlib import Path
import shutil
import unittest

import test_evaluation_suite as lifecycle
from self_correct_dpo.common import read_json
from self_correct_dpo.evaluation.experiment_catalog import draft_catalog, validate_method
from self_correct_dpo.evaluation.metrics import score_panel
from self_correct_dpo.evaluation.suite import (filter_tasks, generate_task, make_plan, parse_args,
                                             score_task, summarize)


class ExperimentCatalogTest(unittest.TestCase):
    def setUp(self):
        self.fixture = lifecycle.EvaluationSuiteTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        info = read_json(f.run / 'training.json')
        info['config'].update(method='dpo', training={'objective': 'dpo', 'noise_views': 1})
        f.write(f.run / 'training.json', info)
        self.flow = f.root / 'flow-runs'
        self.sft = self.flow / 'fresh-sft'
        shutil.copytree(f.run, self.sft)
        info['config']['training']['objective'] = 'sft'
        f.write(self.sft / 'training.json', info)
        self.catalog = f.root / 'experiments.json'
        self.spec = {'version': 1, 'runs_roots': {'legacy': str(f.runs), 'flow': str(self.flow)},
                     'models': {'sd15': f.model}, 'experiments': [
                         {'id': 'dpo-paper', 'family': 'sd15', 'method': 'dpo', 'root': 'legacy',
                          'run': 'group/sd15-trial', 'steps': [1000]},
                         {'id': 'new-sft', 'family': 'sd15', 'method': 'sft', 'root': 'flow',
                          'run_glob': 'fresh-*', 'steps': [1000]}]}
        self.save()

    def save(self):
        self.catalog.write_text(json.dumps(self.spec))

    def args(self, *extra):
        f = self.fixture
        return parse_args(['plan', '--config', str(f.config), '--catalog', str(self.catalog),
                           '--benchmarks', 'pickv2', '--out', str(f.out), *extra])

    def test_multiple_roots_methods_and_exact_per_experiment_steps(self):
        f = self.fixture
        second = f.run / 'checkpoints/step-0002000'
        shutil.copytree(second.with_name('step-0001000'), second)
        cp = read_json(second / 'checkpoint.json')
        from self_correct_dpo.common import digest
        cp['step'] = 2000
        cp['checkpoint_id'] = digest({k: v for k, v in cp.items() if k != 'checkpoint_id'})
        f.write(second / 'checkpoint.json', cp)
        self.spec['experiments'][0]['steps'] = [2000]
        self.save()
        plan = make_plan(self.args())
        self.assertEqual([(t['method'], t['step']) for t in plan['tasks']], [('base', 0), ('dpo', 2000), ('sft', 1000)])
        self.assertEqual(len(plan['runs_roots']), 2)
        self.assertEqual({json.dumps(t['sampling'], sort_keys=True) for t in plan['tasks']},
                         {json.dumps(f.sampling, sort_keys=True)})
        self.assertEqual([t['method_label'] for t in plan['tasks']], ['Base', 'DPO', 'SFT'])
        args = parse_args(['generate', '--out', str(f.out), '--methods', 'sft'])
        self.assertEqual([t['experiment_id'] for t in filter_tasks(plan, args)], ['new-sft'])

    def test_base_does_not_need_any_training_tree_and_score_uses_only_images(self):
        f = self.fixture
        shutil.rmtree(f.runs)
        shutil.rmtree(self.flow)
        plan = make_plan(self.args('--methods', 'base'))
        self.assertEqual(len(plan['tasks']), 1)
        task = plan['tasks'][0]
        self.assertIsNone(task['run_path'])
        generate_task(task, plan, f.out, parse_args(['generate', '--out', str(f.out)]),
                      command_runner=f.command_render)
        self.catalog.unlink()
        def scorer(command):
            def arg(key): return command[command.index(key) + 1]
            score_panel(arg('--run'), arg('--out'), 'pickscore', {},
                        scorer_factory=lambda *a: lambda images, prompts: [.5] * len(images),
                        asset_id={'metric': 'pickscore'})
        score_task(task, 'pickscore', plan, f.out, parse_args(['score', '--out', str(f.out)]), command_runner=scorer)
        result = summarize(plan, f.out, parse_args(['summarize', '--out', str(f.out)]))
        self.assertTrue(result['complete'])
        self.assertEqual(result['rows'][0]['method_label'], 'Base')

    def test_missing_steps_ambiguous_runs_and_conflicting_labels_fail(self):
        self.spec['experiments'][0]['steps'] = [999]
        self.save()
        with self.assertRaisesRegex(ValueError, 'missing exact'):
            make_plan(self.args('--methods', 'dpo'))
        self.spec['experiments'][0]['steps'] = [1000]
        shutil.copytree(self.sft, self.flow / 'fresh-second')
        self.save()
        with self.assertRaisesRegex(ValueError, 'matched 2 runs'):
            make_plan(self.args('--methods', 'sft'))
        self.spec['experiments'][1]['run_glob'] = 'fresh-sft'
        self.spec['experiments'][1]['method'] = 'dpo'
        self.save()
        with self.assertRaisesRegex(ValueError, 'training method/objective/views disagree'):
            make_plan(self.args('--methods', 'dpo'))

    def test_requested_method_cannot_silently_disappear_or_model_revision_change(self):
        with self.assertRaisesRegex(ValueError, 'selections missing'):
            make_plan(self.args('--methods', 'base', 'linear_dpo'))
        self.spec['models']['sd15']['revision'] = 'another-model'
        self.save()
        with self.assertRaisesRegex(ValueError, 'registered Base identity'):
            make_plan(self.args('--methods', 'dpo'))

    def test_legacy_multi_root_base_is_deduplicated_and_outputs_stay_distinct(self):
        f = self.fixture
        args = parse_args(['plan', '--config', str(f.config), '--runs-root', str(f.runs), str(self.flow),
                           '--steps', '0', '1000', '--benchmarks', 'pickv2', '--out', str(f.out)])
        plan = make_plan(args)
        self.assertEqual(len(plan['tasks']), 3)
        self.assertEqual(sum(t['method'] == 'base' for t in plan['tasks']), 1)
        self.assertEqual(len({t['output'] for t in plan['tasks']}), 3)
        self.assertEqual([t['method'] for t in plan['tasks'] if t['step']], ['unregistered', 'unregistered'])

    def test_global_and_stagewise_semantics_cannot_be_swapped(self):
        config = {'method': 'multiview-C-gaussian',
                  'training': {'objective': 'dpo', 'formal_arm': 'C', 'noise_views': 4}}
        validate_method(config, 'global_k4', 'global')
        with self.assertRaisesRegex(ValueError, 'disagree'):
            validate_method(config, 'stagewise_k4', 'stage')
        config['method'] = 'stagewise-C-gaussian'
        config['training']['stagewise'] = {'objective': 'dpo', 'boundary': 500}
        validate_method(config, 'stagewise_k4', 'stage')
        config['training']['stagewise']['objective'] = 'reverse'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            validate_method(config, 'stagewise_k4', 'stage')

    def test_existing_baselines_k6_and_stagewise_ablations_keep_their_labels(self):
        from self_correct_dpo.evaluation.experiment_catalog import METHODS
        for method in ('dmpo', 'dspo', 'kto', 'mapo'):
            config = {'method': 'dpo', 'training': {'objective': method, 'noise_views': 1}}
            validate_method(config, method, method)
            self.assertEqual(METHODS[method], method.upper())
        for method, objective, views, arm, stage_objective in (
            ('global_k6', 'dpo', 6, 'C', None),
            ('gaussian_b_k2', 'dpo', 2, 'B', None),
            ('stagewise_k6', 'dpo', 6, 'C', 'dpo'),
            ('stagewise_dmpo_k6', 'dmpo', 6, 'C', 'dpo'),
            ('stagewise_dspo_k6', 'dspo', 6, 'C', 'dpo'),
            ('stagewise_constant_k2', 'dpo', 2, 'C', 'constant'),
            ('stagewise_phase_k2', 'dpo', 2, 'C', 'phase'),
            ('stagewise_reverse_k2', 'dpo', 2, 'C', 'reverse'),
        ):
            config = {'method': f'{"stagewise" if stage_objective else "multiview"}-{arm}-gaussian',
                      'training': {'objective': objective, 'noise_views': views, 'formal_arm': arm}}
            if stage_objective:
                config['training']['stagewise'] = {'objective': stage_objective, 'boundary': 500}
            validate_method(config, method, method)

    def test_import_current_metadata_discovers_new_sft_without_directory_guessing(self):
        f = self.fixture
        imported = draft_catalog([f.runs, self.flow])
        self.assertEqual({row['method'] for row in imported['experiments']}, {'sft', 'dpo'})
        self.assertEqual(imported['models']['sd15'], f.model)
        self.catalog.write_text(json.dumps(imported))
        plan = make_plan(self.args('--methods', 'sft'))
        self.assertEqual(plan['tasks'][0]['run_path'], str(self.sft))
        # A second new SFT remains a separate candidate; no latest/name preference.
        shutil.copytree(self.sft, self.flow / 'fresh-second')
        imported = draft_catalog([f.runs, self.flow])
        self.assertEqual(sum(row['method'] == 'sft' for row in imported['experiments']), 2)
        self.catalog.write_text(json.dumps(imported))
        with self.assertRaisesRegex(ValueError, 'Ambiguous family/method/step'):
            make_plan(self.args('--methods', 'sft'))

    def test_report_paths_do_not_duplicate_base_but_precision_remains_identity(self):
        f = self.fixture
        self.spec['models']['sd15']['validation_report'] = '/audit/catalog.json'
        for run, report in ((f.run, '/audit/dpo.json'), (self.sft, '/audit/sft.json')):
            saved = read_json(run / 'training.json')
            saved['config']['model']['validation_report'] = report
            f.write(run / 'training.json', saved)
        self.save()
        plan = make_plan(self.args())
        self.assertEqual(sum(t['step'] == 0 for t in plan['tasks']), 1)
        self.assertEqual(plan['tasks'][0]['model_audit']['validation_report'], '/audit/catalog.json')
        self.assertNotIn('validation_report', plan['tasks'][0]['model'])
        saved = read_json(self.sft / 'training.json')
        saved['config']['model']['transformer_dtype'] = 'float32'
        f.write(self.sft / 'training.json', saved)
        with self.assertRaisesRegex(ValueError, 'registered Base identity'):
            make_plan(self.args('--dry-run'))

    def test_import_cli_is_metadata_only_and_does_not_overwrite_a_different_catalog(self):
        import subprocess
        import sys
        f = self.fixture
        target = f.root / 'imported.json'
        script = Path(__file__).resolve().parents[1] / 'evaluate_suite.py'
        command = [sys.executable, '-S', str(script), 'catalog', '--runs-root', str(f.runs), str(self.flow),
                   '--out', str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_json(target), json.loads(result.stdout))
        target.write_text('{}')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_text(), '{}')

    def test_catalog_defaults_all_five_panels_and_rejects_global_step_override(self):
        args = parse_args(['plan', '--catalog', str(self.catalog), '--out', str(self.fixture.out)])
        self.assertEqual(args.benchmarks, ['pickv2', 'hpsv2', 'partiprompts', 'geneval', 'dpg'])
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--steps', '1000')
        legacy = parse_args(['plan', '--runs-root', str(self.fixture.runs), '--out', str(self.fixture.out)])
        self.assertEqual(legacy.benchmarks, ['pickv2', 'hpsv2', 'partiprompts'])


if __name__ == '__main__':
    unittest.main()
