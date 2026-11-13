"""CPU lifecycle checks: freeze pixels independently, add rewards later, reject drift."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from self_correct_dpo.common import digest, load_suite, read_json, read_jsonl, sha256
from self_correct_dpo.evaluation.metrics import score_panel, score_value
from self_correct_dpo.evaluation.suite import (filter_tasks, generate_task, make_plan, metrics_for, parse_args,
    read_plan, run_stage, score_task, selected_steps, summarize, validate_task_render)
from self_correct_dpo.evaluation.gallery import make_gallery


class EvaluationSuiteTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / 'runs'
        self.run = self.runs / 'group' / 'sd15-trial'
        adapter = self.run / 'checkpoints/step-0001000/adapter'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_model.bin').write_bytes(b'tiny-adapter')
        self.model = {'family': 'sd15', 'path': str(self.root / 'unavailable-model')}
        self.write(self.run / 'training.json', {'run_id': 'trial', 'config': {'model': self.model}})
        cp = {'run_id': 'trial', 'step': 1000, 'files': {'adapter/adapter_model.bin': sha256(adapter / 'adapter_model.bin')}}
        cp['checkpoint_id'] = digest(cp)
        self.write(adapter.parent / 'checkpoint.json', cp)
        self.prompts = self.root / 'pickv2.jsonl'
        rows = [{'scene_id': f'p{i}', 'variant_id': 'canonical', 'prompt': text, 'tag': 'pickv2',
                 'metadata': {'prompt': text, 'tag': 'pickv2'}, 'suite_mode': 'canonical-only-v1',
                 'template_family': 'canonical', 'transform_type': 'canonical'}
                for i, text in enumerate(('a red cat', 'two blue cups'))]
        self.prompts.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.config = self.root / 'suite.json'
        self.scorers = self.root / 'scorers.json'
        self.write(self.scorers, {'scorers': {'pickscore': {'model': '/not-downloaded', 'processor': '/not-downloaded'},
                                            'hpsv3': {'model': '/not-downloaded'}}})
        self.sampling = {'seeds': [0, 1], 'height': 8, 'width': 8, 'scheduler': 'EulerDiscreteScheduler',
                         'dtype': 'bfloat16', 'num_inference_steps': 50, 'guidance_scale': 7.5,
                         'negative_prompt': '', 'prompt_truncation': 'truncate', 'safety_checker': 'disabled'}
        self.write(self.config, {'benchmarks': {'pickv2': str(self.prompts)}, 'scorers_config': str(self.scorers),
            'protocols': {'sd15': self.sampling}, 'metric_groups': {'legacy': ['pickscore'], 'modern': ['hpsv3']}})
        self.out = self.root / 'evaluation'
        self.plan_args = parse_args(['plan', '--config', str(self.config), '--runs-root', str(self.runs),
            '--run-glob', 'sd15-*', '--steps', '1000', '--benchmarks', 'pickv2', '--out', str(self.out)])

    def test_plan_preserves_preflight_logs_and_still_rejects_unrelated_outputs(self):
        logs = self.out / 'logs'
        logs.mkdir(parents=True)
        log = logs / 'preflight.log'
        log.write_text('CUDA and assets passed\n')
        # A render without its plan must never be adopted just because logs exist.
        unrelated = self.out / 'render'
        unrelated.mkdir()
        with self.assertRaisesRegex(ValueError, 'nonempty unrelated'):
            make_plan(self.plan_args)
        self.assertFalse((self.out / 'plan.json').exists())
        unrelated.rmdir()
        plan = make_plan(self.plan_args)
        self.assertEqual(make_plan(self.plan_args), plan)
        self.assertEqual(read_plan(self.out)[0], plan)
        self.assertEqual(log.read_text(), 'CUDA and assets passed\n')

    def test_logs_file_or_symlink_cannot_bypass_unrelated_output_guard(self):
        self.out.mkdir()
        logs = self.out / 'logs'
        logs.write_text('unrelated')
        with self.assertRaisesRegex(ValueError, 'nonempty unrelated'):
            make_plan(self.plan_args)
        logs.unlink()
        logs.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'nonempty unrelated'):
            make_plan(self.plan_args)

    @staticmethod
    def write(path, data):
        Path(path).write_text(json.dumps(data))

    def command_render(self, command):
        from PIL import Image
        def arg(name):
            return command[command.index(name) + 1]
        config, rows = read_json(arg('--config')), load_suite(arg('--prompts'))
        out = Path(arg('--out'))
        out.mkdir(parents=True, exist_ok=True)
        (out / 'images').mkdir(exist_ok=True)
        run = {'config': config, 'arm': arg('--arm'), 'suite': rows, 'suite_id': digest(rows),
               'full_suite_id': digest(rows), 'sampling': config['sampling'],
               'expected_images': len(rows) * len(config['sampling']['seeds'])}
        run['run_id'] = digest(run)
        self.write(out / 'run.json', run)
        records = []
        for row in rows:
            for seed in run['sampling']['seeds']:
                name = f'images/{row["scene_id"]}-{seed}.png'
                Image.new('RGB', (8, 8), (seed * 20, 50, 100)).save(out / name)
                record = {k: row[k] for k in ('scene_id', 'variant_id', 'prompt', 'tag')}
                record.update(seed=seed, image=name, image_sha256=sha256(out / name))
                record['image_id'] = digest({'run_id': run['run_id'], **record})
                records.append(record)
        (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        self.write(out / 'complete.json', {'images': len(records), 'manifest_sha256': sha256(out / 'manifest.jsonl')})

    def generated(self):
        plan = make_plan(self.plan_args)
        args = parse_args(['generate', '--out', str(self.out)])
        generate_task(plan['tasks'][0], plan, self.out, args, command_runner=self.command_render)
        return plan, plan['tasks'][0]

    def test_plan_without_assets_dry_run_freeze_and_nested_filter(self):
        self.plan_args.dry_run = True
        plan = make_plan(self.plan_args)
        self.assertFalse(self.out.exists())
        self.assertEqual(sum(t['images'] for t in plan['tasks']), 4)
        self.plan_args.dry_run = False
        self.assertEqual(make_plan(self.plan_args), plan)
        self.assertEqual(make_plan(self.plan_args), plan)
        self.prompts.unlink()  # execution reads its frozen copy
        self.assertEqual(read_plan(self.out)[0], plan)
        args = parse_args(['generate', '--out', str(self.out), '--run-glob', 'sd15-*'])
        self.assertEqual(len(filter_tasks(plan, args)), 1)
        (self.out / 'prompts/pickv2.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'snapshot checksum'):
            read_plan(self.out)

    def test_prompt_scope_provenance_is_verified_and_frozen(self):
        selection = {'mode': 'smoke-only', 'smoke_prompts': 2, 'repository': 'official-fixture',
                     'revision': 'fixed-fixture', 'panels': {
                         'pickv2': {'sha256': sha256(self.prompts), 'prompts': 2, 'source_prompts': 487}}}
        self.write(self.prompts.parent / 'selection.json', selection)
        plan = make_plan(self.plan_args)
        panel = plan['panels']['pickv2']
        self.assertEqual(panel['mode'], 'smoke-only')
        self.assertEqual(panel['provenance']['status'], 'selection-verified')
        self.assertEqual(panel['provenance']['selection'], selection)
        (self.prompts.parent / 'selection.json').unlink()
        self.prompts.unlink()
        self.assertEqual(read_plan(self.out)[0]['panels']['pickv2'], panel)

    def test_prompt_provenance_rejects_stale_hash_or_count_and_marks_custom(self):
        custom = make_plan(parse_args(['plan', '--config', str(self.config), '--runs-root', str(self.runs),
            '--steps', '1000', '--benchmarks', 'pickv2', '--out', str(self.out), '--dry-run']))
        self.assertEqual(custom['panels']['pickv2']['mode'], 'custom-unverified')
        self.assertEqual(custom['panels']['pickv2']['provenance']['status'], 'unverified')
        selection = {'mode': 'full-benchmark', 'panels': {
            'pickv2': {'sha256': '0' * 64, 'prompts': 2}}}
        path = self.prompts.parent / 'selection.json'
        self.write(path, selection)
        with self.assertRaisesRegex(ValueError, 'selection hash/count'):
            make_plan(self.plan_args)
        selection['panels']['pickv2'].update(sha256=sha256(self.prompts), prompts=487)
        self.write(path, selection)
        with self.assertRaisesRegex(ValueError, 'selection hash/count'):
            make_plan(self.plan_args)
        self.assertFalse(self.out.exists())

    def test_compositional_selection_schema_preserves_scope(self):
        from self_correct_dpo.evaluation.suite import panel_provenance
        # Compositional preparation uses one suite.jsonl and a single-panel manifest.
        selection = {'benchmark': 'geneval', 'suite_sha256': sha256(self.prompts),
                     'prompts': 2, 'mode': 'smoke-only', 'source': {'revision': 'fixed-official-fixture'}}
        self.write(self.prompts.parent / 'selection.json', selection)
        info = panel_provenance(self.prompts, 'geneval', sha256(self.prompts), 2)
        self.assertEqual(info['mode'], 'smoke-only')
        self.assertEqual(info['provenance']['selection']['source'], selection['source'])
        with self.assertRaisesRegex(ValueError, 'benchmark/mode mismatch'):
            panel_provenance(self.prompts, 'dpg', sha256(self.prompts), 2)

    def test_step_selection_base_latest_and_missing(self):
        self.assertEqual(selected_steps([0, 1000, 2000, 3000], every=1000, latest=2, base=True), [0, 2000, 3000])
        with self.assertRaisesRegex(ValueError, 'missing'):
            selected_steps([0, 1000], steps=[999])
        self.plan_args.steps = [0, 1000]
        plan = make_plan(self.plan_args)
        self.assertEqual([t['checkpoint']['checkpoint_id'] == 'base' for t in plan['tasks']], [True, False])

    def test_generate_then_score_without_generator_or_training_files(self):
        plan, task = self.generated()
        shutil.rmtree(self.runs)
        args = parse_args(['score', '--out', str(self.out), '--metrics', 'pickscore'])
        def command(command):
            self.assertIn('self_correct_dpo.evaluation.metrics', command)
            self.assertNotIn('self_correct_dpo.evaluation.render', command)
            def arg(name): return command[command.index(name) + 1]
            score_panel(arg('--run'), arg('--out'), 'pickscore', {}, batch_size=2,
                        scorer_factory=lambda *a: lambda images, prompts: [0.25] * len(images),
                        asset_id={'metric': 'pickscore', 'scale': 'test cosine'})
        score_task(task, 'pickscore', plan, self.out, args, command_runner=command)
        score_task(task, 'pickscore', plan, self.out, args, command_runner=command)
        summary = summarize(plan, self.out, parse_args(['summarize', '--out', str(self.out)]))
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['rows'][0]['pickscore_images'], 4)
        self.assertEqual(summary['rows'][0]['pickscore'], 0.25)
        # A new reward is scheduled independently and leaves existing scores intact.
        calls = []
        score_task(task, 'hpsv3', plan, self.out, args, command_runner=calls.append)
        self.assertEqual(calls[0][calls[0].index('--metric') + 1], 'hpsv3')

    def test_alien_valid_render_and_changed_checkpoint_are_rejected(self):
        plan, task = self.generated()
        target = self.out / task['output'] / 'render/run.json'
        run = read_json(target)
        run['config']['model']['path'] = '/another-model'
        run['run_id'] = digest({k: v for k, v in run.items() if k != 'run_id'})
        self.write(target, run)
        with self.assertRaisesRegex(ValueError, 'frozen plan'):
            validate_task_render(task, plan, self.out)
        self.write(self.run / 'training.json', {'run_id': 'different', 'config': {'model': self.model}})
        with self.assertRaises(ValueError):
            generate_task(task, plan, self.out, parse_args(['generate', '--out', str(self.out)]), command_runner=self.command_render)

    def test_incomplete_summary_and_gallery_do_not_select_statistical_samples(self):
        plan, task = self.generated()
        args = parse_args(['summarize', '--out', str(self.out)])
        self.assertFalse(summarize(plan, self.out, args)['complete'])
        render = self.out / task['output'] / 'render'
        before = sha256(render / 'manifest.jsonl')
        args = parse_args(['gallery', '--out', str(self.out), '--scene-id', 'p0', '--seed', '0', '--limit', '1'])
        path = make_gallery(plan, self.out, args)
        self.assertTrue(Path(path).is_file())
        selected = read_json(Path(path).parent / 'selection.json')
        self.assertEqual(len(selected['images']), 1)
        self.assertEqual(len(read_jsonl(render / 'manifest.jsonl')), 4)
        self.assertEqual(sha256(render / 'manifest.jsonl'), before)

    def test_official_metrics_cannot_score_arbitrary_panel(self):
        plan = make_plan(self.plan_args)
        args = parse_args(['score', '--out', str(self.out), '--metrics', 'geneval'])
        with self.assertRaisesRegex(ValueError, 'own prompt panel'):
            metrics_for(plan['tasks'][0], plan, args)

    def test_structured_reward_preserves_details_and_rejects_bad_output(self):
        plan, task = self.generated()
        value = {'score': 3.0, 'components': {'alignment': 2.0, 'coherence': 3.0, 'style': 4.0},
                 'raw_output': 'Alignment: 2, Coherence: 3, Style: 4',
                 'aggregation': 'arithmetic_mean_of_alignment_coherence_style'}
        target = self.out / task['output']
        summary = score_panel(target / 'render', target / 'scores/unifiedreward', 'unifiedreward', {},
            scorer_factory=lambda *a: lambda images, prompts: [value] * len(images), asset_id={'metric': 'unifiedreward', 'scale': 'ACS mean'})
        self.assertEqual(summary['components']['alignment'], 2.0)
        self.assertEqual(read_jsonl(target / 'scores/unifiedreward/scores.jsonl')[0]['raw_output'], value['raw_output'])
        bad = copy.deepcopy(value)
        bad['components']['style'] = float('nan')
        with self.assertRaises(ValueError): score_value(bad, 'unifiedreward')
        with self.assertRaises(ValueError): score_value(0, 'unifiedreward')

    def test_complete_scores_cannot_be_relabelled_as_another_metric(self):
        plan, task = self.generated()
        target = self.out / task['output']
        score_panel(target / 'render', target / 'scores/pickscore', 'pickscore', {},
            scorer_factory=lambda *a: lambda images, prompts: [.25] * len(images),
            asset_id={'metric': 'pickscore', 'scale': 'test cosine'})
        # Every original checksum remains valid, but the directory label is wrong.
        shutil.copytree(target / 'scores/pickscore', target / 'scores/hpsv3')
        summary = summarize(plan, self.out, parse_args(['summarize', '--out', str(self.out), '--metrics', 'hpsv3']))
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['rows'][0]['status'], 'invalid')
        self.assertIn('different metric', summary['rows'][0]['error'])
        self.assertNotIn('hpsv3', summary['rows'][0])

    def test_gallery_rejects_scores_from_another_valid_render(self):
        self.plan_args.steps = [0, 1000]
        plan = make_plan(self.plan_args)
        args = parse_args(['generate', '--out', str(self.out)])
        for task in plan['tasks']:
            generate_task(task, plan, self.out, args, command_runner=self.command_render)
        base, aligned = [self.out / t['output'] for t in plan['tasks']]
        score_panel(base / 'render', base / 'scores/pickscore', 'pickscore', {},
            scorer_factory=lambda *a: lambda images, prompts: [.25] * len(images),
            asset_id={'metric': 'pickscore', 'scale': 'test cosine'})
        # Prompt IDs, seed IDs and score-file checksums match; the render does not.
        shutil.copytree(base / 'scores/pickscore', aligned / 'scores/pickscore')
        args = parse_args(['gallery', '--out', str(self.out), '--steps', '1000'])
        with self.assertRaisesRegex(ValueError, 'different generated image panel'):
            make_gallery(plan, self.out, args)
        self.assertFalse((self.out / 'galleries/gallery/selection.json').exists())

    def test_failed_stages_persist_errors_and_successful_retries_clear_them(self):
        config = read_json(self.config)
        config['metric_groups']['legacy'] = ['pickscore', 'hpsv3']
        self.write(self.config, config)
        plan = make_plan(self.plan_args)
        task = plan['tasks'][0]
        target = self.out / task['output']
        generate_args = parse_args(['generate', '--out', str(self.out)])
        summary_args = parse_args(['summarize', '--out', str(self.out)])
        quiet = io.StringIO()
        with redirect_stdout(quiet), redirect_stderr(quiet), \
             patch('self_correct_dpo.evaluation.suite.generate_task', side_effect=RuntimeError('generation failed')):
            self.assertEqual(run_stage(plan, self.out, generate_args), 1)
        failure = target / 'failures/generate.json'
        self.assertIn('generation failed', read_json(failure)['error'])
        summary = summarize(plan, self.out, summary_args)
        self.assertEqual(summary['rows'][0]['status'], 'failed')
        self.assertIn('generation failed', summary['rows'][0]['error'])
        def generate(*args):
            return generate_task(*args, command_runner=self.command_render)
        with redirect_stdout(quiet), redirect_stderr(quiet), \
             patch('self_correct_dpo.evaluation.suite.generate_task', side_effect=generate):
            self.assertEqual(run_stage(plan, self.out, generate_args), 0)
        self.assertFalse(failure.exists())
        self.assertEqual(summarize(plan, self.out, summary_args)['rows'][0]['status'], 'pending_scores')
        score_args = parse_args(['score', '--out', str(self.out), '--metrics', 'pickscore'])
        with redirect_stdout(quiet), redirect_stderr(quiet), \
             patch('self_correct_dpo.evaluation.suite.score_task', side_effect=RuntimeError('scoring failed')):
            self.assertEqual(run_stage(plan, self.out, score_args), 1)
        failure = target / 'failures/score-pickscore.json'
        summary = summarize(plan, self.out, summary_args)
        # Missing HPSv3 must not hide the earlier PickScore failure.
        self.assertEqual(summary['rows'][0]['status'], 'failed')
        self.assertIn('scoring failed', summary['rows'][0]['error'])
        self.assertTrue(failure.exists())
        def score(task, metric, plan, out, args):
            return score_panel(target / 'render', target / 'scores' / metric, metric, {},
                scorer_factory=lambda *a: lambda images, prompts: [.25] * len(images),
                asset_id={'metric': metric, 'scale': 'test cosine'})
        with redirect_stdout(quiet), redirect_stderr(quiet), \
             patch('self_correct_dpo.evaluation.suite.score_task', side_effect=score):
            self.assertEqual(run_stage(plan, self.out, score_args), 0)
        self.assertFalse(failure.exists())
        summary = summarize(plan, self.out, parse_args(['summarize', '--out', str(self.out), '--metrics', 'pickscore']))
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['rows'][0]['error'], '')

    def test_list_and_plan_cli_without_site_packages(self):
        script = Path(__file__).resolve().parents[1] / 'evaluate_suite.py'
        result = subprocess.run([sys.executable, '-S', str(script), 'list', '--json', '--runs-root', str(self.runs)],
                                text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)[0]['steps'], [0, 1000])


if __name__ == '__main__':
    unittest.main()
