"""Fixed handoff identities, panel order and predeclared report subsets; CPU only."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.evaluation import sd15_main as workflow
from self_correct_dpo.evaluation.suite import read_plan

ROOT = Path(__file__).resolve().parents[1]


class Sd15HandoffTests(unittest.TestCase):
    def test_published_metadata_is_exact_and_has_no_historical_scores(self):
        selection = workflow.load_selection()
        catalog = read_json(workflow.CATALOG)
        self.assertEqual(len(catalog['experiments']), 9)
        self.assertEqual(sum(len(e['steps']) for e in catalog['experiments']), 10)
        self.assertEqual(selection['sampling']['generation_batch_size'], 32)
        self.assertEqual(sum(selection['benchmarks'].values()) * len(selection['variants']), 58509)
        self.assertFalse(any('score' in k for v in selection['variants'] for k in v))
        for entry in catalog['experiments']:
            for step in entry['steps']:
                variant = next(v for v in selection['variants'] if v['catalog_id'] == entry['id'] and v['step'] == step)
                self.assertEqual(entry['expected_run_id'], variant['run_id'])
                self.assertEqual(entry['expected_checkpoint_ids'][str(step)], variant['checkpoint_id'])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = read_json(workflow.CATALOG)
        self.selection = workflow.load_selection()
        runs = self.root / 'runs'
        model = dict(self.selection['model'], path=str(self.root / 'base'))
        self.catalog['models']['sd15'] = model
        self.catalog['runs_roots']['legacy'] = str(runs)
        self.selection['model'] = model
        for entry in self.catalog['experiments']:
            run = runs / entry['run']
            run.mkdir(parents=True)
            method = entry['method']
            stage = method.startswith('stagewise')
            objective = 'dmpo' if 'dmpo' in method else 'dspo' if 'dspo' in method else 'dpo' if stage else method
            training = {'objective': objective, 'noise_views': 6 if stage else 1, 'formal_arm': 'C',
                        'stagewise': {'objective': 'dpo'} if stage else None}
            self.write(run / 'training.json', {'run_id': entry['expected_run_id'], 'config': {
                'model': model, 'method': 'stagewise-C-gaussian' if stage else 'dpo', 'training': training}})
            for step in entry['steps']:
                adapter = run / 'checkpoints' / f'step-{step:07d}' / 'adapter'
                adapter.mkdir(parents=True)
                (adapter / 'adapter_model.bin').write_bytes(b'fixture only; never loaded as tensors')
                checkpoint = {'run_id': entry['expected_run_id'], 'step': step,
                              'files': {'adapter/adapter_model.bin': sha256(adapter / 'adapter_model.bin')}}
                checkpoint['checkpoint_id'] = digest(checkpoint)
                self.write(adapter.parent / 'checkpoint.json', checkpoint)
                entry['expected_checkpoint_ids'][str(step)] = checkpoint['checkpoint_id']
                variant = next(v for v in self.selection['variants'] if v['catalog_id'] == entry['id'] and v['step'] == step)
                variant.update(run_path=str(run), checkpoint_id=checkpoint['checkpoint_id'])
        self.catalog_path, self.selection_path = self.root / 'catalog.json', self.root / 'selection.json'
        self.write(self.catalog_path, self.catalog)
        self.write(self.selection_path, self.selection)
        self.suite = read_json(workflow.SUITE)
        for panel, count in workflow.COUNTS.items():
            folder = self.root / panel
            folder.mkdir()
            prompts = folder / f'{panel}.jsonl'
            records = [{'scene_id': str(i), 'variant_id': 'canonical', 'prompt': f'test {panel} {i}', 'tag': panel,
                'metadata': {'prompt': f'test {panel} {i}', 'tag': panel}, 'suite_mode': 'canonical-only-v1',
                'template_family': 'canonical', 'transform_type': 'canonical'} for i in range(count)]
            prompts.write_text(''.join(json.dumps(r) + '\n' for r in records))
            self.write(folder / 'selection.json', {'mode': 'full-benchmark', 'panels': {
                panel: {'prompts': count, 'sha256': sha256(prompts)}}})
            self.suite['benchmarks'][panel] = str(prompts)
        self.suite_path = self.root / 'suite.json'
        self.write(self.suite_path, self.suite)
        self.out = self.root / 'out'
        self.args = argparse.Namespace(catalog=str(self.catalog_path), suite_config=str(self.suite_path),
            out=str(self.out), resources=None, dry_run=False)

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value))

    def cli(self, stage, *extra):
        return [stage, '--out', str(self.out), '--selection', str(self.selection_path), *extra]

    def test_exact_plan_and_no_latest_fallback_or_identity_substitution(self):
        result = workflow.plan_main(self.args, self.selection)
        self.assertEqual(result['images'], 58509)
        self.assertEqual(result['tasks'], 33)
        plan, _ = read_plan(self.out)
        workflow.validate_plan(plan, self.selection)
        adapter = next(t for t in plan['tasks'] if t['step'])
        checkpoint = Path(adapter['run_path']) / 'checkpoints' / f'step-{adapter["step"]:07d}' / 'checkpoint.json'
        before = checkpoint.read_bytes()
        changed = read_json(checkpoint)
        changed['run_id'] = 'substituted-run'
        changed['checkpoint_id'] = digest({k: v for k, v in changed.items() if k != 'checkpoint_id'})
        self.write(checkpoint, changed)
        with self.assertRaisesRegex(ValueError, 'run/step'):
            workflow.validate_live_checkpoints(plan)
        checkpoint.write_bytes(before)
        (checkpoint.parent / 'adapter/adapter_model.bin').write_bytes(b'wrong weights')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            workflow.validate_live_checkpoints(plan)

    def test_dry_plan_checks_historical_ids_before_any_output(self):
        self.args.dry_run = True
        wrong = copy.deepcopy(self.selection)
        next(v for v in wrong['variants'] if v['step'])['checkpoint_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checkpoint_id'):
            workflow.plan_main(self.args, wrong)
        self.assertFalse(self.out.exists())

    def test_generate_orders_panels_and_never_starts_score(self):
        workflow.plan_main(self.args, self.selection)
        commands = []
        with patch.object(workflow, 'run_command', commands.append):
            self.assertEqual(workflow.main(self.cli('generate', '--gpus', '4,5,6,7')), 0)
        self.assertEqual([c[c.index('--benchmarks') + 1] for c in commands], list(workflow.PANELS))
        self.assertTrue(all('generate' in c and 'score' not in c for c in commands))
        self.assertTrue(all(c[c.index('--gpus') + 1] == '4,5,6,7' for c in commands))
        with self.assertRaisesRegex(ValueError, 'four'):
            workflow.allocated_devices('0,1')

    def test_keep_going_continues_panels_but_reports_failure(self):
        workflow.plan_main(self.args, self.selection)
        commands = []
        def failed_panel(command):
            commands.append(command)
            if command[command.index('--benchmarks') + 1] == 'pickv2':
                raise subprocess.CalledProcessError(1, command)
        with patch.object(workflow, 'run_command', failed_panel):
            self.assertEqual(workflow.main(self.cli('generate', '--gpus', '0,1,2,3', '--keep-going')), 1)
        self.assertEqual(len(commands), 3)

    def test_score_is_separate_and_does_not_read_training_assets(self):
        workflow.plan_main(self.args, self.selection)
        commands = []
        with patch.object(workflow, 'run_command', commands.append), \
             patch.object(workflow, 'validate_task_render') as completed, \
             patch.object(workflow, 'validate_live_checkpoints', side_effect=AssertionError('Training read during scoring')):
            self.assertEqual(workflow.main(self.cli('score', '--gpus', '0,1,2,3')), 0)
            self.assertEqual(completed.call_count, 33)
        self.assertEqual(len(commands), 3)
        self.assertTrue(all(c[c.index('--metric-group') + 1] == 'legacy' for c in commands))
        with patch.object(workflow, 'validate_task_render', side_effect=FileNotFoundError('uncommitted render')), \
             patch.object(workflow, 'run_command', side_effect=AssertionError('Scoring before generation')):
            with self.assertRaises(SystemExit):
                workflow.main(self.cli('score', '--gpus', '0,1,2,3'))

    def test_report_roles_exclude_exact_method_step_without_editing_plan(self):
        workflow.plan_main(self.args, self.selection)
        plan, _ = read_plan(self.out)
        before = digest(plan)
        main = workflow.report_plan(plan, 'main')
        sensitivity = workflow.report_plan(plan, 'sensitivity')
        self.assertEqual(len(main['tasks']), 30)
        self.assertEqual(len(sensitivity['tasks']), 6)
        self.assertFalse(any(t['method'] == 'stagewise_dspo_k6' and t['step'] == 10000 for t in main['tasks']))
        self.assertEqual({t['step'] for t in sensitivity['tasks']}, {6000, 10000})
        self.assertEqual(main['plan_id'], plan['plan_id'])
        self.assertEqual(digest(plan), before)
        self.assertEqual(main['report_selection']['source_plan_id'], plan['plan_id'])

    def test_launcher_passes_stage_and_options_to_local_python(self):
        process = subprocess.run(['bash', str(ROOT / 'scripts/run_sd15_main.sh'), '--help'],
            env={**__import__('os').environ, 'SCD_SD15_PY': sys.executable}, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('generate', process.stdout)
        self.assertIn('score', process.stdout)


if __name__ == '__main__':
    unittest.main()
