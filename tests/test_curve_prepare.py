"""Real tiny checkpoint metadata and streamed caption selection, without GPUs."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from self_correct_dpo.common import digest, read_json, read_jsonl, sha256, write_json, write_jsonl
from self_correct_dpo.evaluation.curve_prepare import CURVE_METHODS, load_prepared, prepare_curve
from self_correct_dpo.data.benchmarks import prompt_key


class CurvePrepareTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / 'runs'
        self.runs.mkdir()
        self.pairs = self.root / 'pairs.jsonl'
        write_jsonl(self.pairs, [{'split': 'train', 'prompts': [{'prompt': 'ＣＡＰＴＩＯＮ ２'}]},
            *[{'split': 'validation_unique' if i % 2 else 'validation',
               **({'caption': f'caption {i}'} if i % 3 else {'prompts': [{'prompt': f'caption {i}'}]})}
              for i in range(180)],
            {'split': 'validation', 'prompt': 'ＣＡＰＴＩＯＮ ３'},
            {'split': 'test', 'prompt': 'must never select test caption'}])
        self.model = {'family': 'sd35_medium', 'path': '/not-loaded/sd35', 'revision': 'fixed'}
        for method in CURVE_METHODS[1:]:
            self.add_run(method, method)
        self.official = {}
        for name in ('pickv2', 'hpsv2', 'partiprompts', 'geneval', 'dpg'):
            path = self.root / f'formal-{name}.jsonl'
            write_jsonl(path, [{'prompt': 'caption 0'}])
            self.official[name] = str(path)
        self.extra = self.root / 'extra.jsonl'
        write_jsonl(self.extra, [{'caption': 'CAPTION 1'}])
        self.scorers = self.root / 'original-scorers.json'
        write_json(self.scorers, {'scorers': {'pickscore': {'model': '/same/weights', 'processor': '/same/processor',
                    'batch_size': 8, 'python': '/same/env/python', 'custom_option': 'preserve'}}})
        self.config = self.root / 'template.json'
        write_json(self.config, {'protocols': {'sd35_medium': {'seeds': [17, 29, 43, 71], 'dtype': 'bfloat16',
                    'height': 1024, 'width': 1024, 'num_inference_steps': 28, 'guidance_scale': 4.5}},
                    'scorers_config': str(self.scorers), 'benchmarks': self.official,
                    'metric_groups': {'modern': ['pickscore', 'hpsv3']}})
        self.args = SimpleNamespace(prepare_dir=str(self.root / 'prepared'), runs_root=[str(self.runs)],
                     catalog=None, run_glob=None, resources=None, config=str(self.config), target_steps=3200,
                     exclude_prompts=[str(self.extra)])

    def add_run(self, name, method, target=3200):
        run = self.runs / name
        run.mkdir()
        training = {'max_steps': target, 'objective': 'dpo', 'noise_views': 1, 'formal_arm': 'C'}
        algorithm = 'dpo'
        if method in ('sft', 'linear_dpo'):
            training['objective'] = method
        elif method.startswith(('global_', 'stagewise_')):
            training['noise_views'] = int(method[-1])
            algorithm = 'stagewise-C-gaussian' if method.startswith('stagewise_') else 'multiview-C-gaussian'
            if method.startswith('stagewise_'):
                training['stagewise'] = {'objective': 'dpo', 'boundary': 500}
        write_json(run / 'training.json', {'run_id': name, 'config': {'model': self.model,
                   'method': algorithm, 'training': training, 'data': {'pairs': str(self.pairs)}}})
        for step in (1000, 3200):
            self.add_checkpoint(run, name, step)
        return run

    def add_checkpoint(self, run, name, step):
        checkpoint = run / f'checkpoints/step-{step:07d}'
        (checkpoint / 'adapter').mkdir(parents=True)
        (checkpoint / 'adapter/weights.bin').write_bytes(b'tiny-adapter')
        record = {'run_id': name, 'step': step, 'files': {'adapter/weights.bin': sha256(checkpoint / 'adapter/weights.bin')}}
        record['checkpoint_id'] = digest(record)
        write_json(checkpoint / 'checkpoint.json', record)

    def test_all_snapshots_and_eight_methods_use_fixed_disjoint_development_panel(self):
        self.add_run('smoke-extra', 'dpo', target=2)
        self.add_run('dpo-backup-v1', 'dpo')
        result = prepare_curve(self.args)
        prepared = Path(result['prepare_dir'])
        catalog = read_json(result['catalog'])
        self.assertEqual({r['method'] for r in catalog['experiments']}, set(CURVE_METHODS[1:]))
        self.assertTrue(all(r['steps'] == [1000, 3200] for r in catalog['experiments']))
        self.assertEqual(catalog['models']['sd35_medium'], self.model)
        rows = read_jsonl(prepared / 'prompts/pickv2.jsonl')
        self.assertEqual(len(rows), 128)
        keys = {prompt_key(row['prompt']) for row in rows}
        self.assertEqual(len(keys), 128)
        self.assertFalse(keys & {'caption 0', 'caption 1', 'caption 2', 'must never select test caption'})
        selection = read_json(prepared / 'prompts/selection.json')
        self.assertEqual(selection['mode'], 'development-curve')
        self.assertEqual(len(selection['sources']), 1)
        self.assertEqual(len(selection['exclude_sources']), 6)
        self.assertEqual(selection['missing_exclusion_panels'], [])
        suite = read_json(result['config'])
        self.assertEqual(suite['protocols']['sd35_medium']['seeds'], [17])
        self.assertEqual(suite['protocols']['sd35_medium']['generation_batch_size'], 4)
        self.assertEqual(suite['metric_groups'], {'legacy': ['pickscore'], 'modern': ['pickscore']})
        spec = read_json(prepared / 'scorers.json')['scorers']['pickscore']
        self.assertEqual(spec['batch_size'], 1)
        self.assertEqual(spec['custom_option'], 'preserve')
        self.assertEqual(spec['model'], '/same/weights')
        self.assertEqual(load_prepared(prepared)['independence'], 'text-disjoint-from-configured-formal-panels')
        from self_correct_dpo.evaluation.suite import make_plan, parse_args
        plan = make_plan(parse_args(['plan', '--catalog', result['catalog'], '--config', result['config'],
                    '--benchmarks', 'pickv2', '--out', str(self.root / 'plan'), '--dry-run']))
        self.assertEqual(len(plan['tasks']), 15)
        self.assertTrue(all(task['images'] == 128 and task['sampling']['seeds'] == [17] for task in plan['tasks']))
        self.assertEqual(plan['panels']['pickv2']['mode'], 'development-curve')


    def test_batch_size_is_bounded_and_cannot_change_frozen_preparation(self):
        for value in (0, -1, True, 2.5):
            self.args.generation_batch_size = value
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                prepare_curve(self.args)
        self.args.generation_batch_size = 8
        prepare_curve(self.args)
        self.args.generation_batch_size = 2
        with self.assertRaises(ValueError):
            prepare_curve(self.args)
        self.args.prepare_dir = str(self.root / 'batch2-prepared')
        result = prepare_curve(self.args)
        self.assertEqual(read_json(result['config'])['protocols']['sd35_medium']['generation_batch_size'], 2)

    def test_replay_is_frozen_without_training_scan_and_rejects_tampering(self):
        first = prepare_curve(self.args)
        shutil.rmtree(self.runs)
        self.pairs.unlink()
        self.extra.unlink()
        with patch('self_correct_dpo.evaluation.curve_prepare._selected_runs', side_effect=AssertionError('rescan')):
            self.assertEqual(prepare_curve(self.args), first)
        (Path(first['prepare_dir']) / 'scorers.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen curve preparation changed'):
            prepare_curve(self.args)

    def test_missing_or_duplicate_methods_are_explicit_errors(self):
        shutil.rmtree(self.runs / 'sft')
        with self.assertRaisesRegex(ValueError, 'Missing SD3.5.*sft'):
            prepare_curve(self.args)
        self.add_run('sft', 'sft')
        self.add_run('new-dpo', 'dpo')
        with self.assertRaisesRegex(ValueError, 'Multiple curve candidates for dpo'):
            prepare_curve(self.args)
        self.assertFalse(Path(self.args.prepare_dir).exists())

    def test_insufficient_validation_never_falls_back_and_missing_formal_is_recorded(self):
        write_jsonl(self.root / 'short.jsonl', [{'split': 'train', 'prompt': 'train'},
                    {'split': 'validation', 'caption': 'one'}, {'split': 'test', 'caption': 'other'}])
        for run in self.runs.iterdir():
            path = run / 'training.json'
            saved = read_json(path)
            saved['config']['data']['pairs'] = str(self.root / 'short.jsonl')
            path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'Need 128.*No training/test fallback'):
            prepare_curve(self.args)
        for run in self.runs.iterdir():
            path = run / 'training.json'
            saved = read_json(path)
            saved['config']['data']['pairs'] = str(self.pairs)
            path.write_text(json.dumps(saved))
        Path(self.official['dpg']).unlink()
        messages = io.StringIO()
        with redirect_stdout(messages):
            result = prepare_curve(self.args)
        manifest = load_prepared(result['prepare_dir'])
        self.assertEqual(manifest['independence'], 'not-established')
        self.assertEqual(manifest['missing_exclusion_panels'], ['dpg'])
        self.assertIn('not established', messages.getvalue())

    def test_new_checkpoints_require_new_prepare_and_order_does_not_change_prompts(self):
        first = prepare_curve(self.args)
        frozen_hash = sha256(Path(first['prompts_root']) / 'pickv2.jsonl')
        self.add_checkpoint(self.runs / 'dpo', 'dpo', 2000)
        self.assertEqual(prepare_curve(self.args), first)
        self.assertNotIn(2000, next(r['steps'] for r in read_json(first['catalog'])['experiments'] if r['method'] == 'dpo'))
        rows = read_jsonl(self.pairs)
        self.pairs.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
        self.args.prepare_dir = str(self.root / 'new-prepared')
        second = prepare_curve(self.args)
        self.assertEqual(sha256(Path(second['prompts_root']) / 'pickv2.jsonl'), frozen_hash)
        self.assertIn(2000, next(r['steps'] for r in read_json(second['catalog'])['experiments'] if r['method'] == 'dpo'))

    def test_explicit_catalog_selects_run_then_expands_all_retained_steps(self):
        first = prepare_curve(self.args)
        catalog = read_json(first['catalog'])
        for entry in catalog['experiments']:
            entry['steps'] = [3200]
        explicit = self.root / 'selected.json'
        write_json(explicit, catalog)
        self.add_run('duplicate-dpo', 'dpo')
        self.args.catalog = str(explicit)
        self.args.prepare_dir = str(self.root / 'explicit-prepared')
        result = prepare_curve(self.args)
        self.assertTrue(all(row['steps'] == [1000, 3200] for row in read_json(result['catalog'])['experiments']))


if __name__ == '__main__':
    unittest.main()
