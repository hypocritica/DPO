"""Sequential benchmark failure isolation, retries and real journal summaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_flow as bench


class BenchmarkTests(unittest.TestCase):
    def test_real_launchers_freeze_matched_budgets_for_both_k_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(tag='fixture', runs_root=root / 'runs')
            environment = dict(os.environ, SCD_FLOW_PYTHON=sys.executable, FLOW_DEBUG_STEPS='20',
                               SCD_RESOURCES=str(ROOT / 'configs/resources-slai-flow.json'))
            for name, views in (('k1-b1', 1), ('k1-b2', 1), ('k4-b1', 4), ('k4-b2', 4)):
                attempt = root / name / '1'
                attempt.mkdir(parents=True)
                result = subprocess.run(bench.debug_command(args, attempt, name, config_only=True),
                                        cwd=ROOT, env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                config = json.loads(next((attempt / 'configs').glob('local-*.json')).read_text())
                training = config['training']
                self.assertEqual(training['noise_views'], views)
                self.assertEqual(training['world_size'], 2)
                self.assertEqual(training['comparison_budget']['per_update'], 4 * views)
                self.assertEqual(training['batch_size'] * training['gradient_accumulation_steps'], 2)
                self.assertEqual(training['logging'], 'jsonl')
                self.assertFalse(config['evaluation']['enabled'])
                self.assertEqual(config['model']['transformer_dtype'], 'float32')

    def test_training_summary_uses_last_ten_updates_and_both_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for rank, name in enumerate(('metrics-1.jsonl', 'rank-1-1.jsonl')):
                rows = [{'step': step, 'seconds_per_update': 100 if step <= 2 else 2 + rank,
                         'pairs_per_second': 2 - rank / 2, 'noise_views': 4,
                         'effective_batch_size_pair_views': 16, 'peak_cuda_bytes': (40 + rank) * 1024**3}
                        for step in range(1, 13)]
                (run / name).write_text(''.join(json.dumps(row) + '\n' for row in rows) + '{interrupted')
            result = bench.training_summary(run, 12)
            self.assertTrue(result['measurement_complete'])
            self.assertEqual(result['ranks']['0']['median_seconds_per_update'], 2)
            self.assertEqual(result['ranks']['1']['median_seconds_per_update'], 3)
            self.assertEqual(result['ranks']['1']['peak_allocated_GiB'], 41)
            self.assertEqual(result['ranks']['0']['median_pair_views_per_second'], 8)
            self.assertFalse(bench.training_summary(run, 20)['measurement_complete'])

    def test_failed_stage_continues_then_retries_without_repeating_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = ['benchmark_flow.py', '--tag', 'fixture', '--out-root', temporary]
            calls = []
            def execute(args, name, attempt, environment):
                calls.append((name, attempt.name))
                self.assertEqual(environment['OMP_NUM_THREADS'], '4')
                self.assertEqual(environment['MKL_NUM_THREADS'], '4')
                return {'status': 'failed' if name == 'k1-b2' and attempt.name == '1' else 'passed',
                        'returncode': 1 if name == 'k1-b2' else 0, 'interrupted': False,
                        'wall_seconds': 1, 'attempt': str(attempt), 'measurement': {}}
            with patch.object(sys, 'argv', command), patch.object(bench, 'execute_stage', side_effect=execute):
                self.assertEqual(bench.main(), 1)
                self.assertEqual(len(calls), 6)
                self.assertEqual(calls[-1][0], 'k4-b2')
                self.assertEqual(bench.main(), 0)
                self.assertEqual(calls[-1], ('k1-b2', '2'))
                self.assertEqual(len(calls), 7)
            root = Path(temporary) / 'fixture'
            self.assertTrue((root / 'k1-b2/1/result.json').is_file())
            self.assertTrue((root / 'k1-b2/2/result.json').is_file())
            self.assertIn('k4-b2', (root / 'summary.md').read_text())
            with patch.object(sys, 'argv', command + ['--threads', '8']):
                with self.assertRaisesRegex(ValueError, 'choose a new tag'):
                    bench.main()

    def test_cancellation_stops_remaining_stages_and_plan_only_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = ['benchmark_flow.py', '--tag', 'cancel', '--out-root', temporary]
            result = {'status': 'failed', 'returncode': -15, 'interrupted': True,
                      'wall_seconds': 1, 'measurement': {}}
            with patch.object(sys, 'argv', command), patch.object(bench, 'execute_stage', return_value=result) as execute:
                self.assertEqual(bench.main(), 75)
                self.assertEqual(execute.call_count, 1)
            untouched = Path(temporary) / 'plan-only'
            child = subprocess.run([sys.executable, str(ROOT / 'scripts/benchmark_flow.py'),
                                    '--tag', 'preview', '--out-root', str(untouched), '--plan-only'],
                                   capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(len(json.loads(child.stdout)['stages']), 6)
            self.assertFalse(untouched.exists())


if __name__ == '__main__':
    unittest.main()
