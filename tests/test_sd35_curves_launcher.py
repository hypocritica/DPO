"""Exercise launcher process failures and argument forwarding without a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/run_sd35_curves.sh'


class CurveLauncherTests(unittest.TestCase):
    def fixture(self, root):
        python = root / 'fake python'
        python.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
stage = sys.argv[3]
print('fixture ' + stage)
raise SystemExit(int(os.environ.get('FAIL_CODE', '7')) if stage == os.environ.get('FAIL_STAGE') else 0)
''')
        python.chmod(0o700)
        env = {**os.environ, 'SCD_FLOW_PY': str(python), 'CALLS': str(root / 'calls.jsonl'),
                'SCD_CURVES_PREP': str(root / 'prepared inputs'), 'SCD_CURVES_OUT': str(root / 'eval output'),
                'SCD_CURVES_LOGS': str(root / 'logs')}
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env.pop('SCD_CURVES_GPUS', None)
        return env

    def test_interruption_stops_all_remaining_stages(self):
        for code in (130, 143):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = self.fixture(root)
                env.update(FAIL_STAGE='generate', FAIL_CODE=str(code))
                result = subprocess.run(['bash', str(LAUNCHER), 'all'], env=env, capture_output=True)
                self.assertEqual(result.returncode, code)
                calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
                self.assertEqual([call[2] for call in calls], ['prepare', 'plan', 'generate'])

    def test_empty_or_disabled_visible_allocation_is_rejected(self):
        for devices in ('', '-1'):
            with self.subTest(devices=devices), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = self.fixture(root)
                env['CUDA_VISIBLE_DEVICES'] = devices
                result = subprocess.run(['bash', str(LAUNCHER), 'generate'], env=env, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse((root / 'calls.jsonl').exists())

    def test_all_keeps_summary_after_stage_failure_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            env['FAIL_STAGE'] = 'generate'
            result = subprocess.run(['bash', str(LAUNCHER), 'all', '--gpus', '2,3'], env=env,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            self.assertEqual([call[2] for call in calls], ['prepare', 'plan', 'generate', 'score', 'summarize'])
            at = calls[0].index('--generation-batch-size')
            self.assertEqual(calls[0][at + 1], '4')
            self.assertIn('2,3', calls[2])
            self.assertIn('--keep-going', calls[2])
            self.assertEqual(len(list((root / 'logs').glob('*.log'))), 5)

    def test_failed_preparation_skips_gpu_stages_but_attempts_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            env['FAIL_STAGE'] = 'prepare'
            result = subprocess.run(['bash', str(LAUNCHER), 'all'], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            self.assertEqual([call[2] for call in calls], ['prepare', 'summarize'])

    def test_catalog_arguments_and_dry_run_are_non_destructive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            command = ['bash', str(LAUNCHER), 'prepare', '--catalog', str(root / 'catalog file.json'),
                       '--run-glob', 'sd35-*', '--exclude-prompts', str(root / 'panel 1.jsonl'),
                       '--exclude-prompts', str(root / 'panel 2.jsonl')]
            result = subprocess.run([*command, '--dry-run'], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'calls.jsonl').exists())
            self.assertFalse((root / 'logs').exists())
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            call = json.loads((root / 'calls.jsonl').read_text())
            self.assertNotIn('--runs-root', call)
            self.assertIn(str(root / 'catalog file.json'), call)
            self.assertEqual(call[-3:], ['--exclude-prompts', str(root / 'panel 1.jsonl'), str(root / 'panel 2.jsonl')])


if __name__ == '__main__':
    unittest.main()
