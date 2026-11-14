"""Exercise the Z-Image job's stage ordering, failures and GPU isolation on CPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_zimage_curves.sh'


class ZImageCurveLauncherTests(unittest.TestCase):
    def run_stage(self, stage, args=(), **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'fake python'
            calls = root / 'calls.jsonl'
            fake.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['TEST_CALLS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[2] == os.environ.get('FAIL_STAGE'):
    raise SystemExit(int(os.environ.get('FAIL_CODE', '7')))
''')
            fake.chmod(0o700)
            env = {k: v for k, v in os.environ.items() if not k.startswith('SCD_') and k != 'CUDA_VISIBLE_DEVICES'}
            env.update(SCD_EVAL_PY=str(fake), TEST_CALLS=str(calls), SCD_Z_CURVES=str(root / 'curves'))
            env.update({k: str(fake) if v == 'FAKE' else v for k, v in overrides.items()})
            result = subprocess.run(['bash', str(SCRIPT), stage, *args], env=env, text=True, capture_output=True)
            rows = [json.loads(row) for row in calls.read_text().splitlines()] if calls.exists() else []
            return result, rows, (root / 'curves').exists()

    def test_all_generates_and_indexes_before_scoring(self):
        result, rows, _ = self.run_stage('all', SCD_FLOW_PY='/flow/python', SCD_SD15_PY='FAKE', SCD_Z_GPUS='GPU-a,GPU-b')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r[1] for r in rows], ['prepare', 'plan', 'generate', 'index', 'score', 'summarize'])
        self.assertEqual(rows[0][rows[0].index('--family') + 1], 'zimage')
        self.assertEqual(rows[0][rows[0].index('--generation-batch-size') + 1], '1')
        self.assertEqual(rows[0][rows[0].index('--cpu-offload') + 1], 'false')
        self.assertIn('--metrics', rows[4])
        self.assertIn('pickscore', rows[4])
        self.assertEqual(rows[2][rows[2].index('--python') + 1], '/flow/python')

    def test_failed_or_interrupted_generation_never_starts_scoring(self):
        for code in ('7', '130', '143'):
            with self.subTest(code=code):
                result, rows, _ = self.run_stage('all', SCD_FLOW_PY='/flow/python', SCD_SD15_PY='FAKE',
                                                SCD_Z_GPUS='0,1', FAIL_STAGE='generate', FAIL_CODE=code)
                self.assertEqual(result.returncode, int(code), result.stderr)
                self.assertEqual([r[1] for r in rows], ['prepare', 'plan', 'generate'])

    def test_generate_without_scorer_uses_dedicated_gpus_and_forwards_filters(self):
        result, rows, _ = self.run_stage('generate', ['--methods', 'dpo', 'stagewise_k4', '--steps', '800', '1600'],
                                        SCD_FLOW_PY='/flow/python', SCD_Z_GPUS='GPU-z', SCD_EVAL_GPUS='0,1,2,3')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0][rows[0].index('--gpus') + 1], 'GPU-z')
        self.assertEqual(rows[0][-6:], ['--methods', 'dpo', 'stagewise_k4', '--steps', '800', '1600'])
        self.assertEqual(rows[1][:2], ['evaluate_suite.py', 'index'])
        result, rows, _ = self.run_stage('generate', ['--check-only'], SCD_FLOW_PY='/flow/python', SCD_Z_GPUS='0,1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 1)

    def test_preview_needs_no_gpu_and_leaves_output_absent(self):
        result, rows, created = self.run_stage('plan', ['--dry-run'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0][:2], ['evaluate_curves.py', 'plan'])
        self.assertFalse(created)

    def test_optional_hpsv3_needs_only_its_worker_and_preserves_metric(self):
        result, rows, _ = self.run_stage('score-hpsv3', SCD_HPSV3_PY='/hps/python', CUDA_VISIBLE_DEVICES='2,3,4,5')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/hps/python', rows[0])
        self.assertEqual(rows[0][-2:], ['--metrics', 'hpsv3'])
        result, rows, _ = self.run_stage('summarize-hpsv3')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0][-2:], ['--metric', 'hpsv3'])

    def test_invalid_gpu_allocation_and_all_options_fail_before_work(self):
        for tokens in ('', '-1', '0,0', '0,', '0,,2,3', '0, 1', '0,1,2'):
            with self.subTest(tokens=tokens):
                result, rows, _ = self.run_stage('generate', SCD_FLOW_PY='/flow/python', SCD_Z_GPUS=tokens)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(rows, [])
        result, rows, _ = self.run_stage('all', ['--dry-run'])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(rows, [])
        result, rows, _ = self.run_stage('generate', SCD_FLOW_PY='/flow/python', SCD_Z_GPUS='', CUDA_VISIBLE_DEVICES='0,1')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
