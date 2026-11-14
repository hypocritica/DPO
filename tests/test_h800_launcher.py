"""CPU shell checks for independent H800 stages and real GPU token forwarding."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_h800_selected.sh'


class H800LauncherTests(unittest.TestCase):
    def run_stage(self, stage, args=(), **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'python'
            calls = root / 'calls'
            fake.write_text('#!/usr/bin/env python3\nimport os,json,sys\nwith open(os.environ["TEST_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            fake.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith('SCD_') and k != 'CUDA_VISIBLE_DEVICES'}
            env.update(SCD_EVAL_PY=str(fake), TEST_CALLS=str(calls), SCD_SELECTED_ROOT=str(root/'formal'))
            env.update(overrides)
            result = subprocess.run(['bash', str(SCRIPT), stage, *args], env=env, capture_output=True, text=True)
            rows = [json.loads(row) for row in calls.read_text().splitlines()] if calls.exists() else []
            return result, rows

    def test_pending_init_and_plan_need_no_gpu_or_model_environment(self):
        for stage in ('init', 'check', 'plan'):
            result, rows = self.run_stage(stage)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rows[0][:2], ['evaluate_selected.py', stage])

    def test_generation_preserves_allocated_uuid_and_indexes_before_scoring(self):
        result, rows = self.run_stage('generate', SCD_FLOW_PY='/existing/flow/python', CUDA_VISIBLE_DEVICES='GPU-abc,GPU-def')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0][rows[0].index('--gpus')+1], 'GPU-abc,GPU-def')
        self.assertEqual(rows[0][rows[0].index('--python')+1], '/existing/flow/python')
        self.assertEqual(rows[1][:2], ['evaluate_suite.py', 'index'])
        self.assertEqual(len(rows), 2)

    def test_selected_scorer_needs_only_its_environment(self):
        for stage, key, metric in [('score-hpsv3', 'SCD_HPSV3_PY', 'hpsv3'), ('score-unifiedreward', 'SCD_UNIFIEDREWARD_PY', 'unifiedreward'), ('score-geneval', 'SCD_GENEVAL_PY', 'geneval'), ('score-legacy', 'SCD_SD15_PY', 'pickscore')]:
            result, rows = self.run_stage(stage, SCD_EVAL_GPUS='3,7', **{key:'/existing/metric/python'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rows[0][:2], ['evaluate_suite.py', 'score'])
            self.assertIn(metric, rows[0])
            self.assertEqual(len(rows), 1)

    def test_missing_duplicate_or_extra_gpu_tokens_fail_before_worker(self):
        for tokens in ('', '0', '0,0', '0,1,2', '0,', '0, 1', '-1,0'):
            result, rows = self.run_stage('generate', SCD_FLOW_PY='/existing/flow/python', SCD_EVAL_GPUS=tokens)
            self.assertNotEqual(result.returncode, 0, tokens)
            self.assertEqual(rows, [])

    def test_explicit_gpu_override_and_check_only(self):
        result, rows = self.run_stage('generate', ['--check-only'], SCD_FLOW_PY='/flow/python', SCD_EVAL_GPUS='5,6', CUDA_VISIBLE_DEVICES='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0][rows[0].index('--gpus')+1], '5,6')
        self.assertEqual(len(rows), 1)
