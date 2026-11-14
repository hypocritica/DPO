"""CPU shell checks for independent H800 stages and real GPU token forwarding."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_h800_selected.sh'


class H800LauncherTests(unittest.TestCase):
    def run_stage(self, stage, args=(), **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'python'
            calls = root / 'calls'
            fake.write_text('#!/usr/bin/env python3\nimport os,json,sys\nwith open(os.environ["TEST_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nif os.environ.get("TEST_FAIL_MATCH") in sys.argv[1:]: sys.exit(9)\n')
            fake.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith('SCD_') and k != 'CUDA_VISIBLE_DEVICES'}
            env.update(SCD_EVAL_PY=str(fake), TEST_CALLS=str(calls), SCD_SELECTED_ROOT=str(root/'formal'), SCD_HPS_ROOT=str(root/'hps'))
            df = root/'df'
            df.write_text('#!/bin/bash\nexit 0\n')
            df.chmod(0o755)
            env['PATH'] = str(root)+os.pathsep+env['PATH']
            env.update(overrides)
            result = subprocess.run(['bash', str(SCRIPT), stage, *args], env=env, capture_output=True, text=True)
            rows = [json.loads(row) for row in calls.read_text().splitlines()] if calls.exists() else []
            return result, rows

    def test_pending_init_and_plan_need_no_gpu_or_model_environment(self):
        for stage in ('init', 'check', 'plan'):
            result, rows = self.run_stage(stage)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rows[0][:2], ['evaluate_selected.py', stage])

    def test_paper_uses_fixed_report_selection_without_gpu(self):
        result, rows = self.run_stage('paper', ['--report', 'all'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0][:2], ['evaluate_selected.py', 'export'])
        self.assertEqual(rows[0][-2:], ['--report', 'all'])

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

    def test_compact_initialization_and_core_scorer_sequence(self):
        compact_script = SCRIPT.with_name('run_compact_evaluation.sh')
        with patch(__name__ + '.SCRIPT', compact_script):
            result, rows = self.run_stage('init')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('configs/evaluation-suite-compact-v2.json', rows[0])
            result, rows = self.run_stage('score', SCD_EVAL_GPUS='0,1',
                SCD_SD15_PY='/legacy/python', SCD_HPSV3_PY='/hpsv3/python', SCD_GENEVAL_PY='/geneval/python')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0][-6:], ['--metrics', 'pickscore', 'image_reward', '--benchmarks', 'partiprompts', 'pickv2'])
            self.assertIn('hpsv3', rows[1])
            self.assertIn('geneval', rows[2])
            self.assertNotIn('unifiedreward', ' '.join(' '.join(row) for row in rows))

    def test_compact_optional_judge_uses_only_three_methods_and_parti(self):
        with patch(__name__ + '.SCRIPT', SCRIPT.with_name('run_compact_evaluation.sh')):
            result, rows = self.run_stage('score-unifiedreward', SCD_EVAL_GPUS='2,3', SCD_UNIFIEDREWARD_PY='/judge/python')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0][-6:], ['--methods', 'base', 'dpo', 'stagewise_k4', '--benchmarks', 'partiprompts'])

    def test_hpsv2_four_gpu_launcher_protocol_and_scoring(self):
        with patch(__name__ + '.SCRIPT', SCRIPT.with_name('run_hpsv2_5090.sh')):
            result, rows = self.run_stage('init')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rows[0][rows[0].index('--cpu-offload')+1], 'true')
            self.assertEqual(rows[0][rows[0].index('--generation-batch-size')+1], '1')
            result, rows = self.run_stage('plan')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rows[0][-2:], ['--benchmarks', 'hpsv2'])
            result, rows = self.run_stage('generate', SCD_EVAL_GPUS='GPU-a,GPU-b,GPU-c,GPU-d', SCD_FLOW_PY='/flow/python')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(rows[0][rows[0].index('--gpus')+1], 'GPU-a,GPU-b,GPU-c,GPU-d')
            self.assertEqual(rows[1][:2], ['evaluate_suite.py', 'index'])
            result, rows = self.run_stage('score', SCD_EVAL_GPUS='0,1,2,3',
                SCD_SD15_PY='/legacy/python', SCD_HPSV3_PY='/hpsv3/python')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][-2:], ['--benchmarks', 'hpsv2'])
            self.assertIn('hpsv3', rows[1])
            result, rows = self.run_stage('generate', SCD_EVAL_GPUS='0,1,1,3', SCD_FLOW_PY='/flow/python')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(rows)

    def test_hpsv2_all_sequence_and_generation_failure_stops_scoring(self):
        with patch(__name__ + '.SCRIPT', SCRIPT.with_name('run_hpsv2_5090.sh')):
            environment = dict(SCD_EVAL_GPUS='0,1,2,3', SCD_FLOW_PY='/flow/python',
                SCD_SD15_PY='/legacy/python', SCD_HPSV3_PY='/hpsv3/python')
            result, rows = self.run_stage('all', **environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([r[1] for r in rows if r[0] in ('evaluate_selected.py', 'evaluate_suite.py')],
                ['init', 'check', 'plan', 'generate', 'index', 'score', 'score', 'summarize', 'export'])
            result, rows = self.run_stage('all', TEST_FAIL_MATCH='generate', **environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual([r[1] for r in rows], ['init', 'check', 'plan', 'generate'])
