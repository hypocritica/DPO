"""Exercise ordering, resume and failure handling without models or GPU workers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompactPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='compact pipeline ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / 'scripts'
        scripts.mkdir()
        shutil.copy(ROOT / 'scripts/run_zimage_then_sd35_compact.sh', scripts)
        self.calls = self.root / 'calls.jsonl'
        stub = '''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
phase = 'zimage' if 'zimage_curves' in sys.argv[0] else 'sd35'
stage = sys.argv[1]
with open(os.environ['TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps([phase,stage,os.environ.get('SCD_Z_BATCH'),sys.argv[2:],os.environ.get('SCD_COMPACT_CONFIG')])+'\\n')
if phase+':'+stage == os.environ.get('TEST_FAIL'):
    sys.exit(9)
if phase == 'sd35':
    root=Path(os.environ['SCD_SELECTED_ROOT'])
    out=Path(os.environ['SCD_SELECTED_OUT'])
    if stage == 'init':
        root.mkdir(parents=True,exist_ok=True)
        (root/'selection.json').write_text('{}')
    if stage == 'plan':
        out.mkdir(parents=True,exist_ok=True)
        (out/'plan.json').write_text(json.dumps({'plan_id':'abcdef1234567890'}))
    if stage == 'paper':
        report=out/'paper/core-abcdef123456'
        report.mkdir(parents=True,exist_ok=True)
        (report/'export.json').write_text('{}')
'''
        for name in ('run_zimage_curves.sh', 'run_compact_evaluation.sh'):
            (scripts / name).write_text(stub)
        # The shell invokes these through bash; route the stub via the current Python.
        for name in ('run_zimage_curves.sh', 'run_compact_evaluation.sh'):
            python_stub = scripts / (name + '.py')
            (scripts / name).rename(python_stub)
            (scripts / name).write_text('#!/bin/bash\nexec ' + self.quote(sys.executable) + ' ' + self.quote(str(python_stub)) + ' "$@"\n')
        (scripts / 'report_generation_sessions.py').write_text('pass\n')
        commands = self.root / 'bin'
        commands.mkdir()
        for name in ('git', 'df'):
            path = commands / name
            path.write_text('#!/bin/bash\nexit 0\n')
            path.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('SCD_') and k != 'CUDA_VISIBLE_DEVICES'}
        self.env.update(PATH=str(commands)+os.pathsep+os.environ['PATH'], TEST_CALLS=str(self.calls),
            SCD_EVAL_GPUS='GPU-a,GPU-b', SCD_Z_CURVES=str(self.root/'existing curves'),
            SCD_SD35_COMPACT_ROOT=str(self.root/'formal'))
        for key in ('SCD_EVAL_PY', 'SCD_FLOW_PY', 'SCD_SD15_PY', 'SCD_HPSV3_PY', 'SCD_GENEVAL_PY'):
            self.env[key] = sys.executable

    @staticmethod
    def quote(value):
        import shlex
        return shlex.quote(value)

    def run_pipeline(self):
        return subprocess.run(['bash', str(self.root/'scripts/run_zimage_then_sd35_compact.sh')],
                              env=self.env, capture_output=True, text=True)

    def test_curve_first_then_formal_and_repeated_command_resumes(self):
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(row) for row in self.calls.read_text().splitlines()]
        self.assertEqual([(r[0], r[1]) for r in calls],
            [('zimage', s) for s in ('prepare', 'plan', 'generate', 'score', 'summarize')] +
            [('sd35', s) for s in ('init', 'check', 'plan', 'generate', 'score', 'summarize', 'paper')])
        self.assertTrue(all(r[2] == '4' for r in calls))
        self.assertIn('--generation-batch-size', calls[5][3])
        self.assertIn('8', calls[5][3])
        self.assertEqual(calls[5][4], 'configs/evaluation-suite-compact-h100-v3.json')
        self.calls.unlink()
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('init', [json.loads(row)[1] for row in self.calls.read_text().splitlines()])

    def test_failed_curve_never_starts_sd35(self):
        self.env['TEST_FAIL'] = 'zimage:score'
        result = self.run_pipeline()
        self.assertNotEqual(result.returncode, 0)
        calls = [json.loads(row) for row in self.calls.read_text().splitlines()]
        self.assertEqual([(r[0], r[1]) for r in calls], [('zimage', s) for s in ('prepare', 'plan', 'generate', 'score')])

    def test_missing_formal_environment_does_not_block_zimage_curve(self):
        self.env['SCD_GENEVAL_PY'] = str(self.root/'missing-python')
        result = self.run_pipeline()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Z-Image curve completed', result.stderr)
        calls = [json.loads(row) for row in self.calls.read_text().splitlines()]
        self.assertEqual([(r[0], r[1]) for r in calls],
                         [('zimage', s) for s in ('prepare', 'plan', 'generate', 'score', 'summarize')])
