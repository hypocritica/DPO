"""Real shell orchestration with CPU-only stand-ins for model entrypoints."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

class H100LauncherTests(unittest.TestCase):
    def test_order_and_separate_full_batch_smoke(self):
        self.run_launcher(probe=False)

    def test_probe_skips_environment_checks_and_stops_before_scores(self):
        self.run_launcher(probe=True)

    def test_sd15_probe_runs_two_batches_without_flow_or_scoring(self):
        self.run_launcher(probe='sd15')

    def run_launcher(self, probe):
        script=Path(__file__).resolve().parents[1]/'scripts/run_h100_two_track.sh'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); calls=root/'calls'; fake=root/'python'
            fake.write_text('''#!/usr/bin/env python3
import sys,os,json
from pathlib import Path
args=sys.argv[1:]
if args[0]=='-u': args=args[1:]
if args[0]!='-': args[0]=Path(args[0]).name
with open(os.environ['TEST_CALLS'],'a') as f:f.write(json.dumps(args)+'\\n')
if args[0]=='-':
    sys.argv=args
    exec(compile(sys.stdin.read(),'<stdin>','exec'))
elif args[:2]==['evaluate_sd15.py','plan']:
    out=Path(args[args.index('--out')+1]); (out/'prompts').mkdir(parents=True,exist_ok=True)
    (out/'prompts/partiprompts.jsonl').write_text(''.join(json.dumps({'prompt':str(i)})+'\\n' for i in range(2048)))
''');fake.chmod(0o755)
            env=dict(os.environ,TEST_CALLS=str(calls),SCD_SD15_PY=str(fake),SCD_FLOW_PY=str(fake),SCD_EVAL_GPUS='0,1',SCD_SD15_OUT=str(root/'sd15'),SCD_CURVES_OUT=str(root/'sd35'),SCD_CURVES_PREP=str(root/'prepared'),SCD_SD15_BATCH='64',SCD_SD35_BATCH='8')
            if probe == 'sd15':
                env.update(SCD_SKIP_ENV_CHECKS='1', SCD_SD15_BATCH='96')
                env.pop('SCD_FLOW_PY')
            elif probe:
                env.update(SCD_SKIP_ENV_CHECKS='1', SCD_SD35_CPU_OFFLOAD='false', SCD_SD35_BATCH='32', SCD_CURVES_SOURCE_PREP=str(root/'old-prepared'))
            result=subprocess.run(['bash',str(script), 'sd15-probe' if probe == 'sd15' else 'probe' if probe else 'all'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(x) for x in calls.read_text().splitlines()]
            if probe == 'sd15':
                self.assertFalse(any(r[0] in ('check_eval_environment.py', 'prepare_eval_assets.py', 'evaluate_curves.py') for r in rows))
                self.assertFalse(any(r[:2] == ['evaluate_sd15.py','generate'] for r in rows))
                self.assertFalse(any(r[:2] == ['evaluate_sd15.py','score'] for r in rows))
                selection=json.loads((root/'sd15-probe/prompts/selection.json').read_text())
                self.assertEqual(selection['prompts'],192)
                self.assertEqual(selection['mode'],'smoke-only')
                native = next(r for r in rows if r[:2] == ['evaluate_suite.py','plan'])
                self.assertEqual(native[native.index('--generation-batch-size')+1], '96')
                return
            if probe:
                self.assertFalse(any(r[0] in ('check_eval_environment.py', 'prepare_eval_assets.py', 'evaluate_sd15.py') for r in rows))
                self.assertFalse(any('--check-only' in r for r in rows))
                prepare = next(r for r in rows if r[:2] == ['evaluate_curves.py', 'prepare'])
                self.assertIn('--from-prepared', prepare)
                self.assertEqual(prepare[prepare.index('--cpu-offload')+1], 'false')
                self.assertEqual(prepare[prepare.index('--generation-batch-size')+1], '32')
                self.assertFalse(any(r[:2] == ['evaluate_curves.py','score'] for r in rows))
                generate = next(r for r in rows if r[:2] == ['evaluate_curves.py','generate'])
                self.assertIn('base', generate)
                self.assertIn('3200', generate)
                return
            def index(file,stage): return next(i for i,r in enumerate(rows) if r[:2]==[file,stage])
            self.assertLess(index('evaluate_curves.py','score'),index('evaluate_sd15.py','generate'))
            self.assertLess(index('evaluate_sd15.py','generate'),index('evaluate_sd15.py','score'))
            selection=json.loads((root/'sd15-smoke/prompts/selection.json').read_text())
            self.assertEqual(selection['prompts'],64)
            self.assertEqual(selection['mode'],'smoke-only')
