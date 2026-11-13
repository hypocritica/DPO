"""Real shell orchestration with CPU-only stand-ins for model entrypoints."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

class H100LauncherTests(unittest.TestCase):
    def test_order_and_separate_full_batch_smoke(self):
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
    (out/'prompts/partiprompts.jsonl').write_text(''.join(json.dumps({'prompt':str(i)})+'\\n' for i in range(100)))
''');fake.chmod(0o755)
            env=dict(os.environ,TEST_CALLS=str(calls),SCD_SD15_PY=str(fake),SCD_FLOW_PY=str(fake),SCD_EVAL_GPUS='0,1',SCD_SD15_OUT=str(root/'sd15'),SCD_CURVES_OUT=str(root/'sd35'),SCD_CURVES_PREP=str(root/'prepared'),SCD_SD15_BATCH='64',SCD_SD35_BATCH='8')
            result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(x) for x in calls.read_text().splitlines()]
            def index(file,stage): return next(i for i,r in enumerate(rows) if r[:2]==[file,stage])
            self.assertLess(index('evaluate_curves.py','score'),index('evaluate_sd15.py','generate'))
            self.assertLess(index('evaluate_sd15.py','generate'),index('evaluate_sd15.py','score'))
            selection=json.loads((root/'sd15-smoke/prompts/selection.json').read_text())
            self.assertEqual(selection['prompts'],64)
            self.assertEqual(selection['mode'],'smoke-only')
