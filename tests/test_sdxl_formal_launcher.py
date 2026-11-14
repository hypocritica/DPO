"""CPU coverage for shared SDXL selection, three planners and independent stages."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_evaluation_suite as lifecycle
from self_correct_dpo.common import read_json

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT/'scripts/run_sdxl_formal.sh'
PREPARE = ROOT/'scripts/prepare_sdxl_formal.py'


class SDXLFormalPlanTests(unittest.TestCase):
    def setUp(self):
        self.fixture = lifecycle.EvaluationSuiteTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.model['family'] = 'sdxl'
        info = read_json(f.run/'training.json')
        info['config'].update(model=f.model, method='dpo', training={'objective':'dpo','noise_views':1})
        f.write(f.run/'training.json', info)
        config = read_json(f.config)
        config['protocols'] = {'sdxl':f.sampling}
        config['benchmarks'] = {p:str(f.prompts) for p in ('partiprompts','pickv2','hpsv2')}
        f.write(f.config, config)
        self.catalog = f.root/'catalog.json'
        # Deliberately relative: frozen copy must retain original resolution.
        f.write(self.catalog, {'version':1,'models':{'sdxl':f.model},'runs_roots':{'runs':'runs'},
            'experiments':[{'id':'chosen-dpo','family':'sdxl','method':'dpo','method_label':'Chosen DPO',
                'root':'runs','run':'group/sd15-trial','steps':[1000]}]})

    def command(self, panel, *extra):
        f = self.fixture
        return [sys.executable,str(PREPARE),'--catalog',str(self.catalog),'--config',str(f.config),
            '--root',str(f.out),'--panel',panel,*extra]

    def test_three_concurrent_plans_share_exact_catalog_and_protocol(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda p: subprocess.run(self.command(p),capture_output=True,text=True),
                ('partiprompts','pickv2','hpsv2')))
        for result in results:
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        f = self.fixture
        plans = [read_json(f.out/p/'plan.json') for p in ('partiprompts','pickv2','hpsv2')]
        self.assertEqual(len({json.dumps(p['catalog'],sort_keys=True) for p in plans}),1)
        for plan in plans:
            self.assertEqual([(t['method'],t['step']) for t in plan['tasks']],[('base',0),('dpo',1000)])
            self.assertEqual(plan['tasks'][1]['method_label'],'Chosen DPO')
            self.assertEqual(plan['tasks'][0]['sampling']['generation_batch_size'],4)
            self.assertFalse(plan['tasks'][0]['sampling']['cpu_offload'])
        frozen = read_json(f.out/'shared/catalog.json')
        self.assertEqual(frozen['runs_roots']['run-1'],str(f.run.parent))
        result = subprocess.run(self.command('pickv2','--batch-size','8'),capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Inputs changed',result.stderr)

    def test_dry_run_does_not_create_output_or_select_a_fallback_step(self):
        result = subprocess.run(self.command('partiprompts','--dry-run'),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(self.fixture.out.exists())
        catalog = read_json(self.catalog)
        catalog['experiments'][0]['steps'] = [999]
        self.fixture.write(self.catalog,catalog)
        result = subprocess.run(self.command('partiprompts'),capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.fixture.out.exists())


class SDXLFormalShellTests(unittest.TestCase):
    def run_stage(self, stage, tokens):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fake=root/'python'; calls=root/'calls'
            fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["TEST_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            fake.chmod(0o755)
            env={k:v for k,v in os.environ.items() if not k.startswith('SCD_') and k!='CUDA_VISIBLE_DEVICES'}
            env.update(SCD_EVAL_PY=str(fake),SCD_SD15_PY='/existing/eval/python',SCD_SDXL_ROOT=str(root/'out'),
                TEST_CALLS=str(calls),CUDA_VISIBLE_DEVICES=tokens)
            result=subprocess.run(['bash',str(LAUNCHER),stage,'--panel','pickv2'],env=env,capture_output=True,text=True)
            rows=[json.loads(x) for x in calls.read_text().splitlines()] if calls.exists() else []
            return result,rows

    def test_generation_only_then_index_and_score_independent_of_catalog(self):
        result,rows=self.run_stage('generate','GPU-a,GPU-b,GPU-c,GPU-d')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([r[:2] for r in rows],[['evaluate_suite.py','generate'],['evaluate_suite.py','index']])
        result,rows=self.run_stage('score','0,2,4,6')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0][:2],['evaluate_suite.py','score'])
        self.assertIn('aesthetic',rows[0])

    def test_bad_allocation_rejected(self):
        for tokens in ('','0,1','0,1,2,2','0,1,2,3,4','0,1,2,'):
            result,rows=self.run_stage('generate',tokens)
            self.assertNotEqual(result.returncode,0,tokens)
            self.assertEqual(rows,[])
