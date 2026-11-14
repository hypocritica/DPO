"""Review exact SDXL selection/counts and restart/failure boundaries on CPU."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.config import load_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_sdxl_formal


class SDXLPaperPlanTest(unittest.TestCase):
    def test_real_plans_exact_counts_seeds_and_supplement_excludes_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalogs = {role: read_json(ROOT / f'configs/evaluation-sdxl-{role}.json')
                        for role in ('main', 'parti-supplement')}
            self.assertEqual(len(catalogs['main']['experiments']), 9)
            self.assertTrue(all(e['steps'] == [2000] for e in catalogs['main']['experiments']))
            extra = catalogs['parti-supplement']['experiments']
            self.assertEqual({e['method'] for e in extra}, {'dmpo', 'stagewise_dmpo_k6', 'dspo', 'stagewise_dspo_k6'})
            self.assertTrue(all(e['steps'] == [3000] for e in extra))
            model = {'family': 'sdxl', 'path': str(root / 'model')}
            for entry in catalogs['main']['experiments']:
                method = entry['method']
                objective = next((x for x in ('dmpo', 'dspo', 'mapo', 'sft') if x in method), 'dpo')
                stagewise = method.startswith('stagewise_')
                views = int(method[-1]) if '_k' in method else 1
                train = {'objective': objective, 'noise_views': views, 'formal_arm': 'C'}
                if stagewise:
                    train['stagewise'] = {'objective': 'dpo', 'boundary': 500}
                config = {'model': model, 'method': 'stagewise-C-gaussian' if stagewise else
                          ('multiview-C-gaussian' if views > 1 else 'dpo'), 'training': train}
                run = root / 'runs' / entry['run']
                run.mkdir(parents=True)
                (run / 'training.json').write_text(json.dumps({'run_id': method, 'config': config}))
                for step in (2000, 3000):
                    folder = run / f'checkpoints/step-{step:07d}'
                    (folder / 'adapter').mkdir(parents=True)
                    weight = folder / 'adapter/adapter_model.bin'
                    weight.write_bytes(f'{method}-{step}'.encode())
                    checkpoint = {'run_id': method, 'step': step, 'files': {'adapter/adapter_model.bin': sha256(weight)}}
                    checkpoint['checkpoint_id'] = digest(checkpoint)
                    (folder / 'checkpoint.json').write_text(json.dumps(checkpoint))
                for catalog in catalogs.values():
                    for target in catalog['experiments']:
                        if target['method'] == method:
                            target['expected_run_id'] = method
                            step = target['steps'][0]
                            target['expected_checkpoint_ids'] = {str(step): read_json(run / f'checkpoints/step-{step:07d}/checkpoint.json')['checkpoint_id']}
            for role, catalog in catalogs.items():
                catalog['models'] = {'sdxl': model}
                catalog['runs_roots'] = {'sdxl': str(root / 'runs')}
                (root / f'{role}.json').write_text(json.dumps(catalog))
            panels = {'partiprompts': 1632, 'pickv2': 487, 'hpsv2': 3200, 'geneval': 553}
            config = load_config(ROOT / 'configs/evaluation-suite-sdxl-paper-v1.json', benchmarks=list(panels))
            for panel, count in panels.items():
                path = root / f'{panel}.jsonl'
                rows = [{'scene_id': f'{panel}-{i}', 'variant_id': 'canonical', 'prompt': f'a cat {i}',
                         'tag': panel, 'metadata': {'prompt': f'a cat {i}', 'tag': panel},
                         'suite_mode': 'canonical-only-v1', 'template_family': 'canonical', 'transform_type': 'canonical'}
                        for i in range(count)]
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                config['benchmarks'][panel] = str(path)
            (root / 'suite.json').write_text(json.dumps(config))
            totals = {}
            for panel in panels:
                argv = ['--catalog', str(root / 'main.json'), '--config', str(root / 'suite.json'),
                        '--root', str(root / 'out/main'), '--panel', panel, '--panels', *panels]
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(prepare_sdxl_formal.main(argv), 0)
                    self.assertEqual(prepare_sdxl_formal.main(argv), 0)
                plan = read_json(root / 'out/main' / panel / 'plan.json')
                self.assertEqual(len(plan['tasks']), 10)
                totals[panel] = sum(t['images'] for t in plan['tasks'])
                for t in plan['tasks']:
                    self.assertEqual(t['step'], 0 if t['method'] == 'base' else 2000)
                    self.assertEqual(t['sampling']['seeds'], [17, 29, 43, 71] if panel == 'geneval' else [17])
                    self.assertEqual(t['sampling']['generation_batch_size'], 4)
                    self.assertFalse(t['sampling']['cpu_offload'])
            self.assertEqual(totals, {'partiprompts': 16320, 'pickv2': 4870, 'hpsv2': 32000, 'geneval': 22120})
            with redirect_stdout(io.StringIO()):
                self.assertEqual(prepare_sdxl_formal.main(['--catalog', str(root / 'parti-supplement.json'),
                    '--config', str(root / 'suite.json'), '--root', str(root / 'out/supplement'),
                    '--panel', 'partiprompts', '--panels', 'partiprompts', '--methods', *[e['method'] for e in extra]]), 0)
            plan = read_json(root / 'out/supplement/partiprompts/plan.json')
            self.assertEqual(len(plan['tasks']), 4)
            self.assertTrue(all(t['step'] == 3000 and t['method'] != 'base' for t in plan['tasks']))
            self.assertEqual(sum(t['images'] for t in plan['tasks']), 6528)


class SDXLPaperShellTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='sdxl paper ')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'scripts').mkdir()
        shutil.copy(ROOT / 'scripts/run_sdxl_paper.sh', self.root / 'scripts')
        self.calls = self.root / 'calls'
        fake = self.root / 'python'
        fake.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['TEST_CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if len(sys.argv)>2 and sys.argv[1]=='evaluate_suite.py' and sys.argv[2]==os.environ.get('TEST_FAIL'):
    sys.exit(9)
''')
        fake.chmod(0o755)
        (self.root / 'bin').mkdir()
        for name in ('git', 'df'):
            path = self.root / 'bin' / name
            path.write_text('#!/bin/bash\nexit 0\n')
            path.chmod(0o755)
        self.env = {k:v for k,v in os.environ.items() if not k.startswith('SCD_') and k != 'CUDA_VISIBLE_DEVICES'}
        self.env.update(SCD_EVAL_PY=str(fake), SCD_SD15_PY=sys.executable, SCD_GENEVAL_PY=str(fake),
                        SCD_SDXL_PAPER_ROOT=str(self.root / 'out'), TEST_CALLS=str(self.calls),
                        CUDA_VISIBLE_DEVICES='GPU-a,GPU-b,GPU-c,GPU-d', PATH=str(self.root/'bin')+os.pathsep+os.environ['PATH'])

    def run_group(self, group, stage='all'):
        return subprocess.run(['bash',str(self.root/'scripts/run_sdxl_paper.sh'),group,stage],
                              env=self.env,capture_output=True,text=True)

    def rows(self):
        return [json.loads(row) for row in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_group_routes_and_repeatable_stages(self):
        expected = {'A': ['main/hpsv2'], 'B': ['main/partiprompts','main/pickv2','supplement/partiprompts'], 'C':['main/geneval']}
        for group, outputs in expected.items():
            self.calls.unlink(missing_ok=True)
            result = self.run_group(group)
            self.assertEqual(result.returncode,0,result.stderr)
            rows = self.rows()
            gens = [r for r in rows if r[:2] == ['evaluate_suite.py','generate']]
            self.assertEqual([Path(r[r.index('--out')+1]).relative_to(self.root/'out').as_posix() for r in gens],outputs)
            scores = [r for r in rows if r[:2] == ['evaluate_suite.py','score']]
            for row in [*gens,*scores]:
                self.assertEqual(row[row.index('--gpus')+1],'GPU-a,GPU-b,GPU-c,GPU-d')
            for row in scores:
                self.assertEqual(row[row.index('--python')+1],self.env['SCD_GENEVAL_PY'] if group=='C' else sys.executable)
            self.assertEqual(self.run_group(group).returncode,0)

    def test_failure_stops_scoring_and_released_lock_allows_restart(self):
        self.env['TEST_FAIL']='generate'
        result=self.run_group('B')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any(r[:2]==['evaluate_suite.py','score'] for r in self.rows()))
        del self.env['TEST_FAIL']
        self.assertEqual(self.run_group('B').returncode,0)

    def test_wrong_gpu_count_and_missing_geneval_python_stop_before_generation(self):
        self.env['CUDA_VISIBLE_DEVICES']='0,1,2,2'
        self.assertNotEqual(self.run_group('A').returncode,0)
        self.assertEqual(self.rows(),[])
        self.env['CUDA_VISIBLE_DEVICES']='0,1,2,3'
        self.env['SCD_GENEVAL_PY']='/missing/python'
        self.assertNotEqual(self.run_group('C').returncode,0)
        self.assertEqual(self.rows(),[])
