"""CPU integration checks: paired requests, source pixel integrity and quiet execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image

from test_multiseed_generation import fixture,RUNTIME
from self_correct_dpo.common import read_json,read_jsonl,sha256,digest,load_suite
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.evaluation.appendix_generation import payload_jsonl
from self_correct_dpo.evaluation.appendix_figures import requests,source_index,worker,results,export,PROMPTS
from self_correct_dpo.evaluation.appendix_progress import run_stage,committed_count
from self_correct_dpo.evaluation.experiment_catalog import validate_method
from self_correct_dpo.evaluation.render import check_prompt_lengths

ROOT=Path(__file__).resolve().parents[1]

class FigureTests(unittest.TestCase):
    def test_requested_scope_and_nonempty_prompts(self):
        spec=read_json(ROOT/'configs/appendix-figures.json')
        self.assertNotIn('kto',spec['methods']['sdxl'])
        self.assertEqual({f:sum(len(j['prompts']) for j in requests(spec,f)[0]) for f in spec['methods']},dict(sd15=1104,sdxl=992,sd35_medium=1284,zimage=360))
        art=load_suite(PROMPTS/'art-100.jsonl');self.assertEqual(len({p['prompt'] for p in art}),100)
        self.assertLess(max(len(p['prompt'].split()) for p in art),35)
        jobs,_=requests(spec,'sd15');self.assertEqual(len(jobs),len({j['name'] for j in jobs}))
        self.assertEqual(len(next(j for j in jobs if j['name']=='dpo-10000-s0')['prompts']),112)

    def test_lossavg_semantics_and_long_prompt_truncation(self):
        for k in (2,4):
            c=dict(method='multiview-B-gaussian',training=dict(objective='dpo',formal_arm='B',noise_views=k,stagewise=None))
            validate_method(c,f'lossavg_k{k}','fixture');c['method']='multiview-C-gaussian'
            with self.assertRaises(ValueError):validate_method(c,f'lossavg_k{k}','fixture')
        class Tokenizer:
            model_max_length=77
            def __call__(self,prompt,**kwargs):return SimpleNamespace(input_ids=list(range(100 if prompt else 2)))
        pipe=SimpleNamespace(tokenizer=Tokenizer());suite=[dict(scene_id='long',variant_id='canonical',prompt='long prompt')]
        report=check_prompt_lengths(pipe,suite,dict(negative_prompt='',prompt_truncation='truncate'),'sd15')
        self.assertEqual(report[0]['tokens'],100)
        with self.assertRaises(ValueError):check_prompt_lengths(pipe,suite,dict(negative_prompt='',prompt_truncation='error'),'sd15')

    def test_progress_preserves_logs_and_propagates_real_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);journal=root/'rows.jsonl'
            script='import pathlib,sys;print("token truncation warning");pathlib.Path(sys.argv[1]).write_text("{}\\n{}\\n")'
            run_stage([sys.executable,'-c',script,str(journal)],log=root/'worker.log',journal=journal,total=2,label='generate')
            self.assertEqual(committed_count(journal),2);self.assertIn('token truncation',(root/'worker.log').read_text())
            progress=read_json(root/'worker.progress.json');self.assertEqual(progress['remaining'],0)
            with self.assertRaisesRegex(RuntimeError,'failed'):
                run_stage([sys.executable,'-c','raise RuntimeError("real failure")'],log=root/'bad.log',journal=journal,total=2,label='score')

    def test_pixels_reuse_missing_generation_resume_and_shallow_export(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);old=output/'old';run,oldrows=fixture(old,4)
            run['truncated_prompts']=[dict(prompt='p2 / canonical',tokens=100),dict(prompt='unrelated / canonical',tokens=90)]
            run['run_id']=digest({k:v for k,v in run.items() if k!='run_id'});atomic_json(old/'render/run.json',run)
            for row in oldrows:row['image_id']=digest({'run_id':run['run_id'],**{k:v for k,v in row.items() if k!='image_id'}})
            payload_jsonl(old/'render/manifest.jsonl',oldrows)
            atomic_json(old/'render/complete.json',dict(images=4,manifest_sha256=sha256(old/'render/manifest.jsonl')))
            job=dict(name='dpo-10000-s0',method='dpo',step=10000,seed=0,prompts=run['suite'],shards=2,blocked=None,
                     adapter=dict(run='/fixture/run',run_id='train',path='/fixture/adapter',checkpoint_id='checkpoint',inventory={}))
            plan=dict(plan_id='figure-test',family='sd15',jobs=[job],history=[str(old/'render/run.json')],
                      display=dict(parti=run['suite'][:1],art=run['suite'],complex=[]),
                      spec=dict(methods={'sd15':{'dpo':{'main_step':10000}}},art_seeds={'sd15':0},progress_methods={}))
            core=dict(model=run['model'],base_identity={},sampling=run['sampling'],render_versions=run['versions'],noise_runtime=RUNTIME,interpreters={'render':sys.executable})
            out=output/'work-figures/sd15';out.mkdir(parents=True)
            (old/'render'/oldrows[0]['image']).unlink();source_index(output,plan,core)
            def render(cmd,**kwargs):
                target=Path(cmd[cmd.index('--out')+1]);target.mkdir();config=read_json(cmd[cmd.index('--config')+1]);prompts=read_jsonl(cmd[cmd.index('--prompts')+1])
                self.assertEqual(len(prompts),1);self.assertEqual(config['sampling']['prompt_truncation'],'truncate')
                new=copy.deepcopy(run);new.update(config=config,sampling=config['sampling'],suite=prompts,suite_id=digest(prompts),full_suite_id=digest(prompts),expected_images=1)
                new['run_id']=digest({k:v for k,v in new.items() if k!='run_id'});atomic_json(target/'run.json',new)
                Image.new('RGB',(64,64),'orange').save(target/'new.png')
                row={k:prompts[0][k] for k in ('scene_id','variant_id','prompt','tag')};row.update(seed=0,image='new.png',image_sha256=sha256(target/'new.png'))
                row['image_id']=digest({'run_id':new['run_id'],**row});payload_jsonl(target/'manifest.jsonl',[row]);atomic_json(target/'complete.json',dict(images=1,manifest_sha256=sha256(target/'manifest.jsonl')))
            prefix='self_correct_dpo.evaluation.appendix_figures.'
            with patch(prefix+'figure_plan',return_value=(plan,core)),patch(prefix+'resolve_adapter',return_value=job['adapter']),patch(prefix+'run_stage',side_effect=render) as called:
                worker(output,'sd15',job['name'],0);worker(output,'sd15',job['name'],0)
                self.assertEqual(called.call_count,1)
                rows=results(output,plan,job,0);self.assertEqual({r['origin'] for r in rows},{'reused','generated'})
                self.assertEqual(next(r for r in rows if r['origin']=='reused')['truncated_prompts'],[dict(prompt='p2 / canonical',tokens=100)])
                quantitative=output/'sd15/report';quantitative.mkdir(parents=True)
                Image.new('RGB',(32,32),'white').save(quantitative/'curves.png')
                report=export(output,'sd15');self.assertEqual(report['exported_images'],2);self.assertEqual(report['missing_shards'],1)
                self.assertTrue((output/'figures/index.html').exists());self.assertTrue((output/'figures/sd15-art-s0-01.png').exists())
                self.assertIn('sd15-quant-curves.png',(output/'figures/index.html').read_text())
                for row in read_jsonl(output/'figures/sd15-image-data.jsonl'):self.assertEqual(sha256(output/'figures'/row['export_image']),row['image_sha256'])

if __name__=='__main__':unittest.main()
