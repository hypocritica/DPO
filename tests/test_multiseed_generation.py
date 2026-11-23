"""CPU checks for the complete appendix design, reuse integrity and report semantics."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from self_correct_dpo.common import digest,sha256,verify_images,read_json,read_jsonl
from self_correct_dpo.evaluation import metrics as metrics_module
from self_correct_dpo.evaluation.appendix_generation import payload_jsonl
from self_correct_dpo.evaluation.generation_inventory import export_inventory
from self_correct_dpo.evaluation.multiseed_generation import (configuration,static_plan,jobs_for,worker,completed,
    result_path,selected_slots,prepare,load_plan,sources)
from self_correct_dpo.evaluation.multiseed_history import read_source,choose_slots,import_pixels,protocol
from self_correct_dpo.evaluation.multiseed_report import compare,interval,family_report
from self_correct_dpo.runtime.checkpoints import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SCORER={'metric':'pickscore','scale':'test-only','spec':{}}
RUNTIME={'gpu':'fixture','torch':'fixture','diffusers':'fixture'}


def fixture(root, count=3):
    render=root/'render';render.mkdir(parents=True)
    suite=[dict(scene_id=f'p{i}',variant_id='canonical',prompt=f'A test object {i}',tag='test',metadata={'tag':'test','prompt':f'A test object {i}'},suite_mode='canonical-only-v1',transform_type='canonical',template_family='canonical') for i in range(count)]
    model=dict(family='sd15',path='/fixture/model',revision='fixture')
    sample=dict(scheduler='EulerDiscreteScheduler',seeds=[0],generation_batch_size=2,height=8,width=8,
                dtype='bfloat16',negative_prompt='',num_inference_steps=2,guidance_scale=1.)
    config=dict(model=model,sampling=sample,aligned=dict(kind='peft',path='/fixture/adapter',revision='checkpoint'))
    run=dict(arm='aligned',config=config,model=model,suite=suite,suite_id=digest(suite),full_suite_id=digest(suite),sampling=sample,
             expected_images=count,shard=dict(index=0,count=1,assignment='sorted-scene-round-robin-v1'),suite_indices=list(range(count)),
             base_identity={},model_identity={},render_sha256=sha256(ROOT/'src/self_correct_dpo/evaluation/render.py'),
             gpu='fixture',versions={'torch':'fixture','diffusers':'fixture'},scheduler_config={'_class_name':'EulerDiscreteScheduler'})
    run['run_id']=digest(run);atomic_json(render/'run.json',run);rows=[]
    for i,p in enumerate(suite):
        image=render/f'images/{i}.png';image.parent.mkdir(exist_ok=True);Image.new('RGB',(8,8),(i*50,0,0)).save(image)
        row={k:p[k] for k in ('scene_id','variant_id','prompt','tag')};row.update(seed=0,image=f'images/{i}.png',image_sha256=sha256(image))
        row['image_id']=digest({'run_id':run['run_id'],**row});rows.append(row)
    payload_jsonl(render/'manifest.jsonl',rows);atomic_json(render/'complete.json',dict(images=count,manifest_sha256=sha256(render/'manifest.jsonl')))
    return run,rows


def score_fixture(root):
    metrics_module.score_panel(root/'render',root/'scores/pickscore','pickscore',{},scorer_factory=lambda *a:lambda imgs,texts:[float(i.getpixel((0,0))[0])/255 for i in imgs],asset_id=SCORER)


class MultiseedTests(unittest.TestCase):
    def test_reviewed_pre_zimage_renderer_reuses_sd15_scores(self):
        from self_correct_dpo.runtime.zimage_precision_compat import LEGACY_RENDER_SHA256
        from self_correct_dpo.evaluation.appendix_generation import noise_runtime
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,_=fixture(root)
            run['render_sha256']=LEGACY_RENDER_SHA256
            run['run_id']=digest({k:v for k,v in run.items() if k!='run_id'})
            atomic_json(root/'render/run.json',run)
            rows=read_jsonl(root/'render/manifest.jsonl')
            for row in rows:
                row['image_id']=digest({'run_id':run['run_id'],**{k:v for k,v in row.items() if k!='image_id'}})
            payload_jsonl(root/'render/manifest.jsonl',rows)
            atomic_json(root/'render/complete.json',{'images':len(rows),'manifest_sha256':sha256(root/'render/manifest.jsonl')})
            score_fixture(root)
            source=read_source(root/'render',{'pickscore':SCORER})
            self.assertEqual(choose_slots(run['suite'],0,['pickscore'],{'old':source})[1]['reuse_scores'],3)
            for family in ('sdxl','sd35_medium'):
                self.assertEqual(noise_runtime({**run,'model':{'family':family}}),RUNTIME)
            with self.assertRaises(ValueError):noise_runtime({**run,'model':{'family':'zimage'}})
            with self.assertRaises(ValueError):noise_runtime({**run,'render_sha256':'unreviewed'})

    def test_full_scope_and_deduplication(self):
        cfg=configuration(ROOT/'configs/appendix-multiseed.json');plan=static_plan(cfg)
        self.assertEqual((plan['main_configurations'],plan['main_images'],plan['unique_images'],len(plan['jobs'])),(18,88128,114240,105))
        self.assertEqual(plan['packs']['E4'],10752)
        jobs=jobs_for(cfg,'sd15',[0,23,42]);self.assertEqual(len({j['name'] for j in jobs}),len(jobs))
        main=[j for j in jobs if j['method']=='stagewise_dspo_k6' and j['panel']=='full'];self.assertEqual({j['step'] for j in main},{6000})
        self.assertTrue(any(j['step']==10000 and j['panel']=='curve' and j['method']=='stagewise_dspo_k6' for j in jobs))
        self.assertFalse(any('lossavg' in j['method'] for j in plan['jobs']))

    def test_sampling_mismatch_and_packing(self):
        a={'model':{'family':'sd15','path':'/fixture','revision':'test'},'sampling':{'seeds':[0],'scheduler':'Euler','generation_batch_size':2}}
        b=copy.deepcopy(a);b['sampling'].update(seeds=[23],generation_batch_size=4,cpu_offload=True)
        self.assertEqual(protocol(a),protocol(b));b['sampling']['scheduler']='PNDM';self.assertNotEqual(protocol(a),protocol(b))

    def test_native_evaluation_scores_reuse_without_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root);score_fixture(root)
            (root/'render'/rows[0]['image']).unlink()
            source=read_source(root/'render',{'pickscore':SCORER})
            slots,counts=choose_slots(run['suite'],0,['pickscore'],{'old':source})
            self.assertEqual(counts,dict(reuse_scores=3,reuse_images=0,generate=0))
            _,counts=choose_slots(run['suite'],0,['pickscore','image_reward'],{'old':source})
            self.assertEqual(counts,dict(reuse_scores=0,reuse_images=2,generate=1))
            self.assertEqual(choose_slots(run['suite'],23,['pickscore'],{'old':source})[1]['generate'],3)

    def test_partial_score_journal_and_bad_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root);score_fixture(root)
            target=root/'scores/pickscore';scores=read_jsonl(target/'scores.jsonl');payload_jsonl(target/'scores.jsonl',scores[:1]);(target/'complete.json').unlink()
            source=read_source(root/'render',{'pickscore':SCORER});_,counts=choose_slots(run['suite'],0,['pickscore'],{'old':source})
            self.assertEqual(counts,dict(reuse_scores=1,reuse_images=2,generate=0))
            rejected=read_source(root/'render',{'pickscore':{**SCORER,'scale':'different'}})
            self.assertFalse(rejected['scores']);self.assertTrue(rejected['rejected'])

    def test_pixel_subset_import_is_native_and_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root/'old');source=read_source(root/'old/render',{})
            before=sha256(root/'old/render/manifest.jsonl');keys=[('p1','canonical',0),('p2','canonical',0)]
            imported,_=import_pixels(source,keys,root/'new/render');actual,records=verify_images(root/'new/render')
            self.assertEqual(actual,imported);self.assertEqual(len(records),2)
            self.assertEqual(actual['imported_from']['run_id'],run['run_id'])
            self.assertEqual(sha256(root/'old/render/manifest.jsonl'),before)
            import_pixels(source,keys,root/'new/render')
            Image.new('RGB',(8,8),(255,0,0)).save(root/'old/render'/rows[1]['image'])
            with self.assertRaisesRegex(ValueError,'Historical pixels changed'):import_pixels(source,keys,root/'new/render')

    def test_corrupt_manifest_and_duplicate_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root);score_fixture(root)
            scorepath=root/'scores/pickscore/scores.jsonl';scores=read_jsonl(scorepath);payload_jsonl(scorepath,scores+[scores[0]])
            self.assertFalse(read_source(root/'render',{'pickscore':SCORER})['scores'])
            rows[0]['prompt']='tampered';payload_jsonl(root/'render/manifest.jsonl',rows)
            with self.assertRaises(ValueError):read_source(root/'render',{})

    def test_worker_reuses_partial_scores_and_only_scores_missing_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';run,rows=fixture(old);score_fixture(old)
            scorepath=old/'scores/pickscore/scores.jsonl';score_rows=read_jsonl(scorepath);payload_jsonl(scorepath,score_rows[:1]);(old/'scores/pickscore/complete.json').unlink()
            source=read_source(old/'render',{'pickscore':SCORER});slots,_=choose_slots(run['suite'],0,['pickscore'],{'old':source})
            out=root/'new';out.mkdir();payload_jsonl(out/'slots.jsonl',slots)
            plan=dict(plan_id='fixture-plan',sources={'old':dict(render=str(old/'render'),files=source['files'])},
                scorers={'pickscore':SCORER},score_versions={'pickscore':read_json(old/'scores/pickscore/identity.json')['identity']['versions']},interpreters={'legacy':sys.executable})
            job=dict(name='dpo-10000-s0',method='dpo',step=10000,seed=0,panel='full',metrics=['pickscore'],shards=1,slots='slots.jsonl')
            def execute(command,**kwargs):
                self.assertIn('self_correct_dpo.evaluation.metrics',command)
                render=Path(command[command.index('--run')+1]);target=Path(command[command.index('--out')+1])
                self.assertEqual(read_json(render/'run.json')['expected_images'],2)
                metrics_module.score_panel(render,target,'pickscore',{},scorer_factory=lambda *a:lambda imgs,texts:[.5]*len(imgs),asset_id=SCORER)
            with patch('self_correct_dpo.evaluation.multiseed_generation.subprocess.run',side_effect=execute) as invoked:
                worker(out,plan,{'full':run['suite']},job,0);self.assertEqual(invoked.call_count,1)
                worker(out,plan,{'full':run['suite']},job,0);self.assertEqual(invoked.call_count,1)
            self.assertTrue(completed(out,plan,job,0));self.assertEqual(len(read_jsonl(result_path(out,job,0)/'scores.jsonl')),3)
            self.assertEqual(len(read_jsonl(scorepath)),1)

    def test_worker_new_generation_uses_native_renderer_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sample_root=root/'sample';run,rows=fixture(sample_root)
            out=root/'out';out.mkdir();slots=[dict(key=[p['scene_id'],'canonical',0],source=None,metrics=[],state='generate') for p in run['suite']]
            payload_jsonl(out/'slots.jsonl',slots)
            adapter=dict(run='/fixture/run',path='/fixture/adapter',run_id='train',checkpoint_id='checkpoint',inventory={})
            plan=dict(plan_id='new-fixture',sources={},model=run['model'],sampling=run['sampling'],base_identity={},
                noise_runtime=RUNTIME,render_versions=run['versions'],scorers={'pickscore':SCORER},score_versions={},interpreters={'render':sys.executable,'legacy':sys.executable})
            score_fixture(sample_root);plan['score_versions']['pickscore']=read_json(sample_root/'scores/pickscore/identity.json')['identity']['versions']
            job=dict(name='dpo-10000-s0',method='dpo',step=10000,seed=0,panel='full',metrics=['pickscore'],shards=1,slots='slots.jsonl',adapter=adapter)
            def execute(command,**kwargs):
                target=Path(command[command.index('--out')+1])
                if 'self_correct_dpo.evaluation.render' in command:
                    fixture(target.parent)
                    self.assertEqual(read_json(target/'run.json')['config'],read_json(command[command.index('--config')+1]))
                else:
                    render=Path(command[command.index('--run')+1])
                    metrics_module.score_panel(render,target,'pickscore',{},scorer_factory=lambda *a:lambda imgs,texts:[.5]*len(imgs),asset_id=SCORER)
            with patch('self_correct_dpo.evaluation.multiseed_generation.resolve_adapter',return_value=adapter),patch('self_correct_dpo.evaluation.multiseed_generation.subprocess.run',side_effect=execute) as invoked:
                worker(out,plan,{'full':run['suite']},job,0);self.assertEqual(invoked.call_count,2)
                worker(out,plan,{'full':run['suite']},job,0);self.assertEqual(invoked.call_count,2)
            self.assertTrue(completed(out,plan,job,0))


    def test_inventory_exports_metadata_only_and_finds_batch_render(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root/'evaluations/old-batch/step-0010000/partiprompts');score_fixture(root/'evaluations/old-batch/step-0010000/partiprompts')
            (root/'evaluations/weights.safetensors').write_bytes(b'not a model')
            info=export_inventory([root/'evaluations',root/'missing'],root/'export',check_images=True)
            self.assertEqual(info['renders'],1);self.assertEqual(info['issues'],1)
            index=read_json(root/'export/inventory.json');self.assertEqual(index['renders'][0]['present_images'],3)
            with tarfile.open(info['archive']) as archive:
                names=archive.getnames();self.assertIn('inventory.json',names)
                self.assertFalse(any(n.endswith(('.png','.safetensors')) for n in names))
            with self.assertRaises(ValueError):export_inventory([root/'evaluations'],root/'export')

    def test_prompt_cluster_statistics(self):
        # Three identical seeds do not create three times as many independent prompts.
        a=np.array([[0.],[1.],[3.],[7.]]);b=np.zeros_like(a)
        one=compare(a,b,400);three=compare(np.repeat(a,3,axis=1),np.repeat(b,3,axis=1),400)
        for key in ('mean_delta','low','high','scorer_win_rate'):self.assertEqual(one[key],three[key])
        self.assertEqual(three['scorer_win_rate'],.875);self.assertEqual(three['seed_delta_sd'],0.)

    def test_prepare_freezes_partial_reuse_and_refuses_protocol_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'history';run,rows=fixture(old);score_fixture(old)
            cfg=configuration(ROOT/'configs/appendix-multiseed.json')
            cfg['full_count']=3;cfg['curve_count']=2
            cfg['experiments']=[e for e in cfg['experiments'] if e['family']=='sd15' and e['method']=='dpo']
            exp=cfg['experiments'][0];exp['checkpoints']['10000']='checkpoint'
            cfg['families']['sd15']['metrics']=['pickscore']
            train=root/'runs'/exp['run'];train.mkdir(parents=True)
            atomic_json(train/'training.json',dict(run_id=exp['run_id'],pairs_id='test-pairs',base_identity={},
                config=dict(model=run['model'],method='dpo',training=dict(objective='dpo',noise_views=1,stagewise=None))))
            prompts=root/'partiprompts.jsonl';payload_jsonl(prompts,run['suite'])
            args=SimpleNamespace(family='sd15',output=root/'out',protocol_source=old/'render/run.json',batch_size=None,
                fresh=False,history_root=[old],prompts=prompts,resources=None)
            registry=dict(SCD_SD15_PY=sys.executable,SCD_HPSV3_PY=sys.executable,RUNS_ROOT=str(root/'runs'))
            suitecfg=dict(protocols={'sd15':run['sampling']},benchmarks={'partiprompts':str(prompts)},scorers_config='fixture')
            def config_load(path,**kwargs):return {'scorers':{'pickscore':{}}} if kwargs.get('scorers') else suitecfg
            def adapter(path,step,runid,cpid):return dict(path='/fixture/adapter',run=str(path),run_id=runid,step=step,checkpoint_id=cpid,inventory={})
            score_versions=read_json(old/'scores/pickscore/identity.json')['identity']['versions']
            prefix='self_correct_dpo.evaluation.multiseed_generation.'
            with patch(prefix+'load_config',side_effect=config_load),patch(prefix+'scorer_identity',return_value=SCORER),patch(prefix+'versions',return_value=score_versions),patch(prefix+'package_versions',return_value=run['versions']),patch(prefix+'model_artifact_identity',return_value={}),patch(prefix+'resolve_adapter',side_effect=adapter):
                summary=prepare(args,cfg,registry)
                self.assertEqual(summary['counts']['reuse_scores'],3)
                self.assertEqual(summary['counts']['generate'],16)  # two new seeds plus five two-prompt curve points
                plan,panels=load_plan(args.output/'sd15');self.assertEqual(len(panels['curve']),2)
                self.assertTrue((args.output/'sd15/reuse-summary.csv').is_file())
                self.assertEqual(prepare(args,cfg,registry)['status'],'already prepared')
                args.batch_size=4
                with self.assertRaisesRegex(ValueError,'Frozen preparation differs'):prepare(args,cfg,registry)


    def test_win_counts_ties_and_complement(self):
        a=np.array([[3.,1.,2.],[0.,5.,4.]])
        b=np.array([[1.,1.,3.],[1.,2.,4.]])
        result=compare(a,b,400);reverse=compare(b,a,400)
        self.assertEqual((result['wins'],result['ties'],result['losses'],result['comparison_count']),(2,2,2,6))
        self.assertEqual(result['scorer_win_rate'],.5)
        self.assertEqual(result['strict_win_rate'],1/3)
        self.assertEqual(result['prompt_count'],2)
        self.assertEqual(result['per_seed_win_rates'],[.5,.75,.25])
        self.assertAlmostEqual(result['scorer_win_rate']+reverse['scorer_win_rate'],1.)
        tied=compare(np.ones((4,3)),np.ones((4,3)),100)
        self.assertEqual((tied['wins'],tied['ties'],tied['losses']),(0,12,0))
        self.assertEqual((tied['win_low'],tied['win_high']),(.5,.5))

    def test_launcher_runs_report_only_after_success(self):
        # Substitute only the child interpreter: exercise the real Bash task dispatch.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'calls.jsonl';fake=root/'python'
            fake.write_text('#!'+sys.executable+'\nimport os,sys,json\n'
                'from pathlib import Path\n'
                'if "-c" in sys.argv: print(os.environ["TEST_OUTPUT"]);sys.exit(0)\n'
                'with open(os.environ["TEST_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                'if "run" in sys.argv:sys.exit(int(os.environ.get("TEST_RUN_EXIT","0")))\n')
            fake.chmod(0o755)
            import os
            env=dict(os.environ,SCD_SD15_PY=str(fake),TEST_LOG=str(log),TEST_OUTPUT=str(root/'output'))
            launcher=ROOT/'scripts/run_multiseed_slai.sh'
            subprocess.run(['bash',str(launcher),'B'],env=env,check=True)
            calls=read_jsonl(log);self.assertEqual([r[1] for r in calls],['run','report'])
            self.assertIn('sd35_medium',calls[-1]);log.unlink()
            fail=subprocess.run(['bash',str(launcher),'A1'],env={**env,'TEST_RUN_EXIT':'7'})
            self.assertEqual(fail.returncode,7);self.assertEqual(len(read_jsonl(log)),1);log.unlink()
            subprocess.run(['bash',str(launcher),'D1'],env={**env,'MVPO_GENERATION_AUTO_REPORT':'0'},check=True)
            self.assertEqual(len(read_jsonl(log)),1)


    def test_partition_and_shared_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);slots=[dict(key=[f'p{i}','canonical',0]) for i in range(11)];payload_jsonl(root/'slots.jsonl',slots)
            job=dict(slots='slots.jsonl',shards=2)
            a=selected_slots(root,job,0);b=selected_slots(root,job,1)
            self.assertEqual(len(a)+len(b),11);self.assertFalse({tuple(s['key']) for s in a}&{tuple(s['key']) for s in b})


if __name__=='__main__':unittest.main()
