"""CPU integration checks for phased scoring, pixel reuse and GPU migration."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from PIL import Image
from test_multiseed_generation import fixture, score_fixture, SCORER
from self_correct_dpo.evaluation import metrics as metrics_module
from appendix_flexible import (
    read_json, read_jsonl, digest, sha256, payload_jsonl, atomic_json,
    refresh_history, generate_shard, score_shard, receipt_rows, shard_root,
    offline_plan, CONFIG, launch, wanted_rows, load_plan, prepare)


def restamp(root, gpu='NVIDIA H100 80GB HBM3', partial=None):
    render = root / 'render'
    run = read_json(render / 'run.json')
    run['gpu'] = gpu
    run['run_id'] = digest({k: v for k, v in run.items() if k != 'run_id'})
    atomic_json(render / 'run.json', run)
    pixels = read_jsonl(render / 'manifest.jsonl')
    for row in pixels:
        row['image_id'] = digest({'run_id': run['run_id'], **{k: v for k, v in row.items() if k != 'image_id'}})
    if partial is not None:
        pixels = pixels[:partial]
        (render / 'complete.json').unlink(missing_ok=True)
    payload_jsonl(render / 'manifest.jsonl', pixels)
    if partial is None:
        atomic_json(render / 'complete.json', dict(images=len(pixels), manifest_sha256=sha256(render / 'manifest.jsonl')))
    return run


def setup_plan(output, run, shards=1):
    adapter = dict(run='/fixture/run', run_id='train', path='/fixture/adapter', checkpoint_id='checkpoint', inventory={})
    job = dict(name='dpo-1000-s0', method='dpo', step=1000, seed=0, panel='curve',
               metrics=['pickscore'], adapter=adapter, shards=shards, blocked=None)
    out = output / 'sd15'
    out.mkdir(parents=True)
    for panel in ['curve', 'full']:
        payload_jsonl(out / (panel + '.jsonl'), run['suite'])
    atomic_json(out / 'scorers.resolved.json', {})
    plan = dict(plan_id='fixture-plan', family='sd15', jobs=[job], model=run['model'], base_identity={},
        sampling=run['sampling'], render_versions=run['versions'], scorers={'pickscore': SCORER}, score_versions=None,
        interpreters={k: sys.executable for k in ['render', 'legacy', 'hpsv3']},
        legacy_root=str(output.parent / 'old'), panels={'full':'fixture', 'curve':'fixture'}, seeds=[0,23,42])
    return plan, job


class FlexibleTests(unittest.TestCase):
    def test_prepare_keeps_legacy_files_and_removes_hardware_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); run,_=fixture(root/'source',4)
            legacy=root/'legacy'; (legacy/'sd15').mkdir(parents=True)
            atomic_json(legacy/'sd15/scorers.resolved.json',{})
            atomic_json(legacy/'sd15/plan.json',{'sentinel':'original frozen A/B plan'})
            checksum=sha256(legacy/'sd15/plan.json')
            scratch=root/'fixture';old,job=setup_plan(scratch,run)
            job['pack']='A1'
            experiment=dict(method='dpo',family='sd15',run_resource='RUNS_ROOT',run='dpo',run_id='train',
                            main_step=1000,checkpoints={'1000':'checkpoint'},curve_steps=[])
            old['config']={'experiments':[experiment]}
            training=root/'runs/dpo';training.mkdir(parents=True)
            atomic_json(training/'training.json',dict(run_id='train',base_identity={},config=dict(model=run['model'],
                method='dpo',training=dict(objective='dpo',noise_views=1,formal_arm='C',stagewise=None))))
            spec=read_json(CONFIG);spec.update(sd15_extra=[],full_count=4,curve_count=4)
            output=root/'new'
            with patch('appendix_flexible.legacy_plan',return_value=(old,{'full':run['suite'],'curve':run['suite']})), \
                 patch('appendix_flexible.resolve_adapter',return_value=job['adapter']):
                prepare(output,legacy,spec,{'RUNS_ROOT':str(root/'runs')},['sd15'])
                first=load_plan(output,'sd15')
                prepare(output,legacy,spec,{'RUNS_ROOT':str(root/'runs')},['sd15'])
            self.assertEqual(sha256(legacy/'sd15/plan.json'),checksum)
            self.assertEqual(first['jobs'][0]['shards'],4)
            self.assertNotIn('noise_runtime',first)
            self.assertEqual(first,load_plan(output,'sd15'))

    def test_six_scopes_counts_and_legacy_ab_unchanged(self):
        spec = read_json(CONFIG)
        expected = {'sd35-table':(10,16320), 'sd15-curves':(42,23744),
            'sd35-curves-dpo-lossavg':(14,7168), 'sd35-curves-global-stage':(14,7168),
            'sd35-curves-k2':(12,6144), 'sd35-lossavg-multiseed':(2,3264),
            'core':(94,63808), 'A':(66,74112), 'B':(79,74048)}
        for task, counts in expected.items():
            result = offline_plan(spec, [task])
            self.assertEqual((len(result['jobs']), result['target_slots']), counts)
        curve_jobs = offline_plan(spec, ['sd35-curves-dpo-lossavg','sd35-curves-k2'])['jobs']
        self.assertFalse(any(j['step']==1600 for j in curve_jobs))
        individual=[{(j['family'],j['name']) for j in offline_plan(spec,[task])['jobs']} for task in spec['tasks']]
        self.assertEqual(sum(map(len,individual)),len(set.union(*individual)))

    def test_cached_scores_need_no_images_no_generation_and_no_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); old=root/'old'; fixture(old,4); run=restamp(old); score_fixture(old)
            output=root/'new'; plan,job=setup_plan(output,run)
            plan['score_versions']={'pickscore':read_json(old/'scores/pickscore/identity.json')['identity']['versions']}
            refresh_history(output,{'sd15':plan},[old])
            for p in (old/'render/images').glob('*.png'):p.unlink()
            with patch('appendix_flexible.run_stage') as stage:
                generate_shard(output,plan,job,0,dict(gpu='NVIDIA GeForce RTX 5090',torch='fixture',diffusers='fixture'),2)
                score_shard(output,plan,job,0)
                score_shard(output,plan,job,0)
                stage.assert_not_called()
            self.assertEqual(len(receipt_rows(output,plan,job,0,'score')),4)

    def test_partial_h100_pixels_survive_5090_restart_and_score_is_separate(self):
        for local_partial in [False,True]:
            with self.subTest(local_partial=local_partial), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); old=root/'old'; fixture(old,4); run=restamp(old,partial=2)
                output=root/'new';plan,job=setup_plan(output,run)
                if local_partial:
                    target=shard_root(output,'sd15',job,0)/'attempts'/'interrupted'
                    target.parent.mkdir(parents=True)
                    old.rename(target)
                    refresh_history(output,{'sd15':plan},[])
                else:refresh_history(output,{'sd15':plan},[old])
                runtime=dict(gpu='NVIDIA GeForce RTX 5090',torch='fixture',diffusers='fixture')
                def render(command,**kwargs):
                    self.assertIn('self_correct_dpo.evaluation.render',command)
                    prompts=read_jsonl(command[command.index('--prompts')+1]);self.assertEqual(len(prompts),2)
                    config=read_json(command[command.index('--config')+1])
                    dest=Path(command[command.index('--out')+1]);dest.mkdir(parents=True)
                    header=copy.deepcopy(run);header.update(config=config,sampling=config['sampling'],gpu=runtime['gpu'],
                        suite=prompts,suite_id=digest(prompts),full_suite_id=digest(prompts),expected_images=2,
                        suite_indices=[0,1])
                    header['run_id']=digest({k:v for k,v in header.items() if k!='run_id'})
                    atomic_json(dest/'run.json',header);pixels=[]
                    for i,p in enumerate(prompts):
                        path=dest/f'{i}.png';Image.new('RGB',(8,8),'orange').save(path)
                        row={k:p[k] for k in ['scene_id','variant_id','prompt','tag']}
                        row.update(seed=0,image=path.name,image_sha256=sha256(path))
                        row['image_id']=digest({'run_id':header['run_id'],**row});pixels.append(row)
                    payload_jsonl(dest/'manifest.jsonl',pixels)
                    atomic_json(dest/'complete.json',dict(images=2,manifest_sha256=sha256(dest/'manifest.jsonl')))
                with patch('appendix_flexible.resolve_adapter',return_value=job['adapter']),patch('appendix_flexible.run_stage',side_effect=render) as stage:
                    generate_shard(output,plan,job,0,runtime,2)
                    generate_shard(output,plan,job,0,runtime,2)
                    self.assertEqual(stage.call_count,1)
                rows=receipt_rows(output,plan,job,0,'generate')
                self.assertEqual({r['generation_gpu'] for r in rows},{'NVIDIA H100 80GB HBM3','NVIDIA GeForce RTX 5090'})
                self.assertIsNone(receipt_rows(output,plan,job,0,'score'))
                def score(command,**kwargs):
                    self.assertIn('self_correct_dpo.evaluation.metrics',command)
                    dest=Path(command[command.index('--out')+1]);render_path=Path(command[command.index('--run')+1])
                    metrics_module.score_panel(render_path,dest,'pickscore',{},
                        scorer_factory=lambda *a:lambda imgs,texts:[float(i.getpixel((0,0))[0])/255 for i in imgs],asset_id=SCORER)
                    plan['score_versions']={'pickscore':read_json(dest/'identity.json')['identity']['versions']}
                with patch('appendix_flexible.run_stage',side_effect=score) as stage:
                    score_shard(output,plan,job,0)
                    self.assertEqual(stage.call_count,2)
                self.assertEqual(len(receipt_rows(output,plan,job,0,'score')),4)

    def test_scoring_barrier_fails_before_gpu_or_any_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,_=fixture(root/'old',4);plan,job=setup_plan(root/'new',run,shards=4)
            args=SimpleNamespace(action='score',output=root/'new')
            with patch('appendix_flexible.gpu_tokens') as gpu,patch('appendix_flexible.run_workers') as workers:
                with self.assertRaisesRegex(ValueError,'Generation barrier'):
                    launch(args,{'sd15':plan},[('sd15',job)])
                gpu.assert_not_called();workers.assert_not_called()

    def test_two_or_four_devices_dispatch_the_same_four_shards(self):
        for device_count in (2,4):
            with self.subTest(device_count=device_count),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);run,_=fixture(root/'old',8);plan,job=setup_plan(root/'new',run,shards=4)
                plan['spec']=read_json(CONFIG)
                args=SimpleNamespace(action='generate',output=root/'new',gpus=None,batch_size=None,tasks=['sd15-curves'],config=CONFIG)
                runtime=dict(gpu='NVIDIA GeForce RTX 5090',torch='fixture',diffusers='fixture')
                with patch('appendix_flexible.gpu_tokens',return_value=[str(i) for i in range(device_count)]), \
                     patch('appendix_flexible.checked_runtime',return_value=runtime),patch('appendix_flexible.run_workers') as worker:
                    launch(args,{'sd15':plan},[('sd15',job)])
                self.assertEqual(worker.call_count,4//device_count)
                commands=[command for call in worker.call_args_list for _,_,command in call.args[0]]
                self.assertEqual({c[c.index('--index')+1] for c in commands},{'0','1','2','3'})
                self.assertTrue(all(c[c.index('--phase')+1]=='generate' for c in commands))

    def test_source_mutation_is_rejected_and_logical_shards_are_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';run,_=fixture(old,8);output=root/'new';plan,job=setup_plan(output,run,shards=4)
            groups=[wanted_rows(output,plan,job,i) for i in range(4)]
            self.assertEqual([len(r) for r in groups],[2,2,2,2])
            self.assertEqual(len({p['scene_id'] for g in groups for p in g}),8)
            refresh_history(output,{'sd15':plan},[old])
            with (old/'render/manifest.jsonl').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Historical files changed'):
                generate_shard(output,plan,job,0,{},2)

    def test_report_keeps_single_seed_and_missing_multiseed_distinct(self):
        from appendix_flexible_report import summarize
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,_=fixture(root/'old',4);plan,job=setup_plan(root/'new',run)
            plan['spec']={'tasks':{}}
            selected=[dict(job,panel='full'),dict(job,name='stagewise_k6-1000-s0',method='stagewise_k6',panel='full')]
            data={}
            for method,offset,gpu in [('dpo',0.,'H100'),('stagewise_k6',1.,'5090')]:
                for i,p in enumerate(run['suite']):
                    data[(method,1000,p['scene_id'],p['variant_id'],0)]=dict(scores={'pickscore':offset+i/10},generation_gpu=gpu)
            means,pairs=summarize(plan,{'full':run['suite'],'curve':run['suite']},data,selected,repeats=20)
            single=next(r for r in pairs if r['seeds']=='0')
            multiple=next(r for r in pairs if r['seeds']=='0,23,42')
            self.assertEqual(single['scorer_win_rate'],1.)
            self.assertFalse(single['paired_same_generation_gpu'])
            self.assertEqual(multiple['status'],'missing')
            self.assertNotIn('scorer_win_rate',multiple)


if __name__=='__main__':unittest.main()
