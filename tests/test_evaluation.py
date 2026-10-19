import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_correct_dpo.benchmarks import normalize_panel, select_monitor, prepare_panels
from self_correct_dpo.checkpoints import atomic_json, progress_steps
from self_correct_dpo.common import digest, read_json, sha256, write_json, write_jsonl
from self_correct_dpo.config import load_config
from self_correct_dpo.evaluation import prune_images, evaluate_checkpoint
from self_correct_dpo.metrics import score_panel
from self_correct_dpo.pipeline import evaluation_steps, keep_images, paired_delta, run_pipeline
from self_correct_dpo.scorers import aesthetic_head, load_scorer, scorer_identity
from self_correct_dpo.render import validate_config as validate_render, check_prompt_lengths


def render_fixture(root, count=3):
    from PIL import Image
    root=Path(root);root.mkdir(parents=True)
    suite=normalize_panel([{'prompt':f'caption {i}'} for i in range(count)],'test')
    run={'suite':suite,'suite_id':digest(suite),'expected_images':count,'sampling':{'seeds':[17]}}
    run['run_id']=digest(run)
    write_json(root/'run.json',run)
    rows=[]
    for i,prompt in enumerate(suite):
        path=root/'images'/f'{i}.png';path.parent.mkdir(exist_ok=True)
        Image.new('RGB',(4,4),(i*20,0,10)).save(path)
        row={k:prompt[k] for k in ('scene_id','variant_id','prompt','tag')}
        row.update(seed=17,image=str(path.relative_to(root)),image_sha256=sha256(path))
        row['image_id']=digest({'run_id':run['run_id'],**row});rows.append(row)
    write_jsonl(root/'manifest.jsonl',rows)
    write_json(root/'complete.json',{'images':count,'manifest_sha256':sha256(root/'manifest.jsonl')})
    return rows


class EvaluationTests(unittest.TestCase):
    def test_default_single_seed_pndm_evaluation_config_and_prompt_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            model=Path(d);(model/'unet').mkdir()
            write_json(model/'model_index.json',{'_class_name':'StableDiffusionPipeline'})
            write_json(model/'unet/config.json',{'cross_attention_dim':768,'in_channels':4})
            config=read_json(Path(__file__).resolve().parents[1]/'configs/evaluate.json')
            config['model']={'family':'sd15','path':str(model),'revision':'fixture'}
            validate_render(config,'base')
            self.assertEqual(config['sampling']['seeds'],[17])
        class Tokenizer:
            model_max_length=4
            def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=text.split())
        pipe=SimpleNamespace(tokenizer=Tokenizer())
        suite=[{'scene_id':'x','variant_id':'canonical','prompt':'a prompt with more than four words'}]
        self.assertEqual(len(check_prompt_lengths(pipe,suite,{'negative_prompt':'','prompt_truncation':'truncate'},'sd15')),1)
        with self.assertRaisesRegex(ValueError,'truncation'):
            check_prompt_lengths(pipe,suite,{'negative_prompt':''},'sd15')

    def test_checkpoint_evaluation_serial_workers_pruning_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);model=r/'model';model.mkdir();(model/'config.json').write_text('{}')
            calls=[]
            inputs={'prompts_sha256':'fixture','scorers_config_sha256':'fixture','prompts':3}
            metrics=['pickscore','clip','hpsv2','aesthetic']
            def worker(command):
                calls.append(command[2])
                target=Path(command[command.index('--out')+1])
                if command[2]=='self_correct_dpo.render':render_fixture(target)
                else:
                    metric=command[command.index('--metric')+1]
                    render=Path(command[command.index('--run')+1])
                    score_panel(render,target,metric,{},scorer_factory=lambda *_:lambda images,prompts:[.2]*len(images),
                                asset_id={'scale':'fixture'})
            with patch('self_correct_dpo.evaluation.validate_evaluation',return_value=inputs),patch('self_correct_dpo.evaluation.require_gpu_allocation'):
                summary=evaluate_checkpoint({'path':str(model),'family':'sd15','revision':'fixture'},None,
                    r/'prompts',r/'scorers',metrics,{'seeds':[17]},r/'evaluation',step=500,keep_images=False,command_runner=worker)
                same=evaluate_checkpoint({'path':str(model),'family':'sd15','revision':'fixture'},None,
                    r/'prompts',r/'scorers',metrics,{'seeds':[17]},r/'evaluation',step=500,keep_images=False,
                    command_runner=lambda *_:self.fail('completed evaluation must not repeat generation/scoring'))
            self.assertEqual(summary,same)
            self.assertEqual(calls,['self_correct_dpo.render']+['self_correct_dpo.metrics']*4)
            self.assertEqual(set(summary['metrics']),set(metrics))
            self.assertTrue((r/'evaluation/images-pruned.json').exists())

    def test_default_schedule_and_permanent_lora_retention(self):
        self.assertEqual(evaluation_steps(2000,{}),[0,200,500,1000,1500,2000])
        self.assertEqual(evaluation_steps(1200,{}),[0,200,500,1000,1200])
        self.assertEqual(progress_steps({'max_steps':2500,'lora_checkpoint_every':1000,'max_lora_checkpoints':None}),[1000,2000,2500])
        self.assertEqual([s for s in evaluation_steps(2000,{}) if keep_images(s,2000,{})],[0,1000,2000])

    def test_deterministic_test_subset_exclusion_and_remaining_panel(self):
        candidates=normalize_panel([{'prompt':f'p{i}'} for i in range(500)]+[{'prompt':' P1 '}],'test',exclude=['p0'])
        self.assertEqual(len(candidates),499)
        a=select_monitor(candidates)
        b=select_monitor(list(reversed(candidates)))
        self.assertEqual(a,b);self.assertEqual(len(a),128)
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            write_json(r/'pick.json',[{'caption':f'p{i}'} for i in range(500)])
            write_json(r/'hpd.json',[{'prompt':f'hpd {i}'} for i in range(400)])
            write_jsonl(r/'train.jsonl',[{'split':'train','prompts':[{'prompt':'p0'}]}])
            prepare_panels(r/'pick.json',r/'hpd.json',r/'out',train_pairs=r/'train.jsonl')
            info=read_json(r/'out/selection.json')
            self.assertEqual(info['panels']['pickapic-monitor']['prompts'],128)
            self.assertEqual(info['panels']['pickapic-test-remaining']['prompts'],371)
            self.assertEqual(info['panels']['hpdv2-test']['prompts'],400)

    def test_score_interrupt_resume_does_not_repeat_committed_images(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);render_fixture(r/'render')
            calls=[]
            def fail(metric,spec):
                def score(images,prompts):
                    calls.extend(prompts)
                    if len(calls)>1: raise RuntimeError('interrupted')
                    return [.25]
                return score
            asset={'scale':'raw cosine','version':None}
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                score_panel(r/'render',r/'scores','pickscore',{},batch_size=1,scorer_factory=fail,asset_id=asset)
            resumed=[]
            def succeed(metric,spec):
                def score(images,prompts): resumed.extend(prompts);return [.5]*len(images)
                return score
            result=score_panel(r/'render',r/'scores','pickscore',{},batch_size=1,scorer_factory=succeed,asset_id=asset)
            self.assertEqual(resumed,['caption 1','caption 2'])
            self.assertEqual(result['images'],3)
            self.assertAlmostEqual(result['mean'],1.25/3)
            self.assertEqual(result,score_panel(r/'render',r/'scores','pickscore',{},scorer_factory=lambda *_:self.fail('must skip'),asset_id=asset))
            with (r/'scores/scores.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                score_panel(r/'render',r/'scores','pickscore',{},scorer_factory=succeed,asset_id=asset)

    def test_pickscore_uses_paired_cosine_without_logit_scale_or_softmax(self):
        import torch
        class Model:
            config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=77))
            logit_scale=torch.tensor(math.log(100))
            def eval(self):return self
            def to(self,*args,**kwargs):return self
            def get_image_features(self,**kw):return torch.tensor([[1.,0.],[0.,1.]])
            def get_text_features(self,**kw):return torch.tensor([[1.,0.],[1.,1.]])
        class Processor:
            def __call__(self,**kw):return {'pixels' if 'images' in kw else 'ids':torch.zeros(2,1)}
        api=SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=lambda *a,**kw:Model()),
                            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**kw:Processor()))
        with patch.dict(sys.modules,{'transformers':api}):
            scorer=load_scorer('pickscore',{'model':'x','processor':'x'},device='cpu')
            scores=scorer([None,None],['a','b'])
        self.assertAlmostEqual(scores[0],1.)
        self.assertAlmostEqual(scores[1],2**-.5,places=6)

    def test_aesthetic_head_matches_checkpoint_shapes_and_hps_version_is_explicit(self):
        import torch
        head=aesthetic_head().eval()
        self.assertEqual(tuple(head.layers[0].weight.shape),(1024,768))
        self.assertEqual(tuple(head.layers[7].weight.shape),(1,16))
        self.assertEqual(tuple(head(torch.zeros(2,768)).shape),(2,))
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'open_clip').mkdir();(r/'open_clip/__init__.py').write_text('')
            (r/'HPS_v2.1_compressed.pt').write_text('fixture')
            with self.assertRaisesRegex(ValueError,'v2.0 requires'):
                scorer_identity('hpsv2',{'checkpoint':str(r/'HPS_v2.1_compressed.pt'),'open_clip_source':str(r),'version':'v2.0'})

    def test_monitor_only_resolves_pickscore_assets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scorers.json'
            p.write_text(json.dumps({'scorers':{'pickscore':{'model':'model','processor':'processor'},
                'hpsv2':{'checkpoint':'${MISSING_HPS}'}}}))
            c=load_config(p,scorers=['pickscore'])
            self.assertEqual(set(c['scorers']),{'pickscore'})

    def test_paired_change_rejects_unpaired_panels(self):
        base=[{'scene_id':str(i),'variant_id':'canonical','seed':17,'prompt':str(i),'score':.2} for i in range(3)]
        new=[dict(row,score=.3) for row in base]
        delta=paired_delta(new,base)
        self.assertAlmostEqual(delta['delta_base'],.1)
        with self.assertRaisesRegex(ValueError,'different'):
            paired_delta(new[:-1],base)

    def test_pruning_deletes_only_owned_generated_images(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);rows=render_fixture(r/'render')
            (r/'render/unrelated.txt').write_text('keep')
            prune_images(r)
            self.assertTrue((r/'render/unrelated.txt').exists())
            self.assertTrue((r/'render/manifest.jsonl').exists())
            self.assertTrue(all(not (r/'render'/row['image']).exists() for row in rows))

    def test_staged_pipeline_resume_finishes_pending_evaluation_before_next_stage(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);out=r/'run'
            (r/'prompts').write_text('fixture');(r/'scorers').write_text('{}')
            config={'model':{'path':'fixture','family':'sd15','revision':'fixture'},
                    'training':{'max_steps':6},'evaluation':{'prompts':str(r/'prompts'),
                    'scorers_config':str(r/'scorers'),'first_step':2,'every':3,'keep_images_every':3,
                    'sampling':{'seeds':[17]},'expected_prompts':2}}
            (r/'config.json').write_text(json.dumps(config))
            trained=[];evaluated=[];interrupted=[False]
            def trainer(command):
                target=int(command[command.index('--stop-after-step')+1]);trained.append(target)
                out.mkdir(exist_ok=True)
                if not (out/'training.json').exists():write_json(out/'training.json',{'config':config,'run_id':'fixture'})
                path=out/'checkpoints'/f'resume-{target}';(path/'adapter').mkdir(parents=True)
                (path/'state.pt').write_text('fake full optimizer state')
                (path/'adapter/config.json').write_text('{}')
                info={'step':target,'run_id':'fixture','full_state':True,'files':{'state.pt':sha256(path/'state.pt'),'adapter/config.json':sha256(path/'adapter/config.json')}}
                info['checkpoint_id']=digest(info);write_json(path/'checkpoint.json',info)
                atomic_json(out/'checkpoints/latest.json',{'directory':path.name,'checkpoint_id':info['checkpoint_id']})
            def evaluator(model,adapter,prompts,scorers,metrics,sampling,path,**kwargs):
                step=kwargs['step'];evaluated.append(step)
                if step==2 and not interrupted[0]:
                    interrupted[0]=True
                    raise subprocess.CalledProcessError(75,['fake scorer interrupt'])
                path=Path(path);target=path/'scores/pickscore';target.mkdir(parents=True)
                rows=[{'scene_id':str(i),'variant_id':'canonical','seed':17,'prompt':str(i),'score':.2+.01*step} for i in range(2)]
                write_jsonl(target/'scores.jsonl',rows)
                summary={'step':step,'metrics':{'pickscore':{'mean':.2+.01*step,'prompts':2,'images':2}}}
                write_json(path/'summary.json',summary)
                write_json(path/'complete.json',{'summary_sha256':sha256(path/'summary.json'),
                    'scores_sha256':{'scores/pickscore/scores.jsonl':sha256(target/'scores.jsonl')}})
                return summary
            frozen={'inputs':{'fixture':'immutable scorer identity'}}
            paused=run_pipeline(config,r/'config.json',out,execution={'mode':'standalone'},command_runner=trainer,evaluator=evaluator,monitor_identity=frozen)
            self.assertFalse(paused['complete']);self.assertEqual(paused['step'],2)
            result=run_pipeline(config,r/'config.json',out,execution={'mode':'standalone'},resume=out/'checkpoints/latest.json',command_runner=trainer,evaluator=evaluator,monitor_identity=frozen)
            self.assertEqual(trained,[2,3,6])
            self.assertEqual(evaluated,[0,2,2,3,6])
            self.assertTrue(result['complete'])
            self.assertTrue((out/'evaluation-complete.json').exists())
            self.assertTrue((out/'pickscore-curve.png').is_file())
            self.assertEqual([row['step'] for row in read_json(out/'pickscore-curve.json')['scores']],[0,2,3,6])
            with self.assertRaisesRegex(ValueError,'inputs changed'):
                run_pipeline(config,r/'config.json',out,execution={'mode':'standalone'},resume=out/'checkpoints/latest.json',
                    command_runner=trainer,evaluator=evaluator,monitor_identity={'inputs':{'fixture':'changed weights'}})


if __name__=='__main__':unittest.main()
