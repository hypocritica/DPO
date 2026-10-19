import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_correct_dpo.benchmarks import prepare_panels
from self_correct_dpo.common import read_json, read_jsonl, write_json, write_jsonl
from self_correct_dpo.config import load_config
from self_correct_dpo.evaluation import evaluate_benchmarks, evaluate_checkpoint, parse_args
from self_correct_dpo.optional_scorers import hps_config, load_optional_scorer, _load_image_reward_state
from self_correct_dpo.scorers import DEFAULT_METRICS, METRICS, scorer_identity
from test_evaluation import render_fixture
from self_correct_dpo.metrics import score_panel


class OptionalEvaluationTests(unittest.TestCase):
    def test_defaults_remain_four_metrics_and_hpdv2_with_unused_paths_unset(self):
        root=Path(__file__).resolve().parents[1]
        args=['--config','fixture','--run','fixture','--step','1000','--out','fixture']
        a=parse_args(args)
        self.assertEqual(a.benchmarks,['hpdv2'])
        self.assertEqual(a.metrics,list(DEFAULT_METRICS))
        self.assertEqual(set(METRICS)-set(a.metrics),{'hpsv3','image_reward'})
        with patch.dict(os.environ,{'HPDV2_PROMPTS':'/tmp/hpdv2.jsonl','SCORERS_CONFIG':str(root/'configs/scorers.json')},clear=True):
            c=load_config(root/'configs/evaluate.json',benchmarks=a.benchmarks)
            self.assertEqual(c['benchmarks'],{'hpdv2':'/tmp/hpdv2.jsonl'})
        selected=parse_args(args+['--benchmarks','pickapic','partiprompts','hpdv3','imagereward',
                                 '--metrics','hpsv3','image_reward'])
        self.assertEqual(len(selected.benchmarks),4)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json'
            write_json(path,{'scorers':{'pickscore':{'model':'model','processor':'processor'},
                'hpsv3':{'python':'${UNSET_OPTIONAL_PYTHON}'},'image_reward':{'source':'${UNSET_SOURCE}'}}})
            self.assertEqual(set(load_config(path,scorers=['pickscore'])['scorers']),{'pickscore'})
            with self.assertRaisesRegex(ValueError,'UNSET_OPTIONAL_PYTHON'):
                load_config(path,scorers=['hpsv3'])

    def test_optional_prompt_inputs_work_without_default_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            (r/'parti.tsv').write_text('Prompt\tCategory\tChallenge\nA bird\tAnimals\tBasic\n TRAIN CAPTION \tObjects\tBasic\n')
            write_json(r/'hpdv3.json',{'test':[{'prompt':'A mountain','category':'nature'},
                {'prompt':'a MOUNTAIN'}, {'prompt':'train caption'}]})
            write_json(r/'ir.json',['A lake','A lake','train caption'])
            write_jsonl(r/'train.jsonl',[{'split':'train','prompts':[{'prompt':'train caption'}]}])
            prepare_panels(None,None,r/'panels',train_pairs=r/'train.jsonl',parti=r/'parti.tsv',
                           hpdv3=r/'hpdv3.json',imagereward=r/'ir.json')
            info=read_json(r/'panels/selection.json')
            self.assertEqual(set(info['panels']),{'partiprompts','hpdv3-test','imagereward-test'})
            self.assertIsNone(info['monitor_count'])
            self.assertTrue(all(p['prompts']==1 for p in info['panels'].values()))
            self.assertEqual(read_jsonl(r/'panels/partiprompts.jsonl')[0]['tag'],'Animals')
            self.assertEqual(read_jsonl(r/'panels/hpdv3-test.jsonl')[0]['tag'],'nature')
            self.assertEqual(len(info['source_sha256']),4)

    def test_all_six_metrics_share_generated_images_and_choose_python(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);model=r/'model';model.mkdir();(model/'config.json').write_text('{}')
            calls=[]
            inputs={'prompts':3,'scorers':{m:{'spec':{'python':f'/env/{m}/bin/python'}} for m in METRICS}}
            def worker(command):
                calls.append(command)
                target=Path(command[command.index('--out')+1])
                if command[2]=='self_correct_dpo.render':render_fixture(target)
                else:
                    metric=command[command.index('--metric')+1]
                    render=Path(command[command.index('--run')+1])
                    score_panel(render,target,metric,{},scorer_factory=lambda *_:lambda images,prompts:[.2]*len(images),
                                asset_id={'scale':'fixture'})
            with patch('self_correct_dpo.evaluation.validate_evaluation',return_value=inputs),patch('self_correct_dpo.evaluation.require_gpu_allocation'):
                result=evaluate_checkpoint({'path':str(model)},None,r/'prompts',r/'scorers',list(METRICS),
                    {'seeds':[17]},r/'out',command_runner=worker)
            self.assertEqual(len(calls),7)
            self.assertEqual(calls[0][0],sys.executable)
            self.assertEqual([c[0] for c in calls[1:]],[f'/env/{m}/bin/python' for m in METRICS])
            self.assertEqual(set(result['metrics']),set(METRICS))
            self.assertEqual(len(list((r/'out/render/images').glob('*.png'))),3)

    def test_multiple_benchmarks_have_separate_outputs_and_reject_changed_request(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);names=['pickapic','partiprompts','hpdv3']
            config={'benchmarks':{n:n+'.jsonl' for n in names},'sampling':{'seeds':[17]},'scorers_config':'scorers'}
            calls=[]
            def evaluator(model,adapter,prompts,scorers,metrics,sampling,out,**kw):
                calls.append(Path(out).name)
                return {'prompts':2,'metrics':{m:{'mean':.2} for m in metrics}}
            with patch('self_correct_dpo.evaluation.evaluate_checkpoint',side_effect=evaluator),patch('self_correct_dpo.evaluation.validate_evaluation'):
                result=evaluate_benchmarks(config,{'path':'model'},None,names,['pickscore'],r/'out',step=1000)
                self.assertEqual(set(result['benchmarks']),set(names))
                self.assertEqual(calls,names)
                self.assertEqual(result,read_json(r/'out/summary.json'))
                with self.assertRaisesRegex(ValueError,'fresh multi-benchmark'):
                    evaluate_benchmarks(config,{'path':'model'},None,names,['hpsv3'],r/'out',step=1000)
                evaluate_benchmarks(config,{'path':'model'},None,names,['pickscore'],r/'check',check_only=True)
                self.assertFalse((r/'check').exists())

    def optional_spec(self,root,metric):
        package='hpsv3' if metric=='hpsv3' else 'ImageReward'
        (root/package).mkdir();(root/package/'__init__.py').write_text('')
        (root/'model').mkdir();(root/'model/config.json').write_text('{}');(root/'checkpoint').write_text('fixture')
        (root/'config.yaml').write_text('model_name_or_path: remote/model\ndisable_flash_attn2: false\noutput_dim: 2\n')
        return {'source':str(root),'model':str(root/'model'),'tokenizer':str(root/'model'),
            'checkpoint':str(root/'checkpoint'),'config':str(root/'config.yaml'),'batch_size':1}

    def test_hpsv3_official_local_constructor_mu_scores_and_config_unchanged(self):
        import torch
        import yaml
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);spec=self.optional_spec(r,'hpsv3');original=(r/'config.yaml').read_bytes()
            captured={}
            class API:
                def __init__(self,**kw):
                    captured.update(kw);captured['config']=yaml.safe_load(Path(kw['config_path']).read_text())
                def reward(self,**kw):
                    captured['batch']=kw
                    return torch.tensor([[.3,9.],[-1.2,8.]])
            images=[Image.new('RGB',(2,2)),Image.new('RGB',(2,2))]
            with patch('self_correct_dpo.optional_scorers.source_package',return_value=SimpleNamespace(HPSv3RewardInferencer=API)):
                scorer=load_optional_scorer('hpsv3',spec,device='cpu')
                values=scorer(images,['a','b'])
                self.assertAlmostEqual(values[0],.3);self.assertAlmostEqual(values[1],-1.2)
                self.assertEqual(captured['batch'],{'image_paths':images,'prompts':['a','b']})
                self.assertEqual(captured['checkpoint_path'],str(r/'checkpoint'))
                self.assertFalse(captured['differentiable'])
                self.assertTrue(captured['config']['disable_flash_attn2'])
                self.assertEqual(captured['config']['model_name_or_path'],str(r/'model'))
                self.assertTrue(Path(captured['config_path']).is_file())
                with self.assertRaisesRegex(ValueError,'two columns'):
                    scorer(images[:1],['a'])
            self.assertEqual((r/'config.yaml').read_bytes(),original)
            identity=scorer_identity('hpsv3',spec)
            self.assertEqual(identity['version'],'v3')
            self.assertIn('adapter_sha256',identity)

    def test_imagereward_local_tokenizer_pairing_and_restores_factory(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);spec=self.optional_spec(r,'image_reward');(r/'config.yaml').write_text('{}')
            captured=[];tokens=[]
            tokenizer=SimpleNamespace(add_special_tokens=tokens.append,additional_special_tokens_ids=[42])
            original=lambda:None
            blip=SimpleNamespace(init_tokenizer=original)
            class Model(torch.nn.Module):
                def __init__(self,**kw):
                    super().__init__();self.weight=torch.nn.Parameter(torch.tensor([0.]))
                    self.tokenizer=blip.init_tokenizer()
                def score(self,prompt,image):
                    captured.append((prompt,image,self.weight.item(),self.training))
                    return self.weight.item()+len(prompt)
            torch.save({'model':{'module.weight':torch.tensor([.5])}},spec['checkpoint'])
            def local_tokenizer(path,**kwargs):
                self.assertEqual(path,spec['tokenizer'])
                self.assertEqual(kwargs,{'local_files_only':True})
                return tokenizer
            factory=SimpleNamespace(from_pretrained=local_tokenizer)
            modules={'ImageReward.ImageReward':SimpleNamespace(ImageReward=Model),
                     'ImageReward.models.BLIP.blip_pretrain':blip}
            images=[Image.new('RGB',(2,2)),Image.new('RGB',(2,2))]
            with patch('self_correct_dpo.optional_scorers.source_package'),patch('self_correct_dpo.optional_scorers.importlib.import_module',side_effect=modules.__getitem__),patch.dict(sys.modules,{'transformers':SimpleNamespace(BertTokenizer=factory)}):
                scorer=load_optional_scorer('image_reward',spec,device='cpu')
                self.assertEqual(scorer(images,['a','bb']),[1.5,2.5])
            self.assertIs(blip.init_tokenizer,original)
            self.assertEqual(tokens,[{'bos_token':'[DEC]'},{'additional_special_tokens':['[ENC]']}])
            self.assertEqual(tokenizer.enc_token_id,42)
            self.assertEqual(captured,[('a',images[0],.5,False),('bb',images[1],.5,False)])

    def test_imagereward_checkpoint_cannot_silently_skip_model_weights(self):
        import torch
        model=torch.nn.Linear(2,1)
        state=model.state_dict();extra={**state,'bert.position_ids':torch.arange(8).unsqueeze(0)}
        _load_image_reward_state(model,extra)
        with self.assertRaises(RuntimeError):_load_image_reward_state(model,{'weight':state['weight']})
        with self.assertRaises(RuntimeError):_load_image_reward_state(model,{**state,'unknown.weight':torch.ones(1)})
        with self.assertRaisesRegex(ValueError,'Noncanonical'):
            _load_image_reward_state(model,{**state,'bert.position_ids':torch.ones(1,8)})

    def test_large_scorer_uses_configured_batch_size_and_allows_override(self):
        from self_correct_dpo.metrics import main
        args=['metrics','--run','fixture','--out','fixture','--scorers-config','fixture','--metric','hpsv3']
        with patch('self_correct_dpo.metrics.load_config',return_value={'scorers':{'hpsv3':{'batch_size':1}}}),patch('self_correct_dpo.metrics.score_panel',return_value={}) as scorer,patch('builtins.print'):
            with patch.object(sys,'argv',args):main()
            self.assertEqual(scorer.call_args.kwargs['batch_size'],1)
            with patch.object(sys,'argv',args+['--batch-size','2']):main()
            self.assertEqual(scorer.call_args.kwargs['batch_size'],2)


if __name__=='__main__':unittest.main()
