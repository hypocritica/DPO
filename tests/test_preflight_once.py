"""Focused CPU checks for explicit schedules and the opt-in once fast path."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import torch
from self_correct_dpo import training
from self_correct_dpo.common import digest,read_json,sha256,write_json
from self_correct_dpo.runtime.checkpoints import atomic_json,resolve_checkpoint
from self_correct_dpo.runtime.pipeline import evaluation_steps,run_pipeline
from self_correct_dpo.evaluation.runner import _evaluate_checkpoint
from test_stage_pause import ToyBackend
import test_validation


class SingleViewToy(ToyBackend):
    def __init__(self,*args):
        super().__init__(*args)
        from types import SimpleNamespace
        self.scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    def score_errors(self,*args,**kwargs):
        values=super().score_errors(*args,**kwargs)
        for key in ('policy_errors','reference_errors'):values[key]=values[key][:,:1]
        return values


class PreflightOnceTests(unittest.TestCase):
    def test_explicit_schedule_legacy_default_and_final_step(self):
        points=[200,400,800,1300,1800,2300,2800,3200]
        self.assertEqual(evaluation_steps(3200,{'steps':points}),[0,*points])
        self.assertEqual(evaluation_steps(3000,{'steps':points,'include_base':False}),points[:-1]+[3000])
        self.assertEqual(evaluation_steps(800,{'steps':[],'include_base':False}),[800])
        self.assertEqual(evaluation_steps(3200,{}),[0,200,500,1000,1500,2000,2500,3000,3200])
        for bad in ([False],[0],['200'],200):
            with self.assertRaises(ValueError):evaluation_steps(3200,{'steps':bad})

    def test_manifest_and_identity_checked_once_then_snapshot_only(self):
        fixture=test_validation.ValidationTests('test_prepared_auto_samples_and_preserves_full_loader_rows')
        fixture.setUp();self.addCleanup(fixture.doCleanups);fixture.ready()
        for row in fixture.rows:row['reviewed']=True
        fixture.save_manifest();fixture.ready()
        base=fixture.root/'base';base.mkdir();(base/'model_index.json').write_text('{}')
        config={'seed':17,'model':{'family':'sd15','path':str(base)},'data':{'pairs':str(fixture.manifest)},
                'training':{'preflight':'once','objective':'dpo','prompt_mode':'canonical'}}
        from self_correct_dpo.data.validation import load_training_pairs
        with redirect_stdout(io.StringIO()),patch.object(training,'validate_config',return_value=config) as validate,patch(
                'self_correct_dpo.data.validation.load_training_pairs',wraps=load_training_pairs) as load,patch.object(
                training,'check_rows',wraps=training.check_rows) as check,patch.object(
                training,'model_artifact_identity',wraps=training.model_artifact_identity) as model:
            first=training.load_preflight(config,fixture.root/'run')
            self.assertEqual((validate.call_count,load.call_count,check.call_count,model.call_count),(1,1,1,1))
        fixture.manifest.unlink()  # The saved snapshot is deliberately trusted.
        with patch.object(training,'validate_config',side_effect=AssertionError('Repeated preflight')),patch.object(
                training,'training_identity',side_effect=AssertionError('Repeated identity scan')):
            again=training.load_preflight(config,fixture.root/'run')
            self.assertEqual(first,again)
            changed=copy.deepcopy(config);changed['seed']=29
            with self.assertRaisesRegex(ValueError,'configuration'):training.load_preflight(changed,fixture.root/'run')

    def test_fast_training_losses_budget_and_resume_match_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir();(base/'model_index.json').write_text('{}')
            rows=[{'pair_id':str(i),'split':'train','reviewed':True,
                   'prompts':[{'variant_id':'canonical','prompt':f'test {i}'}]} for i in range(3)]
            for objective,k,backend in (('dpo',2,ToyBackend),('linear_dpo',1,SingleViewToy)):
                c={'seed':17,'model':{'family':'sd15','path':str(base)},'data':{},
                    'training':{'formal_arm':'C','noise_views':k,'objective':objective,'batch_size':1,
                        'max_steps':4,'gradient_accumulation_steps':2,'checkpoint_every':2,'resolution':64,
                        'beta':2.,'temperature':.1,'learning_rate':.01,'max_grad_norm':1.,'linear_eta':.01,
                        'checkpoint_policy':'bounded','lora_checkpoint_every':4,'max_lora_checkpoints':None,
                        'prompt_mode':'canonical','sampling':'balanced','logging':'jsonl'}}
                image=torch.zeros(1,2,3,1,1)
                with redirect_stdout(io.StringIO()),patch.object(training,'image_batch',return_value=image):
                    training.train(c,rows,root/(objective+'-legacy'),backend_factory=backend,device='cpu')
                c['training']['preflight']='once'
                saved={'config':c,'rows':rows,'validation_rows':[], 'identity':training.training_identity(c,rows,[])}
                console=io.StringIO()
                with redirect_stdout(console),patch.object(training,'image_batch',return_value=image) as images,patch.object(
                        training,'check_rows',side_effect=AssertionError('Repeated row check')),patch.object(
                        training,'training_identity',side_effect=AssertionError('Repeated provenance scan')):
                    training.train(c,rows,root/(objective+'-full'),backend_factory=backend,device='cpu',preflight=saved)
                    training.train(c,rows,root/(objective+'-split'),backend_factory=backend,device='cpu',preflight=saved,stop_after_step=2)
                    training.train(c,rows,root/(objective+'-split'),backend_factory=backend,device='cpu',preflight=saved,
                        resume=root/(objective+'-split')/'checkpoints/latest.json')
                    self.assertTrue(all(call.kwargs['verify_files'] is False for call in images.call_args_list))
                states=[];weights=[]
                for suffix in ('legacy','full','split'):
                    point=resolve_checkpoint(root/(objective+'-'+suffix)/'checkpoints/latest.json')
                    weights.append(torch.load(point/'adapter/weight.pt',weights_only=True))
                    states.append(torch.load(point/'state.pt',weights_only=True))
                for weight in weights[1:]:torch.testing.assert_close(weight,weights[0],rtol=0,atol=0)
                self.assertEqual(states[-1]['cost']['exposure'],{})
                self.assertEqual(states[-1]['cost']['pairs_seen'],8)
                self.assertEqual(states[-1]['cost']['prompt_pair_evaluations'],8*k)
                for key in ('exp_avg','exp_avg_sq','step'):
                    torch.testing.assert_close(states[1]['optimizer']['state'][0][key],states[2]['optimizer']['state'][0][key],rtol=0,atol=0)
                entries=[json.loads(line) for line in console.getvalue().splitlines() if line.startswith('{')]
                for key in ('seconds_per_update','pairs_per_second','peak_cuda_GiB'):self.assertIn(key,entries[-1])

    def test_reissued_certificates_cannot_replace_cached_run_inputs(self):
        from test_flow_data_audit import FlowDataAuditTest, module as data_audit
        from test_flow_model_validation import tiny_model, validate_model
        for changed in ('dataset','model'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                pairs,extraction=FlowDataAuditTest().fixture(root)
                data_report=pairs.parent/'validation-report.json'
                write_json(data_report,data_audit.audit(pairs,extraction=extraction))
                base=root/'base';base.mkdir();tiny_model(base)
                model_report=base/'flow-validation-report.json'
                write_json(model_report,validate_model(base,'zimage','fixture-revision'))
                config={'seed':17,'model':{'family':'zimage','path':str(base),'revision':'fixture-revision',
                            'validation_report':str(model_report)},
                        'data':{'pairs':str(pairs),'validation_report':str(data_report)},
                        'training':{'preflight':'once','objective':'dpo','prompt_mode':'canonical'}}
                with patch.object(training,'validate_config',return_value=config),redirect_stdout(io.StringIO()):
                    saved=training.load_preflight(config,root/'run')
                self.assertEqual(training.load_preflight(config,root/'run'),saved)
                if changed=='dataset':
                    pairs.write_text(pairs.read_text()+'\n')
                    ready=read_json(pairs.parent/'ready.json');ready['pairs_sha256']=sha256(pairs)
                    atomic_json(pairs.parent/'ready.json',ready)
                    report=data_audit.audit(pairs,extraction=extraction)
                    self.assertTrue(report['passed'],report)
                    atomic_json(data_report,report)
                else:
                    weight=base/'transformer/model.safetensors'
                    payload=weight.read_bytes();weight.write_bytes(payload[:-1]+b'\1')
                    atomic_json(model_report,validate_model(base,'zimage','fixture-revision'))
                with self.assertRaisesRegex(ValueError,f'Certified {changed} changed since run preflight'):
                    training.load_preflight(config,root/'run')

    def test_monitor_runs_exact_points_and_reuses_first_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';points=[200,400,800,1300,1800,2300,2800,3200]
            settings={'enabled':True,'steps':points,'include_base':True,'expected_prompts':64,
                'prompts':'panel','scorers_config':'scorers','sampling':{'seeds':[17]}}
            c={'model':{'family':'sd35_medium'},'evaluation':settings,
               'training':{'max_steps':3200,'preflight':'once','logging':'jsonl'}}
            code={n:'frozen-'+n for n in ('runner.py','render.py','metrics.py','scorers.py','common.py')}
            frozen={'settings':settings,'inputs':{'prompts':64},'code_sha256':{}}
            saved={'identity':{'code':code,'base_identity':{'frozen':True}},'monitor_identity':frozen}
            calls=[];evaluated=[]
            def run(command):
                step=int(command[command.index('--stop-after-step')+1]);calls.append(step)
                out.mkdir(exist_ok=True)
                if not (out/'training.json').exists():write_json(out/'training.json',{'run_id':'run','config':c})
                folder=out/'checkpoints'/f'resume-{step}';folder.mkdir(parents=True)
                (folder/'state.pt').write_bytes(b'state')
                info={'run_id':'run','step':step,'full_state':True,'files':{'state.pt':sha256(folder/'state.pt')}}
                info['checkpoint_id']=digest(info);write_json(folder/'checkpoint.json',info)
                from self_correct_dpo.runtime.checkpoints import atomic_json
                atomic_json(out/'checkpoints/latest.json',{'directory':folder.name,'checkpoint_id':info['checkpoint_id']})
            def evaluate(*args,**kwargs):
                evaluated.append(kwargs['step']);self.assertEqual(kwargs['preflight']['inputs'],frozen['inputs'])
                path=Path(args[6]);path.mkdir(parents=True)
                write_json(path/'summary.json',{'step':kwargs['step']})
                write_json(path/'complete.json',{'summary_sha256':sha256(path/'summary.json'),'scores_sha256':{}})
            with patch('self_correct_dpo.runtime.pipeline.update_curve'),patch(
                    'self_correct_dpo.runtime.pipeline.validate_evaluation',side_effect=AssertionError('Repeated scorer scan')):
                result=run_pipeline(c,root/'config.json',out,execution={'mode':'standalone'},
                    command_runner=run,evaluator=evaluate,preflight=saved)
            self.assertTrue(result['complete']);self.assertEqual(calls,points);self.assertEqual(evaluated,[0,*points])
            context={'inputs':{'prompts':64},'base_identity':{'frozen':True},'code_sha256':code}
            with patch('self_correct_dpo.evaluation.runner.validate_evaluation',side_effect=AssertionError('Repeated scorer scan')),patch(
                    'self_correct_dpo.evaluation.runner.model_artifact_identity',side_effect=AssertionError('Repeated base scan')):
                result=_evaluate_checkpoint(c['model'],None,'panel','scorers',['pickscore'],{'seeds':[17]},root/'eval',
                    preflight=context,check_only=True)
                self.assertEqual(result['images'],64)


if __name__=='__main__':unittest.main()
