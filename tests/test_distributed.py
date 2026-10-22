import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from self_correct_dpo.runtime.checkpoints import resolve_checkpoint
from self_correct_dpo.common import read_json, sha256, write_json, write_jsonl
from self_correct_dpo.runtime.distributed import subprocess_environment
from self_correct_dpo.algorithms.formal import formal_loss
from self_correct_dpo.runtime.pipeline import run_pipeline, training_command
from self_correct_dpo.evaluation.runner import validate_evaluation


def fixture(root,world,arm='C',*,monitor=False):
    base=root/'base';base.mkdir(exist_ok=True)
    (base/'model_index.json').write_text('{}')
    model=root/'scorer-model';model.mkdir(exist_ok=True);(model/'config.json').write_text('{}')
    scorer=root/'scorers.json'
    if not scorer.exists():write_json(scorer,{'scorers':{'pickscore':{'model':str(model),'processor':str(model)}}})
    prompts=root/'prompts.jsonl'
    if not prompts.exists():
        from self_correct_dpo.data.benchmarks import normalize_panel
        write_jsonl(prompts,normalize_panel([{'prompt':'held out 1'},{'prompt':'held out 2'}],'test'))
    c={'seed':17,'model':{'family':'sdxl','path':str(base),'revision':'test-fixture'},'data':{},
       'training':{'formal_arm':arm,'noise_views':2,'objective':'dpo','world_size':world,
           'batch_size':1,'max_steps':3,'gradient_accumulation_steps':2,'checkpoint_every':2,'resolution':64,
           'beta':2.,'temperature':.1,'learning_rate':.01,'max_grad_norm':1000.,'lr_schedule':'constant',
           # Scale the independent 200/1000-step schedules to 2/10 for this tiny run.
           'checkpoint_policy':'bounded','lora_checkpoint_every':10,'max_lora_checkpoints':None,
           'prompt_mode':'canonical','sampling':'balanced','independent_rollout':{'state_source':'model'}},
       'evaluation':{'enabled':monitor,'prompts':str(prompts),'scorers_config':str(scorer),
           'expected_prompts':2,'first_step':1,'every':2,'keep_images_every':2,'sampling':{'seeds':[17]}}}
    return c


def execute(command,*,fail_rank=None):
    command=list(command)
    index=command.index('self_correct_dpo.train')
    command[index]=str(ROOT/'tests/ddp_worker.py')
    command.remove('--module')
    env=subprocess_environment(os.environ)
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
               SCD_DISTRIBUTED_TIMEOUT_SECONDS='30')
    if fail_rank is not None:env['SCD_TEST_FAIL_RANK']=str(fail_rank)
    process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:output,_=process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);output,_=process.communicate()
        raise AssertionError('Distributed test timed out: '+output[-5000:])
    if process.returncode:
        class WorkerError(subprocess.CalledProcessError):
            def __str__(self):return super().__str__()+'\n'+self.output[-6000:]
        raise WorkerError(process.returncode,command,output=output)
    return output


def state(out):return torch.load(resolve_checkpoint(Path(out)/'checkpoints/latest.json')/'state.pt',weights_only=True)


class DistributedTests(unittest.TestCase):
    def test_launcher_and_clean_subprocess_identity(self):
        env=subprocess_environment({'RANK':'1','WORLD_SIZE':'4','LOCAL_RANK':'1','MASTER_PORT':'12345',
            'TORCHELASTIC_RUN_ID':'test','CUDA_VISIBLE_DEVICES':'0,1,2,3','HF_HUB_OFFLINE':'1'})
        self.assertEqual(env,{'CUDA_VISIBLE_DEVICES':'0,1,2,3','HF_HUB_OFFLINE':'1'})
        for world in (2,4,8):
            command=training_command({'training':{'world_size':world}},'config','out',execution={'mode':'standalone'},stop_after_step=200)
            self.assertIn(f'--nproc-per-node={world}',command);self.assertIn('--max-restarts=0',command)
            self.assertIn('--stage-id',command)

    def test_multi_rank_rollout_gradients_and_exact_resume(self):
        for world,arm in ((2,'B'),(4,'C'),(8,'B'),(8,'C')):
            with self.subTest(world=world,arm=arm),tempfile.TemporaryDirectory() as d:
                r=Path(d);config=fixture(r,world,arm);write_json(r/'config.json',config)
                def run(out,**kw):execute(training_command(config,r/'config.json',out,execution={'mode':'standalone'},**kw))
                run(r/'full');run(r/'staged',stop_after_step=1)
                paused=state(r/'staged')
                self.assertEqual(len(paused['rank_states']),world)
                self.assertEqual([s['rank'] for s in paused['rank_states']],list(range(world)))
                self.assertTrue(all(s['world_size']==world for s in paused['rank_states']))
                self.assertFalse(torch.equal(paused['rank_states'][0]['generator'],paused['rank_states'][1]['generator']))
                # Direct global B/C autograd versus the production all-view DDP update.
                w=torch.tensor(.2,requires_grad=True);losses=[]
                local=[torch.load(r/'staged'/f'probe-step-1-rank-{rank}.pt',weights_only=True) for rank in range(world)]
                for micro in range(2):
                    policies=[];refs=[]
                    for probe in local:
                        views=probe['microbatches'][micro]['views']
                        policies.append(torch.cat([(w*v['noisy']+.03-v['target']).square().flatten(1).mean(1).reshape(1,1,2) for v in views],1))
                        refs.append(torch.cat([(.1*v['noisy']+.03-v['target']).square().flatten(1).mean(1).reshape(1,1,2) for v in views],1))
                    loss,_=formal_loss({'policy_errors':torch.cat(policies),'reference_errors':torch.cat(refs)},arm,2.)
                    losses.append(loss)
                expected,=torch.autograd.grad(torch.stack(losses).mean(),w)
                torch.testing.assert_close(paused['optimizer']['state'][0]['exp_avg'],.1*expected,rtol=2e-5,atol=2e-7)
                run(r/'staged',resume=r/'staged/checkpoints/latest.json')
                a,b=state(r/'full'),state(r/'staged')
                self.assertEqual(a['scheduler'],b['scheduler'])
                for key in ('exp_avg','exp_avg_sq','step'):
                    torch.testing.assert_close(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key],rtol=0,atol=0)
                self.assertEqual(a['cost']['exposure'],b['cost']['exposure'])
                for rank in range(world):
                    for key in ('generator','cpu_rng'):torch.testing.assert_close(a['rank_states'][rank][key],b['rank_states'][rank][key],rtol=0,atol=0)
                    one=torch.load(r/'full'/f'probe-step-3-rank-{rank}.pt',weights_only=True)
                    two=torch.load(r/'staged'/f'probe-step-3-rank-{rank}.pt',weights_only=True)
                    torch.testing.assert_close(one['weight'],two['weight'],rtol=0,atol=0)
                    torch.testing.assert_close(one['weight'],torch.load(r/'full/probe-step-3-rank-0.pt',weights_only=True)['weight'],rtol=0,atol=0)
                    self.assertEqual(one['random_probe'],two['random_probe'])
                self.assertTrue((r/'staged/complete.json').is_file())

    def test_ddp_stage_evaluation_resume_and_single_rank_signal(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);config=fixture(r,8,monitor=True);write_json(r/'config.json',config)
            evaluated=[];failed=[False]
            def evaluator(model,adapter,prompts,scorers,metrics,sampling,path,**kw):
                step=kw['step'];evaluated.append(step)
                if step==1 and not failed[0]:failed[0]=True;raise subprocess.CalledProcessError(75,['interrupted evaluator'])
                path=Path(path);target=path/'scores/pickscore';target.mkdir(parents=True)
                write_jsonl(target/'scores.jsonl',[{'scene_id':str(i),'variant_id':'canonical','seed':17,'prompt':str(i),'score':.2+.01*step} for i in range(2)])
                summary={'step':step,'metrics':{'pickscore':{'mean':.2+.01*step,'prompts':2,'images':2}}}
                write_json(path/'summary.json',summary)
                write_json(path/'complete.json',{'summary_sha256':sha256(path/'summary.json'),'scores_sha256':{'scores/pickscore/scores.jsonl':sha256(target/'scores.jsonl')}})
            def identity(c):
                s=c['evaluation']
                return {'inputs':validate_evaluation(s['prompts'],s['scorers_config'],['pickscore'],expected_prompts=2)}
            kwargs={'execution':{'mode':'standalone'},'command_runner':execute,'evaluator':evaluator,'monitor_identity':identity(config)}
            result=run_pipeline(config,r/'config.json',r/'pipeline',**kwargs)
            self.assertFalse(result['complete']);self.assertEqual(result['step'],1)
            result=run_pipeline(config,r/'config.json',r/'pipeline',resume=r/'pipeline/checkpoints/latest.json',**kwargs)
            self.assertTrue(result['complete']);self.assertEqual(evaluated,[0,1,1,2,3])
            # Only the last of eight ranks receives TERM: every rank saves the same update.
            config=fixture(r,8,monitor=True);config['training']['test_signal_rank']=7
            write_json(r/'signal.json',config)
            kwargs['monitor_identity']=identity(config)
            result=run_pipeline(config,r/'signal.json',r/'signal',**kwargs)
            self.assertTrue(result['paused']);self.assertEqual(result['step'],1)
            self.assertFalse((r/'signal/complete.json').exists())
            self.assertEqual(len(state(r/'signal')['rank_states']),8)

    def test_worker_failure_preserves_last_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);config=fixture(r,8);write_json(r/'config.json',config)
            execute(training_command(config,r/'config.json',r/'run',execution={'mode':'standalone'},stop_after_step=1))
            before=read_json(r/'run/checkpoints/latest.json')
            with self.assertRaises(subprocess.CalledProcessError):
                execute(training_command(config,r/'config.json',r/'run',execution={'mode':'standalone'},resume=r/'run/checkpoints/latest.json'),fail_rank=7)
            self.assertEqual(before,read_json(r/'run/checkpoints/latest.json'))
            self.assertFalse((r/'run/complete.json').exists())

    def test_config_helper_preserves_global_batch_for_multi_gpu(self):
        for family,target in (('sd15',256),('sdxl',128)):
            for world in (2,4,8):
                with self.subTest(family=family,world=world),tempfile.TemporaryDirectory() as d:
                    p=Path(d)/'config.json'
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/make_config.py'),'--family',family,
                        '--model',d,'--pairs',str(Path(d)/'pairs.jsonl'),'--world-size',str(world),
                        '--out',str(p)],capture_output=True,text=True,timeout=20)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    t=read_json(p)['training']
                    self.assertEqual(t['world_size'],world)
                    self.assertEqual(t['batch_size']*t['gradient_accumulation_steps']*world,target)
                    self.assertEqual(t['checkpoint_every'],200)
                    self.assertEqual(t['lora_checkpoint_every'],1000)


if __name__=='__main__':unittest.main()
