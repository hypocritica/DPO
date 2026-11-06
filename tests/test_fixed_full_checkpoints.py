"""Fixed full checkpoints remain retained while latest rolls independently."""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'scripts')]
import torch
from self_correct_dpo.common import read_json,write_json
from self_correct_dpo.runtime.checkpoints import ManagedCheckpoints,atomic_json,progress_steps,resolve_checkpoint
from self_correct_dpo.training import save_checkpoint
from test_stage_pause import ToyBackend
from run_flow import build_config


class FixedFullCheckpoints(unittest.TestCase):
    def test_pilot_schedule_preserves_all_400_steps_and_eval_snapshots(self):
        c=build_config('zimage','dpo',recipe='paper-pilot',profile='a800')
        t=c['training']
        self.assertEqual((t['checkpoint_every'],t['full_checkpoint_every']),(200,400))
        self.assertEqual(progress_steps(t),[200,400,800,1200,1300,1600,1800,2000,2300,2400,2800,3200])
        self.assertEqual(c['evaluation']['steps'],[200,400,800,1300,1800,2300,2800,3200])

    def test_400_full_600_latest_800_full_survive_pending_transaction_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'checkpoints').mkdir()
            settings={'max_steps':1600,'checkpoint_every':200,'full_checkpoint_every':400,
                'lora_checkpoint_steps':[200,400,800,1300,1600],'max_lora_checkpoints':1}
            manager=ManagedCheckpoints(out,'test-run',settings)
            backend=ToyBackend({},settings,'cpu')
            optimizer=torch.optim.AdamW(backend.trainable_parameters(),lr=.01)
            backend.weight.square().backward();optimizer.step();optimizer.zero_grad()
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            generator=torch.Generator().manual_seed(17)
            backend.resume_rank_states=[{'rank':0,'world_size':2,'generator':generator.get_state()},
                                        {'rank':1,'world_size':2,'generator':torch.Generator().manual_seed(18).get_state()}]
            cost={'exposure':{},'pairs_seen':128}
            manager.save_progress(backend,optimizer,scheduler,generator,200,cost)
            manager.save_resume(backend,optimizer,scheduler,generator,200,cost)
            manager.save_progress(backend,optimizer,scheduler,generator,400,cost)
            fixed400=out/'checkpoints/step-0000400'
            self.assertEqual(manager.save_resume(backend,optimizer,scheduler,generator,400,cost),fixed400)
            self.assertEqual(resolve_checkpoint(out/'checkpoints/latest.json'),fixed400)
            latest600=manager.save_resume(backend,optimizer,scheduler,generator,600,cost)
            self.assertEqual(resolve_checkpoint(out/'checkpoints/latest.json'),latest600)
            self.assertTrue((fixed400/'state.pt').is_file())
            self.assertEqual(len(list((out/'checkpoints').glob('resume-*'))),1)
            # Simulate death after the full 800 directory was committed, before
            # its manager index/latest pointer was advanced.
            staging=manager._begin('step-0000800',True,kind='progress')
            self.assertEqual(read_json(manager.index)['pending']['kind'],'progress')
            fixed800=out/'checkpoints/step-0000800'
            save_checkpoint(fixed800,backend,optimizer,scheduler,generator,800,'test-run',cost,staging_path=staging)
            recovered=ManagedCheckpoints(out,'test-run',settings);recovered.recover()
            self.assertEqual(resolve_checkpoint(out/'checkpoints/latest.json'),fixed800)
            self.assertFalse(latest600.exists());self.assertTrue(fixed400.exists())
            self.assertIn(fixed400.name,recovered.data['progress']);self.assertIn(fixed800.name,recovered.data['progress'])
            state=torch.load(fixed800/'state.pt',weights_only=True)
            self.assertEqual(state['step'],800);self.assertIn('optimizer',state);self.assertIn('scheduler',state)
            self.assertIn('cpu_rng',state);self.assertEqual(len(state['rank_states']),2)
            self.assertTrue(read_json(fixed400/'checkpoint.json')['full_state'])
            # Retention limit applies to LoRA-only snapshots, never fixed fulls.
            recovered.save_progress(backend,optimizer,scheduler,generator,1300,cost)
            self.assertTrue(fixed400.exists());self.assertTrue(fixed800.exists())
            self.assertFalse((out/'checkpoints/step-0001300/state.pt').exists())
            # Older pending journals have no kind: full_state=True meant resume.
            name='resume-0001400-legacy';staging=recovered._begin(name,True)
            old=read_json(recovered.index);old['pending'].pop('kind');atomic_json(recovered.index,old)
            path=out/'checkpoints'/name
            save_checkpoint(path,backend,optimizer,scheduler,generator,1400,'test-run',cost,staging_path=staging)
            recovered=ManagedCheckpoints(out,'test-run',settings);recovered.recover()
            self.assertEqual(resolve_checkpoint(out/'checkpoints/latest.json'),path)
            self.assertTrue(fixed400.exists());self.assertTrue(fixed800.exists())


if __name__=='__main__':unittest.main()
