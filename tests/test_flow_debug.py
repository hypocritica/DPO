"""Concurrent debug snapshots and longer smoke configurations stay separate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_flow_debug_data import prepare
from run_flow import build_config
from self_correct_dpo.common import read_jsonl,sha256
from self_correct_dpo.data.validation import load_training_pairs


class FlowDebugTests(unittest.TestCase):
    def test_snapshot_available_pairs_excludes_heldout_and_leaves_source_untouched(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();images=root/'extract';(images/'images').mkdir(parents=True)
            rows=[]
            for i in range(140):
                row={'prompt':f'caption {i}','path1':f'images/{i}-1.png','path2':f'images/{i}-2.png',
                     'model1':'flux','model2':'kolors','confidence':0.9,'custom':{'original':i}}
                rows.append(row)
                for side in (1,2):
                    path=images/f'images/{i}-{side}.png'
                    if i != 0:Image.new('RGB',(16,16),(i,side,17)).save(path)
            tests=[{'prompt':rows[1]['prompt'],'path1':rows[2]['path1'],'path2':'images/test.png'}]
            (raw/'train.json').write_text(json.dumps(rows));(raw/'test.json').write_text(json.dumps(tests))
            (images/'images/0-1.png.partial').write_bytes(b'not a completed file')
            before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for folder in (raw,images) for p in folder.rglob('*') if p.is_file()}
            out=root/'debug';report=prepare(raw,images,out,count=128)
            self.assertFalse(report['ready_for_paper']);self.assertTrue(report['ready_for_debug'])
            self.assertFalse(report['official_test_image_hash_audited'])
            self.assertFalse((out/'prepared/validation-report.json').exists())
            pairs=load_training_pairs(out/'prepared/pairs.jsonl',mode='full')
            self.assertEqual(len(pairs),128);self.assertEqual({r['split'] for r in pairs},{'train','validation','test'})
            selected=read_jsonl(out/'selection.jsonl')
            self.assertEqual(selected[0]['source_index'],3)
            self.assertEqual(selected[0]['record'],rows[3])
            for pair in pairs:
                self.assertNotIn(pair['prompts'][0]['prompt'],{'caption 1','caption 2'})
                self.assertTrue(Path(pair['winner']).is_relative_to(out))
                self.assertTrue(Path(pair['winner']).is_file())
                self.assertTrue(pair['human_preference']['source_revision'].startswith('debug-snapshot:'))
            for name,(payload,mtime) in before.items():
                self.assertEqual(Path(name).read_bytes(),payload)
                self.assertEqual(Path(name).stat().st_mtime_ns,mtime)
            # Freeze a snapshot even as original annotations/extraction progress.
            checksum=sha256(out/'prepared/pairs.jsonl')
            prepare(raw,images,out,count=128)
            self.assertEqual(sha256(out/'prepared/pairs.jsonl'),checksum)
            with self.assertRaisesRegex(ValueError,'differs'):prepare(raw,images,out,count=64)
            with self.assertRaisesRegex(RuntimeError,'complete pairs'):
                prepare(raw,images,root/'insufficient',count=256)
            self.assertFalse((root/'insufficient').exists())

    def test_longer_smoke_keeps_formal_recipe_and_precision_unchanged(self):
        self.assertEqual(build_config('zimage','dpo',smoke=True)['training']['max_steps'],2)
        for family,profile,world in (('sd35_medium','5090',4),('zimage','h100',2)):
            for algorithm in ('dpo','mvpo','linear_dpo','sft'):
                c=build_config(family,algorithm,smoke=True,smoke_steps=50,profile=profile,world=world)
                t=c['training']
                self.assertEqual(t['max_steps'],50);self.assertEqual(t['checkpoint_every'],10)
                self.assertEqual(t['resolution'],1024);self.assertEqual(t['batch_size'],1)
                self.assertEqual(t['gradient_accumulation_steps'],1);self.assertFalse(c['evaluation']['enabled'])
                if family=='zimage':self.assertEqual(c['model']['transformer_dtype'],'float32')
        for options in ({'smoke':True,'smoke_steps':0},{'smoke':True,'smoke_steps':201},{'smoke_steps':50}):
            with self.assertRaises(ValueError):build_config('zimage','dpo',**options)
        c=build_config('zimage','dpo',recipe='paper-v2',profile='h100')
        self.assertEqual(c['training']['max_steps'],3200)
        self.assertEqual(c['training']['gradient_accumulation_steps'],64)

    def test_real_debug_shell_config_only_profiles_logging_and_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            env={**os.environ,'SCD_FLOW_PYTHON':sys.executable,'SCD_RESOURCES':str(ROOT/'configs/resources-slai-flow.json')}
            for family,profile,world in (('sd35_medium','5090',4),('zimage','h100',2)):
                command=['bash',str(ROOT/'launchers/flow/debug_train.sh'),family,'dpo',
                         '--config-only','--config-dir',str(root/family),'--runs-root',str(root/'runs'),
                         '--stop-after-step','25']
                run=subprocess.run(command,capture_output=True,text=True,env=env)
                self.assertEqual(run.returncode,0,run.stderr)
                config=next((root/family).glob('*.json'));c=json.loads(config.read_text())
                self.assertIn(f'{world}{profile}',config.name);self.assertIn('smoke50',config.name)
                self.assertEqual(c['training']['world_size'],world)
                self.assertEqual(c['training']['max_steps'],50)
                self.assertEqual(c['training']['logging'],'all')
                self.assertEqual(c['training']['wandb']['mode'],'offline')
                self.assertEqual(c['training']['wandb']['project'],'mvpo-flow-debug')
                self.assertIn('flow-debug-v1',c['data']['pairs'])
                self.assertNotIn('validation_report',c['data'])
                self.assertEqual(c['training']['offload_frozen'],family=='sd35_medium')
            formal=subprocess.run(['bash',str(ROOT/'launchers/zimage/01_dpo_k1.sh'),
                                   '--config-only','--config-dir',str(root/'formal'),
                                   '--runs-root',str(root/'runs')],env=env,capture_output=True,text=True)
            self.assertEqual(formal.returncode,0,formal.stderr)
            config=next((root/'formal').glob('*.json'))
            self.assertIn('2h100',config.name)
            self.assertEqual(json.loads(config.read_text())['training']['max_steps'],3200)



if __name__=='__main__':unittest.main()
