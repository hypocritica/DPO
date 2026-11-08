"""Receipt binding and hardware/retention contracts for the new paper recipe."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from self_correct_dpo.common import artifact_identity, sha256
from self_correct_dpo.flow_receipts import data_receipt, file_binding, model_receipt, load_certified_pairs
from self_correct_dpo.runtime.checkpoints import progress_steps

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_flow_receipt_test', ROOT/'scripts/run_flow.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReceiptTests(unittest.TestCase):
    def write(self, path, data):
        path.write_text(json.dumps(data))

    def fixture(self, root):
        pairs = root/'pairs.jsonl'
        self.write(pairs, {'winner':'winner.png', 'loser':'loser.png'})
        self.write(root/'ready.json', {'pairs_sha256':sha256(pairs), 'source_revision':'verified-rev'})
        self.write(root/'audit.json', {'passed':True})
        report = {'schema_version':1, 'kind':'flow_data_validation', 'passed':True, 'status':'passed',
                  'scope':{'manifest_checks':True, 'image_integrity':'verified_during_preparation'},
                  'source_revision':'verified-rev', 'dataset_identity':{
                      'pairs':file_binding(pairs), 'ready':file_binding(root/'ready.json'),
                      'audit':file_binding(root/'audit.json')}}
        self.write(root/'validation-report.json', report)
        return pairs, root/'validation-report.json', report

    def test_passed_receipt_resolves_paths_without_image_reads(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); pairs, receipt, _ = self.fixture(root)
            rows=load_certified_pairs(pairs,receipt)
            self.assertEqual(rows[0]['winner'],str(root/'winner.png'))
            self.assertFalse((root/'winner.png').exists())

    def test_failed_schema_path_stat_and_ready_hash_rejected(self):
        for mutation in ('passed','schema','path','stat','ready_hash'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as t:
                root=Path(t); pairs, receipt, report = self.fixture(root)
                if mutation=='passed': report['passed']=False
                elif mutation=='schema': report['schema_version']=2
                elif mutation=='path':
                    other=root/'other.jsonl'; other.write_bytes(pairs.read_bytes())
                    report['dataset_identity']['pairs']=file_binding(other)
                elif mutation=='stat': pairs.write_text(pairs.read_text()+'\n')
                elif mutation=='ready_hash':
                    ready=root/'ready.json'; stat=ready.stat()
                    ready.write_text(ready.read_text().replace('verified-rev','modified-rev'))
                    os.utime(ready,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                self.write(receipt,report)
                with self.assertRaises(ValueError): data_receipt(receipt,pairs)

    def test_model_config_hash_and_identity_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); config=root/'model_index.json'; self.write(config,{'a':1})
            model={'family':'zimage','revision':'verified-rev','path':str(root)}
            report={'schema_version':1,'kind':'flow_model_validation','passed':True,'status':'passed',
                    'family':model['family'],'revision':model['revision'],'model_path':str(root),
                    'key_files':[file_binding(config)],'artifact_identity':artifact_identity(root)}
            path=root/'flow-validation-report.json'; self.write(path,report)
            self.assertEqual(model_receipt(path,model)['status'],'passed')
            wrong=copy.deepcopy(model); wrong['revision']='another-rev'
            with self.assertRaises(ValueError): model_receipt(path,wrong)
            stat=config.stat(); self.write(config,{'a':2}); os.utime(config,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            with self.assertRaises(ValueError): model_receipt(path,model)

    def test_metadata_rebind_checks_bytes_preserves_original_and_is_readonly_by_default(self):
        from rebind_flow_data_receipt import verify_and_rebind
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); pairs, receipt, report=self.fixture(root)
            original=receipt.read_bytes()
            for name in ('pairs.jsonl','ready.json','audit.json'):
                path=root/name; stat=path.stat()
                os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns//10**9*10**9))
            before={p.name:p.read_bytes() for p in root.iterdir()}
            result=verify_and_rebind(pairs,receipt)
            self.assertTrue(result['metadata_changed'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            out=root/'new-receipt.json'
            result=verify_and_rebind(pairs,receipt,out)
            self.assertEqual(result['status'],'rebound')
            self.assertEqual(receipt.read_bytes(),original)
            self.assertEqual(data_receipt(out,pairs)['dataset_identity']['pairs']['sha256'],sha256(pairs))
            self.assertEqual(load_certified_pairs(pairs,out)[0]['winner'],str(root/'winner.png'))
            self.assertFalse((root/'winner.png').exists())
            saved=out.read_bytes()
            self.assertEqual(verify_and_rebind(pairs,receipt,out)['status'],'reused')
            self.assertEqual(out.read_bytes(),saved)
            unrelated=root/'unrelated.json';unrelated.write_bytes(original)
            with self.assertRaises((ValueError,FileExistsError)):verify_and_rebind(pairs,receipt,unrelated)
            self.assertEqual(unrelated.read_bytes(),original)

    def test_rebind_refuses_changed_content_even_with_original_size_and_mtime(self):
        from rebind_flow_data_receipt import verify_and_rebind
        for name in ('pairs','ready','audit'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);pairs,receipt,report=self.fixture(root)
                path=Path(report['dataset_identity'][name]['path']);stat=path.stat()
                value=bytearray(path.read_bytes());value[-2]^=1;path.write_bytes(value)
                os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,f'SHA256 differs for {name}'):
                    verify_and_rebind(pairs,receipt,root/'new.json')
                self.assertFalse((root/'new.json').exists())

    def test_rebind_refuses_a_file_that_changes_during_hashing(self):
        from rebind_flow_data_receipt import checked_binding
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'file';path.write_text('a')
            def change(value):
                value.write_text('different size')
                return 'a'*64
            with patch('self_correct_dpo.common.sha256',side_effect=change):
                with self.assertRaisesRegex(ValueError,'changed while computing'):
                    checked_binding(path)


class PaperRecipeTests(unittest.TestCase):
    def recipe(self,family='sd35_medium',**kw):
        defaults=dict(recipe='paper-v2',world=4,profile='5090',batch_size=1,budget_unit='pair_views')
        defaults.update(kw)
        return runner.build_config(family,defaults.pop('algorithm','mvpo'),**defaults)

    def test_hardware_budget_and_retention_matrix(self):
        for family in ('sd35_medium','zimage'):
            for profile,world in (('5090',4),('h100',2),('a800',2)):
                for k in (2,4):
                    with self.subTest(family=family,profile=profile,k=k):
                        dtype='bfloat16' if family=='zimage' and profile=='5090' else None
                        c=self.recipe(family,profile=profile,world=world,views=k,weight_dtype=dtype)
                        t=c['training']
                        self.assertEqual(t['batch_size']*world*t['gradient_accumulation_steps']*k,128)
                        self.assertEqual(t['checkpoint_every'],200)
                        self.assertEqual(t['full_checkpoint_every'],400)
                        self.assertTrue(set(range(400,3201,400)).issubset(progress_steps(t)))
                        self.assertEqual(c['data']['validation'],'none')
                        if family=='zimage': self.assertEqual(c['model']['transformer_dtype'],dtype or 'float32')

    def test_b_stage_and_precision_contracts(self):
        self.assertEqual(self.recipe(aggregation='B')['method'],'multiview-B-gaussian')
        for boundary in (250,500,750):
            self.assertEqual(self.recipe(stagewise=True,boundary=boundary)['training']['stagewise']['boundary'],boundary)
        self.assertEqual(self.recipe(aggregation='B',stagewise=True)['method'],'stagewise-B-gaussian')
        with self.assertRaises(ValueError): self.recipe(budget=127)
        with self.assertRaises(ValueError): self.recipe('zimage')
        for algorithm in ('dpo','linear_dpo','sft'):
            c=self.recipe(algorithm=algorithm)
            self.assertEqual(c['training']['noise_views'],1)
            self.assertEqual(c['training']['reference_mode'],'ema' if algorithm=='linear_dpo' else ('none' if algorithm=='sft' else 'base'))


if __name__ == '__main__': unittest.main()
