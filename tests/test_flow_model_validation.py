"""Model receipts validate component/config/header completeness without payload scans."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from validate_flow_model import main, validate_model
from self_correct_dpo.flow_receipts import model_receipt


def write(root,name,value):
    path = root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def tiny_model(root):
    write(root,'model_index.json',{'_class_name':'ZImagePipeline'})
    write(root,'transformer/config.json',{'_class_name':'ZImageTransformer2DModel'})
    write(root,'vae/config.json',{'_class_name':'AutoencoderKL'})
    write(root,'text_encoder/config.json',{'model_type':'qwen3'})
    write(root,'tokenizer/tokenizer_config.json',{'tokenizer_class':'Qwen2Tokenizer'})
    write(root,'tokenizer/vocab.json',{'a':1})
    (root/'tokenizer/merges.txt').write_text('a b\n')
    write(root,'scheduler/scheduler_config.json',{'_class_name':'FlowMatchEulerDiscreteScheduler','shift':6})
    header = json.dumps({'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    for name in ('transformer','vae','text_encoder'):
        (root/name/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)


class ModelValidation(unittest.TestCase):
    def test_local_identity_is_stable_and_receipt_does_not_include_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tiny_model(root)
            first=validate_model(root,'zimage',None)
            report=root/'flow-validation-report.json'
            report.write_text(json.dumps(first))
            second=validate_model(root,'zimage',None)
            self.assertEqual(first['revision'],second['revision'])
            self.assertFalse(second['upstream_revision_verified'])
            model_receipt(report,{'path':str(root),'family':'zimage','revision':first['revision']})
            self.assertFalse(second['scope']['weight_payload_hashed'])

    def test_missing_components_and_bad_shard_tensor_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tiny_model(root)
            (root/'text_encoder/config.json').unlink()
            with self.assertRaises(FileNotFoundError):validate_model(root,'zimage',None)
            write(root,'text_encoder/config.json',{'model_type':'qwen3'})
            write(root,'transformer/model.safetensors.index.json',{'weight_map':{'missing':'model.safetensors'}})
            with self.assertRaisesRegex(ValueError,'Indexed tensor missing'):validate_model(root,'zimage',None)

    def test_truncated_payload_is_rejected_and_download_manifest_revision_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tiny_model(root)
            write(root,'.download-manifest.json',{'status':'complete','revision':'verified-revision'})
            valid=validate_model(root,'zimage','verified-revision')
            self.assertTrue(valid['upstream_revision_verified'])
            with self.assertRaisesRegex(ValueError,'revision mismatch'):validate_model(root,'zimage','another')
            path=root/'vae/model.safetensors';path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'offsets'):validate_model(root,'zimage','verified-revision')

    def test_startup_rejects_changed_payload_without_reading_weights_or_walking_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tiny_model(root)
            validated=validate_model(root,'zimage',None)
            report=root/'flow-validation-report.json';report.write_text(json.dumps(validated))
            model={'path':str(root),'family':'zimage','revision':validated['revision']}
            original_open=Path.open
            def no_payload_read(path,*args,**kwargs):
                if path.suffix == '.safetensors':
                    raise AssertionError('Startup read tensor payload')
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'rglob',side_effect=AssertionError('Startup walked model')),patch.object(Path,'open',no_payload_read):
                model_receipt(report,model)
                weight=root/'transformer/model.safetensors'
                with open(weight,'ab') as handle:handle.write(b'changed')
                with self.assertRaisesRegex(ValueError,'artifact changed'):model_receipt(report,model)

    def test_failed_cli_revalidation_revokes_old_passed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tiny_model(root)
            report=root/'flow-validation-report.json'
            args=['validate_flow_model.py','--model',str(root),'--family','zimage']
            with patch.object(sys,'argv',args):self.assertEqual(main(),0)
            passed=json.loads(report.read_text())
            (root/'vae/model.safetensors').write_bytes(b'broken')
            with patch.object(sys,'argv',args):self.assertEqual(main(),1)
            failed=json.loads(report.read_text())
            self.assertEqual(failed['status'],'failed')
            self.assertFalse(failed['passed'])
            with self.assertRaisesRegex(ValueError,'passed flow_model_validation'):
                model_receipt(report,{'path':str(root),'family':'zimage','revision':passed['revision']})


if __name__=='__main__':unittest.main()
