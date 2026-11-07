"""Native receipts and plain HF folders must retain honest source provenance."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from self_correct_dpo.data.hpdv3_source import source_record, INVENTORY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def raw_folder(root, count=0):
    from PIL import Image
    raw = root / 'raw'; raw.mkdir()
    rows=[]; buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for i in range(count):
            row={'prompt':f'caption {i//2}', 'path1':f'images/{i}-a.png',
                 'path2':f'images/{i}-b.png', 'model1':'flux', 'model2':'kolors'}
            rows.append(row)
            for side,key in enumerate(('path1','path2')):
                image=io.BytesIO();Image.new('RGB',(32,32),(i,side,17)).save(image,format='PNG')
                payload=image.getvalue();info=tarfile.TarInfo(row[key]);info.size=len(payload)
                tar.addfile(info,io.BytesIO(payload))
    (raw/'train.json').write_text(json.dumps(rows));(raw/'test.json').write_text('[]')
    payload=buffer.getvalue();cut=len(payload)//2
    (raw/'images.tar.gz.00').write_bytes(payload[:cut]);(raw/'images.tar.gz.01').write_bytes(payload[cut:])
    return raw


class HPDv3SourceTests(unittest.TestCase):
    def test_plain_hf_inventory_is_honest_and_detects_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw=raw_folder(Path(temporary))
            cache=raw/'.cache/huggingface/download';cache.mkdir(parents=True)
            (cache/'train.json.metadata').write_text('a'*40+'\netag\n0\n')
            (raw/'images.tar.gz.00.incomplete').write_bytes(b'ignored')
            path,record=source_record(raw,create_inventory=True)
            self.assertEqual(path.name,INVENTORY);self.assertEqual(record['status'],'inventoried')
            self.assertNotIn('complete',record)
            self.assertFalse(record['upstream_revision_verified'])
            self.assertFalse(record['download_complete_verified'])
            self.assertEqual(record['hf_cache_commits_observed'],['a'*40])
            self.assertEqual(source_record(raw)[1]['revision'],record['revision'])
            (raw/'train.json').write_text('[ ]')
            with self.assertRaisesRegex(ValueError,'changed'):
                source_record(raw)

    def test_native_scd_receipt_and_incomplete_or_changed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw=raw_folder(Path(temporary));path=raw/'.scd-download.json'
            record={'repo_id':'MizzenAI/HPDv3','revision':'fixture','complete':True,
                    'files':{p.name:p.stat().st_size for p in raw.iterdir()}}
            path.write_text(json.dumps(record))
            self.assertEqual(source_record(raw)[0],path)
            record['complete']=False;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                source_record(raw,create_inventory=True)
            self.assertFalse((raw/INVENTORY).exists())
            record['complete']=True;path.write_text(json.dumps(record))
            (raw/'images.tar.gz.00').write_bytes(b'truncated')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                source_record(raw)

    def test_missing_middle_volume_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw=raw_folder(Path(temporary))
            (raw/'images.tar.gz.01').rename(raw/'images.tar.gz.02')
            with self.assertRaisesRegex(ValueError,'contiguous'):
                source_record(raw,create_inventory=True)
            self.assertFalse((raw/INVENTORY).exists())

    def test_truncated_payload_never_creates_extraction_completion(self):
        from extract_hpdv3_sub import extract
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw=raw_folder(root,2);out=root/'images-view'
            final=raw/'images.tar.gz.01';final.write_bytes(final.read_bytes()[:1])
            with self.assertRaises((EOFError,tarfile.ReadError,OSError)):
                extract(raw,out,reserve_gib=0)
            self.assertFalse((out/'extraction.json').exists())
            record=json.loads((raw/INVENTORY).read_text())
            self.assertEqual(record['status'],'inventoried')
            self.assertFalse(record['download_complete_verified'])
            self.assertNotIn('complete',record)

    def test_no_receipt_full_prepare_wrapper_and_reuse(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw=raw_folder(root,100)
            (raw/'images.tar.gz.00.incomplete').write_bytes(b'ignored leftover')
            resources=json.loads((ROOT/'configs/resources-slai-flow.json').read_text())
            resources.update(HPDV3_RAW=str(raw), HPDV3_SUB_PREPARED=str(root/'prepared'),
                             MONITOR_PROMPTS_FLOW_PILOT=str(root/'panel.jsonl'))
            config=root/'resources.json';config.write_text(json.dumps(resources))
            env={**os.environ,'SCD_RESOURCES':str(config),'SCD_FLOW_PYTHON':sys.executable,
                 'FLOW_IMAGES':str(root/'images-view'),'FLOW_RESERVE_GIB':'0','FLOW_MONITOR_COUNT':'1'}
            command=['bash',str(ROOT/'launchers/flow/prepare_slai_data.sh')]
            run=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
            self.assertIn('[DONE]',run.stdout)
            report=json.loads((root/'prepared/validation-report.json').read_text())
            self.assertTrue(report['passed'],report)
            self.assertEqual(report['counts']['usable_pairs'],100)
            self.assertEqual(report['source_identity']['kind'],'local_inventory')
            self.assertFalse(report['source_identity']['upstream_revision_verified'])
            self.assertFalse(report['source_identity']['download_complete_verified'])
            self.assertFalse((raw/'.download-manifest.json').exists())
            again=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(again.returncode,0,again.stdout+'\n'+again.stderr)
            self.assertIn('Reusing completed extraction',again.stdout)
            self.assertIn('Reusing completed preparation',again.stdout)
            # Source mutation invalidates reuse, and the wrapper names the failed stage.
            (raw/'train.json').write_text('[]')
            failed=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('stage=source',failed.stdout+failed.stderr)


if __name__=='__main__':unittest.main()
