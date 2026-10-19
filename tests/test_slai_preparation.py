"""Offline fixtures for pinned downloads, real Parquet preparation and relocation."""
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import download_assets as assets
import prepare_downloaded_data as prepare
from self_correct_dpo.common import sha256
from self_correct_dpo.config import load_config
from self_correct_dpo.data import load_pairs
from self_correct_dpo.pickapic import validate_human_pair


def png(i):
    from PIL import Image
    stream = BytesIO()
    Image.new('RGB', (3, 3), (i % 256, (i // 256) % 256, 80)).save(stream, format='PNG')
    return stream.getvalue()


def record(i, caption=None):
    return dict(ranking_id=i, caption=caption or f'scene {i}', label_0=1., label_1=0.,
                has_label=True, image_0_uid=f'a{i}', image_1_uid=f'b{i}',
                jpg_0=png(i * 2), jpg_1=png(i * 2 + 1))


class SlaiPreparationTests(unittest.TestCase):
    def test_asset_list_cli_and_placeholder_paths_fail_without_downloading(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/download_assets.py'),'list'],check=True,capture_output=True,text=True)
        self.assertIn('MizzenAI/HPDv3',result.stdout)
        with patch.dict(os.environ,{'SD15_MODEL':'/mnt/afs/CHANGE_ME/sd15'}):
            with self.assertRaisesRegex(ValueError,'paths.sh'):assets.destination('sd15')

    def test_download_selection_avoids_weight_duplicates_and_records_subsets(self):
        entries = [SimpleNamespace(rfilename=n, size=10) for n in (
            'model_index.json', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
            'unet/diffusion_pytorch_model.bin', 'unet/diffusion_pytorch_model.fp16.safetensors',
            'text_encoder/pytorch_model.bin', 'tokenizer/merges.txt', 'v1-5-pruned.ckpt')]
        files = assets.select_files('sd15', entries)
        self.assertIn('text_encoder/pytorch_model.bin', files)
        self.assertNotIn('unet/diffusion_pytorch_model.bin', files)
        self.assertNotIn('unet/diffusion_pytorch_model.fp16.safetensors', files)
        self.assertNotIn('v1-5-pruned.ckpt', files)
        entries = [SimpleNamespace(rfilename=n, size=10) for n in (
            'data/train-00000.parquet', 'data/train-00001.parquet', 'data/test_unique-00000.parquet')]
        files = assets.select_files('pickapic-v2', entries, train_shards=1)
        self.assertEqual(len(files), 2)
        self.assertIn('data/test_unique-00000.parquet', files)

    def test_planning_is_read_only_and_pins_existing_revision(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'SD15_MODEL':str(Path(d)/'model')}):
            info = SimpleNamespace(sha='a'*40, siblings=[SimpleNamespace(rfilename='model_index.json', size=2)])
            api = SimpleNamespace(repo_info=lambda *args, **kwargs: info)
            plan = assets.plan_asset('sd15', api=api)
            root = Path(plan['destination'])
            self.assertFalse(root.exists())
            root.mkdir()
            (root/'.scd-download.json').write_text(json.dumps({**plan, 'complete':False}))
            calls = []
            def query(*args, **kwargs):
                calls.append(kwargs['revision'])
                return info
            assets.plan_asset('sd15', api=SimpleNamespace(repo_info=query))
            self.assertEqual(calls, ['a'*40])
            with self.assertRaisesRegex(ValueError, 'Revision changed'):
                assets.plan_asset('sd15', revision='other', api=SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha='b'*40)))

    def test_preparation_preserves_votes_excludes_heldout_and_relocates(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            storage = Path(d)/'storage'
            raw = storage/'raw'
            raw.mkdir(parents=True)
            training = [record(i) for i in range(200)]
            training += [record(200, 'official only'), record(201, 'hpd prompt 0'), record(202, 'too long')]
            training.append({**record(203), 'jpg_0':png(9000)})
            training.append({**record(204), 'image_1_uid':'heldout-uid'})
            # A contradictory annotation quarantines both copies, with no majority relabeling.
            training.append({**training[0], 'ranking_id':205, 'label_0':0., 'label_1':1.})
            training.append({**record(206), 'label_0':.5, 'label_1':.5})
            training.append({**record(207), 'has_label':False})
            heldout = [record(5000+i, f'official prompt {i}') for i in range(130)]
            heldout += [{**record(6000, 'official only'), 'jpg_0':png(9000), 'image_1_uid':'heldout-uid'}]
            pq.write_table(pa.Table.from_pylist(training), raw/'train-00000.parquet')
            pq.write_table(pa.Table.from_pylist(heldout), raw/'test_unique-00000.parquet')
            hpd = storage/'test.json'
            hpd.write_text(json.dumps([{'prompt':f'hpd prompt {i}'} for i in range(400)]))
            out = storage/'prepared'/'sd15'
            prepare.prepare_pickapic(raw, out, storage/'images', family='sd15', model=storage/'model',
                                    hpdv2_test=hpd, revision='fixture-commit', allowed=lambda p:p!='too long')
            rows = load_pairs(out/'pairs.jsonl')
            for row in rows:
                validate_human_pair(row)
            ids = {r['pair_id'] for r in rows}
            self.assertTrue(ids.isdisjoint({'0','200','201','202','203','204','205','206','207'}))
            audit = json.loads((out/'audit.json').read_text())
            self.assertEqual(audit['exclusions']['conflicting_human_votes'], 2)
            self.assertEqual(audit['exclusions']['official_heldout_caption'], 2)
            self.assertEqual(audit['exclusions']['official_heldout_image'], 2)
            second = storage/'prepared'/'sdxl'
            prepare.prepare_pickapic(raw, second, storage/'images', family='sdxl', model=storage/'model',
                                    hpdv2_test=hpd, revision='fixture-commit', allowed=lambda p:p!='too long')
            self.assertEqual(json.loads((second/'audit.json').read_text())['images_written_bytes'], 0)
            panels = storage/'evaluation'
            prepare.prepare_panels(out/'official-test-unique.jsonl', hpd, panels, train_pairs=out/'pairs.jsonl')
            selection = json.loads((panels/'selection.json').read_text())
            self.assertEqual(selection['panels']['pickapic-monitor']['prompts'], 128)
            self.assertEqual(selection['panels']['hpdv2-test']['prompts'], 400)
            moved = Path(d)/'moved'
            storage.rename(moved)
            self.assertEqual(len(load_pairs(moved/'prepared/sd15/pairs.jsonl')), len(rows))

    def test_portable_config_uses_unified_paths_and_actual_model_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/'model';model.mkdir()
            (model/'.scd-download.json').write_text(json.dumps({'complete':True,'revision':'a'*40}))
            env={**os.environ, 'SDXL_MODEL':str(model), 'PICKAPIC_SDXL_PREPARED':str(root/'prepared'),
                 'MONITOR_PROMPTS':str(root/'monitor.jsonl'), 'SCORERS_CONFIG':str(root/'scorers.json')}
            config=root/'config.json'
            subprocess.run([sys.executable,str(ROOT/'scripts/make_config.py'),'--family','sdxl',
                            '--portable-paths','--out',str(config)],env=env,check=True,capture_output=True,text=True)
            saved=json.loads(config.read_text())
            self.assertEqual(saved['model']['path'],'${SDXL_MODEL}')
            self.assertEqual(saved['model']['revision'],'a'*40)
            with patch.dict(os.environ,env):
                resolved=load_config(config)
            self.assertEqual(resolved['data']['pairs'],str(root/'prepared/pairs.jsonl'))

    def test_hpdv3_preserves_original_preferred_path_and_confidence(self):
        from self_correct_dpo.common import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'hpdv3';images=raw/'images';images.mkdir(parents=True)
            annotations=[]
            for i in range(202):
                names=[f'images/{i}-{j}.png' for j in range(2)]
                for j,name in enumerate(names):(raw/name).write_bytes(png(2*i+j))
                annotations.append({'prompt':f'hpdv3 scene {i}','path1':names[0],'path2':names[1],
                                    'confidence':None if i%2 else .98,'choice_dist':[3,1],'model1':'sd3','model2':'flux'})
            (raw/'test.json').write_text(json.dumps(annotations[-2:]))
            (raw/'train.json').write_text(json.dumps(annotations[:200]+[annotations[-1]]))
            out=root/'prepared'
            prepare.prepare_hpdv3(raw,out,family='unfiltered',revision='official-fixture-revision')
            rows=load_pairs(out/'pairs.jsonl')
            for row in rows:validate_human_pair(row)
            self.assertEqual(len(rows),200)
            evidence=rows[0]['human_preference']
            self.assertEqual(evidence['original_record']['choice_dist'],[3,1])
            self.assertEqual(evidence['preferred_field'],'path1')
            self.assertEqual(evidence['source'],'hpdv3')
            evidence['original_record']['path1']='images/other.png'
            with self.assertRaisesRegex(ValueError,'HPDv3 evidence'):validate_human_pair(rows[0])

    def test_hpdv3_volume_extraction_concatenates_without_intermediate_archive(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'HPDV3_RAW':d}):
            root=Path(d)
            stream=BytesIO()
            with tarfile.open(fileobj=stream,mode='w:gz') as tar:
                member=tarfile.TarInfo('images/sample.png');member.size=len(png(1))
                tar.addfile(member,BytesIO(png(1)))
            payload=stream.getvalue();middle=len(payload)//2
            files={}
            for i,part in enumerate((payload[:middle],payload[middle:])):
                name=f'images.tar.gz.{i:02d}';(root/name).write_bytes(part);files[name]=len(part)
            (root/'.scd-download.json').write_text(json.dumps({'complete':True,'files':files}))
            assets.extract_hpdv3()
            self.assertEqual((root/'images/sample.png').read_bytes(),png(1))
            self.assertFalse((root/'images.tar.gz').exists())
            with self.assertRaises(FileExistsError):assets.extract_hpdv3()

    def test_extraction_rejects_links_and_parent_traversal(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'HPDV2_RAW':d}):
            archive=Path(d)/'train.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                member=tarfile.TarInfo('train/../../outside')
                member.size=1
                tar.addfile(member,BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                assets.extract_hpdv2()
            self.assertFalse((Path(d)/'train').exists())


if __name__ == '__main__':
    unittest.main()
