import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from prepare_hpdv3_sub import select, ALLOWED_MODELS


class HPDv3SubTests(unittest.TestCase):
    def test_both_models_labels_original_paths_and_normalized_heldout(self):
        row={'prompt':'A caption','path1':'assets/preferred.jpg','path2':'assets/rejected.jpg',
             'model1':'flux','model2':'real_images','confidence':.8}
        rows=[row,{**row,'model2':'sdxl'},{**row,'model1':'sdxl'},
              {**row,'prompt':' HELDOUT   prompt '}, {**row,'prompt':'tiny32x32','model1':'sd3'}]
        chosen,excluded=select(rows,[{'prompt':'heldout prompt'}])
        self.assertEqual([r['source_index'] for r in chosen],[0,4])
        self.assertEqual(chosen[0]['record'],row)
        self.assertEqual(excluded,{'model_subset':2,'official_heldout_prompt':1})

    def test_streamed_arrays_chunk_boundaries_and_invalid_input(self):
        from self_correct_dpo.data.json_stream import object_records
        rows=[{'prompt':'中文 \" quote } ] \\ '+('x'*1000),'nested':[1,{'ok':True}]},
              {'prompt':'second','confidence':None}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train.json'
            for content in (json.dumps(rows,ensure_ascii=False),json.dumps(rows), ' [ ] \n'):
                path.write_text(content)
                self.assertEqual(list(object_records(path,chunk_size=7)),json.loads(content))
            for content in ('', '{}', '[{}', '[{},]', '[{}] junk', '[1]', '[{} {}]', '[{"x":"bad'):
                path.write_text(content)
                with self.assertRaises(ValueError,msg=content):list(object_records(path,chunk_size=3))
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'limit'):
                list(object_records(path,chunk_size=17,max_record_chars=100))

    def test_streaming_selection_discards_large_excluded_source(self):
        import tracemalloc
        from collections import Counter
        from prepare_hpdv3_sub import iter_selected
        from self_correct_dpo.data.json_stream import object_records
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train.json'
            bad={'prompt':'x'*512,'model1':'excluded','model2':'flux'}
            good={'prompt':'selected','model1':'flux','model2':'kolors',
                  'path1':'images/a.jpg','path2':'images/b.jpg'}
            with path.open('w') as handle:
                handle.write('[')
                for _ in range(12000):handle.write(json.dumps(bad)+',')
                handle.write(json.dumps(good)+']')
            counts=Counter()
            tracemalloc.start()
            try:
                chosen=list(iter_selected(object_records(path),[],counts))
                _,peak=tracemalloc.get_traced_memory()
            finally:tracemalloc.stop()
            self.assertEqual(chosen,[{'source_index':12000,'record':good}])
            self.assertEqual(counts,{'model_subset':12000})
            self.assertLess(peak,2*1024**2)

    def test_full_prepare_small_images_with_prompt_component_splits(self):
        from PIL import Image
        from prepare_downloaded_data import prepare_hpdv3
        from self_correct_dpo.common import read_jsonl,sha256
        from self_correct_dpo.data.pickapic import prompt_key
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';(raw/'assets').mkdir(parents=True)
            rows=[]
            # Deliberately below 1024: no unsolicited native-resolution filter.
            for i in range(100):
                for side in (1,2):Image.new('RGB',(32,32),(i,side,17)).save(raw/'assets'/f'{i}-{side}.png')
                rows.append({'prompt':f'caption {i//2}','path1':f'assets/{i}-1.png',
                    'path2':f'assets/{i}-2.png','model1':'flux','model2':'kolors'})
            (raw/'train.json').write_text(json.dumps(rows));(raw/'test.json').write_text('[]')
            output=root/'prepared'
            prepare_hpdv3(raw,output,family='unfiltered',revision='fixture',allowed_models=ALLOWED_MODELS)
            pairs=read_jsonl(output/'pairs.jsonl');self.assertEqual(len(pairs),100)
            splits={}
            for pair in pairs:
                key=prompt_key(pair['prompts'][0]['prompt'])
                self.assertEqual(splits.setdefault(key,pair['split']),pair['split'])
                original=pair['human_preference']['original_record']
                self.assertEqual((output/pair['winner']).resolve(),(raw/original['path1']).resolve())
                self.assertEqual((output/pair['loser']).resolve(),(raw/original['path2']).resolve())
            self.assertEqual(set(splits.values()),{'train','validation','test'})
            ready=json.loads((output/'ready.json').read_text());self.assertEqual(ready['pairs_sha256'],sha256(output/'pairs.jsonl'))
            audit=json.loads((output/'audit.json').read_text());self.assertEqual(set(audit['allowed_models']),set(ALLOWED_MODELS))
            parallel=root/'parallel'
            prepare_hpdv3(raw,parallel,family='unfiltered',revision='fixture',allowed_models=ALLOWED_MODELS,workers=4)
            self.assertEqual((output/'pairs.jsonl').read_bytes(),(parallel/'pairs.jsonl').read_bytes())
            self.assertEqual((output/'excluded.jsonl').read_bytes(),(parallel/'excluded.jsonl').read_bytes())

    def test_parallel_split_archive_resume_and_failure(self):
        from extract_hpdv3_sub import extract, _write_image
        from self_correct_dpo.common import output_lock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';raw.mkdir();out=root/'out'
            row={'prompt':'pair','path1':'images/a.jpg','path2':'images/b.jpg',
                 'model1':'flux','model2':'kolors'}
            (raw/'train.json').write_text(json.dumps([row]));(raw/'test.json').write_text('[]')
            (raw/'.download-manifest.json').write_text('{"repo_id":"fixture/HPDv3","revision":"fixture","status":"complete"}')
            payloads={'images/a.jpg':b'a'*12345,'images/b.jpg':b'b'*23456,'images/unused.jpg':b'ignored'}
            buffer=BytesIO()
            with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
                for name,payload in payloads.items():
                    info=tarfile.TarInfo('./'+name);info.size=len(payload)
                    archive.addfile(info,BytesIO(payload))
            data=buffer.getvalue();cut=len(data)//2
            (raw/'images.tar.gz.00').write_bytes(data[:cut]);(raw/'images.tar.gz.01').write_bytes(data[cut:])
            (out/'images').mkdir(parents=True)
            existing=out/'images/a.jpg';existing.write_bytes(payloads['images/a.jpg'])
            mtime=existing.stat().st_mtime_ns
            (out/'images/b.jpg.partial').write_bytes(b'interrupted')
            with output_lock(out):
                with self.assertRaisesRegex(RuntimeError,'Another process'):
                    extract(raw,out,reserve_gib=0)
            def fail_second(target,payload):
                if target.name=='b.jpg':raise OSError('simulated write failure')
                return _write_image(target,payload)
            with patch('extract_hpdv3_sub._write_image',side_effect=fail_second):
                with self.assertRaisesRegex(OSError,'simulated write failure'):
                    extract(raw,out,reserve_gib=0,workers=4)
            self.assertFalse((out/'extraction.json').exists())
            extract(raw,out,reserve_gib=0,workers=4)
            self.assertEqual(existing.stat().st_mtime_ns,mtime)
            self.assertEqual((out/'images/b.jpg').read_bytes(),payloads['images/b.jpg'])
            self.assertFalse((out/'images/b.jpg.partial').exists())
            self.assertFalse((out/'images/unused.jpg').exists())
            report=json.loads((out/'extraction.json').read_text())
            self.assertEqual((report['images'],report['reused_images'],report['written_images']),(2,1,1))


if __name__=='__main__':unittest.main()
