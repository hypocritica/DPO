"""Small real-Parquet checks for interrupted export, legacy adoption and integrity."""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_downloaded_data as prep
from self_correct_dpo.data.preparation import ImagePool, BatchJournal, parquet_batches


class PreparationTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw'
        self.raw.mkdir()
        self.images = self.root/'images'
        self.out = self.root/'prepared'
        self.model = self.root/'model'
        (self.model/'tokenizer').mkdir(parents=True)
        (self.model/'tokenizer/vocab.json').write_text('{}')
        def image(n):
            buf = BytesIO()
            Image.new('RGB', (4, 4), (n % 256, n // 256, n % 17)).save(buf, format='PNG')
            return buf.getvalue()
        self.rows = [dict(ranking_id=f'p{i}', caption=f'caption {i}', label_0=1., label_1=0.,
                          has_label=True, image_0_uid=f'u{2*i}', image_1_uid=f'u{2*i+1}',
                          jpg_0=image(2*i), jpg_1=image(2*i+1)) for i in range(180)]
        heldout = {**self.rows[-1], 'ranking_id':'official-test', 'caption':'official heldout'}
        # Caption, UID and hash overlap are independently excluded before splitting.
        self.rows[2]['caption'] = heldout['caption']
        self.rows[3]['image_0_uid'] = heldout['image_0_uid']
        self.rows[4]['jpg_0'] = heldout['jpg_0']
        self.rows[5]['label_0'] = self.rows[5]['label_1'] = .5
        self.rows[6]['has_label'] = False
        self.rows[7]['jpg_0'] = b'undecodable-image'
        for index, rows in enumerate((self.rows[:90], self.rows[90:])):
            pq.write_table(pa.Table.from_pylist(rows), self.raw/f'train-{index:05d}.parquet', row_group_size=30)
        pq.write_table(pa.Table.from_pylist([heldout]), self.raw/'test_unique-00000.parquet')

    def run_prep(self, **kwargs):
        options = dict(family='sd15', model=self.model, revision='fixture@abc',
                       allowed=lambda p: True, workers=4, batch_size=11)
        options.update(kwargs)
        with redirect_stdout(StringIO()):
            prep.prepare_pickapic(self.raw, self.out, self.images, **options)

    def interrupt_after(self, count):
        original = BatchJournal.commit
        def interrupt(journal, batch):
            original(journal, batch)
            if journal.count == count:
                raise KeyboardInterrupt('simulated interruption after atomic commit')
        return patch.object(BatchJournal, 'commit', interrupt)

    def test_resume_skips_committed_rows_and_preserves_images_and_splits(self):
        with self.interrupt_after(10), self.assertRaises(KeyboardInterrupt):
            self.run_prep()
        saved = [json.loads(p.read_text()) for p in sorted((self.out/'.preparation').glob('batch-*.json'))]
        committed = {row['caption'] for b in saved for row in b['rows']}
        mtimes = {p:p.stat().st_mtime_ns for p in self.images.rglob('*.img')}
        seen = []
        self.run_prep(resume=True, workers=2, batch_size=17, allowed=lambda caption: seen.append(caption) or True)
        self.assertFalse(committed & set(seen))
        self.assertEqual(mtimes, {p:p.stat().st_mtime_ns for p in mtimes})
        resumed = (self.out/'pairs.jsonl').read_bytes()
        excluded = (self.out/'excluded.jsonl').read_bytes()
        # Replay finalization after a partial output and compare exact bytes.
        (self.out/'ready.json').unlink()
        (self.out/'pairs.jsonl').write_text('interrupted final output')
        self.run_prep(resume=True)
        self.assertEqual(resumed, (self.out/'pairs.jsonl').read_bytes())
        # An uninterrupted single-thread run yields identical labels and components.
        self.out = self.root/'baseline'
        self.run_prep(workers=1)
        self.assertEqual(resumed, (self.out/'pairs.jsonl').read_bytes())
        self.assertEqual(excluded, (self.out/'excluded.jsonl').read_bytes())
        audit = json.loads((self.out/'audit.json').read_text())
        self.assertEqual(audit['raw_train_rows'], 180)
        self.assertGreaterEqual(audit['exclusions']['official_heldout_image'], 3)

    def test_adopts_legacy_images_without_rewriting_and_detects_corruption(self):
        pool = ImagePool(self.images)
        for row in self.rows[:30]:
            if row['ranking_id'] == 'p7':
                continue
            for side in (0, 1):
                pool.store(row[f'jpg_{side}'])
        mtimes = {p:p.stat().st_mtime_ns for p in self.images.rglob('*.img')}
        self.out.mkdir()
        (self.out/'preparing.json').write_text(json.dumps({'source':str(self.raw.resolve()), 'revision':'fixture@abc', 'family':'sd15'}))
        with self.assertRaisesRegex(ValueError, 'adopt-legacy'):
            self.run_prep(resume=True)
        self.run_prep(adopt_legacy=True)
        self.assertEqual(mtimes, {p:p.stat().st_mtime_ns for p in mtimes})
        self.assertGreater(json.loads((self.out/'audit.json').read_text())['reused_images'], 0)
        image = Path(pool.store(self.rows[0]['jpg_0'])[0])
        image.write_bytes(b'corrupt')
        self.out = self.root/'corrupt-run'
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.run_prep()
        self.assertFalse((self.out/'ready.json').exists())

    def test_resume_refuses_changed_inputs_and_duplicate_ids_across_checkpoint(self):
        with self.interrupt_after(1), self.assertRaises(KeyboardInterrupt):
            self.run_prep()
        (self.model/'tokenizer/vocab.json').write_text('{"changed":1}')
        with self.assertRaisesRegex(ValueError, 'inputs/code changed'):
            self.run_prep(resume=True)
        (self.model/'tokenizer/vocab.json').write_text('{}')
        real_batches = parquet_batches
        def duplicate(*args, **kwargs):
            for records, cursor in real_batches(*args, **kwargs):
                records[0]['ranking_id'] = 'p0'
                yield records, cursor
        with patch.object(prep, 'parquet_batches', duplicate):
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                self.run_prep(resume=True)

    def test_uncommitted_batch_reuses_orphan_images_and_ignores_partial_journal(self):
        original = BatchJournal.commit
        def interrupt_before_commit(journal, batch):
            if journal.count == 1:
                (journal.root/'batch-00000001.json.partial').write_text('{unfinished')
                raise KeyboardInterrupt('images written, metadata not committed')
            original(journal, batch)
        with patch.object(BatchJournal, 'commit', interrupt_before_commit), self.assertRaises(KeyboardInterrupt):
            self.run_prep()
        mtimes = {p:p.stat().st_mtime_ns for p in self.images.rglob('*.img')}
        seen = []
        self.run_prep(resume=True, allowed=lambda caption: seen.append(caption) or True)
        self.assertIn('caption 11', seen)
        self.assertNotIn('caption 0', seen)
        self.assertEqual(mtimes, {p:p.stat().st_mtime_ns for p in mtimes})
        self.assertTrue((self.out/'ready.json').exists())


if __name__ == '__main__':
    unittest.main()
