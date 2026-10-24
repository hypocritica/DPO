"""Real-file preflight/cache/integrity contracts; standard library only."""
from collections import Counter
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.common import sha256
from self_correct_dpo.data.arrow_images import image_source
from self_correct_dpo.data.pairs import load_pairs
from self_correct_dpo.data.validation import load_training_pairs


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / 'pairs.jsonl'
        self.rows = []
        for i in range(24):
            row = {'pair_id': str(i), 'scene_id': f'scene-{i}',
                   'split': ('train', 'validation', 'test')[i % 3],
                   'prompts': [{'variant_id': 'canonical', 'prompt': f'caption {i}',
                                'template_family': 'canonical', 'transform_type': 'canonical'}]}
            for side in ('winner', 'loser'):
                payload = f'original image bytes: {i} {side}'.encode()
                name = hashlib.sha256(payload).hexdigest() + '.img'
                (self.root / name).write_bytes(payload)
                row[side] = name
                row[side + '_sha256'] = hashlib.sha256(payload).hexdigest()
            self.rows.append(row)
        self.save_manifest()
        self.cache = self.root / '.pairs.jsonl.validation.json'

    def save_manifest(self):
        self.manifest.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def ready(self):
        (self.root / 'ready.json').write_text(json.dumps({'pairs_sha256': sha256(self.manifest),
                   'splits': dict(Counter(r['split'] for r in self.rows)), 'source_revision': 'fixture@abc'}))

    def run_validation(self, **kwargs):
        output = io.StringIO()
        with redirect_stdout(output):
            rows = load_training_pairs(self.manifest, **kwargs)
        return rows, [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]

    def test_prepared_auto_samples_and_preserves_full_loader_rows(self):
        self.ready()
        expected = load_pairs(self.manifest)
        calls = []
        def tracked(path):
            calls.append(str(path))
            return sha256(path)
        with patch('self_correct_dpo.data.validation.sha256', side_effect=tracked):
            actual, logs = self.run_validation(samples=2)
        self.assertEqual(actual, expected)
        self.assertLessEqual(len(calls), 8)  # two random files plus two per split
        self.assertTrue(any(x.get('mode') == 'sampled' for x in logs))
        self.assertEqual(logs[-1]['phase'], 'data_ready')
        self.assertEqual(json.loads(self.cache.read_text())['audit'], 'sampled')

    def test_parallel_full_audit_then_cache_reuse_and_force_full(self):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        threads = set()
        def concurrent_hash(path):
            with lock:
                first = threading.get_ident() not in threads
                threads.add(threading.get_ident())
            if first:
                barrier.wait(timeout=5)
            return sha256(path)
        with patch('self_correct_dpo.data.validation.sha256', side_effect=concurrent_hash):
            expected, logs = self.run_validation(workers=2)
        self.assertEqual(len(threads), 2)
        self.assertTrue(any(x.get('mode') == 'full' for x in logs))
        actual, logs = self.run_validation(samples=1)
        self.assertEqual(actual, expected)
        self.assertTrue(any(x.get('evidence') == 'full-audit-cache' for x in logs))
        _, logs = self.run_validation(mode='full')
        self.assertTrue(any(x.get('mode') == 'full' for x in logs))

    def test_unsampled_corruption_is_rejected_when_image_is_consumed(self):
        self.ready()
        self.run_validation(samples=1)
        first_by_split = {0, 1, 2}
        # Pick a file absent from the deterministic startup sample.
        checked = []
        with patch('self_correct_dpo.data.validation.sha256', side_effect=lambda p: checked.append(str(p)) or sha256(p)):
            rows, _ = self.run_validation(samples=1)
        row = next(row for row in rows if int(row['pair_id']) not in first_by_split and row['winner'] not in checked)
        Path(row['winner']).write_bytes(b'corrupt')
        self.run_validation(samples=1)  # Quick startup is intentionally not a full scan.
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            image_source(row, 'winner', verify_files=True)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.run_validation(mode='full')
        self.assertFalse(self.cache.exists())

    def test_ready_manifest_and_split_changes_are_rejected(self):
        self.ready()
        self.rows[0]['prompts'][0]['prompt'] = 'changed'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'ready.json'):
            self.run_validation()
        self.ready()
        marker = json.loads((self.root / 'ready.json').read_text())
        marker['splits']['train'] += 1
        (self.root / 'ready.json').write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, 'ready.json'):
            self.run_validation()

    def test_cache_does_not_hide_leakage_duplicate_ids_or_missing_images(self):
        self.run_validation()
        original = copy.deepcopy(self.rows)
        for failure in ('leakage', 'duplicate'):
            self.rows = copy.deepcopy(original)
            if failure == 'leakage':
                self.rows[1]['scene_id'] = self.rows[0]['scene_id']
            else:
                self.rows[1]['pair_id'] = self.rows[0]['pair_id']
            self.save_manifest()
            with self.assertRaisesRegex(ValueError, 'leakage|Duplicate'):
                self.run_validation()
        self.rows = original
        self.save_manifest()
        self.ready()
        (self.root / self.rows[0]['winner']).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_validation(samples=1)

    def test_invalid_receipts_and_changed_manifest_fall_back_to_full(self):
        self.cache.write_text('{interrupted')
        self.cache.with_suffix('.partial').write_text('{not a committed receipt')
        _, logs = self.run_validation()
        self.assertTrue(any(x.get('mode') == 'full' for x in logs))
        self.rows[0]['prompts'][0]['prompt'] = 'new caption'
        self.save_manifest()
        _, logs = self.run_validation()
        self.assertTrue(any(x.get('mode') == 'full' for x in logs))

    def test_incomplete_preparation_cannot_be_accepted_as_a_cached_dataset(self):
        (self.root / 'preparing.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_validation()
        self.assertFalse(self.cache.exists())

    def test_missing_hashes_require_full_audit_and_recheck_image_isolation(self):
        for row in self.rows:
            row.pop('winner_sha256')
            row.pop('loser_sha256')
        self.save_manifest()
        expected = load_pairs(self.manifest)
        actual, _ = self.run_validation()
        self.assertEqual(actual, expected)
        self.assertFalse(self.cache.exists())
        # Distinct paths with identical bytes in different splits must still fail.
        (self.root / self.rows[1]['winner']).write_bytes((self.root / self.rows[0]['winner']).read_bytes())
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.run_validation()

    def test_source_revision_and_consumed_image_checks(self):
        for row in self.rows:
            row['human_preference'] = {'source_revision': 'fixture@other'}
        self.save_manifest()
        self.ready()
        with self.assertRaisesRegex(ValueError, 'source'):
            self.run_validation()
        row = {**self.rows[0], 'winner': str(self.root / self.rows[0]['winner'])}
        source = image_source(row, 'winner', verify_files=True)
        self.assertEqual(hashlib.sha256(source.read()).hexdigest(), row['winner_sha256'])

    def test_standalone_audit_cli_reuses_cache_and_reports_progress(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/check_training_data.py'
        for expected in ('full', 'sampled'):
            result = subprocess.run([sys.executable, str(script), '--pairs', str(self.manifest),
                                     '--workers', '2', '--samples', '1'],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            logs = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertTrue(any(x.get('mode') == expected for x in logs))
            self.assertEqual(logs[-1]['phase'], 'data_ready')


if __name__ == '__main__':
    unittest.main()
