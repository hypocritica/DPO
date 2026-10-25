"""Real-file sampled preflight and explicit full-integrity contracts; standard library only."""
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
        self.assertFalse(self.cache.exists())

    def test_explicit_full_audit_reads_every_image_on_two_threads(self):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        threads, checked = set(), []
        def concurrent_hash(path):
            with lock:
                first = threading.get_ident() not in threads
                threads.add(threading.get_ident())
                checked.append(str(path))
            if first:
                barrier.wait(timeout=5)
            return sha256(path)
        with patch('self_correct_dpo.data.validation.sha256', side_effect=concurrent_hash):
            actual, logs = self.run_validation(mode='full', workers=2)
        self.assertEqual(actual, load_pairs(self.manifest))
        self.assertEqual(len(threads), 2)
        self.assertEqual(len(checked), 48)
        self.assertTrue(any(x.get('mode') == 'full' for x in logs))
        self.assertFalse(self.cache.exists())

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

    def test_manifest_checks_reject_leakage_duplicate_ids_or_sampled_missing_image(self):
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

    def test_auto_without_ready_or_with_stale_cache_stays_sampled(self):
        for legacy_cache in (None, '{interrupted', '{"complete": true, "audit": "full"}'):
            with self.subTest(legacy_cache=legacy_cache):
                if legacy_cache is not None:
                    self.cache.write_text(legacy_cache)
                checked = []
                with patch('self_correct_dpo.data.validation.sha256',
                           side_effect=lambda p: checked.append(str(p)) or sha256(p)):
                    _, logs = self.run_validation()
                self.assertLessEqual(len(checked), 14)  # default 8 images plus each split's first pair
                self.assertGreaterEqual(len(checked), 6)
                self.assertTrue(any(x.get('mode') == 'sampled' for x in logs))
                self.assertFalse(any(x.get('mode') == 'full' for x in logs))
                if legacy_cache is None:
                    self.assertFalse(self.cache.exists())
                else:
                    self.assertEqual(self.cache.read_text(), legacy_cache)
                # A changed manifest without ready.json still receives only a bounded sample.
                self.rows[0]['prompts'][0]['prompt'] += ' changed'
                self.save_manifest()

    def test_incomplete_preparation_is_rejected(self):
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
        with patch('self_correct_dpo.data.validation.sha256', wraps=sha256) as hashed:
            with self.assertRaisesRegex(ValueError, 'full|prepar'):
                self.run_validation()
            self.assertEqual(hashed.call_count, 0)
            actual, logs = self.run_validation(mode='full')
            self.assertEqual(hashed.call_count, 48)
        self.assertEqual(actual, expected)
        self.assertTrue(any(x.get('mode') == 'full' for x in logs))
        self.assertFalse(self.cache.exists())
        # Distinct paths with identical bytes in different splits must still fail.
        (self.root / self.rows[1]['winner']).write_bytes((self.root / self.rows[0]['winner']).read_bytes())
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.run_validation(mode='full')

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

    def test_standalone_cli_defaults_to_sampled_and_supports_explicit_full(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/check_training_data.py'
        for arguments, expected in (((), 'sampled'), (('--mode', 'full'), 'full')):
            result = subprocess.run([sys.executable, str(script), '--pairs', str(self.manifest),
                                     '--workers', '2', *arguments],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            logs = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertTrue(any(x.get('mode') == expected for x in logs))
            self.assertEqual(logs[-1]['phase'], 'data_ready')
            self.assertFalse(self.cache.exists())


if __name__ == '__main__':
    unittest.main()
