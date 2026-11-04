"""Real-file sampled preflight and explicit full-integrity contracts; standard library only."""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.common import digest, sha256
from self_correct_dpo.data.arrow_images import image_source
from self_correct_dpo.data.pairs import load_pairs
from self_correct_dpo.data.validation import load_resume_pairs, load_training_pairs


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

    def resume_ids(self, rows):
        return {'pairs_id': digest([r for r in rows if r['split'] == 'train']),
                'validation_pairs_id': digest([r for r in rows if r['split'] == 'validation'])}

    def run_resume(self, identities):
        output = io.StringIO()
        with redirect_stdout(output):
            rows = load_resume_pairs(self.manifest, **identities)
        return rows, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_resume_preserves_normalized_identity_without_image_or_schema_checks(self):
        # The normal startup loader keeps the symlink path, not its target.
        (self.root / 'alias.img').symlink_to(self.root / self.rows[0]['winner'])
        self.rows[0]['winner'] = './alias.img'
        self.rows[0]['metadata'] = {'include': [{'class': 'cat'}], 'prompt': 'ignored wording'}
        self.rows[0]['semantic_scene_key'] = 'untrusted-old-value'
        self.rows[1]['loser'] = str(self.root / self.rows[1]['loser'])
        self.save_manifest()
        expected, _ = self.run_validation(samples=1)
        identities = self.resume_ids(expected)
        for row in self.rows:
            if row['split'] == 'test':
                row.clear()
                row['split'] = 'test'  # Test rows receive no repeated audit.
        self.save_manifest()
        forbidden = AssertionError('Fast resume must not audit schema or read images')
        with patch('self_correct_dpo.data.validation.validate_pair_rows', side_effect=forbidden), \
                patch('self_correct_dpo.data.validation.validate_split_disjointness', side_effect=forbidden), \
                patch('self_correct_dpo.data.validation.sha256', side_effect=forbidden), \
                patch('self_correct_dpo.data.arrow_images.image_source', side_effect=forbidden), \
                patch.object(Path, 'read_bytes', side_effect=forbidden):
            actual, logs = self.run_resume(identities)
        self.assertEqual(actual, [r for r in expected if r['split'] != 'test'])
        self.assertEqual(actual[0]['winner'], str(self.root / 'alias.img'))
        self.assertEqual(logs[0]['phase'], 'data_resume_load')
        self.assertEqual(logs[-1]['phase'], 'data_resume_identity')
        self.assertEqual(logs[-1]['train_pairs'], 8)
        self.assertEqual(logs[-1]['validation_pairs'], 8)
        self.assertEqual(logs[-1]['skipped_test_pairs'], 8)
        self.assertEqual(logs[-1]['checked_images'], 0)
        self.assertFalse(self.cache.exists())
        Path(actual[0]['winner']).write_bytes(b'corrupt after preparation')
        self.run_resume(identities)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            image_source(actual[0], 'winner', verify_files=True)

    def test_resume_rejects_changed_train_or_validation_rows_and_unknown_split(self):
        original = copy.deepcopy(self.rows)
        identities = self.resume_ids(self.run_validation()[0])
        for split, index in (('train', 0), ('validation', 1)):
            for change in ('prompt', 'order', 'hash'):
                with self.subTest(split=split, change=change):
                    self.rows = copy.deepcopy(original)
                    if change == 'prompt':
                        self.rows[index]['prompts'][0]['prompt'] = 'different caption'
                    elif change == 'order':
                        self.rows[index], self.rows[index+3] = self.rows[index+3], self.rows[index]
                    else:
                        self.rows[index]['winner_sha256'] = '0' * 64
                    self.save_manifest()
                    with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                        self.run_resume(identities)
        self.rows = copy.deepcopy(original)
        self.rows[2]['split'] = 'unknown'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Invalid split'):
            self.run_resume(identities)

    def test_resume_requires_saved_identities_and_completed_preparation(self):
        identities = self.resume_ids(self.run_validation()[0])
        for key in identities:
            for invalid in (None, '', 'a'*63, 'A'*64, False, 1):
                with self.subTest(key=key, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, 'valid pairs_id'):
                        self.run_resume(dict(identities, **{key: invalid}))
        (self.root / 'preparing.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_resume(identities)
        self.assertFalse(self.cache.exists())

    def test_resume_arrow_keeps_fields_and_never_opens_or_stats_shards(self):
        shard = self.root / 'nonexistent.arrow'
        for index, row in enumerate(self.rows):
            row['storage'] = {'kind': 'arrow', 'shard': str(shard), 'row': index,
                              'winner_column': 'jpg_0', 'loser_column': 'jpg_1',
                              'shard_size': 123, 'shard_mtime_ns': 456}
            row['winner'] = f'{shard}#row={index}:jpg_0'
            row['loser'] = f'{shard}#row={index}:jpg_1'
        self.save_manifest()
        expected = load_pairs(self.manifest, verify_files=False)
        identities = self.resume_ids(expected)
        stat = Path.stat
        def manifest_stat(path, *args, **kwargs):
            self.assertNotEqual(path, shard, 'Fast resume must not stat Arrow shards')
            return stat(path, *args, **kwargs)
        with patch.object(Path, 'stat', autospec=True, side_effect=manifest_stat), \
                patch('self_correct_dpo.data.arrow_images.arrow_dataset', side_effect=AssertionError('Arrow read')):
            actual, _ = self.run_resume(identities)
        self.assertEqual(actual, [r for r in expected if r['split'] != 'test'])
        self.assertEqual(actual[0]['storage'], self.rows[0]['storage'])
        # A relative shard is accepted only when its locator still resolves to
        # the original identity in the current working directory.
        self.rows[0]['storage']['shard'] = os.path.relpath(shard)
        locator = Path(self.rows[0]['storage']['shard']).expanduser().absolute()
        self.rows[0]['winner'] = f'{locator}#row=0:jpg_0'
        self.rows[0]['loser'] = f'{locator}#row=0:jpg_1'
        self.save_manifest()
        relative = load_pairs(self.manifest, verify_files=False)
        self.assertEqual(self.run_resume(self.resume_ids(relative))[0][0], relative[0])
        self.rows[0]['storage']['shard'] = 'wrong-relative-shard.arrow'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Arrow image identity'):
            self.run_resume(identities)

    def test_resume_rejects_manifest_change_during_read(self):
        identities = self.resume_ids(self.run_validation()[0])
        original_open = Path.open
        @contextmanager
        def changing_open(path, *args, **kwargs):
            with original_open(path, *args, **kwargs) as handle:
                yield handle
            with original_open(path, 'ab') as handle:
                handle.write(b'\n')
        with patch.object(Path, 'open', changing_open):
            with self.assertRaisesRegex(ValueError, 'manifest changed while reading'):
                self.run_resume(identities)

    def test_startup_resume_routes_to_saved_identity_and_rejects_config_change(self):
        from self_correct_dpo.train import load_startup_pairs
        expected, _ = self.run_validation()
        identities = self.resume_ids(expected)
        config = {'data': {'pairs': str(self.manifest)},
                  'training': {'batch_size': 2, 'gradient_accumulation_steps': 4, 'world_size': 2}}
        (self.root / 'training.json').write_text(json.dumps(dict(identities, config=config)))
        current = copy.deepcopy(config)
        current['training'].update(batch_size=4, gradient_accumulation_steps=2)
        with patch.dict(os.environ, {'SCD_FAST_RESUME': '1'}), redirect_stdout(io.StringIO()), \
                patch('self_correct_dpo.data.validation.load_training_pairs', side_effect=AssertionError('Full loader')):
            actual, fast = load_startup_pairs(current, self.root, resume='latest')
            self.assertTrue(fast)
            self.assertEqual(actual, [r for r in expected if r['split'] != 'test'])
            current['training']['world_size'] = 4
            with self.assertRaisesRegex(ValueError, 'permits only'):
                load_startup_pairs(current, self.root, resume='latest')

    def test_startup_first_run_explicit_audits_and_opt_out_keep_normal_loader(self):
        from self_correct_dpo.train import load_startup_pairs
        for case in ('first', 'check-only', 'full', 'opt-out', 'extend'):
            with self.subTest(case=case):
                config = {'data': {'pairs': str(self.manifest)}}
                kwargs = {'resume': None if case == 'first' else 'latest'}
                if case == 'check-only':
                    kwargs['check_only'] = True
                if case == 'extend':
                    kwargs['extend_from'] = 'parent-checkpoint'
                if case == 'full':
                    config['data']['validation'] = 'full'
                with patch.dict(os.environ, {'SCD_FAST_RESUME': '0' if case == 'opt-out' else '1'}), \
                        patch('self_correct_dpo.data.validation.load_training_pairs', return_value=self.rows) as normal, \
                        patch('self_correct_dpo.data.validation.load_resume_pairs', side_effect=AssertionError('Fast loader')):
                    rows, fast = load_startup_pairs(config, self.root, **kwargs)
                self.assertFalse(fast)
                self.assertEqual(rows, self.rows)
                normal.assert_called_once_with(str(self.manifest), mode='full' if case == 'full' else 'auto',
                                               workers=8, samples=8)

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
