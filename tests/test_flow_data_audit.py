"""One-time HPDv3 certificate checks without touching image bytes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from self_correct_dpo.common import sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_flow_data', ROOT / 'scripts/audit_flow_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FlowDataAuditTest(unittest.TestCase):
    def fixture(self, root):
        raw, images, prepared = root / 'raw', root / 'images-view', root / 'prepared'
        for path in (raw, images, prepared):
            path.mkdir()
        def write(path, item):
            path.write_text(json.dumps(item))
        revision = 'MizzenAI/HPDv3@test-revision'
        write(raw / 'train.json', [])
        write(raw / 'test.json', [{'prompt': 'heldout scene', 'path1': 'images/test-a.png',
                                   'path2': 'images/test-b.png'}])
        write(raw / '.download-manifest.json', {
            'repo_id': 'MizzenAI/HPDv3', 'revision': 'test-revision', 'status': 'complete',
            'completed': [{'file': 'train.json', 'hash': sha256(raw / 'train.json')}]})
        rows = []
        for index, split in enumerate(('train', 'validation', 'test')):
            prompt = 'local scene ' + str(index)
            first, second = f'images/{index}-a.png', f'images/{index}-b.png'
            hashes = [f'{index * 2 + 1:064x}', f'{index * 2 + 2:064x}']
            original = {'prompt': prompt, 'path1': first, 'path2': second,
                        'model1': 'flux', 'model2': 'sd3'}
            rows.append({'pair_id': f'hpdv3:{index}', 'scene_id': prompt,
                         'split': split, 'winner': str(images / first), 'loser': str(images / second),
                         'winner_sha256': hashes[0], 'loser_sha256': hashes[1],
                         'prompts': [{'variant_id': 'canonical', 'prompt': prompt,
                                      'template_family': 'canonical', 'transform_type': 'canonical'}],
                         'human_preference': {'source': 'hpdv3', 'source_revision': revision,
                                              'labels': [1, 0], 'winner_index': 0,
                                              'image_uids': [first, second], 'image_sha256': hashes,
                                              'original_prompt': prompt, 'original_record': original,
                                              'preferred_field': 'path1'}})
        pairs = prepared / 'pairs.jsonl'
        pairs.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        counts = {'train': 1, 'validation': 1, 'test': 1}
        evidence = {'train_json_sha256': sha256(raw / 'train.json'),
                    'test_json_sha256': sha256(raw / 'test.json')}
        write(prepared / 'ready.json', {'pairs_sha256': sha256(pairs), 'splits': counts,
                                         'source_revision': revision})
        write(prepared / 'audit.json', {**evidence, 'source_revision': revision,
            'family': 'unfiltered', 'allowed_models': ['flux', 'hunyuan', 'kolors', 'real_images', 'sd3'],
            'seed': 17, 'fractions': [.9, .05, .05], 'splits': counts,
            'usable_pairs': 3, 'raw_train_rows': 4, 'exclusions': {'model_subset': 1},
            'pickapic_heldout_inventory': []})
        extraction = images / 'extraction.json'
        write(extraction, {**evidence, 'source': str(raw), 'revision': 'test-revision', 'images': 8})
        return pairs, extraction

    def test_pass_uses_preparation_evidence_without_image_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs, extraction = self.fixture(root)
            report = module.audit(pairs, extraction=extraction)
            self.assertEqual(report['status'], 'passed', report)
            self.assertTrue(report['passed'])
            self.assertFalse(report['scope']['image_bytes_rehashed'])
            self.assertFalse(report['split_policy']['pickapic_heldout_checked'])
            self.assertEqual(report['dataset_identity']['pairs']['sha256'], sha256(pairs))
            self.assertFalse((extraction.parent / 'images').exists())

    def test_missing_ready_is_pending_and_never_certified(self):
        with tempfile.TemporaryDirectory() as temporary:
            pairs, extraction = self.fixture(Path(temporary))
            (pairs.parent / 'ready.json').unlink()
            report = module.audit(pairs, extraction=extraction)
            self.assertEqual(report['status'], 'in_progress')
            self.assertFalse(report['passed'])
            self.assertFalse(report['scope']['manifest_checks'])

    def test_changed_manifest_and_reversed_winner_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            pairs, extraction = self.fixture(Path(temporary))
            rows = [json.loads(line) for line in pairs.read_text().splitlines()]
            rows[0]['winner'], rows[0]['loser'] = rows[0]['loser'], rows[0]['winner']
            pairs.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            report = module.audit(pairs, extraction=extraction)
            self.assertEqual(report['status'], 'failed')
            ready = json.loads((pairs.parent / 'ready.json').read_text())
            ready['pairs_sha256'] = sha256(pairs)
            (pairs.parent / 'ready.json').write_text(json.dumps(ready))
            report = module.audit(pairs, extraction=extraction)
            self.assertEqual(report['status'], 'failed')
            self.assertIn('path1/path2', report['error'])

    def test_certificate_startup_hashes_only_small_ready_and_rejects_changed_pairs(self):
        from self_correct_dpo import flow_receipts
        with tempfile.TemporaryDirectory() as temporary:
            pairs, extraction = self.fixture(Path(temporary))
            report = module.audit(pairs, extraction=extraction)
            report_path = pairs.parent / 'validation-report.json'
            report_path.write_text(json.dumps(report))
            hashed = []
            def record_hash(path):
                hashed.append(Path(path))
                return sha256(path)
            with patch.object(flow_receipts, 'sha256', side_effect=record_hash):
                rows = flow_receipts.load_certified_pairs(pairs, report_path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(hashed, [pairs.parent / 'ready.json'])
            pairs.write_text(pairs.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Validated file changed'):
                flow_receipts.data_receipt(report_path, pairs)


if __name__ == '__main__':
    unittest.main()
