"""Small CPU checks for strict judge parsing and explicit asset acquisition."""
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from self_correct_dpo.evaluation.unified_reward import (
    AGGREGATION, MODEL_REPOSITORY, MODEL_REVISION, SOURCE_REVISION,
    REFERENCE_SCRIPT, parse_acs, validate_spec,
)

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_assets.py'
module_spec = importlib.util.spec_from_file_location('reward_asset_downloads', SCRIPT)
downloads = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(downloads)


class RewardAssetTests(unittest.TestCase):
    def test_official_decimal_axes_and_invalid_judge_outputs(self):
        text = ('Alignment Score (1-5): 2.694700002670288\n'
                'Coherence Score (1-5): 2.371500015258789\n'
                'Style Score (1-5): 2.802500009536743')
        result = parse_acs(text)
        self.assertAlmostEqual(result['score'], 2.6229000091552734)
        self.assertEqual(result['raw_output'], text)
        self.assertEqual(result['aggregation'], AGGREGATION)
        for bad in (text.replace('2.694700002670288', 'NaN'),
                    text.replace('2.694700002670288', '5.1'),
                    text.replace('2.694700002670288', '0'),
                    text.split('Style')[0], text + '\n' + text,
                    'I cannot score this image.', text + '\nOverall: 3'):
            with self.subTest(output=bad), self.assertRaises(ValueError):
                parse_acs(bad)

    def test_pinned_model_receipt_and_complete_shards_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / 'source', root / 'model'
            (source / REFERENCE_SCRIPT).parent.mkdir(parents=True)
            (source / REFERENCE_SCRIPT).write_text('# fixture reference')
            (source / '.scd-source-revision').write_text(SOURCE_REVISION)
            model.mkdir()
            for name in ('preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.json'):
                (model / name).write_text('{}')
            (model / 'config.json').write_text(json.dumps({'model_type': 'qwen2_5_vl',
                'architectures': ['Qwen2_5_VLForConditionalGeneration']}))
            (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'shard.safetensors'}}))
            (model / 'shard.safetensors').write_bytes(b'test')
            receipt = {'repo_id': MODEL_REPOSITORY, 'repo_type': 'model', 'revision': MODEL_REVISION,
                       'complete': True, 'files': {'shard.safetensors': 4}}
            (model / '.scd-download.json').write_text(json.dumps(receipt))
            spec = {'source': str(source), 'model': str(model), 'source_revision': SOURCE_REVISION,
                    'model_revision': MODEL_REVISION, 'aggregation': AGGREGATION}
            validate_spec(spec)
            receipt['revision'] = 'another-checkpoint'
            (model / '.scd-download.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'pinned model'):
                validate_spec(spec)
            receipt['revision'] = MODEL_REVISION
            (model / '.scd-download.json').write_text(json.dumps(receipt))
            (model / 'shard.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'shard'):
                validate_spec(spec)

    def test_qwen_downloads_require_processor_and_all_selected_shards(self):
        files = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'chat_template.json', 'merges.txt', 'model.safetensors.index.json',
                 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
        siblings = [SimpleNamespace(rfilename=name, size=5) for name in files]
        self.assertEqual(set(downloads.select_files('unifiedreward', siblings)), set(files))
        with self.assertRaisesRegex(ValueError, 'Missing expected'):
            downloads.select_files('unifiedreward', [item for item in siblings if item.rfilename != 'tokenizer.json'])
        self.assertIn('zimage', downloads.ASSETS)
        self.assertIn('zimage-turbo', downloads.ASSETS)

    def test_external_asset_bad_checksum_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'official-small-fixture'
            checksum = hashlib.sha256(payload).hexdigest()
            plan = {'asset': 'fixture', 'repo_id': 'https://official.invalid/weight', 'revision': checksum,
                    'destination': str(root), 'receipt': str(root / '.receipt.json'),
                    'files': {'weight.pt': len(payload)}, 'urls': {'weight.pt': 'https://official.invalid/weight'},
                    'sha256_prefixes': {'weight.pt': '0' * 64}, 'remaining_bytes': len(payload)}
            with patch.object(downloads.urllib.request, 'urlopen', return_value=BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    downloads.download_external(plan)
            self.assertFalse((root / 'weight.pt').exists())
            self.assertFalse(json.loads((root / '.receipt.json').read_text())['complete'])
            # A separate fixed request publishes the exact bytes and a verified hash.
            plan['receipt'] = str(root / '.receipt-good.json')
            plan['sha256_prefixes']['weight.pt'] = checksum
            with patch.object(downloads.urllib.request, 'urlopen', return_value=BytesIO(payload)):
                downloads.download_external(plan)
            self.assertEqual((root / 'weight.pt').read_bytes(), payload)
            self.assertEqual(json.loads((root / '.receipt-good.json').read_text())['verified_sha256']['weight.pt'], checksum)
            (root / 'weight.pt').write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                downloads.download_external(plan)


if __name__ == '__main__':
    unittest.main()
