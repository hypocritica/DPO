"""Local scorer caches are reused without receipts, networking or model imports."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('prepare_eval_assets', ROOT / 'scripts/prepare_eval_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class PrepareAssetsTests(unittest.TestCase):
    def test_manual_pickscore_cache_check_and_download_reuse_without_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, processor = root / 'model', root / 'processor'
            model.mkdir()
            processor.mkdir()
            (model / 'config.json').write_text('{}')
            (model / 'model.safetensors').write_bytes(b'fixture-weight')
            for name in ('preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
                (processor / name).write_text('{}')
            config = root / 'scorers.json'
            config.write_text(json.dumps({'scorers': {'pickscore': {'model': str(model), 'processor': str(processor)}}}))
            with patch.object(assets.downloads, 'plan_asset', side_effect=AssertionError('network forbidden')):
                for action in ('check', 'plan', 'download'):
                    self.assertEqual(assets.main([action, '--profile', 'pickscore', '--scorers-config', str(config)]), 0)
            self.assertFalse((model / '.scd-download.json').exists())
            self.assertFalse((processor / '.scd-download.json').exists())
            (model / 'model.safetensors').write_bytes(b'')
            self.assertEqual(assets.main(['check', '--profile', 'pickscore', '--scorers-config', str(config)]), 1)

    def test_modern_plan_is_explicit_and_never_downloads_or_clones(self):
        plans = []
        def plan(asset, **kwargs):
            plans.append(asset)
            return {'asset': asset, 'destination': str(kwargs['out']), 'files': {}, 'remaining_bytes': 10}
        with patch.object(assets.downloads, 'plan_asset', side_effect=plan), \
             patch.object(assets.downloads, 'download', side_effect=AssertionError('weight download forbidden')), \
             patch.object(assets.subprocess, 'run', side_effect=AssertionError('source clone forbidden')):
            self.assertEqual(assets.main(['plan', '--profile', 'modern']), 0)
        self.assertEqual(plans, ['hpsv3', 'hpsv3-base', 'unifiedreward'])

    def test_existing_legacy_source_is_preserved_and_modern_source_only_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'hpsv2/source').mkdir(parents=True)
            (root / 'unifiedreward').mkdir()
            specs = {'hpsv2': {'open_clip_source': str(root / 'hpsv2/source/hpsv2/src')},
                     'unifiedreward': {'source': str(root / 'unifiedreward')}}
            with patch.object(assets.subprocess, 'run') as run:
                assets.prepare_sources(['hpsv2', 'unifiedreward'], specs)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][2:4], ['--check', 'unifiedreward'])
            self.assertEqual(list((root / 'hpsv2/source').iterdir()), [])

    def test_missing_hpsv2_source_clones_the_official_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'models/preference/hpsv2/source'
            specs = {'hpsv2': {'open_clip_source': str(source / 'hpsv2/src')}}
            with patch.object(assets.subprocess, 'run') as run:
                assets.prepare_sources(['hpsv2'], specs)
            self.assertEqual(run.call_args.args[0], ['bash', str(ROOT / 'scripts/download_sources.sh'),
                                                   'hpsv2', str(source)])


if __name__ == '__main__':
    unittest.main()
