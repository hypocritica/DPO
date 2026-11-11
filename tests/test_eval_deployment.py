"""CPU checks for non-destructive setup and offline asset validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
    def test_setup_dry_run_and_existing_environment_protection(self):
        setup = script('setup_eval_environment')
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'flow env'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/setup_eval_environment.py'),
                                     '--profile', 'flow', '--prefix', str(prefix), '--runtime', 'slai-cu130'],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Dry run', result.stdout)
            self.assertIn('requirements-eval-flow.txt', result.stdout)
            self.assertFalse(prefix.exists())
            prefix.mkdir()
            with self.assertRaises(FileExistsError):
                setup.recipe('flow', prefix, 'slai-cu130')
            with self.assertRaisesRegex(ValueError, '5090'):
                setup.recipe('geneval', Path(directory) / 'geneval', 'slai-cu130')

    def test_offline_check_rejects_incomplete_size_and_missing_indexed_shard(self):
        downloads = script('download_assets')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'layer': 'missing.safetensors'}}))
            receipt = {'repo_id': downloads.ASSETS['pickscore'][0], 'repo_type': 'model', 'revision': 'fixed',
                       'complete': True, 'files': {index.name: index.stat().st_size}}
            marker = root / '.scd-download.json'
            marker.write_text(json.dumps(receipt))
            with patch.object(downloads.urllib.request, 'urlopen', side_effect=AssertionError('network forbidden')):
                with self.assertRaisesRegex(ValueError, 'indexed weight shard'):
                    downloads.check_asset('pickscore', out=root)
                (root / 'missing.safetensors').write_bytes(b'weights')
                receipt['files']['missing.safetensors'] = 7
                marker.write_text(json.dumps(receipt))
                self.assertTrue(downloads.check_asset('pickscore', out=root)['complete'])
                receipt['files']['missing.safetensors'] = 100
                marker.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'incomplete downloaded'):
                    downloads.check_asset('pickscore', out=root)
                receipt['complete'] = False
                marker.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'Download is incomplete'):
                    downloads.check_asset('pickscore', out=root)

    def test_only_documented_profile_runtime_combinations_are_accepted(self):
        setup = script('setup_eval_environment')
        with tempfile.TemporaryDirectory() as directory:
            for profile in setup.PROFILES:
                supported = 'h100-cu118' if profile in ('geneval', 'dpg') else 'slai-cu130'
                for runtime in ('slai-cu130', 'h100-cu118'):
                    with self.subTest(profile=profile, runtime=runtime):
                        prefix = Path(directory) / profile
                        if runtime == supported:
                            _, commands = setup.recipe(profile, prefix, runtime)
                            torch_command = commands[1]
                            expected = 'torch==2.0.0' if runtime == 'h100-cu118' else 'torch==2.14.1+cu130'
                            self.assertIn(expected, torch_command)
                            self.assertFalse(prefix.exists())
                        else:
                            with self.assertRaises(ValueError):
                                setup.recipe(profile, prefix, runtime)

    def test_offline_full_hash_catches_same_size_corruption(self):
        downloads = script('download_assets')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'ViT-L-14.pt'
            target.write_bytes(b'fixture')
            expected = downloads.URL_ASSETS['geneval-clip']
            receipt = {'repo_id': expected['url'], 'repo_type': 'file', 'revision': expected['sha256_prefix'],
                       'complete': True, 'files': {target.name: target.stat().st_size},
                       'sha256_prefixes': {target.name: hashlib.sha256(b'correct').hexdigest()}}
            target.with_name('.' + target.name + '.scd-download.json').write_text(json.dumps(receipt))
            self.assertTrue(downloads.check_asset('geneval-clip', out=target)['complete'])
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                downloads.check_asset('geneval-clip', out=target, full_hash=True)

    def test_legacy_source_check_matches_local_receipt_and_rejects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / 'code.py').write_text('print(1)\n')
            subprocess.run(['git', '-C', str(root), 'add', 'code.py'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            'commit', '-qm', 'fixture'], check=True)
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            (root / '.scd-source-revision').write_text(commit + '\n')
            command = ['bash', str(ROOT / 'scripts/download_sources.sh'), '--check', 'hpsv2', str(root)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            (root / 'code.py').write_text('print(2)\n')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
