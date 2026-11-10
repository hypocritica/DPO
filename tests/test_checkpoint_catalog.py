"""Real-file snapshot discovery and checksum checks, without PyTorch imports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import digest, sha256
from self_correct_dpo.evaluation.checkpoint_catalog import discover_runs, resolve_checkpoint


class CheckpointCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_fixture(self, name, family, full_state=False):
        run = self.root / name
        checkpoint = run / 'checkpoints/step-0000200'
        (checkpoint / 'adapter').mkdir(parents=True)
        (checkpoint / 'adapter/adapter_model.safetensors').write_bytes(b'tiny-adapter')
        if full_state:
            (checkpoint / 'state.pt').write_bytes(b'optimizer-state')
        model = {'family': family, 'path': '/local/model', 'revision': 'revision'}
        (run / 'training.json').write_text(json.dumps({'run_id': name, 'config': {'model': model}}))
        saved = {'run_id': name, 'step': 200, 'full_state': full_state,
                 'files': {p.relative_to(checkpoint).as_posix(): sha256(p)
                           for p in checkpoint.rglob('*') if p.is_file()}}
        self.save_manifest(checkpoint, saved)
        return run, checkpoint

    @staticmethod
    def save_manifest(checkpoint, saved):
        saved['checkpoint_id'] = digest({k: v for k, v in saved.items() if k != 'checkpoint_id'})
        (checkpoint / 'checkpoint.json').write_text(json.dumps(saved))

    def test_recursive_selection_and_baselines_ignore_latest_and_pending(self):
        sd15, _ = self.run_fixture('old/sd15', 'sd15')
        flow, checkpoint = self.run_fixture('new/zimage', 'zimage', full_state=True)
        (flow / 'checkpoints/latest.json').write_text(json.dumps({'path': str(checkpoint)}))
        (flow / 'checkpoints/.pending-step-0000400').mkdir()
        (flow / 'checkpoints/resume-0000400').mkdir()
        records = discover_runs(self.root)
        self.assertEqual([r['name'] for r in records], ['new/zimage', 'old/sd15'])
        self.assertTrue(all(r['steps'] == [0, 200] and not r['files_verified'] for r in records))
        self.assertEqual(discover_runs(self.root, patterns=['new/*'], families=['zimage']), records[:1])
        self.assertEqual(discover_runs(flow)[0]['name'], 'zimage')
        model, adapter, identity = resolve_checkpoint(sd15, 0)
        self.assertEqual(model['family'], 'sd15')
        self.assertIsNone(adapter)
        self.assertEqual(identity['checkpoint_id'], 'base')
        self.assertEqual(resolve_checkpoint(flow, 200)[1], checkpoint / 'adapter')
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(flow, 400)

    def test_corrupt_files_and_changed_manifest_are_errors_not_missing(self):
        run, checkpoint = self.run_fixture('medium', 'sd35_medium', full_state=True)
        (checkpoint / 'state.pt').write_bytes(b'changed')
        self.assertEqual(discover_runs(run)[0]['steps'], [0, 200])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            resolve_checkpoint(run, 200)
        saved = json.loads((checkpoint / 'checkpoint.json').read_text())
        saved['step'] = 400
        self.save_manifest(checkpoint, saved)
        with self.assertRaisesRegex(ValueError, 'run/step'):
            discover_runs(run)
        saved['run_id'] = 'different'
        (checkpoint / 'checkpoint.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'manifest identity'):
            resolve_checkpoint(run, 200)

    def test_unregistered_weights_and_snapshot_symlinks_are_rejected(self):
        run, checkpoint = self.run_fixture('xl', 'sdxl')
        (checkpoint / 'adapter/adapter_model.bin').write_bytes(b'unregistered alternate weights')
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            resolve_checkpoint(run, 200)
        (run / 'checkpoints/step-0000400').symlink_to(checkpoint, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            discover_runs(run)

    def test_module_and_discovery_need_only_standard_library(self):
        self.run_fixture('zimage', 'zimage')
        code = ('import sys; sys.path.insert(0, sys.argv[1]); '
                'from self_correct_dpo.evaluation.checkpoint_catalog import discover_runs, resolve_checkpoint; '
                'runs = discover_runs(sys.argv[2]); resolve_checkpoint(runs[0]["path"], 200); '
                'assert "torch" not in sys.modules and "self_correct_dpo.training" not in sys.modules')
        result = subprocess.run([sys.executable, '-S', '-c', code, str(ROOT / 'src'), str(self.root)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
