"""CPU checks for the standalone SLAI handoff and fair K1/K2 configuration."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from estimate_runtime import estimate
from make_smoke_data import make_smoke_data
from package_code import package
from test_portability import make_pairs
from self_correct_dpo.data import load_pairs


class SlaiMigrationTests(unittest.TestCase):
    def config(self, root, name, *flags):
        out = root / (name + '.json')
        command = [sys.executable, str(ROOT / 'scripts/make_config.py'),
                   '--model', str(root / 'model'), '--pairs', str(root / 'pairs.jsonl'),
                   '--out', str(out), *flags]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(out.read_text())

    def test_fair_dpo_and_pure_k2_configs_for_both_world_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for world, accumulation in ((4, 8), (8, 4)):
                configs = []
                for method, views in (('dpo', 1), ('k2-noise', 2)):
                    config = self.config(root, f'{method}-{world}', '--method', method,
                        '--world-size', str(world), '--batch-size', '8', '--effective-batch', '256')
                    t = config['training']
                    self.assertEqual(t['gradient_accumulation_steps'], accumulation)
                    self.assertEqual(t['noise_views'], views)
                    self.assertNotIn('independent_rollout', t)
                    self.assertEqual(t['max_steps'], 2000)
                    configs.append(config)
                configs[0]['training']['noise_views'] = 2
                configs[0].pop('experiment_note'); configs[1].pop('experiment_note')
                self.assertEqual(configs[0], configs[1])

    def test_benchmark_smoke_and_preserved_rollout_default(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            default = self.config(root, 'default')
            self.assertEqual(default['training']['independent_rollout']['state_source'], 'model')
            benchmark = self.config(root, 'bench', '--method', 'k2-noise', '--benchmark',
                '--world-size', '4', '--batch-size', '8', '--steps', '30')
            t = benchmark['training']
            self.assertEqual((t['max_steps'], t['checkpoint_every'], t['warmup_steps']), (30, 30, 0))
            self.assertEqual(t['diagnostic'], {})
            self.assertEqual(benchmark['evaluation'], {'enabled': False})
            smoke = self.config(root, 'smoke', '--method', 'k2-noise', '--smoke', '--world-size', '4')
            self.assertEqual(smoke['training']['gradient_accumulation_steps'], 1)
            self.assertEqual(smoke['training']['max_steps'], 2)
            rollout = self.config(root, 'rollout', '--method', 'k2-rollout',
                                  '--timestep-sampling', 'uniform_final')
            self.assertEqual(rollout['training']['independent_rollout']['timestep_sampling'], 'uniform_final')

    def test_smoke_selection_preserves_human_labels_and_image_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'source'; source.mkdir()
            original = make_pairs(source)
            selected = make_smoke_data(source / 'pairs.jsonl', root / 'smoke', count=1)
            rows = load_pairs(selected)
            for a, b in zip(original, rows):
                self.assertEqual(a['human_preference'], b['human_preference'])
                self.assertEqual(a['winner_sha256'], b['winner_sha256'])
                self.assertEqual(a['split'], b['split'])
                self.assertTrue(Path(b['winner']).is_file())
            with self.assertRaises(FileExistsError):
                make_smoke_data(source / 'pairs.jsonl', selected.parent, count=1)
            with self.assertRaisesRegex(ValueError, 'Not enough'):
                make_smoke_data(source / 'pairs.jsonl', root / 'insufficient', count=2)
            self.assertFalse((root / 'insufficient').exists())

    def test_runtime_estimate_from_cumulative_time_across_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [{'step': i, 'cost': {'training_seconds': 100 + i * 12},
                     'peak_cuda_bytes': 3 * 1024**3, 'effective_batch_size_pairs': 256}
                    for i in range(1, 21)]
            (root / 'metrics-1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:12]))
            (root / 'metrics-2.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[12:])+'{"step":')
            result = estimate(root, warmup=10, targets=(2000,))
            self.assertEqual(result['median_seconds_per_update'], 12)
            self.assertEqual(result['samples_after_warmup'], 10)
            self.assertEqual(result['peak_rank0_allocated_GiB'], 3)
            self.assertAlmostEqual(result['optimization_hours']['2000'], 2000*12/3600)
            self.assertAlmostEqual(result['remaining_optimization_hours']['2000'], 1980*12/3600)
            (root / 'metrics-3.jsonl').write_text('bad json\n{}\n')
            with self.assertRaises(json.JSONDecodeError): estimate(root)

    def test_code_archive_excludes_runtime_assets_and_checksums_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'code'; root.mkdir()
            for relative in ('train.py', 'pyproject.toml', 'paths.sh',
                             'src/self_correct_dpo/training.py', 'docs/guide.md',
                             'configs/template.json', '.gitignore', 'requirements-5090.txt'):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('source fixture\n')
            for relative in ('.env', '.git/config', 'data/pairs.jsonl', 'models/config.json',
                             'runs/training.json', 'configs/local-secret.json', 'src/.env'):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('must not be exported')
            (root / 'scripts').mkdir()
            (root / 'scripts/link.py').symlink_to(root / '.env')
            out = Path(d) / 'code.zip'
            package(out, root)
            with zipfile.ZipFile(out) as archive:
                names = archive.namelist()
                manifest = json.loads(archive.read('DPO/CODE_MANIFEST.json'))['sha256']
                self.assertEqual(len(names), len(manifest)+1)
                self.assertEqual(len(manifest), 8)
                for relative, digest in manifest.items():
                    self.assertEqual(sha256(archive.read('DPO/'+relative)).hexdigest(), digest)
                self.assertIn('DPO/docs/guide.md', names)
                self.assertNotIn('DPO/scripts/link.py', names)
            with self.assertRaises(FileExistsError): package(out, root)


if __name__ == '__main__':
    unittest.main()
