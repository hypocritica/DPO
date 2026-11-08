"""The one-GPU probe freezes the intended settings and reports allocation failures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import probe_sd35_b2 as probe


class ProbeTests(unittest.TestCase):
    def test_real_shell_config_only_is_single_gpu_b2_resident_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'model-report.json'
            report.write_text(json.dumps({'status': 'passed', 'revision': 'fixture'}))
            command = ['bash', str(ROOT / 'launchers/flow/probe_sd35_b2_5090.sh'),
                       '--tag', 'fixture', '--config-only', '--report-root', str(root / 'reports'),
                       '--runs-root', str(root / 'runs'), '--model-report', str(report)]
            environment = dict(os.environ, SCD_FLOW_PYTHON=sys.executable,
                               SCD_RESOURCES=str(ROOT / 'configs/resources-slai-flow.json'))
            result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config_file = root / 'reports/fixture/config.json'
            original = config_file.read_bytes()
            config = json.loads(original)
            training = config['training']
            for key, expected in {'world_size': 1, 'batch_size': 2, 'noise_views': 1,
                                  'gradient_accumulation_steps': 1, 'max_steps': 6,
                                  'checkpoint_every': 6, 'resolution': 1024, 'dtype': 'bfloat16',
                                  'offload_frozen': False, 'vae_batch_size': 1,
                                  'gradient_checkpointing': True, 'logging': 'jsonl'}.items():
                self.assertEqual(training[key], expected)
            self.assertEqual(training['comparison_budget']['per_update'], 2)
            self.assertFalse(config['evaluation']['enabled'])
            self.assertFalse((root / 'runs').exists())
            variant = command.copy()
            variant[variant.index('fixture')] = 'fixture-b4'
            variant += ['--batch-size', '4']
            check = subprocess.run(variant, cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            config4 = json.loads((root / 'reports/fixture-b4/config.json').read_text())
            self.assertEqual(config4['training']['batch_size'], 4)
            self.assertFalse(config4['training']['offload_frozen'])
            self.assertEqual(config4['training']['comparison_budget']['per_update'], 4)
            self.assertNotEqual(subprocess.run(command, cwd=ROOT, env=environment, capture_output=True).returncode, 0)
            self.assertEqual(config_file.read_bytes(), original)

    def test_zimage_h100_shell_preserves_formal_precision_and_resident_placement(self):
        from run_flow import build_config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / 'model-reports'
            reports.mkdir()
            (reports / 'zimage-model-validation.json').write_text(json.dumps(
                {'status': 'passed', 'revision': 'fixture-zimage'}))
            environment = dict(os.environ, SCD_FLOW_PYTHON=sys.executable,
                               SCD_FLOW_REPORTS_ROOT=str(reports),
                               SCD_RESOURCES=str(ROOT / 'configs/resources-slai-flow.json'))
            result = subprocess.run(['bash', str(ROOT / 'launchers/flow/probe_zimage_b4_h100.sh'),
                '--tag', 'zimage', '--config-only', '--report-root', str(root / 'reports'),
                '--runs-root', str(root / 'runs')], cwd=ROOT, env=environment,
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((root / 'reports/zimage/config.json').read_text())
            formal = build_config('zimage', 'dpo', world=2, profile='h100',
                                  recipe='paper-v2', batch_size=4,
                                  resources=environment['SCD_RESOURCES'])
            self.assertEqual(config['model']['family'], 'zimage')
            self.assertEqual(config['model']['transformer_dtype'], 'float32')
            self.assertEqual(config['model']['revision'], 'fixture-zimage')
            self.assertTrue(config['model']['path'].endswith('/Z-Image'))
            training = config['training']
            for key in ('dtype', 'resolution', 'reference_mode', 'gradient_checkpointing',
                        'offload_frozen', 'vae_batch_size', 'learning_rate', 'beta'):
                self.assertEqual(training[key], formal['training'][key], key)
            self.assertEqual(training['dtype'], 'bfloat16')
            self.assertFalse(training['offload_frozen'])
            self.assertEqual(training['batch_size'], 4)
            self.assertEqual(training['world_size'], 1)
            self.assertEqual(training['noise_views'], 1)
            self.assertEqual(training['gradient_accumulation_steps'], 1)
            self.assertEqual(training['comparison_budget']['per_update'], 4)
            self.assertEqual(training['max_steps'], 20)
            self.assertEqual(training['checkpoint_every'], 20)
            self.assertFalse(config['evaluation']['enabled'])
            self.assertFalse((root / 'runs').exists())

    def test_oom_preserves_traceback_memory_and_plain_text_result(self):
        # The config is real; only CUDA and the final trainer are fixtures.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'model-report.json'
            report.write_text(json.dumps({'status': 'passed', 'revision': 'fixture'}))
            resources = ROOT / 'configs/resources-slai-flow.json'
            config = probe.make_config(6, str(resources), root / 'pairs.jsonl', root / 'model', report)
            class OOM(RuntimeError):
                pass
            fake_cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda device: None,
                mem_get_info=lambda device: (30 * 1024**3, 32 * 1024**3),
                get_device_name=lambda device: 'fixture', device_count=lambda: 1,
                reset_peak_memory_stats=lambda device: None, is_initialized=lambda: True,
                max_memory_allocated=lambda device: 20 * 1024**3,
                max_memory_reserved=lambda device: 24 * 1024**3, OutOfMemoryError=OOM)
            fake_torch = SimpleNamespace(cuda=fake_cuda, __version__='fixture', version=SimpleNamespace(cuda='fixture'))
            def fail():
                raise OOM('fixture allocation failure')
            command = ['probe_sd35_b2.py', '--tag', 'oom', '--report-root', str(root / 'reports'),
                       '--runs-root', str(root / 'runs'), '--resources', str(resources), '--model-report', str(report)]
            with patch.object(sys, 'argv', command), patch.object(probe, 'make_config', return_value=config), \
                 patch('self_correct_dpo.common.require_gpu_allocation'), \
                 patch.dict(sys.modules, {'torch': fake_torch, 'self_correct_dpo.train': SimpleNamespace(main=fail)}):
                with self.assertRaisesRegex(OOM, 'fixture allocation failure'):
                    probe.main()
            summary = json.loads((root / 'reports/oom/summary.json').read_text())
            self.assertEqual(summary['status'], 'oom')
            self.assertEqual(summary['peak_allocated_GiB'], 20)
            self.assertEqual(summary['peak_reserved_GiB'], 24)
            self.assertIn('fixture allocation failure', summary['traceback'])


if __name__ == '__main__':
    unittest.main()
