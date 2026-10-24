"""Configuration contracts that run with the Python standard library only."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.config import add_training_arguments, training_config
from self_correct_dpo.training import validate_config


def compose(*arguments, template='multiview-C.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs' / template))
    add_training_arguments(parser)
    with patch.dict(os.environ, {}, clear=True):
        return training_config(parser.parse_args(arguments))


class GaussianConfigTests(unittest.TestCase):
    def test_data_validation_overrides_and_invalid_settings(self):
        config = compose('--method', 'multiview-C-gaussian', '--data-validation', 'full',
                         '--validation-workers', '4', '--validation-samples', '16')
        self.assertEqual(config['data']['validation'], 'full')
        self.assertEqual(config['data']['validation_workers'], 4)
        self.assertEqual(config['data']['validation_samples'], 16)
        validate_config(config, files=False)
        for key, value in (('validation', 'none'), ('validation_workers', 0), ('validation_samples', False)):
            invalid = copy.deepcopy(config)
            invalid['data'][key] = value
            with self.assertRaises(ValueError):
                validate_config(invalid, files=False)

    def test_gaussian_methods_remove_rollout_and_preserve_family_batch(self):
        for family, global_batch, beta in (('sd15', 256, 5000), ('sdxl', 128, 500)):
            for k in (2, 4, 6, 8):
                for arm in ('B', 'C'):
                    with self.subTest(family=family, k=k, arm=arm):
                        batch = 2 if family == 'sd15' and k == 2 else 1
                        config = compose('--family', family, '--method', f'multiview-{arm}-gaussian',
                                         '--views', str(k), '--world-size', '2',
                                         '--train_batch_size', str(batch))
                        validate_config(config, files=False)
                        t = config['training']
                        self.assertNotIn('independent_rollout', t)
                        self.assertTrue(t['batch_views'])
                        self.assertEqual((t['formal_arm'], t['noise_views'], t['beta']), (arm, k, beta))
                        self.assertEqual(batch * t['gradient_accumulation_steps'] * 2, global_batch)

    def test_selecting_other_methods_clears_gaussian_execution_setting(self):
        for method in ('dpo', 'multiview-C'):
            config = compose('--method', method, template='multiview-C-gaussian.json')
            self.assertNotIn('batch_views', config['training'])
            self.assertEqual('independent_rollout' in config['training'], method != 'dpo')

    def test_gaussian_templates_work_without_method_override(self):
        for arm in ('B', 'C'):
            config = compose(template=f'multiview-{arm}-gaussian.json')
            validate_config(config, files=False)
            self.assertTrue(config['training']['batch_views'])
            self.assertNotIn('independent_rollout', config['training'])

    def test_reject_mislabelled_or_unsupported_batched_algorithms(self):
        config = compose('--method', 'multiview-C-gaussian')
        for key in ('independent_rollout', 'paired_drift'):
            for batching in (False, True):
                invalid = copy.deepcopy(config)
                invalid['training'][key] = {}
                invalid['training']['batch_views'] = batching
                with self.assertRaisesRegex(ValueError, 'Gaussian|ordinary Gaussian'):
                    validate_config(invalid, files=False)
        config['training']['batch_views'] = 'true'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            validate_config(config, files=False)

    def test_make_config_freezes_two_gpu_gaussian_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            for family, k, batch, accumulation in (('sd15', 2, 2, 64), ('sd15', 4, 1, 128),
                                                  ('sd15', 6, 1, 128), ('sd15', 8, 1, 128),
                                                  ('sd15', 2, 1, 128), ('sd15', 4, 2, 64),
                                                  ('sd15', 6, 2, 64), ('sd15', 8, 2, 64),
                                                  ('sdxl', 2, 1, 64), ('sdxl', 4, 1, 64),
                                                  ('sdxl', 6, 1, 64), ('sdxl', 8, 1, 64)):
                output = Path(directory) / f'{family}-k{k}-b{batch}.json'
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/make_config.py'),
                    '--family', family, '--method', 'multiview-C-gaussian', '--views', str(k),
                    '--world-size', '2', '--train_batch_size', str(batch),
                    '--effective-batch', '256' if family == 'sd15' else '128',
                    '--max_train_steps', '10000', '--out', str(output)],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                config = json.loads(output.read_text())
                self.assertNotIn('extends', config)
                self.assertTrue(config['training']['batch_views'])
                self.assertNotIn('independent_rollout', config['training'])
                self.assertEqual(config['training']['gradient_accumulation_steps'], accumulation)


if __name__ == '__main__':
    unittest.main()
