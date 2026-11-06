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
    def test_flow_objective_switch_clears_reference_and_diagnostics(self):
        for family in ('sd35_medium', 'zimage'):
            for objective,old in (('dpo','linear_dpo'),('linear_dpo','dpo')):
                c=compose('--objective',objective,template=f'flow/{family}-{old}.json')
                validate_config(c,files=False)
                self.assertFalse(c['training']['diagnostic'])
                if objective=='dpo':
                    self.assertEqual(c['training'].get('reference_mode','base'),'base')
                    self.assertNotIn('ema_decay',c['training'])
                    self.assertNotIn('linear_eta',c['training'])
                else:
                    self.assertEqual(c['training']['reference_mode'],'ema')
                    self.assertEqual(c['training']['ema_decay'],.995)
                    self.assertEqual(c['training']['linear_eta'],.01)

    def test_sdxl_all_methods_keep_resolution_and_global_batch_on_2_4_8_gpus(self):
        for method in ('dpo', 'multiview-B', 'multiview-C',
                       'multiview-B-gaussian', 'multiview-C-gaussian'):
            for world in (2, 4, 8):
                with self.subTest(method=method, world=world):
                    config = compose('--family', 'sdxl', '--method', method,
                                     '--world-size', str(world), '--max_train_steps', '10000')
                    validate_config(config, files=False)
                    t, evaluation = config['training'], config['evaluation']
                    self.assertEqual((t['resolution'], evaluation['sampling']['height'],
                                      evaluation['sampling']['width']), (1024, 1024, 1024))
                    self.assertEqual((t['batch_size'], t['gradient_accumulation_steps']), (1, 128 // world))
                    self.assertEqual((t['beta'], t['checkpoint_every'], t['lora_checkpoint_every']), (500, 200, 1000))
                    self.assertEqual(t['noise_views'], 1 if method == 'dpo' else 2)
                    self.assertEqual('independent_rollout' in t, method in ('multiview-B', 'multiview-C'))
                    self.assertEqual(bool(t.get('batch_views')), method.endswith('-gaussian'))
                    self.assertTrue(config['data']['pairs'].endswith('prepared-sdxl/pairs.jsonl'))
                    self.assertTrue(evaluation['prompts'].endswith('sdxl/pickapic-monitor.jsonl'))

    def test_two_gpu_dpo_matches_gaussian_baseline_budget(self):
        config = compose('--family', 'sd15', '--method', 'dpo', '--views', '1',
                         '--world-size', '2', '--train_batch_size', '2',
                         '--max_train_steps', '10000', '--mixed_precision', 'bf16',
                         '--checkpointing_steps', '200', '--report_to', 'tensorboard',
                         '--launcher', 'accelerate')
        validate_config(config, files=False)
        t = config['training']
        self.assertEqual((t['noise_views'], t['batch_size'], t['gradient_accumulation_steps']), (1, 2, 64))
        self.assertEqual((t['max_steps'], t['checkpoint_every'], t['learning_rate'], t['beta']),
                         (10000, 200, 1e-5, 5000))
        self.assertEqual((config['seed'], t['lora_rank'], t['lora_alpha']), (17, 16, 16))
        self.assertEqual(t['logging'], 'tensorboard')
        self.assertTrue(config['evaluation']['enabled'])
        self.assertNotIn('independent_rollout', t)
        self.assertNotIn('batch_views', t)

    def test_data_validation_overrides_and_invalid_settings(self):
        config = compose('--method', 'multiview-C-gaussian', '--data-validation', 'full',
                         '--validation-workers', '4', '--validation-samples', '16')
        self.assertEqual(config['data']['validation'], 'full')
        self.assertEqual(config['data']['validation_workers'], 4)
        self.assertEqual(config['data']['validation_samples'], 16)
        validate_config(config, files=False)
        no_images = compose('--data-validation', 'none')
        self.assertEqual(no_images['data']['validation'], 'none')
        validate_config(no_images, files=False)
        for key, value in (('validation', 'unknown'), ('validation_workers', 0), ('validation_samples', False)):
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


class ObjectiveConfigTests(unittest.TestCase):
    def test_mapo_ranking_can_be_disabled_without_allowing_negative_weight(self):
        c = compose('--objective', 'mapo', '--set', 'training.mapo_beta=0')
        validate_config(c, files=False)
        c['training']['mapo_beta'] = -0.1
        with self.assertRaisesRegex(ValueError, 'mapo_beta'):
            validate_config(c, files=False)

    def test_objective_selection_clears_default_or_stagewise_views(self):
        for template in ('multiview-C.json', 'stagewise-C-gaussian.json'):
            for objective in ('sft', 'kto', 'mapo', 'dmpo'):
                c = compose('--objective', objective, template=template)
                self.assertEqual(c['method'], 'dpo')
                self.assertEqual(c['training']['noise_views'], 1)
                for key in ('stagewise', 'independent_rollout', 'batch_views'):
                    self.assertNotIn(key, c['training'])
                validate_config(c, files=False)

    def test_dmpo_scales_and_explicit_override_survive_family_composition(self):
        for family, beta in (('sd15', 2000), ('sdxl', 6000)):
            for template in ('multiview-C.json', 'dmpo.json'):
                c = compose('--family', family, '--objective', 'dmpo',
                            '--method', 'multiview-B-gaussian', '--views', '6', template=template)
                self.assertEqual(c['training']['beta'], beta)
                self.assertEqual(c['training']['objective'], 'dmpo')
                validate_config(c, files=False)
            c = compose('--family', family, '--objective', 'dmpo', '--beta', '123')
            self.assertEqual(c['training']['beta'], 123)
        # Reading a frozen experiment must preserve its chosen hyperparameters.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'frozen.json'
            target.write_text(json.dumps(c))
            restored = compose('--config', str(target))
            self.assertEqual(restored, c)
            restored = compose('--config', str(target), '--family', c['model']['family'])
            self.assertEqual(restored, c)

    def test_switching_kto_to_dpo_restores_dpo_scale_and_diagnostics(self):
        c = compose('--objective', 'dpo', template='kto.json')
        self.assertEqual(c['training']['beta'], 5000)
        self.assertNotIn('kto_desirable_weight', c['training'])
        self.assertTrue(c['training']['diagnostic'])
        validate_config(c, files=False)

    def test_dspo_family_recipes_overrides_and_frozen_preservation(self):
        for family, beta, recipe in (('sd15', .01, 'score_residual'),
                                      ('sdxl', 2000, 'official_sdxl')):
            c = compose('--family', family, '--objective', 'dspo')
            self.assertEqual((c['training']['beta'], c['training']['dspo_recipe']), (beta, recipe))
            validate_config(c, files=False)
            c = compose('--family', family, template='dspo.json')
            self.assertEqual((c['training']['beta'], c['training']['dspo_recipe']), (beta, recipe))
            custom = compose('--family', family, '--objective', 'dspo', '--beta', '123',
                             '--set', 'training.beta=456',
                             '--set', f'training.dspo_recipe={recipe}',
                             '--set', 'training.dspo_sdxl_threshold=0.2')
            self.assertEqual(custom['training']['beta'], 456)
            validate_config(custom, files=False)
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'frozen.json'
                target.write_text(json.dumps(custom))
                self.assertEqual(compose('--config', str(target), '--family', family), custom)
        switched = compose('--objective', 'dpo', template='dspo.json')
        self.assertEqual(switched['training']['beta'], 5000)
        self.assertNotIn('dspo_recipe', switched['training'])

    def test_dmpo_dspo_gaussian_and_stagewise_view_profiles(self):
        for family in ('sd15', 'sdxl'):
            for objective in ('dmpo', 'dspo'):
                for method in ('multiview-B-gaussian', 'multiview-C-gaussian', 'stagewise-C-gaussian'):
                    for k in (2, 4, 6):
                        c = compose('--family', family, '--objective', objective,
                                    '--method', method, '--views', str(k))
                        validate_config(c, files=False)
                        self.assertEqual(c['training']['noise_views'], k)
                        self.assertTrue(c['training']['batch_views'])
                        self.assertNotIn('independent_rollout', c['training'])
                invalid = compose('--family', family, '--objective', objective,
                                  '--method', 'stagewise-C-gaussian',
                                  '--set', 'training.stagewise.objective=phase')
                with self.assertRaises(ValueError):
                    validate_config(invalid, files=False)

    def test_sdxl_stagewise_comparison_entry_budgets_smoke_and_freezing(self):
        with tempfile.TemporaryDirectory() as directory:
            common = [sys.executable, str(ROOT / 'scripts/compare_objectives.py'),
                      '--family', 'sdxl', '--method', 'stagewise-C-gaussian',
                      '--config-only', '--config-dir', directory,
                      '--runs-root', str(Path(directory) / 'runs')]
            for objective, beta, recipe in (('dpo', 500, None), ('dmpo', 6000, None),
                                            ('dspo', 2000, 'official_sdxl')):
                for k in (2, 4, 6):
                    for budget in ('pairs', 'pair_views'):
                        name = f'{objective}-k{k}-{budget}'
                        args = common + ['--objective', objective, '--views', str(k),
                                         '--budget-unit', budget, '--run-name', name]
                        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        c = json.loads((Path(directory) / f'local-{name}.json').read_text())
                        t = c['training']
                        validate_config(c, files=False)
                        self.assertEqual((c['model']['family'], t['resolution']), ('sdxl', 1024))
                        self.assertEqual((t['objective'], t['beta'], t.get('dspo_recipe')),
                                         (objective, beta, recipe))
                        self.assertEqual((t['formal_arm'], t['noise_views'], t['stagewise']['objective']),
                                         ('C', k, 'dpo'))
                        pairs = t['batch_size'] * t['world_size'] * t['gradient_accumulation_steps']
                        self.assertEqual(pairs * (k if budget == 'pair_views' else 1), 192)
                        self.assertEqual(t['max_steps'], 10000)
                        if k == 6 and budget == 'pair_views':
                            result = subprocess.run(args + ['--beta', '123'], capture_output=True,
                                                    text=True, timeout=15)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn('Frozen config differs', result.stderr)
                name = f'{objective}-smoke'
                result = subprocess.run(common + ['--objective', objective, '--views', '6',
                    '--mode', 'smoke', '--run-name', name], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                c = json.loads((Path(directory) / f'local-{name}.json').read_text())
                self.assertEqual(c['training']['max_steps'], 2)
                self.assertEqual(c['training']['gradient_accumulation_steps'], 1)
                self.assertFalse(c['evaluation']['enabled'])

    def test_objective_templates_are_clean_single_view(self):
        for objective in ('sft', 'kto', 'mapo', 'dmpo'):
            c = compose(template=f'{objective}.json')
            self.assertEqual(c['training']['objective'], objective)
            self.assertEqual(c['method'], 'dpo')
            validate_config(c, files=False)


class ComparisonBudgetTests(unittest.TestCase):
    def make_config(self, output, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / 'scripts/make_config.py'),
             *arguments, '--out', str(output)],
            capture_output=True, text=True, timeout=15)

    def test_comparison_profiles_keep_steps_and_declared_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'config.json'
            for family, budget in (('sd15', 384), ('sdxl', 192)):
                for unit in ('pairs', 'pair_views'):
                    for k in (1, 2, 4, 6):
                        for world in (2, 4, 8):
                            with self.subTest(family=family, unit=unit, k=k, world=world):
                                batch = 2 if family == 'sd15' and k <= 2 else 1
                                result = self.make_config(output,
                                    '--family', family, '--method',
                                    'dpo' if k == 1 else 'multiview-C-gaussian',
                                    '--views', str(k), '--world-size', str(world),
                                    '--train_batch_size', str(batch),
                                    '--budget-unit', unit, '--budget-per-update', str(budget),
                                    '--max_train_steps', '10000')
                                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                                config = json.loads(output.read_text())
                                output.unlink()
                                validate_config(config, files=False)
                                t = config['training']
                                self.assertNotIn('extends', config)
                                self.assertEqual(t['comparison_budget'], {'unit': unit, 'per_update': budget})
                                self.assertEqual((t['batch_size'], t['noise_views'], t['max_steps']),
                                                 (batch, k, 10000))
                                pairs = batch * world * t['gradient_accumulation_steps']
                                # Check both exposure counts over the full experiment, not just accumulation.
                                expected_pairs = budget if unit == 'pairs' else budget // k
                                self.assertEqual(pairs * t['max_steps'], expected_pairs * 10000)
                                self.assertEqual(pairs * k * t['max_steps'], expected_pairs * k * 10000)

    def test_budget_cli_rejects_ambiguous_or_impossible_requests(self):
        base = ('--family', 'sd15', '--method', 'multiview-C-gaussian',
                '--views', '6', '--world-size', '2', '--train_batch_size', '1')
        valid = ('--budget-unit', 'pair_views', '--budget-per-update', '384')
        cases = (
            ('--budget-unit', 'pairs'),
            ('--budget-per-update', '384'),
            ('--budget-unit', 'pair_views', '--budget-per-update', '256'),
            (*valid, '--effective-batch', '384'),
            (*valid, '--gradient_accumulation_steps', '32'),
            (*valid, '--method', 'multiview-B'),
            (*valid, '--method', 'multiview-C'),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, arguments in enumerate(cases):
                with self.subTest(arguments=arguments):
                    output = Path(directory) / f'{index}.json'
                    result = self.make_config(output, *base, *arguments)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse(output.exists())

    def test_frozen_budget_rejects_changed_batch_accumulation_or_views(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'frozen.json'
            result = self.make_config(output,
                '--family', 'sd15', '--method', 'multiview-C-gaussian',
                '--views', '6', '--world-size', '2', '--train_batch_size', '1',
                '--budget-unit', 'pair_views', '--budget-per-update', '384')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads(output.read_text())
        validate_config(config, files=False)
        for key, value in (('batch_size', 2), ('gradient_accumulation_steps', 31),
                           ('noise_views', 4), ('world_size', 4)):
            with self.subTest(key=key):
                invalid = copy.deepcopy(config)
                invalid['training'][key] = value
                with self.assertRaises(ValueError):
                    validate_config(invalid, files=False)


if __name__ == '__main__':
    unittest.main()
