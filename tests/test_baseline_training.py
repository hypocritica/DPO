"""CPU checks for actual winner-only and reference-free backend paths."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import torch
from self_correct_dpo.algorithms.joint import BaselineGaussianBackend
from self_correct_dpo.algorithms.objectives import baseline_loss
from test_joint import backend


class BaselineTrainingTests(unittest.TestCase):
    def test_sft_encodes_and_predicts_only_winners(self):
        for family in ('sd15', 'sdxl'):
            b = object.__new__(BaselineGaussianBackend)
            b.family, b.device, b.dtype = family, torch.device('cpu'), torch.float32
            b.training_config = {'objective': 'sft', 'vae_batch_size': 4}
            b.weight = torch.tensor(.2, requires_grad=True)
            encoded, conditioned, predicted = [], [], []
            def encode(x):
                encoded.append(x.clone())
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator: x[:, :1, :2, :2]))
            b.pipe = SimpleNamespace(vae=SimpleNamespace(encode=encode, config=SimpleNamespace(scaling_factor=1.)))
            b._condition = lambda texts: conditioned.append(texts) or {'encoder_hidden_states': torch.ones(len(texts), 1)}
            b.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000, prediction_type='epsilon'),
                                          add_noise=lambda x, n, t: x+n)
            b._autocast = nullcontext
            def predict(x, t, c):
                predicted.append(len(x))
                return b.weight*x
            b._predict = predict
            images = torch.randn(2, 1, 3, 4, 4)
            values = b.score_errors(images, [['a'], ['b']], torch.Generator().manual_seed(7))
            loss, _ = baseline_loss(values, 'sft', 1.)
            loss.backward()
            self.assertEqual(values['policy_errors'].shape, (2, 1, 1))
            self.assertIsNone(values['reference_errors'])
            self.assertEqual(predicted, [2])
            self.assertEqual(conditioned, [['a', 'b']])
            torch.testing.assert_close(encoded[0], images[:, 0])
            self.assertGreater(abs(b.weight.grad.item()), 0)

    def test_mapo_has_no_reference_forward(self):
        b = backend(k=1, rollout=False)
        b.__class__ = BaselineGaussianBackend
        b.training_config['objective'] = 'mapo'
        def forbidden():
            raise AssertionError('MaPO must never disable adapter for reference')
        b.model.disable_adapter = forbidden
        values = b.score_errors(torch.randn(2, 2, 3, 64, 64), [['a'], ['b']], torch.Generator().manual_seed(3))
        loss, _ = baseline_loss(values, 'mapo', 1.)
        loss.backward()
        self.assertIsNone(values['reference_errors'])
        self.assertTrue(torch.isfinite(b.weight.grad))

    def test_dmpo_composes_with_existing_view_samplers(self):
        for family in ('sd15', 'sdxl'):
            for k in (2, 4, 6):
                for rollout, stagewise in ((False, False), (True, False), (False, True)):
                    for arm in ('B', 'C'):
                        if stagewise and arm != 'C':
                            continue
                        b = backend(k=k, rollout=rollout)
                        b.family = family
                        if stagewise:
                            b.training_config['stagewise'] = {'boundary': 437}
                        values = b.score_errors(torch.zeros(2, 2, 3, 64, 64),
                            [['a']*k, ['b']*k], torch.Generator().manual_seed(31))
                        self.assertFalse(values['reference_errors'].requires_grad)
                        self.assertEqual(values['policy_errors'].shape, (2, k, 2))
                        loss, _ = baseline_loss(values, 'dmpo', 2., aggregation=arm)
                        loss.backward()
                        self.assertTrue(torch.isfinite(b.weight.grad))
                        self.assertGreater(abs(b.weight.grad.item()), 0)
                        if stagewise:
                            flags = values['prepared']['low_stage']
                            for view in values['prepared']['views']:
                                self.assertTrue(torch.equal(view['times'][::2] < 437, flags))

    def test_dspo_prediction_gap_and_batched_gradient(self):
        from contextlib import contextmanager
        for family in ('sd15', 'sdxl'):
            for k in (1, 2, 4, 6):
                for rollout, stagewise in ((False, False), (True, False), (False, True)):
                    results = []
                    for packed in ((False,) if rollout else (False, True)):
                        b = backend(k=k, rollout=rollout)
                        b.family = family
                        b.training_config.update(objective='dspo', batch_views=packed)
                        if stagewise:
                            b.training_config['stagewise'] = {'boundary': 437}
                        b.adapter = True
                        @contextmanager
                        def disabled():
                            b.adapter = False
                            try:
                                yield
                            finally:
                                b.adapter = True
                        b.model.disable_adapter = disabled
                        b._predict = lambda x, t, c: (b.weight if b.adapter else .1)*x
                        generator = torch.Generator().manual_seed(31)
                        values = b.score_errors(torch.full((2,2,3,64,64), .3),
                            [['a']*k, ['b']*k], generator)
                        if stagewise:
                            low_stage = values['prepared']['low_stage']
                            for view in values['prepared']['views']:
                                self.assertTrue(torch.equal(view['times'][::2] < 437, low_stage))
                        self.assertEqual(values['winner_prediction_gaps'].shape, (2,k))
                        expected = torch.stack([(.1*v['noisy'][::2]).square().flatten(1).mean(1)
                                                for v in values['prepared']['views']], dim=1)
                        torch.testing.assert_close(values['winner_prediction_gaps'], expected)
                        self.assertFalse(values['reference_errors'].requires_grad)
                        self.assertTrue(values['winner_prediction_gaps'].requires_grad)
                        arm_grad = []
                        for arm in ('B', 'C'):
                            loss, _ = baseline_loss(values, 'dspo', .01 if family == 'sd15' else 2.,
                                aggregation=arm, dspo_recipe='score_residual' if family == 'sd15' else 'official_sdxl',
                                dspo_sdxl_threshold=1e-12)
                            grad, = torch.autograd.grad(loss, b.weight, retain_graph=True)
                            self.assertTrue(torch.isfinite(grad))
                            self.assertGreater(abs(grad.item()), 0)
                            arm_grad.append((loss.detach(), grad))
                        results.append((values, arm_grad, generator.get_state()))
                    if len(results) == 2:
                        for key in ('policy_errors', 'reference_errors', 'winner_prediction_gaps'):
                            torch.testing.assert_close(results[0][0][key], results[1][0][key])
                        for arm in range(2):
                            for value in range(2):
                                torch.testing.assert_close(results[0][1][arm][value], results[1][1][arm][value])
                        self.assertTrue(torch.equal(results[0][2], results[1][2]))

    def test_dspo_rejects_unadapted_velocity_prediction(self):
        b = backend(k=4, rollout=False, kind='v_prediction')
        b.training_config['objective'] = 'dspo'
        with self.assertRaisesRegex(ValueError, 'epsilon prediction'):
            b.score_errors(torch.zeros(2,2,3,64,64), [['a']*4, ['b']*4],
                           torch.Generator().manual_seed(31))

    def test_new_objectives_train_pause_and_resume_exactly(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from test_stage_pause import ToyBackend
        from self_correct_dpo.training import train
        class BaselineToy(ToyBackend):
            def __init__(self, model, training, device):
                super().__init__(model, training, device)
                self.objective = training['objective']
                self.k = training['noise_views']
                self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
            def score_errors(self, images, prompts, generator):
                prepared = torch.randn(len(prompts), self.k, 2, generator=generator)
                values = {'policy_errors': (prepared + self.weight).square(),
                          'reference_errors': prepared.square()}
                if self.objective == 'dspo':
                    values['winner_prediction_gaps'] = self.weight.square().expand(len(prompts), self.k)
                if self.objective in ('sft', 'mapo'):
                    values['reference_errors'] = None
                if self.objective == 'sft':
                    values['policy_errors'] = values['policy_errors'][..., :1]
                return values
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root/'base'; base.mkdir()
            (base/'model_index.json').write_text('{}')
            rows = [{'pair_id': str(i), 'split': 'train', 'reviewed': True,
                     'prompts': [{'variant_id': 'canonical', 'prompt': str(i)}]} for i in range(3)]
            for objective in ('sft', 'mapo', 'kto', 'dmpo', 'dspo'):
                config = {'seed': 17, 'model': {'family': 'sd15', 'path': str(base)}, 'data': {},
                    'training': {'formal_arm': 'C', 'noise_views': 6 if objective in ('dmpo', 'dspo') else 1, 'objective': objective,
                        'batch_size': 1, 'max_steps': 4, 'gradient_accumulation_steps': 2,
                        'checkpoint_every': 2, 'resolution': 64, 'beta': 2., 'temperature': .1,
                        'learning_rate': .01, 'max_grad_norm': 1., 'prompt_mode': 'canonical',
                        'sampling': 'balanced', **({'dspo_recipe': 'score_residual'} if objective == 'dspo' else {})}}
                full, staged = root/(objective+'-full'), root/(objective+'-staged')
                with redirect_stdout(io.StringIO()), patch('self_correct_dpo.training.image_batch',
                        return_value=torch.zeros(1, 2, 3, 1, 1)):
                    complete = train(config, rows, full, backend_factory=BaselineToy, device='cpu')
                    train(config, rows, staged, backend_factory=BaselineToy, device='cpu', stop_after_step=2)
                    resumed = train(config, rows, staged, backend_factory=BaselineToy, device='cpu',
                                    resume=staged/'checkpoints/step-0000002')
                a, b = (path/'checkpoints/step-0000004' for path in (full, staged))
                torch.testing.assert_close(torch.load(a/'adapter/weight.pt', weights_only=True),
                    torch.load(b/'adapter/weight.pt', weights_only=True), rtol=0, atol=0)
                sa, sb = (torch.load(path/'state.pt', weights_only=True) for path in (a, b))
                torch.testing.assert_close(sa['generator'], sb['generator'], rtol=0, atol=0)
                self.assertEqual(sa['scheduler'], sb['scheduler'])
                for key in ('exp_avg', 'exp_avg_sq', 'step'):
                    torch.testing.assert_close(sa['optimizer']['state'][0][key],
                        sb['optimizer']['state'][0][key], rtol=0, atol=0)
                self.assertEqual(complete['cost']['reference_image_evaluations'],
                                 0 if objective in ('sft', 'mapo') else 16 * config['training']['noise_views'])
                self.assertEqual(complete['cost']['policy_image_evaluations'], 8 if objective == 'sft' else 16 * config['training']['noise_views'])


if __name__ == '__main__':
    unittest.main()
