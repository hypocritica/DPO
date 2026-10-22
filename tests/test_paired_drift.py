"""Tiny CPU tests of the actual paired-drift scoring path."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import torch
from test_joint import backend
from self_correct_dpo.algorithms.paired_drift import PairedDriftBackend, bounded_pair_drift, random_pair_direction, validate_paired_drift
from self_correct_dpo.algorithms.formal import formal_loss
from self_correct_dpo.models import PreferenceBackend
from self_correct_dpo.runtime.diagnostics import dpo_diagnostic


def paired(strength=1., equal=False, direction=None):
    b = backend(rollout=False)
    b.__class__ = PairedDriftBackend
    b.training_config['paired_drift'] = {'strength': strength, 'max_noise_rms': .1}
    if direction is not None:
        b.training_config['paired_drift']['direction'] = direction
    b.reference_mode = False
    class Disable:
        def __enter__(self): b.reference_mode = True
        def __exit__(self, *args): b.reference_mode = False
    b.model.disable_adapter = Disable
    b.forward_modes = []
    def predict(noisy, times, condition):
        b.forward_modes.append(torch.is_grad_enabled())
        return (b.weight if equal or not b.reference_mode else .1) * noisy
    b._predict = predict
    return b


class PairedDriftTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.linspace(-.8, .8, 2*2*3*64*64).reshape(2,2,3,64,64)
        self.prompts = [['p', 'p']] * 2

    def test_random_actual_score_amplitude_pair_sharing_rng_and_resume(self):
        contrast, random = paired(), paired(direction='random')
        gc, gr = (torch.Generator().manual_seed(11) for _ in range(2))
        saved = gr.get_state().clone()
        base = contrast.prepare_views(self.images, self.prompts, torch.Generator().manual_seed(11))
        vc = contrast.score_errors(self.images, self.prompts, gc)
        vr = random.score_errors(self.images, self.prompts, gr)
        self.assertTrue(torch.equal(gc.get_state(), gr.get_state()))
        dc = vc['prepared']['views'][1]['target'] - base['views'][1]['target']
        dr = vr['prepared']['views'][1]['target'] - base['views'][1]['target']
        rms = lambda x: x.square().flatten(1).mean(1).sqrt()
        torch.testing.assert_close(rms(dc), rms(dr), rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(dr[::2], dr[1::2])
        torch.testing.assert_close(vr['prepared']['views'][1]['noisy'] - base['views'][1]['noisy'], dr)
        self.assertFalse(torch.allclose(dr, dc))
        unit = dr[::2].flatten(1) / rms(dr[::2])[:, None]
        self.assertFalse(torch.allclose(unit[0], unit[1]))
        self.assertEqual(random.forward_modes, [True, False, True, False])
        self.assertFalse(vr['prepared']['views'][1]['target'].requires_grad)
        self.assertFalse(vr['prepared']['views'][1]['noisy'].requires_grad)
        self.assertFalse(vr['reference_errors'].requires_grad)
        formal_loss(vr, 'C', 2.)[0].backward()
        self.assertTrue(torch.isfinite(random.weight.grad))
        replay = torch.Generator().set_state(saved)
        repeated = paired(direction='random').score_errors(self.images, self.prompts, replay)
        torch.testing.assert_close(repeated['policy_errors'], vr['policy_errors'], rtol=0, atol=0)
        self.assertTrue(torch.equal(gr.get_state(), replay.get_state()))
        # Check the following batch too: local direction draws cannot alter VAE/base draws.
        pc = contrast.prepare_views(self.images, self.prompts, gc)
        pr = random.prepare_views(self.images, self.prompts, gr)
        for c, r in zip(pc['views'], pr['views']):
            for key in ('target', 'times', 'noisy'):
                torch.testing.assert_close(c[key], r[key], rtol=0, atol=0)

    def test_explicit_contrast_is_bitwise_default(self):
        for strength in (0., 1.):
            a, b = paired(strength), paired(strength, direction='contrast')
            ga, gb = (torch.Generator().manual_seed(5) for _ in range(2))
            va = a.score_errors(self.images, self.prompts, ga)
            vb = b.score_errors(self.images, self.prompts, gb)
            for key in ('policy_errors', 'reference_errors'):
                torch.testing.assert_close(va[key], vb[key], rtol=0, atol=0)
            self.assertTrue(torch.equal(ga.get_state(), gb.get_state()))

    def test_random_zero_and_small_signals(self):
        for strength, equal in ((0., False), (1., True), (1e-5, False)):
            a, b = paired(strength, equal), paired(strength, equal, direction='random')
            ga, gb = (torch.Generator().manual_seed(15) for _ in range(2))
            va = a.score_errors(self.images, self.prompts, ga)
            vb = b.score_errors(self.images, self.prompts, gb)
            self.assertTrue(torch.equal(ga.get_state(), gb.get_state()))
            if strength == 0 or equal:
                torch.testing.assert_close(va['policy_errors'], vb['policy_errors'], rtol=0, atol=0)
            else:
                self.assertLess(float(vb['rollout_diagnostics']['paired_drift_delta_rms']), 1e-4)
                self.assertGreater(float(vb['rollout_diagnostics']['paired_drift_delta_rms']), 0.)

    def test_random_direction_preserves_heterogeneous_pair_amplitudes(self):
        original = torch.ones(3, 2, 2, requires_grad=True) * torch.tensor([0., 1e-7, .1])[:, None, None]
        g = torch.Generator().manual_seed(7)
        before = g.get_state().clone()
        delta = random_pair_direction(original, g)
        torch.testing.assert_close(delta.square().mean((1, 2)).sqrt(),
                                   torch.tensor([0., 1e-7, .1]))
        self.assertFalse(delta.requires_grad)
        self.assertTrue(torch.equal(g.get_state(), before))
        # A cloned RNG would draw the future main noise; hashed substreams must differ.
        future = torch.randn(delta.shape, generator=g)
        self.assertFalse(torch.allclose(delta[2] / .1, future[2] / future[2].square().mean().sqrt()))

    def test_pair_difference_clip_and_stopgrad(self):
        p = torch.tensor([[[1.], [-1.]], [[.02], [0.]]], requires_grad=True)
        delta, diag = bounded_pair_drift(p, torch.zeros_like(p), 1., .1, torch.tensor([1., -1.]))
        torch.testing.assert_close(delta, torch.tensor([[.1], [-.01]]))
        self.assertFalse(delta.requires_grad)
        self.assertEqual(float(diag['paired_drift_clip_fraction']), .5)
        # Equal branch residuals must cancel rather than average.
        delta, _ = bounded_pair_drift(torch.ones_like(p), torch.zeros_like(p), 1., .1, torch.ones(2))
        self.assertEqual(float(delta.abs().sum()), 0.)

    def test_zero_and_reference_equality_recover_independent_K2(self):
        for strength, equal in ((0., False), (1., True)):
            b = paired(strength, equal)
            baseline = b.prepare_views(self.images, self.prompts, torch.Generator().manual_seed(8))
            g = torch.Generator().manual_seed(8)
            values = b.score_errors(self.images, self.prompts, g)
            for x, y in zip(baseline['views'], values['prepared']['views']):
                for key in ('noisy', 'target', 'times'):
                    torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
            if strength == 0:
                replay = torch.Generator().manual_seed(8)
                b.prepare_views(self.images, self.prompts, replay)
                self.assertTrue(torch.equal(g.get_state(), replay.get_state()))
            self.assertFalse(torch.equal(baseline['views'][0]['target'], baseline['views'][1]['target']))

    def test_actual_score_shared_pairs_detached_state_and_joint_gradient(self):
        b = paired()
        base = b.prepare_views(self.images, self.prompts, torch.Generator().manual_seed(11))
        b.calls.clear()
        values = b.score_errors(self.images, self.prompts, torch.Generator().manual_seed(11))
        self.assertEqual(b.calls, ['vae'])
        self.assertEqual(b.forward_modes, [True, False, True, False])
        for view in values['prepared']['views']:
            self.assertFalse(view['target'].requires_grad)
            self.assertFalse(view['noisy'].requires_grad)
            torch.testing.assert_close(view['times'][::2], view['times'][1::2])
            torch.testing.assert_close(view['target'][::2], view['target'][1::2])
        shift = values['prepared']['views'][1]['target'] - base['views'][1]['target']
        torch.testing.assert_close(shift[::2], shift[1::2])
        self.assertGreater(float(shift.abs().sum()), 0.)
        loss, _ = formal_loss(values, 'C', 2.)
        grad1, grad2 = torch.autograd.grad(loss, [values['policy_errors']], retain_graph=True)[0].unbind(1)
        self.assertGreater(float(grad1.abs().sum()), 0.)
        self.assertGreater(float(grad2.abs().sum()), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(b.weight.grad))
        self.assertFalse(values['reference_errors'].requires_grad)

    def test_configuration_boundaries(self):
        validate_paired_drift({'family': 'sd15'}, {'paired_drift': {'direction': 'random'}})
        with self.assertRaises(ValueError):
            validate_paired_drift({'family': 'sd15'}, {'paired_drift': {'direction': 'other'}})
        for extra in ({'noise_views': 1}, {'formal_arm': 'B'}, {'independent_rollout': {}}):
            with self.assertRaises(ValueError):
                validate_paired_drift({'family': 'sd15'}, {'paired_drift': {}, **extra})
        with self.assertRaises(ValueError):
            validate_paired_drift({'family': 'sdxl'}, {'paired_drift': {}})

    def test_uniform_diagnostic_routes_to_ordinary_single_view(self):
        b = paired()
        b.eval = lambda: None
        b.train = lambda: None
        values = {'policy_errors': torch.zeros(1, 1, 2),
                  'reference_errors': torch.zeros(1, 1, 2),
                  'timesteps': torch.tensor([[17, 17]])}
        config = {'training': {'resolution': 64, 'beta': 2.,
                              'diagnostic': {'seed': 3, 'pairs': 1}}}
        rows = [{'pair_id': 'test', 'prompts': [{'variant_id': 'canonical', 'prompt': 'p'}]}]
        with TemporaryDirectory() as out, \
             patch('self_correct_dpo.training.image_batch', return_value=self.images[:1]), \
             patch.object(PreferenceBackend, 'pair_errors', return_value=values) as ordinary, \
             patch.object(b, 'score_errors', side_effect=AssertionError('Diagnostic used paired drift')):
            record = dpo_diagnostic(b, rows, config, out, 0, 'cpu', 0)
        ordinary.assert_called_once()
        self.assertIs(ordinary.call_args.args[0], b)
        self.assertEqual(ordinary.call_args.args[2], [['p']])
        self.assertEqual(record['state_distribution'], 'ordinary_uniform_noise')
        self.assertEqual(record['timesteps'], [[[17, 17]]])


if __name__ == '__main__': unittest.main()
