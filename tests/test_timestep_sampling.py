"""Check final-time sampling without loading a diffusion model or CUDA."""
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from self_correct_dpo.algorithms.independent import RolloutConfig, sample_rollout_timesteps, vp_coefficients
from self_correct_dpo.algorithms.joint import JointRolloutBackend


def make_backend(source='model', kind='epsilon'):
    backend = object.__new__(JointRolloutBackend)
    backend.noise_views, backend.resolution = 2, 64
    backend.rollout_config = RolloutConfig(state_source=source, timestep_sampling='uniform_final')
    backend.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000, prediction_type=kind),
                                        alphas_cumprod=torch.linspace(.999, .001, 1000))
    backend.weight = torch.tensor(.2, requires_grad=True)
    backend._encode_images = lambda images, generator: images[:, :, :1, :2, :2].detach()
    backend._condition = lambda texts: {'emb': torch.ones(len(texts), 1)}
    backend._predict = lambda noisy, times, condition: backend.weight * noisy
    backend._autocast = nullcontext
    return backend


class TimestepSamplingTests(unittest.TestCase):
    def test_original_default_preserves_draws_and_rng_state(self):
        config = RolloutConfig()
        self.assertEqual(config.timestep_sampling, 'original')
        for batch in (1, 8, 37):
            actual_rng = torch.Generator().manual_seed(17)
            expected_rng = torch.Generator().manual_seed(17)
            actual = sample_rollout_timesteps(config, 1000, batch, 'cpu', actual_rng)
            up = torch.randint(53, 1000, (batch,), generator=expected_rng)
            sp = up - 33
            rp = sp + (torch.rand(batch, generator=expected_rng) * (1000-sp)).floor().long()
            for a, b in zip(actual, (up, sp, rp)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(torch.equal(actual_rng.get_state(), expected_rng.get_state()))
            torch.testing.assert_close(torch.randn(batch, 4, generator=actual_rng),
                                       torch.randn(batch, 4, generator=expected_rng), rtol=0, atol=0)

    def test_uniform_final_marginal_and_support(self):
        config = RolloutConfig(timestep_sampling='uniform_final')
        up, sp, rp = sample_rollout_timesteps(config, 1000, 196000, 'cpu', torch.Generator().manual_seed(31))
        self.assertTrue(((20 <= rp) & (rp < 1000)).all())
        self.assertTrue(((20 <= sp) & (sp <= rp) & (sp <= 966)).all())
        self.assertTrue(torch.equal(up, sp + 33))
        counts = torch.bincount(rp - 20, minlength=980).double()
        # 980 categories with expected count 200; fixed-seed broad chi-square bound.
        self.assertLess(float(((counts-200).square()/200).sum()), 1250.)
        bins = (rp - 20) // 98
        fractions = torch.bincount(bins, minlength=10).double() / len(rp)
        self.assertLess(float((fractions-.1).abs().max()), .004)

    def test_conditional_anchor_law_and_boundaries(self):
        total, stride, minimum = 11, 3, 2
        config = RolloutConfig(stride=stride, min_anchor_t=minimum, timestep_sampling='uniform_final')
        up, sp, rp = sample_rollout_timesteps(config, total, 180000, 'cpu', torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(sp[rp == minimum], torch.full_like(sp[rp == minimum], minimum)))
        self.assertEqual(int(sp.max()), total-1-stride)
        self.assertEqual(int(up.max()), total-1)
        for final in range(minimum, total):
            upper = min(final, total-1-stride)
            selected = sp[rp == final]
            self.assertTrue(((minimum <= selected) & (selected <= upper)).all())
            expected = 1. / (total-torch.arange(minimum, upper+1, dtype=torch.float64))
            expected /= expected.sum()
            observed = torch.bincount(selected-minimum, minlength=len(expected)).double() / len(selected)
            self.assertLess(float((observed-expected).abs().max()), .015)
        # Single admissible anchor remains valid for every final time.
        edge = RolloutConfig(stride=3, min_anchor_t=2, timestep_sampling='uniform_final')
        u, s, r = sample_rollout_timesteps(edge, 6, 1000, 'cpu', torch.Generator().manual_seed(8))
        self.assertTrue((s == 2).all())
        self.assertTrue((u == 5).all())
        self.assertTrue(((2 <= r) & (r <= 5)).all())
        with self.assertRaisesRegex(ValueError, 'timestep_sampling'):
            RolloutConfig(timestep_sampling='invalid').validate(1000)

    def test_k2_pair_sharing_independence_stopgrad_replay_and_bins(self):
        backend = make_backend()
        images = torch.linspace(-.8, .8, 8*2*3*64*64).reshape(8, 2, 3, 64, 64)
        generator = torch.Generator().manual_seed(17)
        before = generator.get_state()
        prepared = backend.prepare_views(images, [['p', 'p']]*8, generator)
        after = generator.get_state()
        views = prepared['views']
        for view in views:
            for key in ('noisy', 'target', 'x0', 'epsilon'):
                self.assertFalse(view[key].requires_grad)
                self.assertTrue(torch.isfinite(view[key]).all())
            torch.testing.assert_close(view['epsilon'][::2], view['epsilon'][1::2])
            torch.testing.assert_close(view['times'][::2], view['times'][1::2])
            self.assertFalse(torch.equal(view['x0'][::2], view['x0'][1::2]))
            expected_bins = (view['times'][::2]-20) // 98
            expected_fractions = torch.bincount(expected_bins, minlength=10).float() / 8
            observed = torch.stack([view['diagnostics'][f'rollout_r_bin_{j}_fraction'] for j in range(10)])
            torch.testing.assert_close(observed, expected_fractions)
            self.assertEqual(float(observed.sum()), 1.)
        self.assertFalse(torch.equal(views[0]['epsilon'], views[1]['epsilon']))
        self.assertFalse(torch.equal(views[0]['times'], views[1]['times']))
        generator.set_state(before)
        replay = backend.prepare_views(images, [['p', 'p']]*8, generator)
        self.assertTrue(torch.equal(after, generator.get_state()))
        for view, other in zip(views, replay['views']):
            for key in ('noisy', 'target', 'times', 'epsilon'):
                torch.testing.assert_close(view[key], other[key], rtol=0, atol=0)

    def test_uniform_final_oracle_degenerates_to_forward(self):
        images = torch.linspace(-.8, .8, 2*2*3*64*64).reshape(2, 2, 3, 64, 64)
        for kind in ('epsilon', 'v_prediction'):
            backend = make_backend(source='oracle', kind=kind)
            views = backend.prepare_views(images, [['p', 'p']]*2, torch.Generator().manual_seed(8))['views']
            for view in views:
                alpha, sigma = vp_coefficients(backend.scheduler.alphas_cumprod, view['times'], view['x0'])
                torch.testing.assert_close(view['noisy'], alpha*view['x0']+sigma*view['epsilon'], rtol=2e-5, atol=2e-6)
                expected = view['epsilon'] if kind == 'epsilon' else alpha*view['epsilon']-sigma*view['x0']
                torch.testing.assert_close(view['target'], expected, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
