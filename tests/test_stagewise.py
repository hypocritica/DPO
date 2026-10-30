"""Small CPU contracts for within-stage Gaussian aggregation."""
import copy
import unittest
import torch
import torch.nn.functional as F
from test_joint import backend
from test_config import compose
from self_correct_dpo.algorithms.formal import formal_loss, stagewise_terms
from self_correct_dpo.models import shared_pair_noise
from self_correct_dpo.training import validate_config

SETTINGS = dict(boundary=500, objective='dpo', target_margin=1., huber_delta=1., huber_weight=.5)


class StagewiseTests(unittest.TestCase):
    def test_sampling_support_independence_uniform_marginal_and_replay(self):
        for boundary in (230, 500):
            b = backend(2, rollout=False)
            b.training_config['stagewise'] = dict(SETTINGS, boundary=boundary)
            images = torch.zeros(24000, 2, 1, 2, 2)
            g = torch.Generator().manual_seed(12)
            state = g.get_state()
            p = b.prepare_views(images, [['p', 'p']] * len(images), g)
            after = g.get_state()
            low = p['low_stage']
            self.assertAlmostEqual(low.float().mean().item(), boundary/1000, delta=.012)
            times = []
            for v in p['views']:
                t = v['times'][::2]
                times.append(t)
                self.assertTrue(torch.equal(t < boundary, low))
                self.assertTrue(torch.equal(v['times'][::2], v['times'][1::2]))
                self.assertTrue(torch.equal(v['target'][::2], v['target'][1::2]))
                histogram = torch.histc(t.float(), bins=10, min=0, max=1000) / len(t)
                self.assertTrue(torch.all((histogram - .1).abs() < .012))
            for mask, width, offset in ((low, boundary, 0), (~low, 1000-boundary, boundary)):
                a, c = [(t[mask].float()-offset) / width for t in times]
                correlation = torch.corrcoef(torch.stack([a, c]))[0, 1]
                self.assertLess(abs(correlation.item()), .05)
                self.assertAlmostEqual((times[0][mask] == times[1][mask]).float().mean().item(), 1/width, delta=.003)
            noises = [v['target'][::2].flatten() for v in p['views']]
            self.assertLess(abs(torch.corrcoef(torch.stack(noises))[0, 1].item()), .02)
            g.set_state(state)
            q = b.prepare_views(images, [['p', 'p']] * len(images), g)
            self.assertTrue(torch.equal(after, g.get_state()))
            for v, w in zip(p['views'], q['views']):
                for key in ('times', 'noisy', 'target'):
                    self.assertTrue(torch.equal(v[key], w[key]))

    def test_four_losses_signed_gradients_and_initial_coefficient(self):
        for objective in ('dpo', 'phase', 'constant', 'reverse'):
            m = torch.tensor([-2., 0., .5, 1., 2., 4.], requires_grad=True)
            low = torch.tensor([True, False, True, False, True, False])
            settings = dict(SETTINGS, objective=objective)
            loss, coefficient = stagewise_terms(m, low, settings)
            d = F.softplus(-m)
            h = .5 * F.huber_loss(m, torch.ones_like(m), reduction='none')
            expected = {'dpo': d, 'phase': torch.where(low, h, d),
                        'constant': .5*(d+h), 'reverse': torch.where(low, d, h)}[objective]
            torch.testing.assert_close(loss, expected)
            gradient, = torch.autograd.grad(loss.sum(), m)
            torch.testing.assert_close(gradient, coefficient)
            initial = torch.zeros(2, requires_grad=True)
            _, c = stagewise_terms(initial, torch.tensor([True, False]), settings)
            torch.testing.assert_close(c, torch.full_like(c, -.5))
        m = torch.tensor([2., 4.], requires_grad=True)
        _, c = stagewise_terms(m, torch.tensor([True, False]), dict(SETTINGS, objective='phase'))
        self.assertGreater(c[0].item(), 0)
        self.assertLess(c[1].item(), 0)

    def test_batched_sequential_and_streamed_gradients(self):
        images = torch.linspace(-.8,.8,2*2*3*64*64).reshape(2,2,3,64,64)
        for objective in ('dpo', 'phase', 'constant', 'reverse'):
            results = []
            for batched in (False, True):
                b = backend(2, rollout=False)
                settings = dict(SETTINGS, objective=objective)
                b.training_config.update(stagewise=settings, batch_views=batched)
                g = torch.Generator().manual_seed(17)
                v = b.score_errors(images, [['p']*2]*2, g)
                loss, metrics = formal_loss(v, 'C', 2., settings)
                loss.backward()
                results.append((loss.detach(), b.weight.grad.clone(), g.get_state()))
                self.assertEqual(b.calls, ['vae'])
                self.assertIn('stage_low_fraction', metrics)
                # Recompute view graphs using the detached analytical coefficient.
                b.weight.grad = None
                from self_correct_dpo.algorithms.losses import preference_logits
                margins = preference_logits(v['policy_errors'].detach(), v['reference_errors'], 2.)
                _, coefficient = stagewise_terms(margins.mean(1), v['prepared']['low_stage'], settings)
                for j in range(2):
                    logit = preference_logits(b.view_errors(v['prepared'], j), v['reference_errors'][:, j:j+1], 2.).squeeze(1)
                    (coefficient.detach() * logit).mean().div(2).backward()
                torch.testing.assert_close(b.weight.grad, results[-1][1])
            for a, c in zip(*results):
                torch.testing.assert_close(a, c)

    def test_existing_gaussian_draws_and_loss_unchanged(self):
        b = backend(2, rollout=False)
        images = torch.zeros(3, 2, 1, 2, 2)
        g, old = [torch.Generator().manual_seed(8) for _ in range(2)]
        p = b.prepare_views(images, [['p']*2]*3, g)
        for v in p['views']:
            n, t = shared_pair_noise(images, 1000, old)
            torch.testing.assert_close(v['target'], n.flatten(0, 1), rtol=0, atol=0)
            self.assertTrue(torch.equal(v['times'], t.flatten()))
        self.assertTrue(torch.equal(g.get_state(), old.get_state()))
        values = dict(policy_errors=torch.randn(3, 2, 2), reference_errors=torch.randn(3, 2, 2))
        from self_correct_dpo.algorithms.losses import preference_logits
        logits = preference_logits(values['policy_errors'], values['reference_errors'], 5000)
        for arm in ('B', 'C'):
            loss, _ = formal_loss(values, arm, 5000)
            torch.testing.assert_close(loss, F.softplus(-logits if arm == 'B' else -logits.mean(1)).mean(), rtol=0, atol=0)

    def test_configuration_switch_validation_and_budget(self):
        config = compose('--method', 'stagewise-C-gaussian')
        validate_config(config, files=False)
        self.assertEqual(config['training']['stagewise'], SETTINGS)
        self.assertEqual(config['training']['beta'], 5000)
        for method in ('dpo', 'multiview-B', 'multiview-C-gaussian'):
            changed = compose('--method', method, template='stagewise-C-gaussian.json')
            validate_config(changed, files=False)
            self.assertNotIn('stagewise', changed['training'])
        for key, value in (('boundary', 0), ('boundary', 1000), ('boundary', True),
                           ('objective', 'bad'), ('target_margin', float('nan')),
                           ('huber_delta', 0), ('huber_weight', -1)):
            bad = copy.deepcopy(config)
            bad['training']['stagewise'][key] = value
            with self.assertRaises(ValueError): validate_config(bad, files=False)
        for key, value in (('formal_arm', 'B'), ('independent_rollout', {}), ('paired_drift', {})):
            bad = copy.deepcopy(config); bad['training'][key] = value
            with self.assertRaises(ValueError): validate_config(bad, files=False)
        config['training']['comparison_budget'] = dict(unit='pair_views', per_update=512)
        validate_config(config, files=False)
        config['training']['comparison_budget']['per_update'] = 511
        with self.assertRaises(ValueError): validate_config(config, files=False)


if __name__ == '__main__': unittest.main()
