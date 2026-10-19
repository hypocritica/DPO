import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F
from self_correct_dpo.joint import JointRolloutBackend
from self_correct_dpo.independent import RolloutConfig, SingleRolloutBackend, vp_coefficients
from self_correct_dpo.losses import preference_logits, preference_loss
from self_correct_dpo.formal import formal_loss, MultiViewBackend


def backend(k=2, source='model', kind='epsilon', rollout=True):
    b = object.__new__(JointRolloutBackend if rollout else MultiViewBackend)
    b.family, b.training_config = 'sd15', {'noise_views': k}
    b.noise_views, b.resolution = k, 64
    b.rollout_config = RolloutConfig(state_source=source)
    b.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000, prediction_type=kind),
                                  alphas_cumprod=torch.linspace(.999, .001, 1000),
                                  add_noise=lambda x, n, t: x+n)
    b.weight = torch.tensor(.2, requires_grad=True)
    b.calls = []
    def encode(images, generator):
        b.calls.append('vae')
        return images[:, :, :1, :2, :2].detach()
    b._encode_images = encode
    b._condition = lambda texts: {'emb': torch.ones(len(texts), 1)}
    b._predict = lambda noisy, times, cond: b.weight * noisy
    b._autocast = nullcontext
    b.model = SimpleNamespace(training=True, eval=lambda: None, train=lambda flag: None, disable_adapter=nullcontext)
    return b


class JointTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.linspace(-.8,.8,2*2*3*64*64).reshape(2,2,3,64,64)

    def test_pool_encoding_pair_sharing_stopgrad_and_replay(self):
        b = backend()
        g = torch.Generator().manual_seed(17)
        before = g.get_state()
        prepared = b.prepare_views(self.images, [['p','p']]*2, g)
        after = g.get_state()
        self.assertEqual(b.calls, ['vae'])
        for view in prepared['views']:
            for key in ('noisy','target','x0','epsilon'):
                self.assertFalse(view[key].requires_grad)
                self.assertTrue(torch.isfinite(view[key]).all())
            torch.testing.assert_close(view['epsilon'][::2], view['epsilon'][1::2])
            torch.testing.assert_close(view['times'][::2], view['times'][1::2])
            self.assertFalse(torch.equal(view['x0'][::2],view['x0'][1::2]))
        self.assertFalse(torch.equal(prepared['views'][0]['epsilon'],prepared['views'][1]['epsilon']))
        g.set_state(before)
        repeated = b.prepare_views(self.images, [['p','p']]*2, g)
        self.assertTrue(torch.equal(after, g.get_state()))
        for v,w in zip(prepared['views'],repeated['views']):
            torch.testing.assert_close(v['noisy'],w['noisy'])

    def test_oracle_degenerates_to_matched_forward(self):
        for kind in ('epsilon','v_prediction'):
            b = backend(source='oracle', kind=kind)
            p = b.prepare_views(self.images, [['p','p']]*2, torch.Generator().manual_seed(8))
            for v in p['views']:
                a,s = vp_coefficients(b.scheduler.alphas_cumprod,v['times'],v['x0'])
                torch.testing.assert_close(v['noisy'],a*v['x0']+s*v['epsilon'],rtol=2e-5,atol=2e-6)
                expected = v['epsilon'] if kind == 'epsilon' else a*v['epsilon']-s*v['x0']
                torch.testing.assert_close(v['target'],expected,rtol=2e-5,atol=2e-6)

    def test_K1_exact_original_draw_state_and_loss(self):
        b = backend(k=1)
        single = SingleRolloutBackend.prepare_rollout(b,self.images,[['p']]*2,torch.Generator().manual_seed(11))
        joint = b.prepare_views(self.images,[['p']]*2,torch.Generator().manual_seed(11))['views'][0]
        for key in ('noisy','target','times','epsilon','x0'):
            torch.testing.assert_close(single[key],joint[key],rtol=0,atol=0)
        errors = torch.randn(3,1,2)
        refs = torch.randn_like(errors)
        c,_ = formal_loss({'policy_errors':errors,'reference_errors':refs},'C',2.)
        a,_ = preference_loss(errors,refs,'dpo',2.)
        torch.testing.assert_close(a,c)

    def test_streamed_all_view_parameter_gradient_equals_direct_autograd(self):
        for k in (1,2):
            for arm in ('B','C'):
                b = backend(k=k)
                p = b.prepare_views(self.images,[['p']*k]*2,torch.Generator().manual_seed(4))
                refs = torch.ones(2,k,2)
                errors = torch.cat([b.view_errors(p,j) for j in range(k)],1)
                full,_ = formal_loss({'policy_errors':errors,'reference_errors':refs},arm,2.)
                expected, = torch.autograd.grad(full,b.weight)
                with torch.no_grad():
                    logits = preference_logits(errors,refs,2.)
                    weight = torch.sigmoid(-logits.mean(1) if arm=='C' else -logits)
                for j in range(k):
                    logit = preference_logits(b.view_errors(p,j),refs[:,j:j+1],2.).squeeze(1)
                    (-((weight if arm=='C' else weight[:,j])*logit).mean()/k).backward()
                torch.testing.assert_close(b.weight.grad,expected,rtol=1e-5,atol=1e-7)

    def test_joint_backward_matches_recomputation_with_fewer_policy_forwards(self):
        for rollout, source in ((False, 'model'), (True, 'model'), (True, 'oracle')):
            for k in (1, 2):
                for arm in ('B', 'C'):
                    with self.subTest(rollout=rollout, source=source, k=k, arm=arm):
                        direct = backend(k=k, source=source, rollout=rollout)
                        modes = []
                        predict = direct._predict
                        def tracked(*args):
                            modes.append(torch.is_grad_enabled())
                            return predict(*args)
                        direct._predict = tracked
                        generator = torch.Generator().manual_seed(19)
                        values = direct.score_errors(self.images, [['p']*k]*2, generator)
                        self.assertTrue(values['policy_errors'].requires_grad)
                        self.assertFalse(values['reference_errors'].requires_grad)
                        loss, _ = formal_loss(values, arm, 2.)
                        loss.backward()
                        self.assertEqual(modes.count(True), k)
                        self.assertEqual(modes.count(False), k*(2 if rollout and source == 'model' else 1))

                        streamed = backend(k=k, source=source, rollout=rollout)
                        replay = torch.Generator().manual_seed(19)
                        with torch.no_grad():
                            old = streamed.score_errors(self.images, [['p']*k]*2, replay)
                            logits = preference_logits(old['policy_errors'], old['reference_errors'], 2.)
                            weight = torch.sigmoid(-logits.mean(1) if arm == 'C' else -logits)
                        for view in range(k):
                            errors = streamed.view_errors(old['prepared'], view)
                            logit = preference_logits(errors, old['reference_errors'][:, view:view+1], 2.).squeeze(1)
                            (-((weight if arm == 'C' else weight[:, view])*logit).mean()/k).backward()
                        torch.testing.assert_close(direct.weight.grad, streamed.weight.grad, rtol=1e-5, atol=1e-7)
                        torch.testing.assert_close(values['policy_errors'], old['policy_errors'], rtol=0, atol=0)
                        torch.testing.assert_close(values['reference_errors'], old['reference_errors'], rtol=0, atol=0)
                        self.assertTrue(torch.equal(generator.get_state(), replay.get_state()))


if __name__ == '__main__': unittest.main()
