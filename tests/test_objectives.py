"""Small mathematical checks for the paper objectives and their gradients."""
import math
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.algorithms.objectives import baseline_loss, dmpo_loss_from_logits, mapo_score


class ObjectiveTests(unittest.TestCase):
    def test_dmpo_is_reverse_kl_with_finite_optimum_and_gradient(self):
        alpha = .01
        u = torch.tensor([-20., 0., 2., math.log(99), 20.], dtype=torch.double, requires_grad=True)
        p = torch.stack((u.sigmoid(), (-u).sigmoid()), -1)
        target = torch.tensor([1-alpha, alpha], dtype=torch.double)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs=p), torch.distributions.Categorical(probs=target))
        actual = dmpo_loss_from_logits(u, alpha)
        torch.testing.assert_close(actual, expected)
        gradient, = torch.autograd.grad(actual.sum(), u)
        torch.testing.assert_close(gradient, u.sigmoid()*(-u).sigmoid()*(u-math.log(99)))
        self.assertAlmostEqual(actual[3].item(), 0., places=12)
        self.assertTrue(gradient[0] < 0 and gradient[-1] > 0)
        extreme = torch.tensor([-1e4, 1e4], requires_grad=True)
        dmpo_loss_from_logits(extreme).sum().backward()
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_dmpo_view_composition_and_reference_stop_gradient(self):
        for k in (1, 2, 4):
            for arm in ('B', 'C'):
                p = torch.linspace(.4, 1.4, 4*k, dtype=torch.double).reshape(2,k,2).requires_grad_()
                r = torch.ones_like(p, requires_grad=True)
                loss, _ = baseline_loss({'policy_errors':p,'reference_errors':r}, 'dmpo', 3., aggregation=arm)
                logits = -3*(p[...,0]-p[...,1])
                if arm == 'C': logits=logits.mean(1)
                expected = dmpo_loss_from_logits(logits).mean()
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(torch.autograd.grad(loss,p,retain_graph=True)[0],torch.autograd.grad(expected,p)[0])
                self.assertIsNone(r.grad)

    def test_mapo_matches_official_formula_and_stable_boundary(self):
        p = torch.tensor([[[.6,1.2]], [[1.1,.8]]],dtype=torch.double,requires_grad=True)
        score = .5*p / torch.expm1(.5*p)
        expected = p[...,0].mean() + .1*torch.nn.functional.softplus(-1000*(score[...,0]-score[...,1])).mean()
        loss, _ = baseline_loss({'policy_errors':p}, 'mapo', 1000)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(torch.autograd.grad(loss,p)[0],torch.autograd.grad(expected,p)[0])
        x=torch.tensor([0.,1e-8,1e4],dtype=torch.double,requires_grad=True)
        y=mapo_score(x)
        y.sum().backward()
        torch.testing.assert_close(y,torch.tensor([1.,1.-5e-9,0.],dtype=torch.double))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(x.grad[0].item(), -.5)

    def test_kto_individual_utilities_global_detached_reference_point(self):
        p=torch.tensor([[[.6,1.1]],[[.8,1.2]]],dtype=torch.double,requires_grad=True)
        r=torch.ones_like(p,requires_grad=True)
        kl=torch.tensor(.2,requires_grad=True)
        loss,_=baseline_loss({'policy_errors':p,'reference_errors':r},'kto',2.,kto_kl=kl)
        expected=(torch.sigmoid(-2*(1-p[...,0]-.2)).mean()+torch.sigmoid(2*(1-p[...,1]-.2)).mean())/2
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue((p.grad[...,0]>0).all() and (p.grad[...,1]<0).all())
        self.assertIsNone(r.grad)
        self.assertIsNone(kl.grad)

    def test_dspo_sufficient_statistics_match_vector_loss_and_gradients(self):
        generator = torch.Generator().manual_seed(41)
        for k in (1, 4, 6):
            for arm in ('B', 'C'):
                for recipe in ('score_residual', 'official_sdxl'):
                    for dtype in (torch.float32, torch.float64):
                        with self.subTest(k=k, arm=arm, recipe=recipe, dtype=dtype):
                            prediction = torch.randn(3, k, 2, 7, generator=generator, dtype=dtype, requires_grad=True)
                            reference = torch.randn(3, k, 2, 7, generator=generator, dtype=dtype, requires_grad=True)
                            target = torch.randn(3, k, 2, 7, generator=generator, dtype=dtype)
                            errors = (prediction-target).square().mean(-1)
                            refs = (reference.detach()-target).square().mean(-1)
                            gap = (prediction[..., 0, :]-reference.detach()[..., 0, :]).square().mean(-1)
                            # Exercise both SDXL branches within the same orbit.
                            threshold_means = torch.tensor([.2 if j%2 else .05 for j in range(k)],dtype=dtype,requires_grad=True)
                            beta = .3
                            actual, _ = baseline_loss({'policy_errors':errors, 'reference_errors':refs,
                                'winner_prediction_gaps':gap}, 'dspo', beta, aggregation=arm,
                                dspo_recipe=recipe, dspo_gap_mean=threshold_means)
                            z = -.5*beta*((errors[..., 0]-errors[..., 1])-(refs[..., 0]-refs[..., 1]))
                            if arm == 'C': z=z.mean(1,keepdim=True)
                            gate = torch.sigmoid(-z)
                            delta = prediction[..., 0, :]-reference.detach()[..., 0, :]
                            if recipe == 'score_residual':
                                expected = (prediction[..., 0, :]-target[..., 0, :]-beta*gate[...,None]*delta).square().mean()
                            else:
                                norm = (gate[...,None]*delta).square().mean(-1)
                                expected = torch.where(threshold_means.detach()>=.1,norm,gate.square().expand_as(norm)).mean()
                            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
                            gradient, ref_grad, branch_grad = torch.autograd.grad(actual,(prediction,reference,threshold_means),allow_unused=True,retain_graph=True)
                            torch.testing.assert_close(gradient,torch.autograd.grad(expected,prediction)[0],rtol=3e-6,atol=2e-7)
                            self.assertIsNone(ref_grad)
                            self.assertIsNone(branch_grad)

    def test_dspo_k1_equivalence_and_per_view_sdxl_branch(self):
        p=torch.tensor([[[.4,.8]],[[.6,.9]]],dtype=torch.double,requires_grad=True)
        r=torch.tensor([[[.5,.7]],[[.7,1.]]],dtype=torch.double)
        d=torch.tensor([[.02],[.08]],dtype=torch.double,requires_grad=True)
        for recipe in ('score_residual','official_sdxl'):
            b,_=baseline_loss({'policy_errors':p,'reference_errors':r,'winner_prediction_gaps':d},'dspo',.1,aggregation='B',dspo_recipe=recipe)
            c,_=baseline_loss({'policy_errors':p,'reference_errors':r,'winner_prediction_gaps':d},'dspo',.1,aggregation='C',dspo_recipe=recipe)
            torch.testing.assert_close(b,c,rtol=0,atol=0)
        # Same A/R and different D: scalar MSE pairs alone cannot define DSPO.
        values={'policy_errors':p.repeat(1,2,1),'reference_errors':r.repeat(1,2,1),
                'winner_prediction_gaps':torch.tensor([[.05,.2],[.05,.2]],dtype=torch.double)}
        loss,metrics=baseline_loss(values,'dspo',1.,dspo_recipe='official_sdxl')
        z=-.5*((values['policy_errors'][...,0]-values['policy_errors'][...,1])-(values['reference_errors'][...,0]-values['reference_errors'][...,1]))
        expected=(torch.sigmoid(-z).square()*torch.tensor([1.,.2],dtype=torch.double)).mean()
        torch.testing.assert_close(loss,expected)
        self.assertEqual(metrics['dspo_active_view_fraction'].item(),.5)
        with self.assertRaises(ValueError): baseline_loss(values,'dspo',1.)
        values.pop('winner_prediction_gaps')
        with self.assertRaises(ValueError): baseline_loss(values,'dspo',1.,dspo_recipe='score_residual')

    def test_dspo_regrouped_residual_avoids_beta_one_cancellation(self):
        # Equal margins imply g=.5; beta=2 gives scale=1 and exact ref error.
        # Naive A-scale*(A-R+D)+scale**2*D loses tiny R in FP32.
        a=torch.tensor([[[1e8,1e8]]],requires_grad=True)
        r=torch.tensor([[[.001,.001]]])
        d=torch.tensor([[1e8]],requires_grad=True)
        loss,_=baseline_loss({'policy_errors':a,'reference_errors':r,
            'winner_prediction_gaps':d},'dspo',2.,dspo_recipe='score_residual')
        torch.testing.assert_close(loss,r[...,0].mean(),rtol=0,atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(d.grad).all())

    def test_sft_uses_only_winners(self):
        p=torch.tensor([[[.5,1.]],[[.7,2.]]],requires_grad=True)
        loss,_=baseline_loss({'policy_errors':p},'sft',0.)
        loss.backward()
        torch.testing.assert_close(p.grad,torch.tensor([[[.5,0.]],[[.5,0.]]]))
        single,_=baseline_loss({'policy_errors':p[...,:1]},'sft',0.)
        torch.testing.assert_close(single,loss)


if __name__=='__main__': unittest.main()
