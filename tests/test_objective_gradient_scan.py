"""Native loss identities, archive replay and task 4/5 guards on CPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from objective_scan import configuration, plan, location
from self_correct_dpo.common import digest
from self_correct_dpo.evaluation.appendix_native_gradients import (
    coefficients, collect_native_statistics, loss_value, objective_spec, validate_native_statistics,
)
from self_correct_dpo.evaluation.appendix_gradient_reporting import summarize_gradient_statistics
from self_correct_dpo.evaluation.appendix_native_probe import NativePairProbe, native_measurement
from self_correct_dpo.evaluation.appendix_probe import batch_check, load_checkpoint, check_adapter_metadata
from self_correct_dpo.evaluation.appendix_scan import fixed_conditions


def direct_loss(policy, reference, target, objective, beta, alpha, aggregation):
    policy_mse=(policy-target).square().mean(-1)
    reference_mse=(reference-target).square().mean(-1)
    rel=reference_mse-policy_mse
    if objective=='dmpo':
        u=beta*(rel[:,0]-rel[:,1])
        if aggregation=='C':u=u.mean()[None]
        return (torch.sigmoid(u)*(F.logsigmoid(u)-np.log1p(-alpha))+
                torch.sigmoid(-u)*(F.logsigmoid(-u)-np.log(alpha))).mean()
    m=.5*beta*(rel[:,0]-rel[:,1])
    gate=torch.sigmoid(-m.mean()) if aggregation=='C' else torch.sigmoid(-m)
    residual=policy[:,0]-target[:,0]-beta*gate[...,None]*(policy[:,0]-reference[:,0])
    return residual.square().mean()


class NativeGradientsTests(unittest.TestCase):
    def test_native_backend_batch_repartition_keeps_exact_objective_and_effective_batch(self):
        settings=dict(objective='dspo', beta=.01, batch_size=2, gradient_accumulation_steps=96)
        model=dict(family='sd15', path='/model')
        saved=dict(training={**settings, 'batch_size':16, 'gradient_accumulation_steps':12}, model=model)
        changes=check_adapter_metadata(saved,settings,model,'/adapter/backend.json',allow_batch_layout=True)
        self.assertEqual(changes,{'batch_size':{'training':2,'adapter':16},
                                 'gradient_accumulation_steps':{'training':96,'adapter':12}})
        self.assertEqual(settings['batch_size'],2)
        self.assertEqual(saved['training']['batch_size'],16)
        self.assertEqual(check_adapter_metadata(dict(training=settings,model=model),settings,model,'/x'),{})
        for overrides in ({'beta':.02}, {'gradient_accumulation_steps':13}, {'batch_size':False}, {'batch_size':16.0}):
            bad={**saved,'training':{**saved['training'],**overrides}}
            with self.assertRaisesRegex(ValueError,'/adapter/backend.json'):
                check_adapter_metadata(bad,settings,model,'/adapter/backend.json',allow_batch_layout=True)
        with self.assertRaisesRegex(ValueError,'training.batch_size'):
            check_adapter_metadata(saved,settings,model,'/x')
        with self.assertRaises(ValueError):
            check_adapter_metadata({**saved,'model':{**model,'path':'/other'}},settings,model,'/x',allow_batch_layout=True)

    def test_plan_covers_matched_steps_and_dspo_main_checkpoint(self):
        config=configuration(ROOT/'configs/appendix-objective-scan.json')
        p4,p5=plan(config,4),plan(config,5)
        self.assertEqual((p4['conditions'],p4['pair_view_observations'],p4['raw_component_VJPs']),(40,573440,1720320))
        self.assertEqual((p5['conditions'],p5['pair_view_observations']),(30,430080))
        self.assertEqual([r['step'] for r in p4['checkpoints']].count(6000),2)
        self.assertNotEqual(location(Path('/x'),4,'dspo',1000),location(Path('/x'),4,'stagewise_dspo_k6',1000))
        with self.assertRaises(ValueError):configuration(ROOT/'configs/appendix-objective-scan.json',3)

    def test_derivative_and_aggregation_match_original_reverse_kl_and_score_residual(self):
        gen=torch.Generator().manual_seed(41)
        slopes=torch.randn(8,2,3,2,generator=gen,dtype=torch.float64)
        base=torch.randn(8,2,3,generator=gen,dtype=torch.float64)
        reference=torch.randn(8,2,3,generator=gen,dtype=torch.float64)
        target=torch.randn(8,2,3,generator=gen,dtype=torch.float64)
        for objective,beta in [('dmpo',1.3),('dspo',.01),('dspo',2.)]:
            spec=objective_spec(dict(objective=objective,beta=beta,dmpo_alpha=.01,dspo_recipe='score_residual'))
            theta=torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
            policy=base+torch.einsum('bijk,k->bij',slopes,theta)
            a=(policy-target).square().mean(-1);r=(reference-target).square().mean(-1)
            rel=r-a
            z=(beta if objective=='dmpo' else .5*beta)*(rel[:,0]-rel[:,1])
            d=(policy[:,0]-reference[:,0]).square().mean(-1)
            x=z[:,None] if objective=='dmpo' else torch.stack((z,a[:,0],d,r[:,0]),1)
            h=torch.stack([torch.stack([torch.autograd.grad(x[i,j],theta,retain_graph=True)[0]
                                       for j in range(len(spec['components']))]) for i in range(8)])
            for mode in ('B','C'):
                for k in (1,2,4,8):
                    xx=x[:k].detach().numpy()
                    if mode=='B':g=(torch.tensor(coefficients(xx,spec))[:,:,None]*h[:k]).sum(1).mean(0)
                    else:g=(torch.tensor(coefficients(xx.mean(0),spec))[:,None]*h[:k].mean(0)).sum(0)
                    direct=direct_loss(policy[:k],reference[:k],target[:k],objective,beta,.01,mode)
                    expected=torch.autograd.grad(direct,theta,retain_graph=True)[0]
                    torch.testing.assert_close(g,expected,atol=1e-11,rtol=1e-10)
                    value=loss_value(xx,spec).mean() if mode=='B' else loss_value(xx.mean(0),spec)
                    self.assertAlmostEqual(float(direct.detach()),float(value),places=11)

    def test_dmpo_gradient_changes_sign_at_finite_optimum(self):
        spec=objective_spec(dict(objective='dmpo',beta=2000,dmpo_alpha=.01))
        optimum=np.log(99.)
        c=coefficients([[optimum-1],[optimum],[optimum+1]],spec)[:,0]
        self.assertLess(c[0],0);self.assertAlmostEqual(c[1],0);self.assertGreater(c[2],0)
        self.assertTrue(np.isfinite(coefficients([[-10000],[10000]],spec)).all())

    def test_native_component_batches_and_archive_replay_preserve_loss_gradients(self):
        class Toy(NativePairProbe):
            def __init__(self,objective):
                self.parameters=(torch.tensor([.2,-.1],requires_grad=True),)
                self.spec=objective_spec(dict(objective=objective,beta=.8,dmpo_alpha=.01,dspo_recipe='score_residual'))
            def moment_tensors(self,condition,seeds):
                slopes=torch.tensor([[s%5-2.,s%7-3.] for s in seeds])
                v=slopes@self.parameters[0]
                # A differentiable four-moment map; direct residual tests cover the model-specific reduction.
                x=v[:,None] if self.spec['name']=='dmpo' else torch.stack((v,v.square()+1,(v-.4).square(),v.detach()*0+1.2),1)
                return x,torch.full((len(seeds),),condition['index_start'])
        condition=fixed_conditions('sd15',[299])[0]
        sampling=dict(view_seed=23,ks=[1,2,4,6,8],source_blocks=2,source_views_per_block=24,audit_blocks=2,audit_views_per_block=8)
        scan=dict(rng_scope={'family':'sd15'},view_batch_size=1)
        for obj in ('dspo','dmpo'):
            probe=Toy(obj)
            self.assertTrue(batch_check(probe,[condition],4,17)['passed'])
            a=native_measurement(probe,condition,sampling,'p',None,scan)
            for size in (2,4):
                b=native_measurement(probe,condition,sampling,'p',None,{**scan,'view_batch_size':size})
                self.assertEqual(a['source_views'],b['source_views'])
                np.testing.assert_allclose(a['statistics']['block_mean_gram'],b['statistics']['block_mean_gram'],rtol=1e-6,atol=1e-7)
            raw=a['statistics']
            replay=summarize_gradient_statistics([dict(pair_id='p',condition=condition['condition'],statistics=raw)],bootstrap_repeats=8)
            metrics={(r['objective'],r['k']):r['metrics'] for r in replay['pairs'][0]['objectives']}
            self.assertEqual(metrics['lossavg',1],metrics['mvpo',1])
            audit_seeds=[v['seed'] for block in a['audit_views'] for v in block]
            av=list(probe.stat_gradients(condition,audit_seeds))
            am=np.mean([v[0] for v in av],axis=0);ah=torch.stack([v[1] for v in av]).mean(0)
            ref=(torch.tensor(coefficients(am,probe.spec),dtype=ah.dtype)[:,None]*ah).sum(0)
            source_seeds=[v['seed'] for block in a['source_views'] for v in block]
            sv=list(probe.stat_gradients(condition,source_seeds))
            xs=np.stack([v[0] for v in sv]);hs=torch.stack([v[1] for v in sv])
            gs=(torch.tensor(coefficients(xs,probe.spec),dtype=hs.dtype)[:,:,None]*hs).sum(1)
            for k in sampling['ks']:
                for method in ('lossavg','mvpo'):
                    groups=[]
                    for start in range(0,len(xs),k):
                        if method=='lossavg':g=gs[start:start+k].mean(0)
                        else:
                            c=torch.tensor(coefficients(xs[start:start+k].mean(0),probe.spec),dtype=hs.dtype)
                            g=(c[:,None]*hs[start:start+k].mean(0)).sum(0)
                        groups.append(g)
                    groups=torch.stack(groups);mean=groups.mean(0)
                    self.assertAlmostEqual(metrics[method,k]['distance_squared']['estimate'],float((mean-ref).square().sum()),delta=2e-6)
                    self.assertAlmostEqual(metrics[method,k]['stochastic_variance']['estimate'],float((groups-mean).square().sum()/(len(groups)-1)),delta=2e-6)
                    self.assertAlmostEqual(metrics[method,k]['descent_cosine']['estimate'],float(F.cosine_similarity(mean[None],ref[None])),delta=2e-6)

    def test_native_method_identity_cannot_use_the_dpo_shortcut(self):
        identity=dict(checkpoint_id='expected',run_id='wrong',step=1000)
        with patch('self_correct_dpo.evaluation.appendix_probe.resolve_checkpoint',return_value=({'family':'sd15'},None,identity)):
            with self.assertRaisesRegex(ValueError,'run differs'):
                load_checkpoint('sd15','/fake/checkpoints/step-0001000',expected={**identity,'run_id':'correct'},method='dspo')


if __name__=='__main__':unittest.main()
