"""Actual shift-3/6 flow schedules: stage sharing and marginal distribution."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from self_correct_dpo.algorithms.formal import MultiViewBackend, stagewise_metrics


class FlowStagewise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shifted_schedules_preserve_marginals_and_share_pair_stage(self):
        count = 24000
        for family, shift in (('sd35_medium', 3), ('zimage', 6)):
            for boundary in (250, 500, 750):
                with self.subTest(family=family, boundary=boundary):
                    backend = object.__new__(MultiViewBackend)
                    backend.family = family
                    backend.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
                    backend.training_config = {'noise_views':4, 'stagewise':{'boundary':boundary}}
                    latents = torch.zeros(count,2,1,1,1)
                    backend._encode_images = lambda images,generator: latents
                    backend._condition = lambda texts: None
                    prepared = backend.prepare_views(None,[['caption']]*count,torch.Generator().manual_seed(17))
                    mask = prepared['high_noise_stage']
                    self.assertEqual(prepared['stage_axis'], 'descending_flow_schedule_index')
                    torch.testing.assert_close(mask,prepared['low_stage'])
                    self.assertAlmostEqual(mask.float().mean().item(),boundary/1000,delta=.012)
                    indices = []
                    for view in prepared['views']:
                        # Recover real schedule index from shifted sigma, rather
                        # than treating native timestep as a uniform integer.
                        sigma = view['times'][::2]/1000
                        unshifted = sigma/(shift-(shift-1)*sigma)
                        index = (1000-1000*unshifted).round().long()
                        indices.append(index)
                        torch.testing.assert_close(view['times'][::2],view['times'][1::2])
                        torch.testing.assert_close(view['target'][::2],view['target'][1::2])
                        self.assertTrue(torch.equal(index < boundary,mask))
                        bins = torch.bincount(index//100,minlength=10).float()/count
                        torch.testing.assert_close(bins,torch.full((10,),.1),atol=.009,rtol=0)
                    for selected in (mask,~mask):
                        a,b = indices[0][selected].float(),indices[1][selected].float()
                        correlation = ((a-a.mean())*(b-b.mean())).mean()/(a.std(unbiased=False)*b.std(unbiased=False))
                        self.assertLess(abs(correlation.item()),.04)
                    self.assertFalse(torch.equal(prepared['views'][0]['target'],prepared['views'][1]['target']))
                    # Exact replay includes all stage/time/noise RNG draws.
                    replay = backend.prepare_views(None,[['caption']]*count,torch.Generator().manual_seed(17))
                    torch.testing.assert_close(replay['views'][0]['noisy'],prepared['views'][0]['noisy'],rtol=0,atol=0)
                    if boundary == 500:
                        expected = .75 if shift == 3 else 6/7
                        self.assertAlmostEqual(backend.scheduler.sigmas[500].item(),expected,delta=.002)

    def test_flow_metrics_use_physical_noise_names_without_changing_loss(self):
        logits = torch.tensor([[1.,3.],[-1.,-3.]],requires_grad=True)
        mask = torch.tensor([True,False])
        settings = {'objective':'dpo','target_margin':1.,'huber_delta':1.,'huber_weight':1.}
        flow_loss,metrics = stagewise_metrics(logits,{'low_stage':mask,'stage_axis':'descending_flow_schedule_index'},settings)
        legacy_loss,legacy = stagewise_metrics(logits,{'low_stage':mask},settings)
        torch.testing.assert_close(flow_loss,legacy_loss,rtol=0,atol=0)
        self.assertEqual(metrics['stage_high_noise_margin_weighted'].item(),1.)
        self.assertEqual(metrics['stage_low_noise_margin_weighted'].item(),-1.)
        self.assertNotIn('stage_low_fraction',metrics)
        self.assertIn('stage_low_fraction',legacy)


if __name__ == '__main__':
    unittest.main()
