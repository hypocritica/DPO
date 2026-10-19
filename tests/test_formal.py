import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import torch
import torch.nn.functional as F
from self_correct_dpo.formal import formal_loss, sample_views
from self_correct_dpo.training import sample_batch
from self_correct_dpo.losses import preference_logits

class FormalTests(unittest.TestCase):
    def test_prepare_once_independent_draws_and_resume(self):
        from self_correct_dpo.formal import MultiViewBackend
        from types import SimpleNamespace
        backend = object.__new__(MultiViewBackend)
        backend.family = 'sd15'
        calls = []
        def encode(images, generator):
            calls.append('vae')
            return torch.randn((2, 2, 1, 2, 2), generator=generator)
        backend._encode_images = encode
        backend._condition = lambda texts: {'texts': texts}
        backend.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000, prediction_type='epsilon'),
                                             add_noise=lambda x, n, t: x+n)
        generator = torch.Generator().manual_seed(18)
        before = generator.get_state()
        prepared = backend.prepare_views(torch.zeros(2,2,3,4,4), [['canonical']]*2, generator)
        after = generator.get_state()
        self.assertEqual(calls, ['vae'])
        self.assertFalse(torch.equal(prepared['views'][0]['target'], prepared['views'][1]['target']))
        for view in prepared['views']:
            torch.testing.assert_close(view['target'][::2], view['target'][1::2])
            torch.testing.assert_close(view['times'][::2], view['times'][1::2])
        generator.set_state(before)
        replay = backend.prepare_views(torch.zeros(2,2,3,4,4), [['canonical']]*2, generator)
        self.assertTrue(torch.equal(generator.get_state(), after))
        for v, r in zip(prepared['views'], replay['views']):
            for key in v: torch.testing.assert_close(v[key], r[key])

    def test_full_and_streamed_C_gradients(self):
        p = torch.tensor([[[1.2, .9], [.8, 1.1]], [[.6, 1.3], [1.4, .5]]], requires_grad=True)
        r = torch.ones_like(p)
        loss, _ = formal_loss({'policy_errors': p, 'reference_errors': r}, 'C', 2.)
        expected, = torch.autograd.grad(loss, p)
        logits = preference_logits(p, r, 2.)
        weight = torch.sigmoid(-logits.mean(1)).detach()
        streamed = sum(-(weight * logits[:, j]).mean()/2 for j in range(2))
        actual, = torch.autograd.grad(streamed, p)
        torch.testing.assert_close(expected, actual)

    def test_stateless_pair_alignment_resume(self):
        rows=[{'pair_id':str(i),'prompts':[{'variant_id':v,'prompt':v+str(i)} for v in ['canonical','v1','v2']]} for i in range(9)]
        config={'seed':17,'training':{'batch_size':3,'sampling':'balanced','objective':'dpo','prompt_mode':'canonical','noise_views':2},'data':{}}
        for rank in range(2):
            a=sample_batch(rows,config,11,world=2,rank=rank)
            c=sample_views(rows,config,11,world=2,rank=rank)
            self.assertEqual(a[0],c[0])
            self.assertEqual(c,sample_views(rows,config,11,world=2,rank=rank))
            self.assertTrue(all(ids == ["canonical", "canonical"] for ids in c[2]))

if __name__=='__main__': unittest.main()
