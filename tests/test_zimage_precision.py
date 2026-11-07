"""Real tiny Z-Image regression for FP32 weights with BF16 autocast."""
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import diffusers
from diffusers import ZImageTransformer2DModel
from packaging.version import Version
from peft import LoraConfig, get_peft_model
import torch

from self_correct_dpo.algorithms.adapter_ema import AdapterEMA
from self_correct_dpo.models import PreferenceBackend
from self_correct_dpo.zimage_compat import fix_zimage_padding


def tiny_transformer():
    torch.manual_seed(17)
    model = ZImageTransformer2DModel(in_channels=2, dim=32, n_layers=1,
        n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=16,
        axes_dims=[4, 6, 6], axes_lens=[128, 32, 32])
    with torch.no_grad():
        model.x_pad_token.normal_(std=.02)
        model.cap_pad_token.normal_(std=.02)
    return model


class ZImagePrecisionTests(unittest.TestCase):
    def test_real_reference_policy_backward_and_ema(self):
        for weight_dtype, activation_dtype in ((torch.float32, torch.bfloat16),
                (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)):
            with self.subTest(weights=weight_dtype, activations=activation_dtype):
                model = tiny_transformer().to(weight_dtype)
                noisy = torch.randn(2, 2, 4, 4).to(activation_dtype)
                condition = {'cap_feats': [torch.randn(n, 16).to(activation_dtype) for n in (3, 5)]}
                times = torch.tensor([800, 200])
                backend = object.__new__(PreferenceBackend)
                backend.family, backend.model = 'zimage', model
                def autocast():
                    return (torch.autocast('cpu', dtype=torch.bfloat16)
                            if activation_dtype == torch.bfloat16 else nullcontext())

                if (weight_dtype == torch.float32 and activation_dtype == torch.bfloat16
                        and Version(diffusers.__version__) < Version('0.40.0')):
                    with self.assertRaisesRegex(RuntimeError, 'dtypes match'):
                        with torch.no_grad(), autocast():
                            backend._predict(noisy, times, condition)

                fix_zimage_padding(model)
                model.requires_grad_(False)
                backend.model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2,
                    target_modules=['attention.to_q', 'attention.to_v']))
                for parameter in backend.model.parameters():
                    if parameter.requires_grad:
                        parameter.data = parameter.data.float()
                backend.model.enable_gradient_checkpointing()
                backend.ema = AdapterEMA(backend.model, .9)

                backend.model.eval()
                with torch.no_grad(), backend.model.disable_adapter(), autocast():
                    reference = backend._predict(noisy, times, condition)
                backend.model.train()
                with autocast():
                    prediction = backend._predict(noisy, times, condition)
                loss = (prediction.float() - torch.randn_like(prediction).float()).square().mean()
                loss.backward()
                gradients = [p.grad for p in backend.model.parameters() if p.requires_grad]
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(reference).all())
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
                backend.ema.update(backend.model)
                backend._using_ema_reference = True
                with autocast():
                    ema_prediction = backend._predict(noisy, times, condition)
                self.assertTrue(torch.isfinite(ema_prediction).all())
                self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_sequence_masks_values_and_fp32_equivalence(self):
        model = tiny_transformer()
        feats = [torch.randn(n, 32, requires_grad=True) for n in (4, 3)]
        ids = [torch.zeros(n, 3, dtype=torch.long) for n in (4, 3)]
        masks = [torch.tensor([False, False, True, True]), torch.tensor([False, False, True])]
        noise = [[1, 1, 0, 0], [1, 0, 0]]
        args = (feats, ids, masks, model.x_pad_token, noise, torch.device('cpu'))
        expected = model._prepare_sequence(*args)
        fix_zimage_padding(model)
        actual = model._prepare_sequence(*args)
        for old, new in zip(expected, actual):
            if isinstance(old, torch.Tensor):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            else:
                self.assertEqual(old, new)
        mixed = model._prepare_sequence([f.to(torch.bfloat16) for f in feats], *args[1:])
        self.assertEqual(mixed[0].dtype, torch.float32)
        torch.testing.assert_close(mixed[0][0, 2], model.x_pad_token[0], rtol=0, atol=0)
        mixed[0].sum().backward()
        torch.testing.assert_close(model.x_pad_token.grad, torch.full_like(model.x_pad_token, 3))
        self.assertEqual(feats[0].grad[2:].abs().sum().item(), 0)

    def test_modern_diffusers_keeps_native_method(self):
        model = tiny_transformer()
        native = model._prepare_sequence.__func__
        with patch.object(diffusers, '__version__', '0.40.0'):
            fix_zimage_padding(model)
        self.assertIs(model._prepare_sequence.__func__, native)


if __name__ == '__main__':
    unittest.main()
