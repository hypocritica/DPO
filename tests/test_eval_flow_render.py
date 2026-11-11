"""Flow renderer precision and prompt handling; full models are never loaded."""
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.evaluation.render import (cast_vae_decode_input,
                                               check_prompt_lengths, effective_sampling,
                                               validate_zimage_base, load_peft_adapter)


class FlowPromptTests(unittest.TestCase):
    def test_zimage_rejects_turbo_even_when_local_directory_is_renamed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {'path': str(root), 'revision': 'fixed-revision'}
            receipt = root / '.scd-download.json'
            receipt.write_text(json.dumps({'repo_id': 'Tongyi-MAI/Z-Image-Turbo',
                'revision': model['revision'], 'complete': True}))
            with self.assertRaisesRegex(ValueError, 'Base'):
                validate_zimage_base(model)
            receipt.write_text(json.dumps({'repo_id': 'Tongyi-MAI/Z-Image',
                'revision': model['revision'], 'complete': True}))
            validate_zimage_base(model)
            validate_zimage_base(dict(model, revision='local-inventory:artifact-hash'))
            with self.assertRaisesRegex(ValueError, 'revision'):
                validate_zimage_base(dict(model, revision='another-revision'))
            receipt.write_text(json.dumps({'repo_id': 'Tongyi-MAI/Z-Image',
                'revision': 'local-inventory:first', 'complete': True}))
            with self.assertRaisesRegex(ValueError, 'revision'):
                validate_zimage_base(dict(model, revision='local-inventory:second'))
            receipt.unlink()
            (root / '.download-manifest.json').write_text(json.dumps({
                'model_id': 'Tongyi-MAI/Z-Image-Turbo', 'revision': model['revision'], 'status': 'complete'}))
            with self.assertRaisesRegex(ValueError, 'Base'):
                validate_zimage_base(model)
            with self.assertRaisesRegex(ValueError, 'Turbo'):
                validate_zimage_base({'path': str(root / 'Z-Image-Turbo'), 'revision': 'r'})

    def test_qwen_chat_template_counts_toward_token_limit(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return 'start user ' + messages[0]['content'] + ' assistant think'
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=text.split())
        pipe = SimpleNamespace(tokenizer=Tokenizer())
        suite = [{'scene_id': 'a', 'variant_id': 'canonical', 'prompt': 'one two three'}]
        sampling = {'max_sequence_length': 6, 'negative_prompt': '', 'prompt_truncation': 'truncate'}
        truncated = check_prompt_lengths(pipe, suite, sampling, 'zimage')
        self.assertEqual(truncated, [{'prompt': 'a / canonical', 'tokenizer': 'Qwen3_chat', 'tokens': 7, 'limit': 6}])
        with self.assertRaisesRegex(ValueError, 'Qwen3'):
            check_prompt_lengths(pipe, suite, dict(sampling, prompt_truncation='error'), 'zimage')
        config = {'model': {'family': 'zimage', 'path': '/model', 'revision': 'r', 'max_sequence_length': 256},
                  'sampling': {'max_sequence_length': 512}}
        with self.assertRaisesRegex(ValueError, 'disagree'):
            effective_sampling(config)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('peft'),
                     'Optional PEFT environment is not installed')
class AdapterLoadTests(unittest.TestCase):
    def test_real_peft_rejects_missing_or_unexpected_adapter_weights(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file, save_file
        class Denoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 4)
            def forward(self, value):
                return self.projection(value)
        with tempfile.TemporaryDirectory() as directory:
            source = get_peft_model(Denoiser(), LoraConfig(r=2, target_modules=['projection']))
            source.save_pretrained(directory, safe_serialization=True)
            path = Path(directory) / 'adapter_model.safetensors'
            weights = load_file(str(path))
            loaded = load_peft_adapter(Denoiser(), directory)
            self.assertTrue(any('lora_' in name for name, _ in loaded.named_parameters()))
            incomplete = dict(weights)
            del incomplete[next(iter(incomplete))]
            save_file(incomplete, str(path))
            with self.assertRaisesRegex(ValueError, 'missing='):
                load_peft_adapter(Denoiser(), directory)
            save_file(dict(weights, unexpected=torch.ones(1)), str(path))
            with self.assertRaisesRegex(ValueError, 'unexpected='):
                load_peft_adapter(Denoiser(), directory)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('diffusers'),
                     'Optional Flow environment is not installed')
class FlowPrecisionTests(unittest.TestCase):
    def test_real_tiny_zimage_accepts_bf16_captions_with_fp32_weights(self):
        import diffusers
        if not hasattr(diffusers, 'ZImageTransformer2DModel'):
            self.skipTest('Z-Image needs the separate modern generation environment')
        import torch
        from self_correct_dpo.zimage_compat import fix_zimage_padding
        torch.manual_seed(17)
        model = diffusers.ZImageTransformer2DModel(in_channels=2, dim=32, n_layers=1,
            n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=16,
            axes_dims=[4, 6, 6], axes_lens=[128, 32, 32]).eval()
        images = [torch.randn(2, 1, 4, 4)]
        captions = [torch.randn(3, 16, dtype=torch.bfloat16)]
        times = torch.tensor([.5])
        with torch.inference_mode():
            expected = model(images, times, captions, return_dict=False)[0][0]
        fix_zimage_padding(model)
        with torch.inference_mode():
            output = model(images, times, captions, return_dict=False)[0][0]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.shape, images[0].shape)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_sd3_vae_decode_preserves_fp32_decoder(self):
        import torch
        class VAE(torch.nn.Module):
            dtype = torch.float32
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 3, 1)
            def decode(self, sample):
                return self.projection(sample)
        vae = cast_vae_decode_input(VAE())
        with torch.inference_mode():
            output = vae.decode(torch.ones(1, 2, 4, 4, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
