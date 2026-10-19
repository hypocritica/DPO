"""Single-device SD1.5 / SDXL / SD3.5 Medium LoRA preference backends.

The training parameterizations follow Diffusers v0.35.1 examples:
https://github.com/huggingface/diffusers/blob/v0.35.1/examples/dreambooth/train_dreambooth_lora_sdxl.py
https://github.com/huggingface/diffusers/blob/v0.35.1/examples/dreambooth/train_dreambooth_lora_sd3.py
The local 0.33.1 pipeline and PEFT 0.10.0 source APIs were also inspected.
Real model execution still requires the project's GPU integration checks.
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path

import torch


def detach_condition(value):
    """Detach tensors throughout SDXL's nested added_cond_kwargs."""
    if isinstance(value,torch.Tensor):return value.detach()
    if isinstance(value,dict):return {key:detach_condition(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(detach_condition(item) for item in value)
    return value


def shared_pair_noise(latents, num_timesteps, generator=None):
    """One Gaussian noise/timestep draw per pair, shared by its two images.

    Input/output noise layout is [B,2,C,H,W]. The caller reuses these same
    tensors for every expression; there is no resampling inside the K loop.
    """
    if latents.ndim != 5 or latents.shape[1] != 2:
        raise ValueError("Expected latents [B,2,C,H,W]")
    if int(num_timesteps) < 1:
        raise ValueError("num_timesteps must be positive")
    noise = torch.randn(
        latents[:, 0].shape, dtype=latents.dtype, device=latents.device,
        generator=generator,
    )
    indices = torch.randint(
        int(num_timesteps), (latents.shape[0],), device=latents.device,
        generator=generator,
    )
    return noise.unsqueeze(1).expand_as(latents), indices[:, None].expand(-1, 2)


def flow_inputs(latents, noise, sigmas):
    """Forward flow x_sigma=(1-sigma)x+sigma*noise; velocity=noise-x."""
    if latents.shape != noise.shape:
        raise ValueError("Latents and noise must have identical shapes")
    while sigmas.ndim < latents.ndim:
        sigmas = sigmas.unsqueeze(-1)
    return (1 - sigmas) * latents + sigmas * noise, noise - latents


def scale_vae_latents(latents, scaling_factor, shift_factor=0.0):
    scale, shift = float(scaling_factor), float(shift_factor)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shift):
        raise ValueError("Invalid VAE scale/shift")
    return (latents - shift) * scale


def check_prompt_lengths(tokenizer, prompts, max_length, label):
    """Fail rather than silently change semantics through token truncation."""
    if tokenizer is None:
        raise ValueError(f"Required tokenizer missing: {label}")
    tokenized = tokenizer(prompts, padding=False, truncation=False, add_special_tokens=True)
    ids = tokenized["input_ids"]
    if any(len(row) > int(max_length) for row in ids):
        lengths = [len(row) for row in ids]
        raise ValueError(f"{label} truncation would occur: lengths={lengths}, limit={max_length}")


def validate_sd35_medium_config(config):
    """Reject old SD3 Medium and SD3.5 Large, despite the shared pipeline class.

    Architecture checks do not establish weight provenance; record upstream
    revision separately. Stability AI's Medium card specifies QK normalization,
    dual attention in the first 13 layers, and the extended positional grid.
    """
    expected = {"num_layers": 24, "num_attention_heads": 24,
                "qk_norm": "rms_norm", "pos_embed_max_size": 384}
    get = config.get if hasattr(config, "get") else lambda k, default=None: getattr(config, k, default)
    for key, value in expected.items():
        if get(key) != value:
            raise ValueError(f"sd35_medium architecture mismatch: {key}={get(key)!r}, expected {value!r}")
    if list(get("dual_attention_layers", [])) != list(range(13)):
        raise ValueError("sd35_medium requires dual_attention_layers=0..12")


class PreferenceBackend:
    """Frozen local base with one trainable PEFT adapter; no second base copy.

    Images must be RGB tensors in [-1,1], [B,2,3,resolution,resolution].
    Prompts are a rectangular B x K list, canonical first when K > 1.
    Only float32 and bfloat16 are supported; no implicit FP16 without scaling.
    This class contains no optimizer, model downloads, or distributed launcher.
    """

    def __init__(self, model_config, training_config, device):
        from diffusers import (
            AutoencoderKL, DDPMScheduler, FlowMatchEulerDiscreteScheduler,
            StableDiffusion3Pipeline, StableDiffusionXLPipeline, StableDiffusionPipeline,
        )
        from peft import LoraConfig, get_peft_model

        self.model_config = dict(model_config)
        self.training_config = dict(training_config)
        self.family = model_config["family"]
        if self.family not in {"sd15", "sdxl", "sd35_medium"}:
            raise ValueError(f"Unsupported model family: {self.family}")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Real PreferenceBackend requires an allocated CUDA device")
        dtype_name = training_config.get("dtype", "bfloat16")
        dtypes = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                  "float32": torch.float32, "fp32": torch.float32}
        if dtype_name not in dtypes:
            raise ValueError("Training dtype must be bfloat16 or float32")
        self.dtype = dtypes[dtype_name]
        self.resolution = int(training_config.get("resolution", 1024))
        if self.resolution <= 0 or self.resolution % 16:
            raise ValueError("resolution must be a positive multiple of 16")
        self.max_sequence_length = int(model_config.get("max_sequence_length", 256))
        if not 1 <= self.max_sequence_length <= 512:
            raise ValueError("max_sequence_length must be between 1 and 512")
        model_path = Path(model_config["path"]).expanduser().resolve()
        if not (model_path / "model_index.json").is_file():
            raise FileNotFoundError(f"Expected a complete local Diffusers pipeline: {model_path}")
        from self_correct_dpo.architecture import validate_sdxl_base, validate_vae_override
        validate_vae_override(model_config)
        if self.family=='sdxl':validate_sdxl_base(model_path)
        kwargs = {"local_files_only": True, "torch_dtype": self.dtype}
        if model_config.get("revision"):
            kwargs["revision"] = model_config["revision"]
        if self.family in {"sd15", "sdxl"}:
            pipeline = StableDiffusionPipeline if self.family == "sd15" else StableDiffusionXLPipeline
            extra = {} if self.family == "sd15" else {"add_watermarker": False}
            if self.family=='sdxl':
                # Load VAE directly in FP32; casting a BF16-loaded VAE back to
                # FP32 cannot recover the precision discarded at load time.
                vae_path=model_config['vae']['path'] if model_config.get('vae') else str(model_path/'vae')
                extra['vae']=AutoencoderKL.from_pretrained(vae_path,
                    local_files_only=True,torch_dtype=torch.float32)
            self.pipe = pipeline.from_pretrained(
                str(model_path), **extra, **kwargs,
            )
            self.scheduler = DDPMScheduler.from_pretrained(
                str(model_path), subfolder="scheduler", local_files_only=True,
            )
            if self.scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
                raise ValueError("SD1.5/SDXL supports only epsilon/v_prediction schedulers")
            component = "unet"
            targets = ["to_q", "to_k", "to_v", "to_out.0"]
            self.error_parameterization = f"diffusion_{self.scheduler.config.prediction_type}_mse"
        else:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(str(model_path), **kwargs)
            if self.pipe.text_encoder_3 is None or self.pipe.tokenizer_3 is None:
                raise ValueError("SD3.5 Medium requires the complete T5 conditioning branch")
            validate_sd35_medium_config(self.pipe.transformer.config)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(model_path), subfolder="scheduler", local_files_only=True,
            )
            if getattr(self.scheduler.config, "use_dynamic_shifting", False):
                raise ValueError("This backend requires the checkpoint's static flow shift schedule")
            component = "transformer"
            targets = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                       "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out"]
            self.error_parameterization = "flow_velocity_mse_uniform_schedule_index"
        self.component_name = component
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.frozen_modules = [self.pipe.vae, self.pipe.text_encoder]
        if self.family != "sd15":
            self.frozen_modules.append(self.pipe.text_encoder_2)
        if self.family == "sd35_medium":
            self.frozen_modules.append(self.pipe.text_encoder_3)
        for module in self.frozen_modules:
            if module is None:
                raise ValueError("A required frozen pipeline component is missing")
            module.requires_grad_(False).eval()
        # Both official LoRA examples keep VAE arithmetic in float32.
        self.pipe.vae.to(dtype=torch.float32)
        base = getattr(self.pipe, component)
        base.requires_grad_(False)
        rank = int(training_config.get("lora_rank", 16))
        alpha = float(training_config.get("lora_alpha", rank))
        if rank < 1 or alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("LoRA rank/alpha must be positive")
        self.model = get_peft_model(base, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0,
            target_modules=targets, init_lora_weights="gaussian", bias="none",
        ))
        setattr(self.pipe, component, self.model)
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"Unexpected trainable base parameter: {name}")
                parameter.data = parameter.data.float()
        if not self.trainable_parameters():
            raise RuntimeError("No LoRA parameters were attached")
        if training_config.get("gradient_checkpointing", True):
            self.model.enable_gradient_checkpointing(gradient_checkpointing_func=partial(torch.utils.checkpoint.checkpoint, use_reentrant=False))
        self.train()

    def trainable_parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def train(self):
        self.model.train()
        for module in self.frozen_modules:
            module.eval()
        return self

    def eval(self):
        self.model.eval()
        for module in self.frozen_modules:
            module.eval()
        return self

    def _autocast(self):
        return (torch.autocast(device_type=self.device.type, dtype=self.dtype)
                if self.dtype == torch.bfloat16 else nullcontext())

    @torch.no_grad()
    def _encode_images(self, images, generator):
        flat = images.flatten(0, 1).to(device=self.device, dtype=torch.float32)
        # Encode once so every expression uses the same VAE posterior sample.
        chunk_size = int(self.training_config.get("vae_batch_size", 4))
        latent = torch.cat([self.pipe.vae.encode(chunk).latent_dist.sample(generator=generator)
                            for chunk in flat.split(chunk_size)], dim=0)
        shift = self.pipe.vae.config.shift_factor if self.family == "sd35_medium" else 0.0
        latent = scale_vae_latents(latent, self.pipe.vae.config.scaling_factor, shift)
        return latent.to(self.dtype).reshape(images.shape[0], 2, *latent.shape[1:])

    @torch.no_grad()
    def _condition(self, prompts):
        if self.family == "sd15":
            check_prompt_lengths(self.pipe.tokenizer, prompts, self.pipe.tokenizer.model_max_length, "CLIP")
            embeds, _ = self.pipe.encode_prompt(prompt=prompts, device=self.device,
                num_images_per_prompt=1, do_classifier_free_guidance=False)
            return {"encoder_hidden_states": embeds.to(self.dtype)}
        for index, tokenizer in enumerate([self.pipe.tokenizer, self.pipe.tokenizer_2]):
            limit = self.pipe.tokenizer_max_length if self.family == "sd35_medium" else tokenizer.model_max_length
            check_prompt_lengths(tokenizer, prompts, limit, f"CLIP-{index + 1}")
        kwargs = dict(prompt=prompts, prompt_2=prompts, device=self.device,
                      num_images_per_prompt=1, do_classifier_free_guidance=False)
        if self.family == "sd35_medium":
            check_prompt_lengths(self.pipe.tokenizer_3, prompts, self.max_sequence_length, "T5")
            kwargs.update(prompt_3=prompts, max_sequence_length=self.max_sequence_length)
        embeds, _, pooled, _ = self.pipe.encode_prompt(**kwargs)
        embeds, pooled = embeds.to(self.dtype), pooled.to(self.dtype)
        if self.family == "sd35_medium":
            return {"encoder_hidden_states": embeds, "pooled_projections": pooled}
        # The data contract provides already resized square images, no random crop.
        time_ids = torch.tensor(
            [[self.resolution, self.resolution, 0, 0, self.resolution, self.resolution]],
            device=self.device, dtype=self.dtype,
        ).expand(len(prompts), -1)
        return {"encoder_hidden_states": embeds,
                "added_cond_kwargs": {"text_embeds": pooled, "time_ids": time_ids}}

    def enable_distributed(self):
        from torch.nn.parallel import DistributedDataParallel
        self.distributed_model = DistributedDataParallel(self.model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            broadcast_buffers=False, find_unused_parameters=False)
        return self

    def _predict(self, noisy, timesteps, condition):
        model = self.distributed_model if torch.is_grad_enabled() and hasattr(self, "distributed_model") else self.model
        if self.family in {"sd15", "sdxl"}:
            return model(noisy, timesteps, **condition, return_dict=False)[0]
        return model(hidden_states=noisy, timestep=timesteps,
                          **condition, return_dict=False)[0]

    def pair_errors(self, images, prompts, generator, reference=True):
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError("images must have shape [B,2,3,H,W]")
        if images.shape[0] == 0:
            raise ValueError("Empty image batch")
        if images.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError("Image dimensions differ from configured square resolution")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("images must be finite floating-point RGB tensors in [-1,1]")
        if images.min() < -1.00001 or images.max() > 1.00001:
            raise ValueError("images must be normalized to [-1,1]")
        batch_size = images.shape[0]
        if len(prompts) != batch_size or batch_size == 0:
            raise ValueError("Prompt/image batch size mismatch")
        if any(not isinstance(row, (list, tuple)) or not row for row in prompts):
            raise ValueError("prompts must contain nonempty expression lists")
        variants = len(prompts[0])
        if any(len(row) != variants or any(not isinstance(p, str) or not p.strip() for p in row)
               for row in prompts):
            raise ValueError("prompts must be a rectangular B x K matrix of nonempty strings")
        latents = self._encode_images(images, generator)
        noise, indices = shared_pair_noise(latents, self.scheduler.config.num_train_timesteps, generator)
        latents, noise, indices = latents.flatten(0, 1), noise.flatten(0, 1), indices.flatten()
        if self.family in {"sd15", "sdxl"}:
            timesteps = indices
            noisy = self.scheduler.add_noise(latents, noise, timesteps)
            target = noise if self.scheduler.config.prediction_type == "epsilon" else self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            timesteps = self.scheduler.timesteps.to(self.device)[indices]
            sigmas = self.scheduler.sigmas.to(device=self.device, dtype=self.dtype)[indices]
            noisy, target = flow_inputs(latents, noise, sigmas)
        policy, references = [], []
        for variant in range(variants):
            # Flattening order is [b0 winner, b0 loser, b1 winner, b1 loser, ...].
            texts = [prompts[b][variant] for b in range(batch_size) for _ in range(2)]
            condition = self._condition(texts)
            if reference:
                was_training = self.model.training
                self.model.eval()
                try:
                    with torch.no_grad(), self.model.disable_adapter(), self._autocast():
                        ref_prediction = self._predict(noisy, timesteps, condition)
                        ref_error = (ref_prediction.float() - target.float()).square().flatten(1).mean(1)
                        references.append(ref_error.reshape(batch_size, 2).detach())
                finally:
                    self.model.train(was_training)
            with self._autocast():
                prediction = self._predict(noisy, timesteps, condition)
            error = (prediction.float() - target.float()).square().flatten(1).mean(1)
            policy.append(error.reshape(batch_size, 2))
        return {"policy_errors": torch.stack(policy, dim=1),
                "reference_errors": torch.stack(references, dim=1) if reference else None,
                "timesteps": timesteps.reshape(batch_size, 2).detach()}

    def save_adapter(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path), safe_serialization=True)
        metadata = {
            "family": self.family, "component": self.component_name,
            "model": self.model_config, "error_parameterization": self.error_parameterization,
            "training": self.training_config,
        }
        (path / "backend.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def load_adapter(self, path):
        """Load into the existing parameters, retaining optimizer references."""
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file

        path = Path(path)
        metadata = json.loads((path / "backend.json").read_text(encoding="utf-8"))
        if metadata["family"] != self.family or metadata["component"] != self.component_name:
            raise ValueError("Adapter family/component differs from the backend")
        saved_model = metadata.get("model", {})
        # A remote machine may place the same traced base at a different path.
        for key in ("revision", "max_sequence_length"):
            if saved_model.get(key) != self.model_config.get(key):
                raise ValueError(f"Adapter base configuration mismatch: {key}")
        from self_correct_dpo.architecture import same_vae_revision
        if not same_vae_revision(saved_model,self.model_config):
            raise ValueError('Adapter VAE revision differs from the backend')
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        active = self.model.peft_config["default"]
        if config["r"] != active.r or float(config["lora_alpha"]) != float(active.lora_alpha):
            raise ValueError("Adapter rank/alpha differs from the backend")
        state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
        expected = get_peft_model_state_dict(self.model)
        if not state or set(state) != set(expected):
            raise ValueError("Adapter parameter names do not match the current model")
        for name, tensor in state.items():
            if tensor.shape != expected[name].shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid adapter tensor: {name}")
        set_peft_model_state_dict(self.model, state, adapter_name="default")
        return self
