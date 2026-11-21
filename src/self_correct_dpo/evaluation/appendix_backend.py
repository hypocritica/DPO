"""The two archived training backends used by the appendix probe."""
from functools import partial
from pathlib import Path

import torch

from self_correct_dpo.algorithms.formal import MultiViewBackend
from self_correct_dpo.models import flow_inputs, validate_sd35_medium_config


class AppendixBackend(MultiViewBackend):
    """Reuse the training backend and restore the archived SD3.5 VAE/text path."""

    def __init__(self, model, training, device):
        self.last_prompt_truncations = {}
        if model["family"] == "sd15":
            super().__init__(model, training, device)
        else:
            self._load_sd35(model, training, device)

    def _load_sd35(self, model, training, device):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
        from peft import LoraConfig, get_peft_model

        # SD3.5 training backend source recorded at 8cc8d24. The VAE weights
        # are loaded directly in FP32 before loading the remaining BF16 pipeline.
        self.model_config = dict(model)
        self.training_config = dict(training)
        self.family = model["family"]
        self.device = torch.device(device)
        self.dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[training["dtype"]]
        self.resolution = int(training["resolution"])
        self.max_sequence_length = int(model["max_sequence_length"])
        path = str(Path(model["path"]).expanduser().resolve(strict=True))
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", local_files_only=True, torch_dtype=torch.float32)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            path, vae=vae, local_files_only=True, torch_dtype=self.dtype, revision=model["revision"],
        )
        if self.pipe.text_encoder_3 is None or self.pipe.tokenizer_3 is None:
            raise ValueError("SD3.5 Medium requires the complete T5 conditioning branch")
        validate_sd35_medium_config(self.pipe.transformer.config)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler", local_files_only=True)
        if getattr(self.scheduler.config, "use_dynamic_shifting", False):
            raise ValueError("SD3.5 appendix probe requires the archived static schedule")
        self.component_name = "transformer"
        self.error_parameterization = "flow_velocity_mse_uniform_schedule_index"
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.frozen_modules = [vae, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.text_encoder_3]
        for module in self.frozen_modules:
            module.requires_grad_(False).eval()
        self.model = get_peft_model(self.pipe.transformer.requires_grad_(False), LoraConfig(
            r=training["lora_rank"], lora_alpha=training["lora_alpha"], lora_dropout=0.0,
            target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out"],
            init_lora_weights="gaussian", bias="none",
        ))
        self.pipe.transformer = self.model
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"Unexpected trainable parameter: {name}")
                parameter.data = parameter.data.float()
        if training["gradient_checkpointing"]:
            self.model.enable_gradient_checkpointing(
                gradient_checkpointing_func=partial(torch.utils.checkpoint.checkpoint, use_reentrant=False),
            )
        self.train()

    @torch.no_grad()
    def _condition(self, prompts):
        if self.family == "sd15":
            return super()._condition(prompts)
        unique = list(dict.fromkeys(prompts))
        lookup = {text: index for index, text in enumerate(unique)}
        order = torch.tensor([lookup[text] for text in prompts], device=self.device)
        weights = [prompts.count(text) for text in unique]
        policy = self.model_config["prompt_truncation"]
        limits = [
            ("CLIP-1", self.pipe.tokenizer, self.pipe.tokenizer_max_length),
            ("CLIP-2", self.pipe.tokenizer_2, self.pipe.tokenizer_max_length),
            ("T5", self.pipe.tokenizer_3, self.max_sequence_length),
        ]
        for label, tokenizer, limit in limits:
            ids = tokenizer(unique, padding=False, truncation=False, add_special_tokens=True)["input_ids"]
            count = sum(weight for row, weight in zip(ids, weights) if len(row) > limit)
            if count and policy != "truncate":
                raise ValueError(f"{label} would truncate {count} captions")
            self.last_prompt_truncations[label] = count
        embeds, _, pooled, _ = self.pipe.encode_prompt(
            prompt=unique, prompt_2=unique, prompt_3=unique, device=self.device,
            max_sequence_length=self.max_sequence_length, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {"encoder_hidden_states": embeds.to(self.dtype).index_select(0, order),
                "pooled_projections": pooled.to(self.dtype).index_select(0, order)}

    def pair_view_inputs(self, latents, noise, indices):
        latents, noise, indices = latents.flatten(0, 1), noise.flatten(0, 1), indices.flatten()
        if self.family == "sd15":
            return self.scheduler.add_noise(latents, noise, indices), indices, noise
        timesteps = self.scheduler.timesteps.to(self.device)[indices]
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=self.dtype)[indices]
        noisy, target = flow_inputs(latents, noise, sigmas)
        return noisy, timesteps, target

    def pair_mse(self, noisy, timesteps, target, condition, *, reference=False):
        # MultiViewBackend's reduction applies to a prepared view in both
        # archived runs. Its policy input stays differentiable.
        return self.view_errors(
            {"batch_size": noisy.shape[0] // 2, "condition": condition},
            {"noisy": noisy, "times": timesteps, "target": target}, reference=reference,
        )
