"""Independent detached rollout per view, then DPO on the mean margin."""
import torch
from self_correct_dpo.models import PreferenceBackend, detach_condition
from self_correct_dpo.algorithms.independent import SingleRolloutBackend, RolloutConfig, _require_finite
from self_correct_dpo.algorithms.formal import MultiViewBackend


class JointRolloutBackend(SingleRolloutBackend, MultiViewBackend):
    def __init__(self, model_config, training_config, device):
        k = training_config.get('noise_views')
        if model_config['family'] not in ('sd15','sdxl') or type(k) is not int or k < 1:
            raise ValueError('Joint rollout requires SD1.5/SDXL and a positive integer K')
        PreferenceBackend.__init__(self, model_config, training_config, device)
        self.noise_views = training_config['noise_views']
        self.rollout_config = RolloutConfig(**training_config['independent_rollout'])
        self.rollout_config.validate(self.scheduler.config.num_train_timesteps)

    @torch.no_grad()
    def prepare_views(self, images, prompts, generator):
        if not prompts or images.ndim != 5 or images.shape[:3] != (len(prompts), 2, 3):
            raise ValueError('Expected nonempty images [B,2,3,H,W] and B prompt rows')
        if images.shape[-2:] != (self.resolution, self.resolution) or not images.is_floating_point() or not torch.isfinite(images).all() or images.min() < -1.00001 or images.max() > 1.00001:
            raise ValueError('Expected configured resolution and finite normalized images')
        if any(not p or not isinstance(p[0], str) or not p[0].strip() for p in prompts):
            raise ValueError('Expected nonempty canonical prompts')
        canonical = [[p[0]] for p in prompts]
        if any(len(p) != self.noise_views or any(q != p[0] for q in p) for p in prompts):
            raise ValueError('Views must share the same canonical prompt')
        # One posterior draw per image pool; view randomness starts after encoding.
        latents = self._encode_images(images, generator).detach().float()
        condition = detach_condition(self._condition([p[0] for p in canonical for _ in range(2)]))
        views = [self.prepare_rollout(images, canonical, generator, encoded=latents,
                                     condition=condition) for _ in range(self.noise_views)]
        return {'views': views, 'condition': condition, 'batch_size': len(prompts)}

    def view_errors(self, prepared, view, reference=False):
        errors = MultiViewBackend.view_errors(self, prepared, view, reference=reference)
        _require_finite('reference MSE' if reference else 'policy MSE', errors)
        return errors.detach() if reference else errors

    def score_errors(self, images, prompts, generator):
        """Keep policy graphs for all views; rollout and reference are detached."""
        prepared = self.prepare_views(images, prompts, generator)
        gaps = {}
        if self.training_config.get('objective') == 'dspo':
            values = [self.view_errors_with_gap(prepared, j) for j in range(self.noise_views)]
            policy, refs, gap = [torch.cat([value[i] for value in values], 1) for i in range(3)]
            gaps['winner_prediction_gaps'] = gap
        else:
            policy = torch.cat([self.view_errors(prepared, j) for j in range(self.noise_views)], 1)
            refs = torch.cat([self.view_errors(prepared, j, reference=True) for j in range(self.noise_views)], 1)
        diagnostics = {}
        for j, view in enumerate(prepared['views']):
            for key, value in view['diagnostics'].items():
                diagnostics[f'{key}_view_{j}'] = value
                diagnostics[key] = diagnostics.get(key, 0) + value / self.noise_views
        return {'policy_errors': policy, 'reference_errors': refs, 'prepared': prepared,
                'rollout_diagnostics': diagnostics, **gaps}

    def pair_errors(self, images, prompts, generator, reference=True):
        # Diagnostics also use the joint distribution and the same frozen states.
        expanded = [[p[0]] * self.noise_views for p in prompts]
        prepared = self.prepare_views(images, expanded, generator)
        errors = torch.cat([self.view_errors(prepared, j) for j in range(self.noise_views)], 1)
        refs = torch.cat([self.view_errors(prepared, j, reference=True)
                          for j in range(self.noise_views)], 1) if reference else None
        return {'policy_errors': errors, 'reference_errors': refs}


class BaselineGaussianBackend(MultiViewBackend):
    """Gaussian baselines, including true winner-only supervised denoising."""

    def score_errors(self, images, prompts, generator):
        objective = self.training_config['objective']
        if objective == 'sft':
            from self_correct_dpo.models import flow_inputs
            # No loser VAE encoding, conditioning, UNet pass, or reference pass.
            with torch.no_grad():
                # The shared encoder handles FP32 VAE arithmetic, model scale/
                # shift and frozen-component placement, with one image per row.
                latents = self._encode_images(images[:, :1], generator).flatten(0, 1)
                condition = self._condition([p[0] for p in prompts])
                noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
                times = torch.randint(self.scheduler.config.num_train_timesteps,
                                      (len(prompts),), device=latents.device, generator=generator)
                if self.family in ('sd35_medium', 'zimage'):
                    indices = times
                    times = self.scheduler.timesteps.to(self.device)[indices]
                    sigmas = self.scheduler.sigmas.to(device=self.device, dtype=self.dtype)[indices]
                    noisy, target = flow_inputs(latents, noise, sigmas)
                else:
                    noisy = self.scheduler.add_noise(latents, noise, times)
                    target = noise if self.scheduler.config.prediction_type == 'epsilon' else self.scheduler.get_velocity(latents, noise, times)
            with self._autocast():
                prediction = self._predict(noisy, times, condition)
            errors = (prediction.float() - target.float()).square().flatten(1).mean(1)
            return {'policy_errors': errors[:, None, None], 'reference_errors': None,
                    'prepared': {'conditioning_diagnostics': dict(getattr(self, 'last_prompt_truncations', {}))}}
        if objective != 'mapo':
            return super().score_errors(images, prompts, generator)
        prepared = self.prepare_views(images, prompts, generator)
        policy = torch.cat([self.view_errors(prepared, j) for j in range(len(prepared['views']))], 1)
        return {'policy_errors': policy, 'reference_errors': None, 'prepared': prepared}
