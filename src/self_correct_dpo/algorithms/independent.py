"""Single detached independent-branch DDIM rollout for SD1.5/SDXL DPO (K=1)."""
from dataclasses import dataclass
from functools import lru_cache
import torch
from self_correct_dpo.models import PreferenceBackend, detach_condition

@dataclass(frozen=True)
class RolloutConfig:
    stride: int = 33
    min_anchor_t: int = 20
    state_source: str = 'model'
    cfg_scale: float = 1.0
    timestep_sampling: str = 'original'

    def validate(self, total):
        if not isinstance(self.stride, int) or self.stride < 1:
            raise ValueError('stride must be a positive integer')
        if not isinstance(self.min_anchor_t, int) or self.min_anchor_t < 0:
            raise ValueError('min_anchor_t must be a nonnegative integer')
        if self.stride + self.min_anchor_t >= total:
            raise ValueError('Empty rollout timestep support')
        if self.state_source not in ('model', 'oracle'):
            raise ValueError('state_source must be model or oracle')
        if self.cfg_scale != 1.0:
            raise ValueError('Only cfg_scale=1 is implemented')
        if self.timestep_sampling not in ('original', 'uniform_final'):
            raise ValueError('timestep_sampling must be original or uniform_final')


@lru_cache(maxsize=16)
def _anchor_cdf(total, stride, min_anchor_t, device):
    anchors = torch.arange(min_anchor_t, total - stride, device=device, dtype=torch.float64)
    return (1.0 / (total - anchors)).cumsum(0)


@torch.no_grad()
def sample_rollout_timesteps(config, total, batch, device, generator):
    """Sample pair-level u/s/r; uniform_final preserves the old s given r law."""
    config.validate(total)
    if config.timestep_sampling == 'original':
        # Keep the historical draw order and dtypes, including RNG consumption.
        up = torch.randint(config.stride + config.min_anchor_t, total, (batch,), device=device, generator=generator)
        sp = up - config.stride
        rp = sp + (torch.rand(batch, device=device, generator=generator) * (total-sp)).floor().long()
    else:
        rp = torch.randint(config.min_anchor_t, total, (batch,), device=device, generator=generator)
        upper = rp.clamp(max=total - 1 - config.stride) - config.min_anchor_t
        cdf = _anchor_cdf(total, config.stride, config.min_anchor_t, str(torch.device(device)))
        quantile = torch.rand(batch, device=device, dtype=torch.float64, generator=generator) * cdf[upper]
        sp = torch.searchsorted(cdf, quantile) + config.min_anchor_t
        up = sp + config.stride
    return up, sp, rp


def vp_coefficients(alphas_cumprod, times, like):
    """Return broadcast FP32 alpha/sigma on like.device for flattened samples."""
    cumulative = torch.as_tensor(alphas_cumprod, device=like.device, dtype=torch.float32)
    if cumulative.ndim != 1 or not torch.isfinite(cumulative).all() or not ((cumulative > 0) & (cumulative < 1)).all():
        raise ValueError('VP alphas_cumprod must lie strictly between zero and one')
    if times.ndim != 1 or times.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('times must be a one-dimensional integer tensor')
    if times.numel() != like.shape[0] or not ((times >= 0) & (times < cumulative.numel())).all():
        raise ValueError('times length/range is invalid')
    values = cumulative[times.to(device=like.device, dtype=torch.long)]
    shape = (values.numel(),) + (1,) * (like.ndim - 1)
    return values.sqrt().reshape(shape), (1 - values).sqrt().reshape(shape)


@torch.no_grad()
def prediction_to_epsilon_x0(noisy, prediction, alpha, sigma, prediction_type):
    noisy, prediction, alpha, sigma = [v.float() for v in (noisy, prediction, alpha, sigma)]
    if prediction_type == 'epsilon':
        epsilon_hat = prediction
        x0_hat = (noisy - sigma * epsilon_hat) / alpha
    elif prediction_type == 'v_prediction':
        epsilon_hat = alpha * prediction + sigma * noisy
        x0_hat = alpha * noisy - sigma * prediction
    else:
        raise ValueError('Only epsilon and v_prediction are supported')
    return epsilon_hat.detach(), x0_hat.detach()


@torch.no_grad()
def independent_rollout_state(x0, epsilon, prediction, alpha_u, sigma_u,
                              alpha_s, sigma_s, alpha_r, sigma_r,
                              prediction_type='epsilon'):
    """FP32 DDIM eta=0 followed by re-noising along the original epsilon.

    Every input sample has its own x0 and prediction; pair sharing is solely in
    epsilon and times. No clipping, rewards, correction term, or gradient path.
    """
    x0, epsilon = x0.float(), epsilon.float()
    alpha_u, sigma_u, alpha_s, sigma_s, alpha_r, sigma_r = [
        v.float() for v in (alpha_u, sigma_u, alpha_s, sigma_s, alpha_r, sigma_r)]
    x_u = alpha_u * x0 + sigma_u * epsilon
    epsilon_hat, x0_hat = prediction_to_epsilon_x0(x_u, prediction, alpha_u, sigma_u, prediction_type)
    z_s = alpha_s * x0_hat + sigma_s * epsilon_hat
    ratio = alpha_r / alpha_s
    x_r = ratio * z_s + (sigma_r - ratio * sigma_s) * epsilon
    epsilon_star = (x_r - alpha_r * x0) / sigma_r
    target = epsilon_star if prediction_type == 'epsilon' else alpha_r * epsilon_star - sigma_r * x0
    return {name: value.detach() for name, value in locals().copy().items()
            if name in ('x_u', 'z_s', 'x_r', 'epsilon_hat', 'x0_hat', 'epsilon_star', 'target')}


def _require_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f'Nonfinite {name}')



class SingleRolloutBackend(PreferenceBackend):
    def __init__(self, model_config, training_config, device):
        if model_config['family'] not in ('sd15','sdxl') or training_config.get('formal_arm', 'C') not in ('B','C') or training_config.get('noise_views') != 1:
            raise ValueError('Independent rollout requires SD1.5/SDXL, formal_arm=B/C, noise_views=1')
        super().__init__(model_config, training_config, device)
        self.rollout_config = RolloutConfig(**training_config['independent_rollout'])
        self.rollout_config.validate(self.scheduler.config.num_train_timesteps)

    def ordinary_pair_errors(self, images, prompts, generator, reference=True):
        return super().pair_errors(images, prompts, generator, reference=reference)

    @torch.no_grad()
    def prepare_rollout(self, images, prompts, generator, *, encoded=None, condition=None):
        if images.ndim != 5 or images.shape[1:3] != (2, 3) or images.shape[0] != len(prompts) or not prompts:
            raise ValueError('Expected nonempty images [B,2,3,H,W] and B prompt rows')
        if images.shape[-2:] != (self.resolution, self.resolution) or not images.is_floating_point() or not torch.isfinite(images).all() or images.min() < -1.00001 or images.max() > 1.00001:
            raise ValueError('Expected configured resolution and finite normalized images')
        if any(len(p) != 1 or not isinstance(p[0], str) or not p[0].strip() for p in prompts):
            raise ValueError('Independent rollout requires exactly one nonempty prompt per pair')
        cfg = self.rollout_config
        total = self.scheduler.config.num_train_timesteps
        cfg.validate(total)
        kind = self.scheduler.config.prediction_type
        if kind not in ('epsilon', 'v_prediction'):
            raise ValueError('Unsupported prediction type')
        latents = (self._encode_images(images, generator) if encoded is None else encoded).detach().float()
        x0 = latents.flatten(0, 1)
        _require_finite('encoded latents', x0)
        condition = self._condition([p[0] for p in prompts for _ in range(2)]) if condition is None else condition
        condition = detach_condition(condition)
        b = len(prompts)
        up, sp, rp = sample_rollout_timesteps(cfg, total, b, x0.device, generator)
        u, s, r = [v.repeat_interleave(2) for v in (up, sp, rp)]
        epsilon = torch.randn((b, *latents.shape[2:]), device=x0.device, dtype=torch.float32, generator=generator).repeat_interleave(2, 0)
        au, su = vp_coefficients(self.scheduler.alphas_cumprod, u, x0)
        ass, ss = vp_coefficients(self.scheduler.alphas_cumprod, s, x0)
        ar, sr = vp_coefficients(self.scheduler.alphas_cumprod, r, x0)
        xu = au*x0 + su*epsilon
        if cfg.state_source == 'model':
            with self._autocast():
                prediction = self._predict(xu, u, condition).detach().float()
        else:
            prediction = epsilon if kind == 'epsilon' else au*epsilon-su*x0
        _require_finite('rollout prediction', prediction)
        state = independent_rollout_state(x0, epsilon, prediction, au, su, ass, ss, ar, sr, kind)
        for name, value in state.items():
            _require_finite(name, value)
        displacement = (state['z_s']-(ass*x0+ss*epsilon)).square().flatten(1).mean(1).sqrt().reshape(b, 2)
        norm = state['target'].square().flatten(1).mean(1).sqrt().reshape(b, 2)
        diagnostics = {'rollout_u_mean': up.float().mean(), 'rollout_s_mean': sp.float().mean(), 'rollout_r_mean': rp.float().mean()}
        if cfg.timestep_sampling == 'uniform_final':
            bins = ((rp - cfg.min_anchor_t) * 10 // (total - cfg.min_anchor_t)).clamp(max=9)
            fractions = torch.bincount(bins, minlength=10).float() / b
            diagnostics.update({f'rollout_r_bin_{j}_fraction': fractions[j] for j in range(10)})
        for j, branch in enumerate(('winner', 'loser')):
            diagnostics['rollout_displacement_rms_'+branch] = displacement[:, j].mean()
            diagnostics['effective_label_rms_'+branch] = norm[:, j].mean()
        return {'noisy': state['x_r'], 'target': state['target'], 'times': r, 'condition': condition, 'diagnostics': diagnostics, 'x0': x0, 'epsilon': epsilon}

    def pair_errors(self, images, prompts, generator, reference=True):
        prepared = self.prepare_rollout(images, prompts, generator)
        b = len(prompts)
        refs = None
        if reference:
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad(), self.model.disable_adapter(), self._autocast():
                    prediction = self._predict(prepared['noisy'], prepared['times'], prepared['condition'])
                _require_finite('reference prediction', prediction)
                refs = (prediction.float()-prepared['target']).square().flatten(1).mean(1).reshape(b, 1, 2).detach()
                _require_finite('reference MSE', refs)
            finally:
                self.model.train(was_training)
        with self._autocast():
            prediction = self._predict(prepared['noisy'], prepared['times'], prepared['condition'])
        _require_finite('policy prediction', prediction)
        errors = (prediction.float()-prepared['target']).square().flatten(1).mean(1).reshape(b, 1, 2)
        _require_finite('policy MSE', errors)
        return {'policy_errors': errors, 'reference_errors': refs, 'timesteps': prepared['times'].reshape(b, 2), 'rollout_diagnostics': prepared['diagnostics']}
