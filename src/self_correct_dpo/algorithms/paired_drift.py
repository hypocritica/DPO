"""Two independent noise views with a detached, pair-shared second-view drift."""
import math
import hashlib
import torch
from self_correct_dpo.algorithms.formal import MultiViewBackend
from self_correct_dpo.models import PreferenceBackend


def validate_paired_drift(model, training):
    settings = training['paired_drift']
    if model['family'] != 'sd15' or training.get('formal_arm', 'C') != 'C' or training.get('noise_views', 2) != 2:
        raise ValueError('paired_drift requires SD1.5, formal_arm=C, noise_views=2')
    if 'independent_rollout' in training:
        raise ValueError('paired_drift and independent_rollout are mutually exclusive')
    if not isinstance(settings, dict) or set(settings) - {'strength', 'max_noise_rms', 'direction'}:
        raise ValueError('paired_drift accepts only strength, max_noise_rms and direction')
    if settings.get('direction', 'contrast') not in ('contrast', 'random'):
        raise ValueError('paired_drift.direction must be contrast or random')
    for key, default in (('strength', 1.0), ('max_noise_rms', 0.1)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f'paired_drift.{key} must be finite and nonnegative')


@torch.no_grad()
def bounded_pair_drift(policy, reference, strength, max_noise_rms, signs):
    """Inputs [B,2,...]; clip each pair's RMS without amplifying small drift."""
    residual = policy.detach().float() - reference.detach().float()
    velocity = 0.5 * (residual[:, 0] - residual[:, 1])
    raw = float(strength) * velocity
    dims = tuple(range(1, raw.ndim))
    rms = raw.square().mean(dims, keepdim=True).sqrt()
    scale = (float(max_noise_rms) / (rms + 1e-12)).clamp(max=1.0)
    delta = raw * scale * signs.reshape(-1, *([1] * (raw.ndim - 1)))
    return delta.detach(), {
        'paired_drift_v_rms': velocity.square().mean(dims).sqrt().mean(),
        'paired_drift_delta_rms': delta.square().mean(dims).sqrt().mean(),
        'paired_drift_clip_fraction': (rms > float(max_noise_rms)).float().mean(),
    }


@torch.no_grad()
def random_pair_direction(delta, generator):
    """Match each clipped pair RMS using an isolated, reproducible RNG stream."""
    # Domain separation avoids reusing the main stream's future base-view noise.
    state = generator.get_state().cpu().numpy().tobytes()
    digest = hashlib.sha256(b'paired-drift-random-direction-v1\0' + state).digest()
    local = torch.Generator(device=delta.device)
    local.manual_seed(int.from_bytes(digest[:8], 'little'))
    noise = torch.randn(delta.shape, device=delta.device, dtype=torch.float32, generator=local)
    dims = tuple(range(1, delta.ndim))
    amplitude = delta.float().square().mean(dims, keepdim=True).sqrt()
    noise_rms = noise.square().mean(dims, keepdim=True).sqrt()
    return (noise * (amplitude / noise_rms.clamp_min(torch.finfo(noise.dtype).tiny))).detach()


class PairedDriftBackend(MultiViewBackend):
    def ordinary_pair_errors(self, images, prompts, generator, reference=True):
        """Keep uniform-noise diagnostics identical to the single-view baseline."""
        return PreferenceBackend.pair_errors(self, images, prompts, generator, reference=reference)

    def __init__(self, model_config, training_config, device):
        validate_paired_drift(model_config, training_config)
        super().__init__(model_config, training_config, device)
        if self.scheduler.config.prediction_type != 'epsilon':
            raise ValueError('paired_drift supports epsilon prediction only')

    def _view_prediction(self, prepared, index, reference=False):
        view = prepared['views'][index]
        if reference:
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad(), self.model.disable_adapter(), self._autocast():
                    return self._predict(view['noisy'], view['times'], prepared['condition'])
            finally:
                self.model.train(was_training)
        with self._autocast():
            return self._predict(view['noisy'], view['times'], prepared['condition'])

    def score_errors(self, images, prompts, generator):
        # Draw both views before the sign so strength=0 exactly replays K2 RNG.
        prepared = self.prepare_views(images, prompts, generator)
        b = prepared['batch_size']
        policy1 = self._view_prediction(prepared, 0)
        ref1 = self._view_prediction(prepared, 0, reference=True)
        settings = self.training_config['paired_drift']
        strength, bound = settings.get('strength', 1.0), settings.get('max_noise_rms', 0.1)
        signs = torch.ones(b, device=policy1.device)
        if strength != 0:
            signs = torch.randint(0, 2, (b,), device=policy1.device, generator=generator) * 2 - 1
        delta, diagnostics = bounded_pair_drift(policy1.reshape(b, 2, *policy1.shape[1:]),
            ref1.reshape(b, 2, *ref1.shape[1:]), strength, bound, signs)
        if settings.get('direction', 'contrast') == 'random':
            delta = random_pair_direction(delta, generator)
        with torch.no_grad():
            second = prepared['views'][1]
            # add_noise is linear in noise; preserve the original base state and dtype.
            shift = delta[:, None].expand(-1, 2, *delta.shape[1:]).flatten(0, 1)
            shift = shift.to(second['target'].dtype)
            second['target'] = (second['target'] + shift).detach()
            second['noisy'] = (second['noisy'] + self.scheduler.add_noise(
                torch.zeros_like(shift), shift, second['times'])).detach()
        policy2 = self._view_prediction(prepared, 1)
        ref2 = self._view_prediction(prepared, 1, reference=True)
        def error(prediction, index):
            return (prediction.float() - prepared['views'][index]['target'].float()).square().flatten(1).mean(1).reshape(b, 1, 2)
        return {'policy_errors': torch.cat([error(policy1, 0), error(policy2, 1)], 1),
                'reference_errors': torch.cat([error(ref1, 0), error(ref2, 1)], 1),
                'prepared': prepared, 'rollout_diagnostics': diagnostics}

    def pair_errors(self, images, prompts, generator, reference=True):
        return self.score_errors(images, [[p[0]] * 2 for p in prompts], generator)
