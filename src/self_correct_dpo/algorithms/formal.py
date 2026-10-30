"""B averages per-view losses; C applies the loss to the mean margin."""
import torch
import torch.nn.functional as F
from self_correct_dpo.models import PreferenceBackend, shared_pair_noise
from self_correct_dpo.algorithms.losses import preference_logits


def _repeat_condition(condition, views):
    """Repeat SD1.5/SDXL conditioning in view-major order."""
    if isinstance(condition, torch.Tensor):
        return torch.cat([condition] * views, dim=0)
    if isinstance(condition, dict):
        return {key: _repeat_condition(value, views) for key, value in condition.items()}
    return condition


def sample_views(rows, config, microbatch, *, world=1, rank=0):
    from self_correct_dpo.training import sample_batch
    batch, prompts, ids = sample_batch(rows, config, microbatch, world=world, rank=rank)
    k = config["training"]["noise_views"]
    return batch, [[p[0]] * k for p in prompts], [[p[0]] * k for p in ids]


def stagewise_terms(mean, low_stage, settings):
    """Per-pair loss and signed derivative with respect to the mean margin."""
    dpo = F.softplus(-mean)
    residual = mean - settings['target_margin']
    delta, weight = settings['huber_delta'], settings['huber_weight']
    huber = weight * F.huber_loss(mean, torch.full_like(mean, settings['target_margin']),
                                  delta=delta, reduction='none')
    dpo_coefficient = -torch.sigmoid(-mean)
    huber_coefficient = weight * residual.clamp(-delta, delta)
    objective = settings['objective']
    if objective == 'dpo':
        return dpo, dpo_coefficient
    if objective == 'constant':
        return .5 * (dpo + huber), .5 * (dpo_coefficient + huber_coefficient)
    if objective not in ('phase', 'reverse'):
        raise ValueError(f'Unknown stagewise objective: {objective}')
    use_huber = low_stage if objective == 'phase' else ~low_stage
    return torch.where(use_huber, huber, dpo), torch.where(use_huber, huber_coefficient, dpo_coefficient)


def stagewise_loss(logits, low_stage, settings):
    mean = logits.mean(1)
    per_pair, coefficient = stagewise_terms(mean, low_stage, settings)
    # All bucket sums divide by the full pair count. Fractions retain the
    # denominator needed to interpret them after microbatch/rank averaging.
    metrics = {'stage_low_fraction': low_stage.float().mean(),
               'stage_coefficient_mean': coefficient.detach().mean()}
    for label, mask in (('low', low_stage), ('high', ~low_stage)):
        for name, value in (('margin', mean), ('loss', per_pair), ('coefficient', coefficient)):
            metrics[f'stage_{label}_{name}_weighted'] = (value.detach() * mask).mean()
    return per_pair.mean(), metrics


def formal_loss(values, arm, beta, stagewise=None):
    logits = preference_logits(values['policy_errors'], values['reference_errors'], beta)
    mean = logits.mean(1)
    stage_metrics = {}
    if stagewise is not None:
        if arm != 'C':
            raise ValueError('stagewise requires formal_arm=C')
        loss, stage_metrics = stagewise_loss(logits, values['prepared']['low_stage'], stagewise)
    elif arm == 'B':
        loss = F.softplus(-logits).mean()
    elif arm == 'C':
        loss = F.softplus(-mean).mean()
    else:
        raise ValueError(arm)
    return loss, {'loss': loss.detach(), 'margin_mean': mean.detach().mean(),
                  'margin_variance': logits.detach().var(1, unbiased=False).mean(),
                  'implicit_accuracy': (logits.detach() > 0).float().mean(), **stage_metrics}


class MultiViewBackend(PreferenceBackend):
    def prepare_views(self, images, prompts, generator):
        if self.family not in ('sd15','sdxl'):
            raise ValueError('Formal comparison requires SD1.5/SDXL')
        b = len(prompts)
        latents = self._encode_images(images, generator)
        texts = [ps[0] for ps in prompts for _ in range(2)]
        condition = self._condition(texts)
        views = []
        k = getattr(self, 'training_config', {}).get('noise_views', 2)
        stagewise = getattr(self, 'training_config', {}).get('stagewise')
        low_stage = None
        total = self.scheduler.config.num_train_timesteps
        if stagewise is not None:
            boundary = stagewise['boundary']
            if type(boundary) is not int or not 0 < boundary < total:
                raise ValueError('stagewise boundary must be inside the scheduler timestep range')
            low_stage = torch.randint(total, (b,), device=latents.device, generator=generator) < boundary
        for _ in range(k):
            if stagewise is None:
                noise, indices = shared_pair_noise(latents, total, generator)
            else:
                noise = torch.randn(latents[:, 0].shape, dtype=latents.dtype,
                                    device=latents.device, generator=generator)
                # Both integer draws are exact uniform; each view samples
                # independently inside its pair's shared stage.
                low = torch.randint(boundary, (b,), device=latents.device, generator=generator)
                high = torch.randint(boundary, total, (b,), device=latents.device, generator=generator)
                indices = torch.where(low_stage, low, high)[:, None].expand(-1, 2)
                noise = noise[:, None].expand_as(latents)
            flat, noise, times = latents.flatten(0, 1), noise.flatten(0, 1), indices.flatten()
            noisy = self.scheduler.add_noise(flat, noise, times)
            target = noise if self.scheduler.config.prediction_type == 'epsilon' else self.scheduler.get_velocity(flat, noise, times)
            views.append({'noisy': noisy, 'times': times, 'target': target})
        return {'views': views, 'condition': condition, 'batch_size': b, **({'low_stage': low_stage} if stagewise is not None else {})}

    def view_errors(self, prepared, view, reference=False):
        v = prepared['views'][view] if isinstance(view, int) else view
        b = prepared['batch_size']
        if reference:
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad(), self.model.disable_adapter(), self._autocast():
                    prediction = self._predict(v['noisy'], v['times'], prepared['condition'])
            finally:
                self.model.train(was_training)
        else:
            with self._autocast():
                prediction = self._predict(v['noisy'], v['times'], prepared['condition'])
        return (prediction.float() - v['target'].float()).square().flatten(1).mean(1).reshape(b, 1, 2)

    def score_errors(self, images, prompts, generator):
        """Score each view, optionally in one UNet batch; reference stays frozen."""
        prepared = self.prepare_views(images, prompts, generator)
        if self.training_config.get('batch_views', False):
            # Each view is [B*2,...], ordered winner/loser within each pair.
            # Concatenation gives [K*B*2,...]; restore [B,K,2] before the loss.
            b, k = prepared['batch_size'], len(prepared['views'])
            packed = {**prepared, 'batch_size': b * k,
                      'condition': _repeat_condition(prepared['condition'], k)}
            view = {key: torch.cat([v[key] for v in prepared['views']], dim=0)
                    for key in ('noisy', 'times', 'target')}
            # Score the frozen reference before retaining policy activations.
            refs = self.view_errors(packed, view, reference=True).reshape(k, b, 2).permute(1, 0, 2)
            policy = self.view_errors(packed, view).reshape(k, b, 2).permute(1, 0, 2)
            return {'policy_errors': policy, 'reference_errors': refs, 'prepared': prepared}
        policy = [self.view_errors(prepared, j) for j in range(len(prepared['views']))]
        refs = [self.view_errors(prepared, j, reference=True) for j in range(len(prepared['views']))]
        return {'policy_errors': torch.cat(policy, 1), 'reference_errors': torch.cat(refs, 1), 'prepared': prepared}

    def pair_errors(self, images, prompts, generator, reference=True):
        if len(prompts[0]) == 1:
            return super().pair_errors(images, prompts, generator, reference=reference)
        raise ValueError('Multiple views use score_errors with a joint all-view backward')
