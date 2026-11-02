"""Paper-defined diffusion baselines, independent of model/view construction.

DMPO: arxiv:2507.07510v1, Eq. 14 and Appendix B.3 (beta absorbs T*w).
MaPO: official mapo-t2i/mapo revision 374b3d3, utils.compute_loss.
KTO: official jacklishufan/diffusion-kto revision 226bbd0, sigmoid utility.
DSPO: official huaishengzhu/DSPO revision 8b4e09b, separate SD1.5/SDXL recipes.
The input errors are per-image mean denoising MSE, not exact log likelihoods.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


BASELINE_OBJECTIVES = frozenset({'sft', 'mapo', 'kto', 'dmpo', 'dspo'})


def _positive(name, value, allow_zero=False):
    if not math.isfinite(float(value)) or (float(value) < 0 if allow_zero else float(value) <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')


def _errors(values, reference=False):
    errors = values['policy_errors']
    if errors.ndim != 3 or min(errors.shape) < 1 or errors.shape[-1] not in (1, 2):
        raise ValueError('policy_errors must be nonempty [B,K,1 or 2]')
    if not torch.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('Expected finite nonnegative denoising errors')
    # Preserve double precision for mathematical checks; promote low precision.
    errors = errors if errors.dtype == torch.float64 else errors.float()
    if not reference:
        return errors, None
    refs = values.get('reference_errors')
    if refs is None or refs.shape != errors.shape or not torch.isfinite(refs).all() or (refs < 0).any():
        raise ValueError('Matching finite nonnegative reference_errors required')
    return errors, refs.detach().to(errors)


def dmpo_loss_from_logits(logits, alpha=0.01):
    """Elementwise reverse KL(Bernoulli(sigmoid(u)) || Bernoulli(1-alpha)).

    Its finite optimum u=log((1-alpha)/alpha) is essential to DMPO; this is
    neither label-smoothed DPO cross entropy nor a dynamic-beta objective.
    """
    if not math.isfinite(float(alpha)) or not 0 < float(alpha) < .5:
        raise ValueError('dmpo_alpha must be in (0, 0.5)')
    if not torch.isfinite(logits).all():
        raise ValueError('DMPO logits must be finite')
    return (torch.sigmoid(logits) * (F.logsigmoid(logits) - math.log1p(-float(alpha)))
            + torch.sigmoid(-logits) * (F.logsigmoid(-logits) - math.log(float(alpha))))


def mapo_score(x):
    """Stable x/(exp(x)-1), including the removable singularity at zero."""
    # Use a Taylor branch near zero and exp(-x) elsewhere to avoid overflow.
    small = x.abs() < 1e-3
    safe = torch.where(small, torch.ones_like(x), x)
    regular = safe * torch.exp(-safe) / (-torch.expm1(-safe))
    near_zero = torch.where(small, x, torch.zeros_like(x))
    series = 1 - near_zero / 2 + near_zero.square() / 12 - near_zero.pow(4) / 720
    return torch.where(small, series, regular)


def dspo_loss(values, beta, *, aggregation='B', recipe=None, sdxl_threshold=0.1,
              gap_mean=None):
    """Official DSPO score residual or SDXL recipe (author revision 8b4e09b).

    B averages full per-view losses. C shares the sigmoid gate of the mean
    margin, retaining each view's residual and SDXL threshold branch. The
    latter is a project extension, not a claim of a paper-defined C objective.
    winner_prediction_gaps must retain policy gradients; only the reference
    prediction is frozen. gap_mean optionally supplies detached, across-rank
    mean_B(gaps) with shape [K]. A scalar deliberately shares the branch.
    """
    _positive('beta', beta)
    if aggregation not in ('B', 'C'):
        raise ValueError('aggregation must be B or C')
    if recipe not in ('score_residual', 'official_sdxl'):
        raise ValueError('DSPO requires explicit recipe=score_residual or official_sdxl')
    _positive('dspo_sdxl_threshold', sdxl_threshold)
    errors, refs = _errors(values, reference=True)
    gaps = values.get('winner_prediction_gaps')
    if (errors.shape[-1] != 2 or gaps is None or gaps.shape != errors.shape[:2]
            or not torch.isfinite(gaps).all() or (gaps < 0).any()):
        raise ValueError('DSPO requires finite nonnegative winner_prediction_gaps [B,K]')
    # Expanded squared residuals can suffer cancellation when beta*g is near
    # one. Evaluate the small sufficient statistics in float64, without a
    # detached/clamped loss: these would change gradients. FP32 input moments
    # still retain their original precision; the backend should reduce errors
    # and prediction gaps in FP32 or better, never BF16.
    a, r, d = errors[..., 0].double(), refs[..., 0].double(), gaps.double()
    relative = refs.double() - errors.double()
    logits = .5 * float(beta) * (relative[..., 0] - relative[..., 1])
    selected = logits.mean(1, keepdim=True) if aggregation == 'C' else logits
    gate = torch.sigmoid(-selected)
    metrics = {'margin_mean': logits.mean().detach(),
               'winner_prediction_gap': d.mean().detach()}
    if recipe == 'score_residual':
        scaled = float(beta) * gate
        # a - scaled*(a-r+d) + scaled**2*d, regrouped to reduce operations.
        terms = (1-scaled)*a + scaled*r + scaled*(scaled-1)*d
    else:
        mean = d.detach().mean(0) if gap_mean is None else torch.as_tensor(
            gap_mean, device=d.device, dtype=d.dtype).detach()
        if (mean.ndim > 1 or mean.numel() not in (1, d.shape[1])
                or not torch.isfinite(mean).all() or (mean < 0).any()):
            raise ValueError('dspo_gap_mean must be a finite nonnegative scalar or [K]')
        active = mean >= float(sdxl_threshold)
        terms = gate.square() * torch.where(active, d, torch.ones_like(d))
        metrics['dspo_active_view_fraction'] = active.double().mean()
    loss = terms.mean()
    dtype = torch.float64 if errors.dtype == torch.float64 else torch.float32
    return loss.to(dtype), {key: value.to(dtype) for key, value in metrics.items()}


def baseline_loss(values, objective, beta, *, aggregation='B', dmpo_alpha=0.01,
                  mapo_beta=0.1, mapo_snr=0.5, num_train_timesteps=1000,
                  kto_desirable_weight=1.0, kto_undesirable_weight=1.0, kto_kl=None,
                  dspo_recipe=None, dspo_sdxl_threshold=0.1, dspo_gap_mean=None):
    """Return scalar loss and detached metrics from [B,K,winner/loser] errors.

    DMPO and DSPO support K>1: B averages per-view losses; C uses the mean
    margin for DMPO's reverse KL or DSPO's shared gate. These compositions
    are project extensions.
    KTO converts each pair into binary labels; it does not compare pair gaps.
    Its detached KL reference point must be reduced across ranks by the caller
    (kto_kl); default estimation is suitable for single-process callers.
    """
    if objective not in BASELINE_OBJECTIVES:
        raise ValueError(f'Unknown baseline objective: {objective}')
    if aggregation not in ('B', 'C'):
        raise ValueError('aggregation must be B or C')
    errors, refs = _errors(values, reference=objective in ('kto', 'dmpo', 'dspo'))
    if objective not in ('dmpo', 'dspo') and errors.shape[1] != 1:
        raise ValueError('SFT, MaPO and KTO baseline require K=1')
    metrics = {'policy_error': errors.mean().detach(), 'winner_error': errors[..., 0].mean().detach()}
    if objective == 'dspo':
        loss, details = dspo_loss(values, beta, aggregation=aggregation, recipe=dspo_recipe,
                                 sdxl_threshold=dspo_sdxl_threshold, gap_mean=dspo_gap_mean)
        metrics.update(details)
        metrics['reference_error'] = refs.mean().detach()
        metrics['loss'] = loss.detach()
        return loss, metrics
    if objective == 'sft':
        loss = errors[..., 0].mean()
    else:
        if errors.shape[-1] != 2:
            raise ValueError('Preference baselines require winner/loser errors')
        if objective == 'mapo':
            _positive('mapo_beta', mapo_beta, allow_zero=True)
            _positive('mapo_snr', mapo_snr)
            if type(num_train_timesteps) is not int or num_train_timesteps <= 0:
                raise ValueError('num_train_timesteps must be a positive integer')
            scores = mapo_score(float(mapo_snr) * errors)
            logits = num_train_timesteps * (scores[..., 0] - scores[..., 1])
            ranking = F.softplus(-logits).mean()
            loss = errors[..., 0].mean() + float(mapo_beta) * ranking
            metrics['mapo_ranking_loss'] = ranking.detach()
        else:
            _positive('beta', beta)
            relative = refs - errors
            metrics['reference_error'] = refs.mean().detach()
            if objective == 'dmpo':
                logits = float(beta) * (relative[..., 0] - relative[..., 1])
                selected = logits.mean(dim=1) if aggregation == 'C' else logits
                loss = dmpo_loss_from_logits(selected, dmpo_alpha).mean()
                metrics['dmpo_optimal_logit'] = errors.new_tensor(math.log((1-dmpo_alpha)/dmpo_alpha))
            else:
                _positive('kto_desirable_weight', kto_desirable_weight)
                _positive('kto_undesirable_weight', kto_undesirable_weight)
                kl = relative.mean().detach() if kto_kl is None else torch.as_tensor(kto_kl, device=errors.device, dtype=errors.dtype).detach()
                if kl.numel() != 1 or not torch.isfinite(kl).all():
                    raise ValueError('kto_kl must be a finite scalar')
                kl = kl.clamp_min(0)
                logits = float(beta) * (relative - kl)
                losses_w = float(kto_desirable_weight) * torch.sigmoid(-logits[..., 0])
                losses_l = float(kto_undesirable_weight) * torch.sigmoid(logits[..., 1])
                loss = (losses_w.mean() + losses_l.mean()) / 2
                metrics['kto_kl'] = kl
        metrics['margin_mean'] = logits.mean().detach()
    metrics['loss'] = loss.detach()
    return loss, metrics
