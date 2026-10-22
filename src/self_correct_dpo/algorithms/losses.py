"""Preference objectives on denoising/flow errors, not exact log likelihoods."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


OBJECTIVES = frozenset({
    "sft", "dpo", "augmentation", "orbit_mean", "orbit_worst",
    "orbit_softmax", "orbit_consistency",
})


def preference_logits(policy_errors, reference_errors, beta):
    """-0.5*beta*((Pw-Pl)-(Rw-Rl)), a denoising/flow error surrogate.

    This is not an exact image log-likelihood ratio; reference is detached.
    """
    if policy_errors.ndim != 3 or policy_errors.shape[-1] != 2:
        raise ValueError("policy_errors must be [B,K,2]")
    if reference_errors is None or policy_errors.shape != reference_errors.shape:
        raise ValueError("Matching reference errors are required")
    if min(policy_errors.shape[:2]) < 1 or not torch.isfinite(policy_errors).all() or not torch.isfinite(reference_errors).all():
        raise ValueError("Preference logits require nonempty, finite errors")
    if not math.isfinite(float(beta)) or float(beta) <= 0:
        raise ValueError("beta must be finite and positive")
    relative = policy_errors.float() - reference_errors.detach().float()
    return -0.5 * float(beta) * (relative[..., 0] - relative[..., 1])


def preference_loss(policy_errors, reference_errors, objective, beta,
                    temperature=0.1, consistency_weight=0.0):
    """Consume [batch, expressions, winner/loser] per-image mean errors.

    The Diffusion-DPO convention is logit = -beta/2 * (policy gap -
    reference gap). For SD3.5 these are velocity-regression errors; this
    defines a flow preference surrogate and is not an exact likelihood ratio.
    orbit_softmax is normalized log-mean-exp of per-expression DPO losses.
    Consistency penalizes the population variance of the scaled logits.
    Single-expression selection for dpo/augmentation belongs to the trainer.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if policy_errors.ndim != 3 or policy_errors.shape[-1] != 2:
        raise ValueError("policy_errors must be [B,K,2], ordered winner then loser")
    if min(policy_errors.shape[:2]) < 1:
        raise ValueError("Empty batch or orbit")
    if not torch.isfinite(policy_errors).all():
        raise ValueError("Non-finite policy errors")
    if not math.isfinite(float(beta)) or float(beta) <= 0:
        raise ValueError("beta must be finite and positive")
    if not math.isfinite(float(consistency_weight)) or consistency_weight < 0:
        raise ValueError("consistency_weight must be finite and nonnegative")
    if objective != "orbit_consistency" and consistency_weight != 0:
        raise ValueError("consistency_weight applies only to orbit_consistency")
    errors = policy_errors.float()
    if objective == "sft":
        loss = errors[..., 0].mean()
        return loss, {"loss": loss.detach(), "winner_error": loss.detach()}
    if reference_errors is None or reference_errors.shape != errors.shape:
        raise ValueError("Preference objectives require matching reference errors")
    if not torch.isfinite(reference_errors).all():
        raise ValueError("Non-finite reference errors")
    if objective in {"dpo", "augmentation"} and errors.shape[1] != 1:
        raise ValueError("dpo/augmentation require trainer-selected K=1")
    logits = preference_logits(errors, reference_errors, beta)
    per_expression = F.softplus(-logits)
    consistency = logits.var(dim=1, unbiased=False).mean()
    if objective == "orbit_worst":
        per_pair = per_expression.max(dim=1).values
    elif objective == "orbit_softmax":
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        # Centering also avoids overflow for extreme preference margins.
        center = per_expression.max(dim=1, keepdim=True).values
        per_pair = center.squeeze(1) + float(temperature) * (
            torch.logsumexp((per_expression - center) / float(temperature), dim=1)
            - math.log(per_expression.shape[1])
        )
    else:
        per_pair = per_expression.mean(dim=1)
    loss = per_pair.mean()
    if objective == "orbit_consistency":
        loss = loss + float(consistency_weight) * consistency
    metrics = {
        "loss": loss.detach(), "dpo_loss": per_expression.mean().detach(),
        "margin_mean": logits.mean().detach(),
        "margin_variance": consistency.detach(),
        "worst_margin": logits.min(dim=1).values.mean().detach(),
        "implicit_accuracy": (logits > 0).float().mean().detach(),
        "policy_error": errors.mean().detach(),
        "reference_error": reference_errors.float().mean().detach(),
    }
    return loss, metrics
