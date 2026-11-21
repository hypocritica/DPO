"""Collect and replay the appendix comparison in the full LoRA parameter space."""

from __future__ import annotations

import math
import shutil
import tempfile
from itertools import chain
from pathlib import Path

import numpy as np


def _loss_derivative(margin):
    return -np.exp(-np.logaddexp(0, margin))


def _bank_layout(ks, source_blocks, audit_blocks):
    roles = ["margin", "lossavg", *(f"mvpo:{k}" for k in ks if k > 1)]
    source = np.arange(source_blocks * len(roles)).reshape(source_blocks, len(roles))
    audit = np.arange(source.size, source.size + audit_blocks)
    return roles, dict(zip(roles, source.T)), audit


def collect_gradient_statistics(
    observe, *, source_seeds, audit_seeds, ks,
    source_views_per_block=96, audit_views_per_block=128, scratch_dir=None,
):
    """Archive exact block dot products for one pair and one view distribution.

    Each unique seed is evaluated once. ``observe(seed)`` returns the scalar margin and
    its detached flat FP32 gradient over all trainable LoRA parameters in fixed order.
    The source and audit banks are sequences of equal-sized independent blocks.
    """
    import torch

    ks = tuple(ks)
    source_seeds = [tuple(block) for block in source_seeds]
    audit_seeds = [tuple(block) for block in audit_seeds]
    if not ks or tuple(sorted(set(ks))) != ks or ks[0] != 1:
        raise ValueError("ks must be distinct increasing positive integers starting at 1")
    if any(type(k) is not int for k in ks):
        raise ValueError("ks must contain integers")
    if source_views_per_block < 1 or any(source_views_per_block % k for k in ks):
        raise ValueError("each k must evenly divide source_views_per_block")
    if audit_views_per_block < 1 or len(source_seeds) < 2 or len(audit_seeds) < 2 or len(audit_seeds) % 2:
        raise ValueError("require two or more source blocks and a positive even number of audit blocks")
    for bank, size in ((source_seeds, source_views_per_block), (audit_seeds, audit_views_per_block)):
        if any(len(block) != size for block in bank):
            raise ValueError("seed block lengths must agree with views_per_block")
    seeds = [seed for block in source_seeds + audit_seeds for seed in block]
    if len(set(seeds)) != len(seeds):
        raise ValueError("source and audit view seeds must be distinct")

    roles, source_index, audit_index = _bank_layout(ks, len(source_seeds), len(audit_seeds))
    parameter_count = None
    device = None

    def read_view(seed):
        nonlocal parameter_count, device
        margin, gradient = observe(seed)
        margin = float(margin)
        if (not math.isfinite(margin) or not isinstance(gradient, torch.Tensor)
                or gradient.ndim != 1 or gradient.dtype != torch.float32 or not gradient.numel()):
            raise ValueError(f"invalid margin or FP32 gradient for view seed {seed}")
        if parameter_count is None:
            parameter_count = gradient.numel()
            device = gradient.device
        if gradient.numel() != parameter_count or gradient.device != device:
            raise ValueError(f"gradient shape or device changed for view seed {seed}")
        return margin, gradient.detach().contiguous()

    with tempfile.TemporaryDirectory(prefix="mvpo-gradients-", dir=scratch_dir) as scratch:
        first_view = read_view(source_seeds[0][0])
        n_rows = int(audit_index[-1]) + 1
        scratch_bytes = n_rows * parameter_count * np.dtype(np.float32).itemsize
        if shutil.disk_usage(scratch).free < scratch_bytes:
            raise OSError(f"gradient scratch needs {scratch_bytes} free bytes in {scratch}")
        vectors = np.memmap(Path(scratch) / "block_means", mode="w+", dtype=np.float32,
                            shape=(n_rows, parameter_count))
        source_blocks = []
        for block_id, block in enumerate(source_seeds):
            observations = (
                first_view if block_id == 0 and i == 0 else read_view(seed)
                for i, seed in enumerate(block)
            )
            means, block_record = _source_block(observations, ks, source_views_per_block)
            for role in roles:
                vectors[source_index[role][block_id]] = means[role].cpu().numpy()
            source_blocks.append(block_record)
            del means
            first_view = None

        audit_margins = []
        for row, block in zip(audit_index, audit_seeds):
            margin_sum = 0.0
            gradient_sum = torch.zeros(parameter_count, dtype=torch.float32, device=device)
            for seed in block:
                margin, gradient = read_view(seed)
                margin_sum += margin
                gradient_sum.add_(gradient)
            vectors[row] = gradient_sum.div_(audit_views_per_block).cpu().numpy()
            audit_margins.append(margin_sum / audit_views_per_block)

        gram = np.zeros((n_rows, n_rows), dtype=np.float64)
        width = max(1, (24 * 1024**2) // (n_rows * np.dtype(np.float64).itemsize))
        for start in range(0, parameter_count, width):
            panel = np.ascontiguousarray(vectors[:, start:start + width], dtype=np.float64)
            gram += panel.dot(panel.T)
        del vectors

    if not np.isfinite(gram).all():
        raise ValueError("non-finite dot product in gradient banks")
    return {
        "schema_version": 1,
        "parameter_count": parameter_count,
        "ks": list(ks),
        "source_views_per_block": source_views_per_block,
        "audit_views_per_block": audit_views_per_block,
        "source_blocks": source_blocks,
        "audit_block_margin_means": audit_margins,
        "block_mean_gram_source_roles": roles,
        "block_mean_gram": gram.tolist(),
    }


def _source_block(observations, ks, size):
    import torch

    first_margin, first_gradient = next(observations)
    def empty_vector():
        return torch.zeros_like(first_gradient)

    means = {"margin": empty_vector(), "lossavg": empty_vector()}
    means.update({f"mvpo:{k}": empty_vector() for k in ks if k > 1})
    pending_h = {k: empty_vector() for k in ks if k > 1}
    pending_g = {k: empty_vector() for k in pending_h}
    pending_m = dict.fromkeys(pending_h, 0.0)
    norm_sums = torch.zeros((2, len(ks)), dtype=torch.float64, device=first_gradient.device)
    margin_sum = derivative_sum = 0.0

    for i, (margin, h) in enumerate(chain(((first_margin, first_gradient),), observations), 1):
        derivative = float(_loss_derivative(margin))
        g = h * derivative
        margin_sum += margin
        derivative_sum += derivative
        means["margin"].add_(h)
        means["lossavg"].add_(g)
        norm_sums[0, 0].add_(torch.dot(g, g))
        for col, k in enumerate(ks[1:], 1):
            pending_m[k] += margin
            pending_h[k].add_(h)
            pending_g[k].add_(g)
            if i % k == 0:
                pending_h[k].mul_(float(_loss_derivative(pending_m[k] / k)) / k)
                pending_g[k].div_(k)
                means[f"mvpo:{k}"].add_(pending_h[k])
                norm_sums[0, col].add_(torch.dot(pending_g[k], pending_g[k]))
                norm_sums[1, col].add_(torch.dot(pending_h[k], pending_h[k]))
                pending_m[k] = 0.0
                pending_h[k].zero_()
                pending_g[k].zero_()

    norm_sums[1, 0] = norm_sums[0, 0]
    norm_sums = norm_sums.cpu().numpy()
    if not np.isfinite(norm_sums).all():
        raise ValueError("non-finite stochastic gradient norm in source block")
    for role, vector in means.items():
        groups = size // int(role.split(":")[1]) if role.startswith("mvpo:") else size
        vector.div_(groups)
    return means, {
        "margin_mean": margin_sum / size,
        "loss_derivative_mean": derivative_sum / size,
        "group_squared_norm_sums": {
            method: dict(zip(map(str, ks), norm_sums[row].tolist()))
            for row, method in enumerate(("lossavg", "mvpo"))
        },
    }
