"""Exact sufficient-statistic gradients for the archived SD1.5 DMPO/DSPO losses.

Training objectives.py SHA256 b5e29226cb929ae49d8538f40f768737e64564c9117a10cf0b08080ec482dcda
(retrieved from repository commit 9573c1e). DSPO includes the differentiable
winner prediction gap and gate; it cannot be represented by a DPO weight.
"""
import shutil
import tempfile
from pathlib import Path
import numpy as np

ARCHIVED_OBJECTIVES_SHA = "b5e29226cb929ae49d8538f40f768737e64564c9117a10cf0b08080ec482dcda"


def objective_spec(settings):
    name = settings["objective"]
    if not np.isfinite(float(settings["beta"])) or float(settings["beta"]) <= 0:
        raise ValueError("native beta must be finite and positive")
    if name == "dmpo":
        alpha = float(settings["dmpo_alpha"])
        if not 0 < alpha < .5:
            raise ValueError("DMPO alpha must lie in (0, .5)")
        return dict(name=name, beta=float(settings["beta"]), alpha=alpha,
                    moments=["u"], components=["u"])
    if name == "dspo" and settings.get("dspo_recipe") == "score_residual":
        return dict(name=name, beta=float(settings["beta"]), recipe="score_residual",
                    moments=["m", "policy_winner_mse", "winner_prediction_gap", "reference_winner_mse"],
                    components=["m", "policy_winner_mse", "winner_prediction_gap"])
    raise ValueError("only archived SD1.5 DMPO and score-residual DSPO are supported")


def coefficients(moments, spec):
    x = np.asarray(moments, dtype=np.float64)
    z = x[..., 0]
    if spec["name"] == "dmpo":
        # Stable sigmoid and its complement, including the finite optimum.
        p = np.exp(-np.logaddexp(0, -z))
        q = np.exp(-np.logaddexp(0, z))
        return (p * q * (z - np.log((1 - spec["alpha"]) / spec["alpha"])))[..., None]
    gate = np.exp(-np.logaddexp(0, z))
    scale = spec["beta"] * gate
    derivative = -spec["beta"] * gate * np.exp(-np.logaddexp(0, -z))
    m_coeff = derivative * (-x[..., 1] + x[..., 3] + (2 * scale - 1) * x[..., 2])
    return np.stack((m_coeff, 1 - scale, scale * (scale - 1)), axis=-1)


def loss_value(moments, spec):
    x = np.asarray(moments, dtype=np.float64)
    z = x[..., 0]
    if spec["name"] == "dmpo":
        p = np.exp(-np.logaddexp(0, -z))
        q = np.exp(-np.logaddexp(0, z))
        return p * (-np.logaddexp(0, -z) - np.log1p(-spec["alpha"])) + q * (-np.logaddexp(0, z) - np.log(spec["alpha"]))
    scale = spec["beta"] * np.exp(-np.logaddexp(0, z))
    return (1-scale)*x[..., 1] + scale*x[..., 3] + scale*(scale-1)*x[..., 2]


def layout(ks, source_blocks, audit_blocks, components):
    roles = ["margin", "lossavg", *(f"mvpo:{k}" for k in ks if k > 1),
             *(f"component:{i}" for i in range(1, components))]
    indices = np.arange(source_blocks * len(roles)).reshape(source_blocks, len(roles))
    source = dict(zip(roles, indices.T))
    audit = np.arange(indices.size, indices.size + components * audit_blocks).reshape(components, audit_blocks)
    return roles, source, audit


def collect_native_statistics(observe, *, source_seeds, audit_seeds, ks, spec, scratch_dir=None):
    """Each observation provides moment values and every required raw component gradient."""
    import torch
    source_size, audit_size = len(source_seeds[0]), len(audit_seeds[0])
    if (len(source_seeds) < 2 or len(audit_seeds) < 2 or len(audit_seeds) % 2
            or any(source_size % k for k in ks) or len(set(ks)) != len(ks) or ks[0] != 1
            or any(len(b) != source_size for b in source_seeds)
            or any(len(b) != audit_size for b in audit_seeds)):
        raise ValueError("invalid native gradient block layout")
    all_seeds = [s for block in source_seeds + audit_seeds for s in block]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("source/audit seeds must be distinct")
    components = len(spec["components"])
    roles, indices, audit_index = layout(ks, len(source_seeds), len(audit_seeds), components)
    first = observe(source_seeds[0][0])
    parameter_count = first[1].shape[-1]
    device = first[1].device
    def checked(value):
        moments, gradients = value
        x = np.asarray(moments, dtype=np.float64)
        if (x.shape != (len(spec["moments"]),) or not np.isfinite(x).all()
                or gradients.shape != (components, parameter_count) or gradients.dtype != torch.float32
                or gradients.device != device or not bool(torch.isfinite(gradients).all())):
            raise ValueError("invalid native moments/component gradients")
        return x, gradients
    first = checked(first)
    with tempfile.TemporaryDirectory(prefix="mvpo-native-", dir=scratch_dir) as scratch:
        n_rows = int(audit_index[-1, -1]) + 1
        if shutil.disk_usage(scratch).free < n_rows * parameter_count * 4:
            raise OSError("insufficient local scratch for native gradient block means")
        vectors = np.memmap(Path(scratch) / "block-means", mode="w+", dtype=np.float32,
                            shape=(n_rows, parameter_count))
        blocks = []
        for bi, seeds in enumerate(source_seeds):
            h_sum = torch.zeros((components, parameter_count), device=device)
            g_sums = {k: torch.zeros(parameter_count, device=device) for k in ks}
            pending_h = {k: torch.zeros_like(h_sum) for k in ks if k > 1}
            pending_g = {k: torch.zeros(parameter_count, device=device) for k in ks if k > 1}
            pending_x = {k: np.zeros(len(spec["moments"])) for k in ks if k > 1}
            norms = {method: {str(k): 0. for k in ks} for method in ("lossavg", "mvpo")}
            x_sum, c_sum = np.zeros(len(spec["moments"])), np.zeros(components)
            for vi, seed in enumerate(seeds):
                x, h = first if bi == 0 and vi == 0 else checked(observe(seed))
                c = coefficients(x, spec)
                g = (h * torch.as_tensor(c, device=device, dtype=h.dtype)[:, None]).sum(0)
                h_sum.add_(h); x_sum += x; c_sum += c; g_sums[1].add_(g)
                norms["lossavg"]["1"] += float(torch.dot(g, g).double())
                for k in ks[1:]:
                    pending_h[k].add_(h); pending_g[k].add_(g); pending_x[k] += x
                    if (vi + 1) % k == 0:
                        c_k = coefficients(pending_x[k] / k, spec)
                        native = (pending_h[k] * torch.as_tensor(c_k, device=device, dtype=h.dtype)[:, None]).sum(0) / k
                        avg = pending_g[k] / k
                        g_sums[k].add_(native)
                        norms["mvpo"][str(k)] += float(torch.dot(native, native).double())
                        norms["lossavg"][str(k)] += float(torch.dot(avg, avg).double())
                        pending_h[k].zero_(); pending_g[k].zero_(); pending_x[k].fill(0)
            norms["mvpo"]["1"] = norms["lossavg"]["1"]
            for j in range(components):
                role = "margin" if j == 0 else f"component:{j}"
                vectors[indices[role][bi]] = (h_sum[j] / source_size).cpu().numpy()
            vectors[indices["lossavg"][bi]] = (g_sums[1] / source_size).cpu().numpy()
            for k in ks[1:]:
                vectors[indices[f"mvpo:{k}"][bi]] = (g_sums[k] / (source_size // k)).cpu().numpy()
            blocks.append(dict(margin_mean=float(x_sum[0] / source_size), moment_means=(x_sum/source_size).tolist(),
                               coefficient_means=(c_sum/source_size).tolist(), group_squared_norm_sums=norms))
            # Free group accumulators before the next source/audit block.
            del h_sum, g_sums, pending_h, pending_g, h, g
            first = None
        audit_means = []
        for bi, seeds in enumerate(audit_seeds):
            h_sum = torch.zeros((components, parameter_count), device=device)
            x_sum = np.zeros(len(spec["moments"]))
            for seed in seeds:
                x, h = checked(observe(seed)); x_sum += x; h_sum.add_(h)
            audit_means.append((x_sum / audit_size).tolist())
            for j in range(components):
                vectors[audit_index[j, bi]] = (h_sum[j] / audit_size).cpu().numpy()
            del h_sum, h
        gram = np.zeros((n_rows, n_rows), dtype=np.float64)
        width = max(1, (24 * 1024**2) // (n_rows * 8))
        for start in range(0, parameter_count, width):
            chunk = np.asarray(vectors[:, start:start+width], dtype=np.float64)
            gram += chunk @ chunk.T
        del vectors
    if not np.isfinite(gram).all():
        raise ValueError("non-finite native Gram")
    return dict(schema_version=2, native_objective=spec, parameter_count=parameter_count, ks=list(ks),
                source_views_per_block=source_size, audit_views_per_block=audit_size, source_blocks=blocks,
                audit_block_margin_means=[x[0] for x in audit_means], audit_block_moment_means=audit_means,
                block_mean_gram_source_roles=roles, block_mean_gram=gram.tolist())


def audit_coefficients(raw, counts):
    spec = raw["native_objective"]
    _, _, audit = layout(raw["ks"], len(raw["source_blocks"]), len(raw["audit_block_margin_means"]), len(spec["components"]))
    weights = counts / counts.sum(axis=1, keepdims=True)
    moments = weights @ np.asarray(raw["audit_block_moment_means"])
    c = coefficients(moments, spec)
    result = np.zeros((len(counts), len(raw["block_mean_gram"])))
    for j, rows in enumerate(audit):
        result[:, rows] = weights * c[:, j, None]
    return result, moments


def evaluate_native_pair(raw, source_counts, audit_counts):
    from self_correct_dpo.evaluation.appendix_gradient_reporting import _dot, _cosine, _sqrt, _ratio, _mean_coefficients
    gram = np.asarray(raw["block_mean_gram"], dtype=np.float64)
    source = raw["source_blocks"]
    spec, ks = raw["native_objective"], raw["ks"]
    _, index, _ = layout(ks, len(source), len(raw["audit_block_margin_means"]), len(spec["components"]))
    weights = source_counts / source_counts.sum(axis=1, keepdims=True)
    n = raw["source_views_per_block"] * source_counts.sum(axis=1)
    audit_g, audit_x = audit_coefficients(raw, audit_counts)
    norm = _sqrt(_dot(audit_g, gram, audit_g))
    def alignment(g):
        projection = _dot(g, gram, audit_g)
        return dict(descent_cosine=_cosine(g, gram, audit_g), signed_projection=projection,
                    signed_unit_projection=_ratio(projection, norm))
    coupled = _mean_coefficients(weights, index["lossavg"], len(gram))
    c = weights @ np.asarray([b["coefficient_means"] for b in source])
    independent = np.zeros_like(coupled)
    for j in range(len(spec["components"])):
        role = "margin" if j == 0 else f"component:{j}"
        independent += c[:, j, None] * _mean_coefficients(weights, index[role], len(gram))
    decoupled = (n[:, None] * independent - coupled) / (n-1)[:, None]
    covariance = coupled-decoupled
    parallel = _ratio(_dot(covariance, gram, audit_g), norm)
    decoupled_alignment = alignment(decoupled)
    coupling = dict(covariance_norm=_sqrt(_dot(covariance, gram, covariance)), covariance_parallel=parallel,
                    covariance_perpendicular=_sqrt(_dot(covariance, gram, covariance)-parallel**2),
                    coupled_decoupled_cosine=_cosine(coupled, gram, decoupled),
                    descent_cosine_change=decoupled_alignment["descent_cosine"]-alignment(coupled)["descent_cosine"],
                    **{f"decoupled_{k}": v for k,v in decoupled_alignment.items()})
    _, _, audit_rows = layout(ks, len(source), len(raw["audit_block_margin_means"]), len(spec["components"]))
    h0 = _mean_coefficients(audit_counts / audit_counts.sum(axis=1, keepdims=True), audit_rows[0], len(gram))
    audit = dict(mean_margin=audit_x[:, 0], reference_loss=loss_value(audit_x, spec),
                 reference_gradient_norm=norm, margin_gradient_norm=_sqrt(_dot(h0, gram, h0)))
    metrics = {}
    diagonal = np.diag(gram)
    for method in ("lossavg", "mvpo"):
        for k in ks:
            rows = index[f"mvpo:{k}" if method == "mvpo" and k > 1 else "lossavg"]
            g = _mean_coefficients(weights, rows, len(gram))
            norm2 = _dot(g, gram, g)
            dist = np.maximum(_dot(g-audit_g, gram, g-audit_g), 0)
            mc = np.maximum(weights @ diagonal[rows]-norm2, 0)/(len(source)-1)
            groups = n // k
            sums = source_counts @ [b["group_squared_norm_sums"][method][str(k)] for b in source]
            metrics[(method,k)] = dict(distance=_sqrt(dist), distance_squared=dist,
                source_corrected_distance_squared=dist-mc, stochastic_variance=np.maximum(sums-groups*norm2,0)/(groups-1),
                mean_gradient_norm=_sqrt(norm2), source_mean_gradient_mc_rmse=_sqrt(mc), **alignment(g))
    return metrics, coupling, audit


def validate_native_statistics(raw, protocol):
    sampling = protocol["sampling"]
    spec = objective_spec(protocol["training"]["config"])
    if (protocol.get("native_objective") != spec or raw["schema_version"] != 2 or raw["native_objective"] != spec or raw["ks"] != sampling["ks"]
            or raw["parameter_count"] != protocol["model"]["lora_parameter_count"]
            or len(raw["source_blocks"]) != sampling["source_blocks"]
            or len(raw["audit_block_moment_means"]) != sampling["audit_blocks"]
            or len(raw["audit_block_margin_means"]) != sampling["audit_blocks"]
            or any(raw[k] != sampling[k] for k in ("source_views_per_block", "audit_views_per_block"))):
        raise ValueError("native objective/sampling differs from frozen protocol")
    roles, _, audit = layout(raw["ks"], sampling["source_blocks"], sampling["audit_blocks"], len(spec["components"]))
    gram = np.asarray(raw["block_mean_gram"])
    if (raw["block_mean_gram_source_roles"] != roles or gram.shape != (audit[-1,-1]+1,)*2
            or not np.isfinite(gram).all() or not np.allclose(gram, gram.T) or (np.diag(gram)<0).any()):
        raise ValueError("invalid native Gram")
    moments = np.asarray(raw["audit_block_moment_means"])
    if (moments.shape != (sampling["audit_blocks"],len(spec["moments"])) or not np.isfinite(moments).all()
            or not np.array_equal(moments[:,0],raw["audit_block_margin_means"])):
        raise ValueError("invalid native audit moments")
    for block in raw["source_blocks"]:
        x, c = np.asarray(block["moment_means"]), np.asarray(block["coefficient_means"])
        if (x.shape != (len(spec["moments"]),) or c.shape != (len(spec["components"]),)
                or not np.isfinite(x).all() or not np.isfinite(c).all() or x[0] != block["margin_mean"]):
            raise ValueError("invalid native source moments")
        for method in ("lossavg", "mvpo"):
            norms = block["group_squared_norm_sums"][method]
            if set(norms) != set(map(str,raw["ks"])) or any(not np.isfinite(v) or v < 0 for v in norms.values()):
                raise ValueError("invalid native gradient norm sums")
