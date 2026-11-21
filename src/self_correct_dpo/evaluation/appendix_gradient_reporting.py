"""Recompute gradient geometry and uncertainty from portable block statistics."""

from pathlib import Path

import numpy as np

from self_correct_dpo.evaluation.appendix_gradients import _bank_layout, _loss_derivative


def _dot(left, gram, right):
    return np.sum(left.dot(gram) * right, axis=1)


def _sqrt(value):
    return np.sqrt(np.maximum(value, 0))


def _ratio(value, divisor):
    return np.divide(value, divisor, out=np.full_like(value, np.nan), where=divisor > 0)


def _cosine(left, gram, right):
    divisor = _sqrt(_dot(left, gram, left)) * _sqrt(_dot(right, gram, right))
    return np.clip(_ratio(_dot(left, gram, right), divisor), -1, 1)


def _mean_coefficients(weights, indices, n_rows):
    coeff = np.zeros((len(weights), n_rows))
    coeff[:, indices] = weights
    return coeff


def _evaluate_pair(raw, source_counts, audit_counts):
    """Evaluate vectors represented as linear combinations of archived block means."""
    gram = np.asarray(raw["block_mean_gram"], dtype=np.float64)
    source = raw["source_blocks"]
    audit_margins = np.asarray(raw["audit_block_margin_means"], dtype=np.float64)
    ks = raw["ks"]
    _, source_index, audit_index = _bank_layout(ks, len(source), len(audit_margins))
    source_weights = source_counts / source_counts.sum(axis=1, keepdims=True)
    audit_weights = audit_counts / audit_counts.sum(axis=1, keepdims=True)
    source_views = raw["source_views_per_block"] * source_counts.sum(axis=1)
    audit_h = _mean_coefficients(audit_weights, audit_index, len(gram))
    audit_mu = audit_weights.dot(audit_margins)
    audit_g = _loss_derivative(audit_mu)[:, None] * audit_h
    audit_norm = _sqrt(_dot(audit_h, gram, audit_h))
    diagonal = np.diag(gram)

    def alignment(estimate):
        projection = -_dot(estimate, gram, audit_h)
        return {
            "descent_cosine": _cosine(-estimate, gram, audit_h),
            "signed_projection": projection,
            "signed_unit_projection": _ratio(projection, audit_norm),
        }

    source_h = _mean_coefficients(source_weights, source_index["margin"], len(gram))
    coupled_g = _mean_coefficients(source_weights, source_index["lossavg"], len(gram))
    derivative = source_weights.dot([block["loss_derivative_mean"] for block in source])
    # Sum a_i h_j for i != j using only the two marginal sums and the diagonal sum.
    decoupled_g = (source_views[:, None] * derivative[:, None] * source_h - coupled_g) / (source_views - 1)[:, None]
    covariance = coupled_g - decoupled_g
    covariance_parallel = _ratio(_dot(covariance, gram, audit_h), audit_norm)
    decoupled_alignment = alignment(decoupled_g)
    coupling = {
        "covariance_norm": _sqrt(_dot(covariance, gram, covariance)),
        "covariance_parallel": covariance_parallel,
        "covariance_perpendicular": _sqrt(_dot(covariance, gram, covariance) - covariance_parallel**2),
        "coupled_decoupled_cosine": _cosine(coupled_g, gram, decoupled_g),
        "descent_cosine_change": decoupled_alignment["descent_cosine"] - alignment(coupled_g)["descent_cosine"],
        **{f"decoupled_{key}": value for key, value in decoupled_alignment.items()},
    }
    audit = {
        "mean_margin": audit_mu,
        "margin_gradient_norm": audit_norm,
        "reference_gradient_norm": _sqrt(_dot(audit_g, gram, audit_g)),
    }
    metrics = {}
    for method in ("lossavg", "mvpo"):
        for k in ks:
            indices = source_index[f"mvpo:{k}" if method == "mvpo" and k > 1 else "lossavg"]
            estimate = _mean_coefficients(source_weights, indices, len(gram))
            norm_squared = _dot(estimate, gram, estimate)
            distance = estimate - audit_g
            distance_squared = np.maximum(_dot(distance, gram, distance), 0)
            mean_mc_variance = np.maximum(source_weights.dot(diagonal[indices]) - norm_squared, 0) / (len(source) - 1)
            block_norm_sums = [block["group_squared_norm_sums"][method][str(k)] for block in source]
            group_count = source_views // k
            sum_norm_squared = source_counts.dot(block_norm_sums)
            variance = np.maximum(sum_norm_squared - group_count * norm_squared, 0) / (group_count - 1)
            metrics[(method, k)] = {
                "distance": _sqrt(distance_squared),
                "distance_squared": distance_squared,
                "source_corrected_distance_squared": distance_squared - mean_mc_variance,
                "stochastic_variance": variance,
                "mean_gradient_norm": _sqrt(norm_squared),
                "source_mean_gradient_mc_rmse": _sqrt(mean_mc_variance),
                **alignment(estimate),
            }
    return metrics, coupling, audit


def _estimate(value):
    return float(value) if np.isfinite(value) else None


def _interval(samples):
    valid = samples[np.isfinite(samples)]
    return np.quantile(valid, (0.025, 0.975)).tolist() if valid.size else None


def _describe(samples):
    return {"estimate": _estimate(samples[0]), "monte_carlo_95": _interval(samples[1:])}


def _mean_defined(values, axis):
    valid = np.isfinite(values)
    return _ratio(np.where(valid, values, 0).sum(axis=axis), valid.sum(axis=axis))


def _summarize_collection(blocks, pair_draws):
    values = np.stack(blocks)
    average = _mean_defined(values, axis=0)
    result = _describe(average)
    result["n_pairs_defined"] = int(np.isfinite(values[:, 0]).sum())
    result["pair_bootstrap_95"] = (
        _interval(_mean_defined(values[pair_draws, 0], axis=1)) if len(values) > 1 else None
    )
    return result


def _draw_block_counts(rng, blocks, repeats):
    counts = rng.multinomial(blocks, np.full(blocks, 1 / blocks), size=repeats)
    return np.vstack((np.ones(blocks, dtype=np.int64), counts))


def summarize_gradient_statistics(records, *, bootstrap_repeats=2000, seed=17):
    """Replay raw geometry, with view-block and pair uncertainty kept separate."""
    if bootstrap_repeats < 2:
        raise ValueError("at least two bootstrap repetitions are required")
    grouped = {}
    for record in records:
        condition = record["condition"]
        pairs = grouped.setdefault(condition, {})
        if record["pair_id"] in pairs:
            raise ValueError(f"duplicate pair {record['pair_id']} in condition {condition}")
        pairs[record["pair_id"]] = record["statistics"]
    if not grouped:
        raise ValueError("at least one gradient record is required")

    rng = np.random.default_rng(seed)
    conditions = []
    pair_results = []
    for condition, pairs in grouped.items():
        collection = {}
        raw_first = next(iter(pairs.values()))
        ks = raw_first["ks"]
        source_blocks = len(raw_first["source_blocks"])
        audit_blocks = len(raw_first["audit_block_margin_means"])
        for pair_id, raw in pairs.items():
            if (raw["ks"] != ks or len(raw["source_blocks"]) != source_blocks
                    or len(raw["audit_block_margin_means"]) != audit_blocks
                    or any(raw[key] != raw_first[key] for key in (
                        "source_views_per_block", "audit_views_per_block", "parameter_count"
                    ))):
                raise ValueError(f"gradient sampling or parameter count changed in {condition}")
            source_counts = _draw_block_counts(rng, source_blocks, bootstrap_repeats)
            audit_counts = _draw_block_counts(rng, audit_blocks, bootstrap_repeats)
            metrics, coupling, audit = _evaluate_pair(raw, source_counts, audit_counts)
            split_counts = np.repeat(np.eye(2, dtype=np.int64), audit_blocks // 2, axis=1)
            split, split_coupling, split_audit = _evaluate_pair(raw, source_counts[[0, 0]], split_counts)
            gram = np.asarray(raw["block_mean_gram"], dtype=np.float64)
            _, _, audit_index = _bank_layout(ks, source_blocks, audit_blocks)
            half_h = _mean_coefficients(split_counts / (audit_blocks // 2), audit_index, len(gram))
            groups = {
                "objectives": metrics,
                "paired_differences": {
                    k: {name: value - metrics[("lossavg", k)][name]
                        for name, value in metrics[("mvpo", k)].items()}
                    for k in ks
                },
                "coupling": {None: coupling},
                "audit": {None: audit},
            }
            descriptions = {}
            for group, entries in groups.items():
                descriptions[group] = {}
                for key, entry in entries.items():
                    descriptions[group][key] = {name: _describe(samples) for name, samples in entry.items()}
                    for name, samples in entry.items():
                        collection.setdefault((group, key, name), []).append(samples)

            pair_results.append({
                "condition": condition,
                "pair_id": pair_id,
                "objectives": _objective_rows(descriptions["objectives"]),
                "paired_differences": _difference_rows(descriptions["paired_differences"]),
                "coupling": descriptions["coupling"][None],
                "audit": descriptions["audit"][None],
                "audit_split": {
                    "direction_cosine": _estimate(_cosine(half_h[:1], gram, half_h[1:])[0]),
                    "objectives": _objective_rows({
                        key: {metric: [_estimate(x) for x in entry[metric]]
                              for metric in ("distance", "descent_cosine", "signed_projection")}
                        for key, entry in split.items()
                    }),
                    "coupling": {key: [_estimate(x) for x in samples] for key, samples in split_coupling.items()},
                    "audit": {key: [_estimate(x) for x in samples] for key, samples in split_audit.items()},
                },
            })

        pair_draws = rng.integers(0, len(pairs), size=(bootstrap_repeats, len(pairs)))
        aggregate = {}
        for (group, key, name), values in collection.items():
            aggregate.setdefault(group, {}).setdefault(key, {})[name] = _summarize_collection(values, pair_draws)
        conditions.append({
            "condition": condition,
            "n_pairs": len(pairs),
            "source_blocks": source_blocks,
            "source_views_per_block": raw_first["source_views_per_block"],
            "audit_blocks": audit_blocks,
            "audit_views_per_block": raw_first["audit_views_per_block"],
            "objectives": _objective_rows(aggregate["objectives"]),
            "paired_differences": _difference_rows(aggregate["paired_differences"]),
            "coupling": aggregate["coupling"][None],
            "audit": aggregate["audit"][None],
        })

    return {
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": seed,
        "monte_carlo_95_unit": "Independent source and audit blocks within each fixed pair",
        "pair_bootstrap_95_unit": "Preference pairs, with objectives matched within each pair",
        "paired_difference": "MVPO minus LossAvg at the same K",
        "covariance_sign": "Loss gradient convention, with derivative -sigmoid(-margin)",
        "source_correction": "Squared distance minus the estimated trace variance of the source mean",
        "conditions": conditions,
        "pairs": pair_results,
    }


def _objective_rows(table):
    return [{"objective": method, "k": k, "metrics": metrics} for (method, k), metrics in table.items()]


def _difference_rows(table):
    return [{"k": k, "metrics": metrics} for k, metrics in table.items()]


def plot_gradient_statistics(summary, output_dir):
    """Draw the two appendix comparison figures for each measured condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    titles = {
        "sd15_t499": "SD1.5 at t=499",
        "sd35_stage0": "SD3.5-M scheduler indices 0–499",
        "sd35_stage1": "SD3.5-M scheduler indices 500–999",
    }
    styles = {"lossavg": ("LossAvg", "#bb5032", "s"), "mvpo": ("MVPO", "#286b9d", "o")}
    for condition in summary["conditions"]:
        name = condition["condition"]
        interval_key = "pair_bootstrap_95" if condition["n_pairs"] > 1 else "monte_carlo_95"
        interval_label = "pair bootstrap" if condition["n_pairs"] > 1 else "source and audit block bootstrap"
        randomness = "Gaussian noise draws" if name == "sd15_t499" else "Gaussian noise and scheduler-index draws"
        title = (
            f'{titles.get(name, name)}  ·  {condition["n_pairs"]} preference pairs\n'
            f'{condition["source_blocks"]}×{condition["source_views_per_block"]} source and '
            f'{condition["audit_blocks"]}×{condition["audit_views_per_block"]} audit views per pair'
            f'\n{randomness}  ·  95% {interval_label} intervals'
        )

        def draw_ci(axis, x, description, color):
            interval = description[interval_key]
            if interval is not None:
                axis.vlines(x, *interval, color=color, linewidth=1.15, alpha=0.64)

        figure, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout="constrained")
        for axis, metric, ylabel in zip(
            axes,
            ("distance", "stochastic_variance", "descent_cosine"),
            ("Distance to audit gradient", "Trace of gradient variance", "Descent cosine to audit direction"),
        ):
            for method, (label, color, marker) in styles.items():
                rows = [row for row in condition["objectives"] if row["objective"] == method]
                x = [row["k"] for row in rows]
                offset = -0.06 if method == "lossavg" else 0.06
                x_plot = np.asarray(x) + offset
                points = [row["metrics"][metric] for row in rows]
                axis.plot(x_plot, [point["estimate"] for point in points], color=color, marker=marker, label=label)
                for pos, point in zip(x_plot, points):
                    draw_ci(axis, pos, point, color)
                axis.set_xticks(x)
            axis.set(xlabel="Views per group (K)", ylabel=ylabel)
            axis.grid(axis="y", alpha=0.24)
        axes[0].legend(frameon=False)
        figure.suptitle(title, fontsize=11)
        figure.savefig(output_dir / f"{name}-gradients.png", dpi=180)
        plt.close(figure)

        figure, (covariance, alignment) = plt.subplots(1, 2, figsize=(8, 3.7), layout="constrained")
        mvpo = [row for row in condition["objectives"] if row["objective"] == "mvpo"]
        controls = condition["coupling"]
        covariance_points = [controls["covariance_parallel"], controls["covariance_perpendicular"]]
        alignment_points = [mvpo[0]["metrics"]["descent_cosine"], controls["decoupled_descent_cosine"],
                            mvpo[-1]["metrics"]["descent_cosine"]]
        for axis, points, labels, ylabel in (
            (covariance, covariance_points, ["Parallel (signed)", "Perpendicular (norm)"], "Covariance component"),
            (alignment, alignment_points, ["Single view", "Decoupled", f'MVPO K={mvpo[-1]["k"]}'], "Descent cosine"),
        ):
            for x, point in enumerate(points):
                if point["estimate"] is not None:
                    axis.scatter(x, point["estimate"], color="#286b9d", zorder=3)
                    draw_ci(axis, x, point, "#286b9d")
            axis.axhline(0, color="#555555", linewidth=0.7)
            axis.set_xticks(range(len(labels)), labels)
            axis.set(ylabel=ylabel, xlim=(-0.45, len(labels) - 0.55))
        figure.suptitle(title, fontsize=11)
        figure.savefig(output_dir / f"{name}-coupling.png", dpi=180)
        plt.close(figure)
