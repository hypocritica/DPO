"""Recompute SD3.5-M conditional margins and gates from the two raw view banks."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_correct_dpo.common import digest, read_json


def gate(margin):
    e = math.exp(-abs(margin))
    return e / (1 + e) if margin >= 0 else 1 / (1 + e)


def finite_k4(banks, stage, draws, rng):
    """Condition on a tagged view from this stage. The other three views are independent."""
    values = np.asarray(banks, dtype=float)
    positions = rng.integers(values.shape[1], size=(draws, 4))
    stage_margins = values[stage, positions]
    global_stages = rng.integers(2, size=(draws, 4))
    global_stages[:, 0] = stage
    global_margins = values[global_stages, positions]
    # Common random numbers: both groups contain exactly the same tagged view.
    stage_gate = np.exp(-np.logaddexp(0, stage_margins.mean(axis=1)))
    global_gate = np.exp(-np.logaddexp(0, global_margins.mean(axis=1)))
    delta = stage_gate - global_gate
    result = {}
    for name, data in (("stage", stage_gate), ("global", global_gate), ("delta", delta)):
        result[f"k4_{name}_mean"] = float(data.mean())
        for q in (5, 50, 95):
            result[f"k4_{name}_q{q:02d}"] = float(np.percentile(data, q))
    result["k4_delta_mc_se"] = float(delta.std(ddof=1) / math.sqrt(draws))
    return result


def pair_rows(root, protocol, k4_draws=10000, k4_seed=17):
    """Only compare both stages of the exact preselected pairs on the frozen run."""
    if protocol["experiment"] != "sd35-stage" or protocol["protocol_id"] != digest(
            {k: v for k, v in protocol.items() if k != "protocol_id"}):
        raise ValueError("Stage protocol identity mismatch")
    expected = {f"{p['pair_index']:04d}.json" for p in protocol["pairs"]}
    actual = {p.name for p in (root / "pairs").glob("*.json")}
    if expected != actual:
        raise ValueError(f"Stage panel incomplete or mixed: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    n_views = protocol["sampling"]["stage_views"]
    rows, stages = [], []
    for selection in protocol["pairs"]:
        i, pair_id = selection["pair_index"], selection["pair_id"]
        record = read_json(root / "pairs" / f"{i:04d}.json")
        if record.pop("result_id") != digest(record):
            raise ValueError(f"Stage result checksum mismatch for {pair_id}")
        if (record["protocol_id"], record["pair_index"], record["pair_id"]) != (
                protocol["protocol_id"], i, pair_id):
            raise ValueError(f"Unexpected protocol or pair in record {i}")
        banks = record["stages"]
        if len(banks) != 2:
            raise ValueError(f"Expected two stages for {pair_id}")
        means, errors, seen, margins = [], [], set(), []
        for j, bank in enumerate(banks):
            lo, hi = 500 * j, 500 * (j + 1)
            if (bank["stage_index"], bank["index_start"], bank["index_stop"], bank["probability"]) != (j, lo, hi, 0.5):
                raise ValueError(f"Unexpected scheduler partition for {pair_id}")
            views = bank["views"]
            if len(views) != n_views or n_views < 2 or {v["view_index"] for v in views} != set(range(n_views)):
                raise ValueError(f"Incomplete stage {j} for {pair_id}")
            vals = []
            for v in views:
                if not lo <= v["scheduler_index"] < hi or v["seed"] in seen or not math.isfinite(v["margin"]):
                    raise ValueError(f"Invalid or repeated view for {pair_id}")
                seen.add(v["seed"])
                vals.append(v["margin"])
            margins.append(vals)
            means.append(float(np.mean(vals)))
            errors.append(float(np.std(vals, ddof=1) / math.sqrt(n_views)))
        mu = 0.5 * (means[0] + means[1])
        opposite = min(means) < 0 < max(means)
        rows.append(dict(pair_index=i, pair_id=pair_id, stage0_margin=means[0], stage1_margin=means[1],
                         stage0_margin_mc_se=errors[0], stage1_margin_mc_se=errors[1], global_margin=mu,
                         opposite_sign=int(opposite), global_positive_with_negative=int(mu > 0 and min(means) < 0)))
        for j, m in enumerate(means):
            stages.append(dict(pair_index=i, pair_id=pair_id, stage=j, views=n_views, stage_margin=m,
                               global_margin=mu, negative_stage=int(m < 0), stage_gate=gate(m),
                               global_gate=gate(mu), gate_difference=gate(m) - gate(mu),
                               **finite_k4(margins, j, k4_draws, np.random.default_rng([k4_seed, i, j]))))
    if not rows or len({r["pair_id"] for r in rows}) != len(rows) or len({r["pair_index"] for r in rows}) != len(rows):
        raise ValueError("Expected a nonempty frozen panel of distinct preference pairs")
    return rows, stages


def summarize(rows, stages, repeats, seed):
    """Resample whole pairs; retain both stages of every resampled pair."""
    if repeats < 1:
        raise ValueError("bootstrap repetitions must be positive")
    n = len(rows)
    opposite = np.array([r["opposite_sign"] for r in rows])
    masked = np.array([r["global_positive_with_negative"] for r in rows])
    values = np.array([r["gate_difference"] for r in stages]).reshape(n, 2)
    negative = np.array([r["negative_stage"] for r in stages], dtype=bool).reshape(n, 2)
    weight_sum = np.where(negative, values, 0).sum(axis=1)
    counts = negative.sum(axis=1)
    draws = np.random.default_rng(seed).integers(n, size=(repeats, n))
    n_weak = counts[draws].sum(axis=1)
    valid = n_weak > 0

    def estimate(point, samples):
        return dict(estimate=point, ci95=np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) else None,
                    valid_bootstrap_draws=len(samples))

    k4 = np.array([r["k4_delta_mean"] for r in stages]).reshape(n, 2)
    k4_sum = np.where(negative, k4, 0).sum(axis=1)
    return dict(negative_stage_k4_mean_delta=estimate(
                    float(k4[negative].mean()) if negative.any() else None,
                    k4_sum[draws[valid]].sum(axis=1) / n_weak[valid]),
                all_stages_k4_mean_delta=estimate(float(k4.mean()), k4.mean(axis=1)[draws].mean(axis=1)),
                n_pairs=n, n_negative_stages=int(counts.sum()), n_pairs_with_negative_stage=int((counts > 0).sum()),
                n_opposite_sign=int(opposite.sum()), n_global_positive_with_negative=int(masked.sum()),
                bootstrap=dict(unit="preference pair, both stages kept together", repeats=repeats, seed=seed),
                opposite_fraction=estimate(float(opposite.mean()), opposite[draws].mean(axis=1)),
                global_positive_with_negative_fraction=estimate(float(masked.mean()), masked[draws].mean(axis=1)),
                negative_stage_mean_gate_difference=estimate(
                    float(values[negative].mean()) if negative.any() else None,
                    weight_sum[draws[valid]].sum(axis=1) / n_weak[valid]),
                negative_stage_gate_difference_quantiles=dict(zip(
                    ("q05", "q25", "median", "q75", "q95"),
                    np.quantile(values[negative], [.05, .25, .5, .75, .95]).tolist())) if negative.any() else None)


def plot(report, rows, stages, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, hist, finite) = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for key, color, label in (
            (0, "#798791", "Other pairs"), (1, "#306c98", "Opposite signs"),
            (2, "#c45728", "Opposite signs, global mean > 0")):
        chosen = [r for r in rows if r["opposite_sign"] + r["global_positive_with_negative"] == key]
        if chosen:
            ax.scatter([r["stage0_margin"] for r in chosen], [r["stage1_margin"] for r in chosen],
                       s=21, alpha=.8, color=color, label=label)
    ax.axvline(0, color="black", linewidth=.7)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(xlabel="High-noise stage mean margin (indices 0–499)",
           ylabel="Low-noise stage mean margin (indices 500–999)", title=f'{len(rows)} preference pairs')
    ax.legend(fontsize=8)
    weak = [r["gate_difference"] for r in stages if r["negative_stage"]]
    if weak:
        hist.hist(weak, bins="auto", color="#306c98")
    hist.axvline(0, color="black", linewidth=.8)
    hist.set(xlabel="Stage gate − global gate", ylabel="Number of stages",
             title=f'{summary["n_negative_stages"]} negative-margin stages')
    finite.scatter(weak, [r["k4_delta_mean"] for r in stages if r["negative_stage"]], s=18, alpha=.7)
    finite.axhline(0, color="0.5", linewidth=.7)
    finite.axvline(0, color="0.5", linewidth=.7)
    finite.set(xlabel="Gate difference at bank means", ylabel="Mean K=4 gate difference",
               title="Tagged view from a negative stage")
    fig.suptitle(f"SD3.5-M Flow-DPO at update 1,600 · {len(stages) // len(rows)} stages, "
                 f"{stages[0]['views']} views per pair and stage\n"
                 f"View RNG seed {summary['view_seed']} · one point per preference pair", fontsize=10)
    fig.savefig(report / "stage-mechanism.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261121, help="whole-pair bootstrap RNG seed")
    parser.add_argument("--k4-draws", type=int, default=10000)
    parser.add_argument("--k4-seed", type=int, default=17)
    args = parser.parse_args()
    if args.k4_draws < 2 or args.k4_seed < 0:
        parser.error("require at least two K4 draws and a nonnegative K4 seed")
    protocol = read_json(args.input / "protocol.json")
    rows, stages = pair_rows(args.input, protocol, args.k4_draws, args.k4_seed)
    summary = summarize(rows, stages, args.repeats, args.seed)
    summary.update(k4_draws_per_pair_stage=args.k4_draws, k4_seed=args.k4_seed,
                   k4_design="Empirical banks sampled with replacement. Same tagged first view. Stage-wise companions from its stage; global companions independently select either stage with probability 0.5. Negative status from bank mean. Quantiles of paired deltas use common random numbers. Pair intervals condition on saved banks.",
                   protocol_id=protocol["protocol_id"], view_seed=protocol["sampling"]["view_seed"],
                   pair_selection_seed=protocol["sampling"]["pair_seed"], views_per_stage=protocol["sampling"]["stage_views"],
                   sampling_unit="independent noise and scheduler-index draw conditional on a fixed latent pair",
                   fraction_denominator="all preselected preference pairs",
                   gate_definition="sigmoid(-mean margin); global averages both stages with equal probability")
    report = args.input / "report"
    report.mkdir(exist_ok=True)
    for name, table in (("pair-margins", rows), ("stage-weights", stages)):
        with (report / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    plot(report, rows, stages, summary)
    (report / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
