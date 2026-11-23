"""Rebuild the appendix gradient tables and figures from archived pair statistics."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import digest, read_json
from self_correct_dpo.evaluation.appendix_gradient_reporting import (
    audit_reliability_rows, paired_change_rows, plot_gradient_statistics, plot_pair_diagnostics,
    summarize_gradient_statistics,
)
from self_correct_dpo.runtime.checkpoints import atomic_json


def read_pair_records(root, protocol):
    if digest({key: value for key, value in protocol.items() if key != "protocol_id"}) != protocol["protocol_id"]:
        raise ValueError("gradient protocol metadata failed its digest check")
    sampling = protocol["sampling"]
    ks = sampling["ks"]
    if bool(protocol.get("native_objective")) != bool(protocol.get("scan", {}).get("method")):
        raise ValueError("native objective label and checkpoint method must agree")
    conditions = {row["condition"]: row for row in protocol["conditions"]}
    expected_ranges = {"sd15_t499": (499, 500), "sd35_stage0": (0, 500), "sd35_stage1": (500, 1000)}
    if "scan" in protocol:
        from self_correct_dpo.evaluation.appendix_scan import validate_job
        job = validate_job(protocol["scan"], protocol["model"]["config"]["family"])
        if protocol["conditions"] != job["conditions"] or protocol["checkpoint"] != job["checkpoint"]:
            raise ValueError("protocol differs from frozen scan checkpoint/conditions")
        expected_ranges = {row["condition"]: (row["index_start"], row["index_stop"]) for row in job["conditions"]}
    if (("scan" not in protocol and set(conditions) not in ({"sd15_t499"}, {"sd35_stage0", "sd35_stage1"}))
            or len(conditions) != len(protocol["conditions"])
            or any((row["index_start"], row["index_stop"]) != expected_ranges[name] for name, row in conditions.items())):
        raise ValueError("unexpected or incomplete gradient conditions in protocol")
    pairs = protocol["pairs"]
    if not pairs or any(len({pair[key] for pair in pairs}) != len(pairs) for key in ("pair_id", "pair_index")):
        raise ValueError("expected a nonempty frozen panel of distinct preference pairs")
    expected = {
        f"{pair['pair_index']:04d}-{condition}.json": (pair, condition)
        for pair in pairs for condition in conditions
    }
    actual = {file.name for file in (root / "pairs").glob("*.json")}
    if missing := expected.keys() - actual:
        raise ValueError(f"gradient records still missing: {', '.join(sorted(missing))}")
    if unexpected := actual - expected.keys():
        raise ValueError(f"gradient records outside the frozen protocol: {', '.join(sorted(unexpected))}")

    records = []
    for name, (pair, condition) in expected.items():
        record = read_json(root / "pairs" / name)
        result_id = digest({key: value for key, value in record.items() if key != "result_id"})
        if result_id != record["result_id"]:
            raise ValueError(f"gradient record failed its digest check in {name}")
        identity = (record["protocol_id"], record["pair_index"], record["pair_id"], record["condition"])
        if identity != (protocol["protocol_id"], pair["pair_index"], pair["pair_id"], condition):
            raise ValueError(f"unexpected protocol, pair, or condition in {name}")
        raw = record["statistics"]
        if protocol.get("native_objective"):
            from self_correct_dpo.evaluation.appendix_native_gradients import validate_native_statistics, coefficients
            validate_native_statistics(raw, protocol)
            check_view_banks(record, sampling, *expected_ranges[condition], name)
            for role in ("source", "audit"):
                means = ([b["moment_means"] for b in raw["source_blocks"]] if role == "source"
                         else raw["audit_block_moment_means"])
                for bi, (views, expected) in enumerate(zip(record[f"{role}_views"], means)):
                    moments = np.asarray([v["moments"] for v in views])
                    if (moments.shape != (sampling[f"{role}_views_per_block"], len(expected))
                            or not np.isfinite(moments).all() or not np.allclose(moments.mean(0), expected, rtol=1e-10, atol=1e-10)):
                        raise ValueError(f"native moment means disagree with view bank: {name}")
                    if role == "source" and not np.allclose(coefficients(moments, raw["native_objective"]).mean(0),
                                                            raw["source_blocks"][bi]["coefficient_means"], rtol=1e-10, atol=1e-10):
                        raise ValueError(f"native coefficient means disagree: {name}")
            records.append(record)
            continue
        source = raw["source_blocks"]
        audit = raw["audit_block_margin_means"]
        if (raw["schema_version"] != 1 or raw["ks"] != ks
                or raw["parameter_count"] != protocol["model"]["lora_parameter_count"]
                or len(source) != sampling["source_blocks"] or len(audit) != sampling["audit_blocks"]
                or any(raw[key] != sampling[key] for key in ("source_views_per_block", "audit_views_per_block"))):
            raise ValueError(f"unexpected sampling or LoRA dimension in {name}")
        roles = ["margin", "lossavg", *(f"mvpo:{k}" for k in ks if k > 1)]
        n_rows = len(source) * len(roles) + len(audit)
        gram = np.asarray(raw["block_mean_gram"], dtype=np.float64)
        if (raw["block_mean_gram_source_roles"] != roles or gram.shape != (n_rows, n_rows)
                or not np.isfinite(gram).all() or not np.allclose(gram, gram.T)
                or np.any(np.diag(gram) < 0)):
            raise ValueError(f"invalid block-mean Gram matrix in {name}")
        if not all(math.isfinite(margin) for margin in audit):
            raise ValueError(f"invalid audit margins in {name}")
        for block in source:
            squared_norms = block["group_squared_norm_sums"]
            if (not math.isfinite(block["margin_mean"]) or not -1 <= block["loss_derivative_mean"] <= 0
                    or set(squared_norms) != {"lossavg", "mvpo"}
                    or any(set(norms) != set(map(str, ks)) or not all(math.isfinite(x) and x >= 0 for x in norms.values())
                           for norms in squared_norms.values())):
                raise ValueError(f"invalid source-block statistics in {name}")
        check_view_banks(record, sampling, *expected_ranges[condition], name)
        records.append(record)
    return records


def check_view_banks(record, sampling, lo, hi, name):
    raw = record["statistics"]
    means = {
        "source": [block["margin_mean"] for block in raw["source_blocks"]],
        "audit": raw["audit_block_margin_means"],
    }
    seen = set()
    for bank in ("source", "audit"):
        blocks = record[f"{bank}_views"]
        if len(blocks) != len(means[bank]):
            raise ValueError(f"missing {bank} view blocks in {name}")
        for block, expected_mean in zip(blocks, means[bank]):
            if len(block) != sampling[f"{bank}_views_per_block"]:
                raise ValueError(f"wrong {bank} view count in {name}")
            margins = []
            for view in block:
                seed, index, margin = view["seed"], view["scheduler_index"], view["margin"]
                if seed in seen or not lo <= index < hi or not math.isfinite(margin):
                    raise ValueError(f"invalid or repeated {bank} view in {name}")
                seen.add(seed)
                margins.append(margin)
            if not math.isclose(sum(margins) / len(margins), expected_mean, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"{bank} block margin does not match its view bank in {name}")


def _table_rows(summaries, sampling, conditions=()):
    fixed = {row["condition"] for row in conditions if row["index_stop"] == row["index_start"] + 1} | {"sd15_t499"}
    def rows(entry, group, metrics, k=None, objective=None):
        for name, metric in metrics.items():
            mc = metric["monte_carlo_95"] or (None, None)
            pair = metric.get("pair_bootstrap_95") or (None, None)
            yield {
                "condition": entry["condition"], "pair_id": entry.get("pair_id"),
                "group": group, "objective": objective, "k": k, "metric": name,
                "estimate": metric["estimate"], "mc95_low": mc[0], "mc95_high": mc[1],
                "pair95_low": pair[0], "pair95_high": pair[1],
                "n_pairs": entry.get("n_pairs", 1), "n_pairs_defined": metric.get("n_pairs_defined"),
                "view_randomness": "Gaussian noise only" if entry["condition"] in fixed
                else "Stage-conditional scheduler index and Gaussian noise",
                "view_rng_seed": sampling.get("view_seed"),
                "source_blocks": sampling["source_blocks"],
                "source_views_per_block": sampling["source_views_per_block"],
                "audit_blocks": sampling["audit_blocks"],
                "audit_views_per_block": sampling["audit_views_per_block"],
            }

    for entry in summaries:
        for objective in entry["objectives"]:
            yield from rows(entry, "objective", objective["metrics"], objective["k"], objective["objective"])
        for difference in entry["paired_differences"]:
            yield from rows(entry, "mvpo_minus_lossavg", difference["metrics"], difference["k"])
        for group in ("audit", "coupling"):
            yield from rows(entry, group, entry[group])


def _write_csv(path, rows, numerical_validation="not_skipped"):
    rows = ({"numerical_validation": numerical_validation, **row} for row in rows)
    first = next(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=first.keys())
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2000, help="bootstrap repetitions")
    parser.add_argument("--seed", type=int, default=17, help="bootstrap RNG seed")
    args = parser.parse_args()
    protocol = read_json(args.input / "protocol.json")
    records = read_pair_records(args.input, protocol)
    summary = summarize_gradient_statistics(records, bootstrap_repeats=args.repeats, seed=args.seed)
    summary["protocol_id"] = protocol["protocol_id"]
    summary["lora_parameter_count"] = protocol["model"]["lora_parameter_count"]
    summary["sampling"] = protocol["sampling"]
    summary["numerical_validation"] = "skipped" if protocol.get("scan", {}).get("skip_numerical_checks") else "not_skipped"
    if protocol.get("native_objective"):
        summary["native_objective"] = protocol["native_objective"]
        summary["checkpoint_method"] = protocol["scan"]["method"]
        summary["covariance_sign"] = "Native loss-gradient convention; axis is independent audit loss gradient"
        summary["alignment_reference"] = "Negative independent audit native-loss gradient, not positive margin gradient"
    summary["condition_definitions"] = protocol["conditions"]
    summary["checkpoint"] = protocol["checkpoint"]
    summary["view_unit"] = "Independent Gaussian noise; fixed index or uniform within the frozen interval"
    report = args.input / "report"
    report.mkdir(exist_ok=True)
    validation = summary["numerical_validation"]
    _write_csv(report / "aggregate.csv", _table_rows(summary["conditions"], protocol["sampling"], protocol["conditions"]), validation)
    _write_csv(report / "pair-metrics.csv", _table_rows(summary["pairs"], protocol["sampling"], protocol["conditions"]), validation)
    _write_csv(report / "paired-changes.csv", iter(paired_change_rows(summary)), validation)
    _write_csv(report / "audit-reliability.csv", iter(audit_reliability_rows(summary)), validation)
    plot_gradient_statistics(summary, report)
    plot_pair_diagnostics(summary, report)
    atomic_json(report / "summary.json", summary)
    print(report)


if __name__ == "__main__":
    main()
