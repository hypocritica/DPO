#!/usr/bin/env python3
"""Summarize paired rewards already scored by a native evaluate_suite plan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from self_correct_dpo.common import canonical_scheduler_config, digest, model_artifact_identity, output_lock, read_json, read_jsonl, sha256
from self_correct_dpo.evaluation.appendix_generation import checked_scores, image_key, resolve_adapter
from self_correct_dpo.evaluation.appendix_generation_report import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap, csv_write
from self_correct_dpo.evaluation.render import effective_sampling, expected_records
from self_correct_dpo.evaluation.scorers import METRICS
from self_correct_dpo.evaluation.suite import PANELS, freeze_json, read_plan, safe_target, validate_task_render


def select_task(plan, selector, benchmark, step):
    matches = [
        task for task in plan["tasks"]
        if task["benchmark"] == benchmark and task["step"] == step
        and selector in (task["method"], task["experiment_id"])
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one task for {selector!r}, {benchmark}, step {step}; "
                         f"found {[task['experiment_id'] for task in matches]}. Select its exact experiment ID.")
    task = matches[0]
    if task["task_id"] != digest({key: v for key, v in task.items() if key != "task_id"}):
        raise ValueError(f"Invalid task identity: {task['task_id']}")
    return task


def load_source(root, plan, task):
    target = safe_target(root, task["output"])
    run, receipt = validate_task_render(task, plan, root)
    base = model_artifact_identity(task["model"])
    if run["sampling"] != effective_sampling(run["config"]) or run["base_identity"] != base:
        raise ValueError(f"Render sampling or base model inventory changed: {target}")

    checkpoint = task["checkpoint"]
    adapter_hash = None
    if task["step"]:
        training_run = Path(task["run_path"]).expanduser().resolve(strict=True)
        adapter = resolve_adapter(training_run, task["step"], checkpoint["run_id"], checkpoint["checkpoint_id"])
        if (any(adapter[key] != checkpoint[key] for key in ("checkpoint_sha256", "training_sha256"))
                or run["model_identity"] != adapter["inventory"]):
            raise ValueError(f"Adapter bytes or render inventory differ from the native task: {target}")
        adapter_hash = adapter["files"]["adapter/adapter_model.safetensors"]
    elif (checkpoint["checkpoint_id"] != "base" or checkpoint["base_model_id"] != digest(task["model"])
          or run["model_identity"] != base):
        raise ValueError(f"Invalid pretrained-model render identity: {target}")

    images = {}
    expected = expected_records(run)
    for row in read_jsonl(target / "render" / "manifest.jsonl"):
        key = image_key(row)
        payload = {name: value for name, value in row.items() if name != "image_id"}
        if (key in images or key not in expected or any(row.get(k) != v for k, v in expected[key].items())
                or row["image_id"] != digest({"run_id": run["run_id"], **payload})):
            raise ValueError(f"Unexpected rendered prompt, seed or image identity: {target}, {key}")
        images[key] = row
    if images.keys() != expected.keys():
        raise ValueError(f"Rendered image journal does not cover the complete design: {target}")
    source = {
        "task_id": task["task_id"], "method": task["method"], "experiment_id": task["experiment_id"],
        "step": task["step"], "checkpoint": checkpoint, "adapter_sha256": adapter_hash,
        "render_run_id": run["run_id"], "render_manifest_sha256": receipt["manifest_sha256"],
        "target": str(target),
    }
    return target, run, receipt, images, source


def load_scores(metric, arm, scorer=None):
    target, run, receipt, images, _ = arm
    if scorer is None:
        scorer = read_json(target / "scores" / metric / "identity.json")["identity"]["scorer"]
    rows, identity, complete = checked_scores(target, metric, run, receipt, images, scorer)
    return {image_key(row): row for row in rows}, identity, complete


def report(args):
    import numpy as np

    plan_path = args.plan.expanduser().resolve(strict=True)
    if plan_path.is_dir():
        plan_path /= "plan.json"
    if plan_path.name != "plan.json":
        raise ValueError("Expected a native evaluate_suite plan.json or its containing directory")
    plan, root = read_plan(plan_path.parent)
    baseline_step = args.baseline_step if args.baseline_step is not None else args.step
    tasks = [select_task(plan, selector, args.benchmark, step)
             for selector, step in ((args.method, args.step), (args.baseline, baseline_step))]
    if tasks[0]["task_id"] == tasks[1]["task_id"] or tasks[0]["family"] != tasks[1]["family"]:
        raise ValueError("Select two distinct tasks in the same model family")
    left, right = [load_source(root, plan, task) for task in tasks]
    first, second = left[1], right[1]
    generation_keys = ("base_identity", "suite", "sampling", "gpu", "versions",
                       "render_sha256", "common_sha256", "zimage_compat_sha256")
    if (any(first.get(key) != second.get(key) for key in generation_keys)
            or canonical_scheduler_config(first["scheduler_config"]) != canonical_scheduler_config(second["scheduler_config"])
            or left[3].keys() != right[3].keys()):
        raise ValueError("Base, prompts, seed design, renderer or scheduler protocol differs")
    seeds = first["sampling"]["seeds"]
    prompts = first["suite"]
    ordered_keys = list(expected_records(first))
    paired_rows, metric_rows = [], []
    scorers = {}
    for metric in args.metrics:
        method_scores, method_identity, method_receipt = load_scores(metric, left)
        scorer = method_identity["identity"]["scorer"]
        baseline_scores, baseline_identity, baseline_receipt = load_scores(metric, right, scorer=scorer)
        for name in ("versions", "code_sha256"):
            if method_identity["identity"][name] != baseline_identity["identity"][name]:
                raise ValueError(f"{metric} scorer implementation or package versions differ")
        method_values = np.array([float(method_scores[key]["score"]) for key in ordered_keys])
        baseline_values = np.array([float(baseline_scores[key]["score"]) for key in ordered_keys])
        deltas = method_values - baseline_values
        statistics = bootstrap(np, deltas.reshape(len(prompts), len(seeds)))
        metric_rows.append({
            "metric": metric, "prompts": len(prompts), "images": len(deltas), "seed_count": len(seeds),
            "method_mean": float(method_values.mean()), "baseline_mean": float(baseline_values.mean()), **statistics,
        })
        paired_rows.extend({
            "metric": metric, "scene_id": key[0], "variant_id": key[1], "generation_seed": key[2],
            "prompt": method_scores[key]["prompt"],
            "method_image_sha256": method_scores[key]["image_sha256"],
            "baseline_image_sha256": baseline_scores[key]["image_sha256"],
            "method_score": float(m), "baseline_score": float(b), "delta": float(d),
        } for key, m, b, d in zip(ordered_keys, method_values, baseline_values, deltas))
        scorers[metric] = {"identity": scorer, "method": {**method_receipt, **method_identity},
                           "baseline": {**baseline_receipt, **baseline_identity}}

    payload = {
        "schema_version": 1, "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
        "plan_id": plan["plan_id"], "family": tasks[0]["family"], "benchmark": args.benchmark,
        "panel": plan["panels"][args.benchmark], "method": left[4], "baseline": right[4],
        "sampling": first["sampling"], "scheduler": first["scheduler_config"],
        "render_source_sha256": first["render_sha256"], "scorers": scorers, "comparisons": metric_rows,
        "bootstrap": {"unit": "prompt with all included generation seeds kept together",
                      "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "interval": "pointwise paired percentile 95%",
                      "win_rate": "one point per win and half per tie, then averaged over seeds and prompts",
                      "conditioned_on": "the listed model checkpoints and generation seeds"},
    }
    payload["report_id"] = digest(payload)
    out = args.out.expanduser().resolve()
    with output_lock(out / "paired-summary.json"):
        freeze_json(out / "paired-summary.json", payload)
        csv_write(out / "paired-differences.csv", metric_rows)
        csv_write(out / "paired-scores.csv", paired_rows)
    print(json.dumps({"report_id": payload["report_id"], "output": str(out),
                      "prompts": len(prompts), "generation_seeds": seeds}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--method", required=True, help="Unique native method or exact experiment ID")
    parser.add_argument("--baseline", required=True, help="Unique native method or exact experiment ID")
    parser.add_argument("--benchmark", required=True, choices=tuple(p for p in PANELS if p not in {"geneval", "dpg"}))
    parser.add_argument("--step", required=True, type=int, help="Optimizer step for --method")
    parser.add_argument("--baseline-step", type=int, help="Optimizer step for --baseline if different")
    parser.add_argument("--metrics", required=True, nargs="+", choices=METRICS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.metrics)) != len(args.metrics) or args.step < 0 or (args.baseline_step is not None and args.baseline_step < 0):
        parser.error("require distinct metrics and nonnegative steps")
    report(args)


if __name__ == "__main__":
    main()
