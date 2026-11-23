#!/usr/bin/env python3
"""Three independent two-GPU jobs for the frozen MVPO checkpoint/index survey."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import digest, output_lock, read_json, sha256
from self_correct_dpo.evaluation.appendix_probe import load_checkpoint
from self_correct_dpo.evaluation.appendix_scan import fixed_conditions, validate_job
from self_correct_dpo.evaluation.checkpoint_catalog import resolve_checkpoint
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.runtime.distributed import subprocess_environment
from appendix_jobs import gpu_tokens, run_probes
from gradient_experiment import checked_json, deep_validate, inspect, worker_shards

FAMILIES = {"sd15": ("sd15-gradient", "MVPO_SD15_PROBE_PY"),
            "sd35_medium": ("sd35-gradient", "MVPO_FLOW_PROBE_PY")}
FIELDS = ("pair_count", "source_blocks", "source_views_per_block", "audit_blocks", "audit_views_per_block")


def configuration(path, batch_sd15=None, batch_flow=None):
    config = read_json(path)
    if config["schema_version"] != 1 or config["research_phase"] != "exploration":
        raise ValueError("this three-task launcher runs the exploration survey; freeze confirmation separately")
    for family, batch in (("sd15", batch_sd15), ("sd35_medium", batch_flow)):
        if batch is not None:
            config["view_batch_sizes"][family] = batch
        if config["view_batch_sizes"][family] not in (1, 2, 4):
            raise ValueError("view batch must be 1, 2 or 4")
        fixed_conditions(family, config["indices"])
    if (len(config["tasks"]) != 3 or [t["task"] for t in config["tasks"]] != [1, 2, 3]
            or any(type(t[f]) is not int or t[f] < 1 for t in config["tasks"] for f in FAMILIES)
            or any(len({t[f] for t in config["tasks"]}) != 3 for f in FAMILIES)):
        raise ValueError("require three distinct positive checkpoint steps per model, tasks 1/2/3")
    if (any(type(config[k]) is not int or config[k] < 1 for k in FIELDS)
            or config["source_blocks"] < 2 or config["audit_blocks"] < 2 or config["audit_blocks"] % 2
            or any(config["source_views_per_block"] % k for k in (1, 2, 4, 6, 8))):
        raise ValueError("invalid pair/block sampling")
    return config


def task_plan(config, task):
    row = config["tasks"][task - 1]
    count = config["pair_count"] * len(config["indices"]) * (
        config["source_blocks"] * config["source_views_per_block"] +
        config["audit_blocks"] * config["audit_views_per_block"])
    return dict(task=task, allocated_gpus=2,
                checkpoints=[dict(family=f, step=row[f], indices=config["indices"],
                                  view_batch_size=config["view_batch_sizes"][f]) for f in FAMILIES],
                formal_pair_view_observations=2 * count,
                per_checkpoint_pair_view_observations=count,
                extra=("numerical checks and pilot skipped" if config.get("skip_numerical_checks") else
                       "engineering pilot, initialization, repeat and batch numerical checks"))


def source_identity():
    paths = sorted((ROOT / "src/self_correct_dpo").rglob("*.py"))
    paths += [ROOT / "scripts" / name for name in ("gradient_scan.py", "gradient_experiment.py",
              "mvpo_appendix_probe.py", "appendix_jobs.py", "appendix_gradient_report.py")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def frozen(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen inputs/code changed: {path}; use original code/settings or a new output root")
    else:
        atomic_json(path, value)


def shared_frozen(path, value):
    """Short shared metadata writes wait for sibling tasks, rather than failing a race."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ("." + path.name + ".lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        frozen(path, value)


def freeze_task(output, config, task, registry, resource_path):
    """Anchor each model to the already known DPO run before selecting other steps."""
    signature = dict(config=config, sources=source_identity(), resources=str(resource_path),
                     interpreters={f: registry[key] for f, (_, key) in FAMILIES.items()},
                     run_roots={f: registry[key] for f, key in (("sd15", "MVPO_APPENDIX_SD15_DPO_RUN"),
                                                              ("sd35_medium", "MVPO_APPENDIX_SD35_DPO_RUN"))})
    # All three tasks must share a scan definition/code, even when started concurrently.
    shared_frozen(output / "scan.json", signature)
    folder = output / f"task-{task}"
    selection_path = folder / "selection.json"
    if selection_path.exists():
        selection = checked_json(selection_path, "selection_id")
        if selection["signature"] != signature:
            raise ValueError("task identity differs from frozen scan")
        return selection
    jobs = []
    for family in FAMILIES:
        _, _, training, anchor = load_checkpoint(family, resources=resource_path)
        step = config["tasks"][task - 1][family]
        run = Path(anchor["path"]).parent.parent
        _, _, identity = resolve_checkpoint(run, step)
        if identity["run_id"] != anchor["run_id"]:
            raise ValueError("target checkpoint belongs to a different training run")
        _, _, _, identity = load_checkpoint(family, run / "checkpoints" / f"step-{step:07d}", resource_path, identity)
        job = dict(schema_version=1, family=family, research_phase="exploration", checkpoint=identity,
                   conditions=fixed_conditions(family, config["indices"]),
                   rng_scope=dict(phase="exploration", family=family, pairs_id=training["pairs_id"],
                                  panel_seed=config["panel_seed"]),
                   latent_seed=config["latent_seed"], excluded_pair_ids=[],
                   view_batch_size=config["view_batch_sizes"][family])
        if config.get("skip_numerical_checks"):
            job["skip_numerical_checks"] = True
        job["job_id"] = digest(job)
        jobs.append(validate_job(job, family))
    selection = dict(signature=signature, task=task, jobs=jobs)
    selection["selection_id"] = digest(selection)
    atomic_json(selection_path, selection)
    return selection


def assert_checks(output):
    protocol, records = deep_validate(output)
    if (protocol.get("scan", {}).get("view_batch_size", 1) > 1
            and not protocol.get("scan", {}).get("skip_numerical_checks")):
        batch = checked_json(output / "batch-check.json", "result_id")
        if batch["protocol_id"] != protocol["protocol_id"] or batch["passed"] is not True:
            raise ValueError("batch numerical check missing or failed")
    return protocol, records


def check_panel(output, family, prepared):
    # CPU preparation across independent tasks must produce byte-identical panels.
    panel = dict(dataset=prepared["dataset"], pairs=prepared["pairs"])
    shared_frozen(output / "panels" / f"{family}.json", panel)


def run_checkpoint(args, config, registry, job, env):
    family = job["family"]
    experiment, key = FAMILIES[family]
    interpreter = registry[key]
    devices = gpu_tokens(args.gpus, env, interpreter)
    if len(devices) != 2:
        raise ValueError(f"each submitted task requires exactly 2 allocated GPUs, received {devices}")
    parent = args.output / f"task-{args.task}" / family
    if job.get("method"):
        parent = parent / job["method"]
    parent = parent / f"step-{job['checkpoint']['step']:07d}"
    parent.mkdir(parents=True, exist_ok=True)
    job_path = parent / "scan-job.json"
    frozen(job_path, job)
    skip_checks = job.get("skip_numerical_checks", False)
    for phase in (("formal",) if skip_checks else ("pilot", "formal")):
        output = parent / phase
        values = {key: config[key] for key in FIELDS}
        if phase == "pilot":
            values.update(pair_count=4, source_blocks=2, source_views_per_block=24,
                          audit_blocks=2, audit_views_per_block=8)
        cmd = [interpreter, str(ROOT / "scripts/mvpo_appendix_probe.py"), experiment,
               "--output", str(output), "--scan-job", str(job_path), "--resources", str(args.resources),
               "--pair-seed", str(config["panel_seed"]), "--view-seed", str(config["view_seed"]),
               "--scratch-dir", str(args.scratch_dir)]
        if not skip_checks:
            cmd.append("--check-initialization")
        for name, value in values.items():
            cmd += ["--" + name.replace("_", "-"), str(value)]
        subprocess.run(cmd + ["--prepare-only"], env=env, cwd=ROOT, check=True)
        prepared = checked_json(output / "prepared.json", "prepared_id")
        if phase == "formal":
            check_panel(args.output, family, prepared)
        if phase == "formal" and not skip_checks:
            pilot_protocol, _ = assert_checks(parent / "pilot")
            if (pilot_protocol["scan"] != job or pilot_protocol["checkpoint"] != prepared["request"]["checkpoint"]
                    or pilot_protocol["dataset"] != prepared["dataset"]
                    or pilot_protocol["pairs"] != prepared["pairs"][:len(pilot_protocol["pairs"])]):
                raise ValueError("pilot does not match formal checkpoint/conditions/panel")
        state = inspect(output)
        if not state.get("ready_to_collect"):
            commands = [(gpu, index, cmd + ["--shard-index", str(index), "--shard-count", str(count)])
                        for gpu, index, count, _ in worker_shards(values["pair_count"], (0, 1), devices)]
            run_probes(commands, output, env)
        protocol, records = assert_checks(output)
        # A frozen source mismatch must never be silently skipped during resume.
        for name, expected in protocol["source"]["probe"].items():
            if sha256(ROOT / "src/self_correct_dpo" / name) != expected:
                raise ValueError(f"measurement source drift: {name}")
        print(f"{family} step {job['checkpoint']['step']} {phase}: {len(records)} records checked", flush=True)
        if phase == "formal":
            subprocess.run([sys.executable, str(ROOT / "scripts/appendix_gradient_report.py"),
                            "--input", str(output), "--repeats", str(args.repeats)], env=env, cwd=ROOT, check=True)
        else:
            for path in sorted((output / "pairs").glob("*.json")):
                execution = read_json(path)["execution"]
                print(f"pilot {path.name}: {execution['measurement_seconds']:.1f}s, "
                      f"{execution['peak_reserved_bytes']/2**30:.2f} GiB reserved", flush=True)
        if phase == "pilot" and args.pilot_only:
            break


def scan_status(output, config):
    states = []
    for task in config["tasks"]:
        for family in FAMILIES:
            for phase in (("formal",) if config.get("skip_numerical_checks") else ("pilot", "formal")):
                path = output / f"task-{task['task']}" / family / f"step-{task[family]:07d}" / phase
                state = inspect(path)
                states.append(dict(task=task["task"], family=family, step=task[family], phase=phase,
                                   ready=state.get("ready_to_collect", False), checks=state["checks"],
                                   complete=len(state["complete"]), missing=len(state["missing"]),
                                   prepared=state["prepared"], failed=state["failed"]))
    return states


def report_scan(args, config):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = args.output / "scan-report"
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in config["tasks"]:
        for family in FAMILIES:
            root = args.output / f"task-{task['task']}" / family / f"step-{task[family]:07d}" / "formal"
            if not inspect(root).get("ready_to_collect"):
                rows.extend(dict(family=family, step=task[family], index=index, status="missing_or_failed",
                                 k=None, metric=None, estimate=None, low=None, high=None, audit_half_cosine=None)
                            for index in config["indices"])
                continue
            protocol, _ = assert_checks(root)
            selection = checked_json(args.output / f"task-{task['task']}" / "selection.json", "selection_id")
            expected_job = next(job for job in selection["jobs"] if job["family"] == family)
            expected_sampling = {key: config[key] for key in FIELDS if key != "pair_count"}
            expected_sampling.update(pair_seed=config["panel_seed"], view_seed=config["view_seed"])
            if (selection["signature"]["config"] != config or protocol["scan"] != expected_job
                    or len(protocol["pairs"]) != config["pair_count"]
                    or any(protocol["sampling"][key] != value for key, value in expected_sampling.items())):
                raise ValueError("reported checkpoint/sampling differs from scan catalog")
            check_panel(args.output, family, checked_json(root / "prepared.json", "prepared_id"))
            summary_path = root / "report/summary.json"
            if not summary_path.exists() or read_json(summary_path)["protocol_id"] != protocol["protocol_id"]:
                subprocess.run([sys.executable, str(ROOT / "scripts/appendix_gradient_report.py"), "--input", str(root),
                                "--repeats", str(args.repeats)], check=True)
            summary = read_json(summary_path)
            definitions = {c["condition"]: c for c in protocol["conditions"]}
            for condition in summary["conditions"]:
                audits = [p["audit_split"]["direction_cosine"] for p in summary["pairs"]
                          if p["condition"] == condition["condition"] and p["audit_split"]["direction_cosine"] is not None]
                for diff in condition["paired_differences"]:
                    for metric, result in diff["metrics"].items():
                        interval = result.get("pair_bootstrap_95") or [None, None]
                        rows.append(dict(family=family, step=task[family], index=definitions[condition["condition"]]["index_start"],
                                         status="measured", k=diff["k"], metric=metric, estimate=result["estimate"],
                                         low=interval[0], high=interval[1],
                                         audit_half_cosine=float(np.mean(audits)) if audits else None))
    for row in rows:
        row["numerical_validation"] = "skipped" if config.get("skip_numerical_checks") else "required"
    with (report / "scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(report / "status.json", scan_status(args.output, config))
    for family, k in (("sd15", 6), ("sd35_medium", 4)):
        metrics = ("source_corrected_distance_squared", "descent_cosine", "signed_projection", "stochastic_variance")
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
        steps = [task[family] for task in config["tasks"]]
        selected = [r for r in rows if r["family"] == family and r["k"] == k]
        for axis, metric in zip(axes.flat, (*metrics, "audit_half_cosine")):
            values = np.full((len(steps), len(config["indices"])), np.nan)
            for row in selected:
                value = row["audit_half_cosine"] if metric == "audit_half_cosine" else row["estimate"]
                if (metric == "audit_half_cosine" or row["metric"] == metric) and value is not None:
                    values[steps.index(row["step"]), config["indices"].index(row["index"])] = value
            scale = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1, 1e-12)
            art = axis.imshow(np.ma.masked_invalid(values), cmap="coolwarm", vmin=-scale, vmax=scale, aspect="auto")
            for i in range(len(steps)):
                for j in range(len(config["indices"])):
                    axis.text(j, i, "N/A" if not np.isfinite(values[i,j]) else f"{values[i,j]:.3g}",
                              ha="center", va="center", fontsize=7)
            axis.set(xticks=range(len(config["indices"])), xticklabels=config["indices"],
                     yticks=range(len(steps)), yticklabels=steps, xlabel="Scheduler index", ylabel="Training step",
                     title=metric.replace("_", " "))
            fig.colorbar(art, ax=axis, shrink=.7)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, .8, "MVPO minus LossAvg at the same K\nAudit panel: mean split-half cosine\n"
                          "All conditions retained, including null/negative.\nPointwise intervals are in scan.csv.\n"
                          "Missing/undefined cells are not zero.\nAudit reliability is diagnostic, not a truth guarantee.", fontsize=10)
        notice = "\nExploratory: numerical checks skipped" if config.get("skip_numerical_checks") else ""
        fig.suptitle(f"{family} exploration · K={k} · finite audit reference" + notice)
        for suffix in ("png", "pdf"):
            fig.savefig(report / f"{family}-scan.{suffix}", dpi=160)
        plt.close(fig)
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status", "report"))
    parser.add_argument("--task", type=int, choices=(1, 2, 3))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/appendix-gradient-scan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=Path(os.environ.get("SCD_RESOURCES", ROOT / "configs/resources.json")))
    parser.add_argument("--sd15-view-batch-size", type=int)
    parser.add_argument("--flow-view-batch-size", type=int)
    parser.add_argument("--gpus")
    parser.add_argument("--scratch-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--skip-numerical-checks", action="store_true",
                        help="Exploration: skip initialization/repeat/batch checks and pilot")
    parser.add_argument("--family", choices=tuple(FAMILIES), help="Run only this model; default runs both")
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    args.output, args.resources = args.output.expanduser().resolve(), args.resources.expanduser().resolve()
    try:
        config = configuration(args.config, args.sd15_view_batch_size, args.flow_view_batch_size)
        if args.skip_numerical_checks:
            if args.pilot_only:
                parser.error("--skip-numerical-checks cannot be combined with --pilot-only")
            config["skip_numerical_checks"] = True
        if args.action == "plan":
            print(json.dumps([task_plan(config, i) for i in ([args.task] if args.task else (1, 2, 3))], indent=2))
            return
        if args.action in ("status", "report"):
            saved = read_json(args.output / "scan.json")
            config = saved["config"]
            if args.action == "status":
                print(json.dumps(scan_status(args.output, config), indent=2))
            else:
                report_scan(args, config)
            return
        if not args.task:
            parser.error("run requires --task 1, 2 or 3")
        registry = load_resources(args.resources)
        env = dict(subprocess_environment(os.environ), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.setdefault("OMP_NUM_THREADS", "2")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
        with output_lock(args.output / f"task-{args.task}" / "execution"):
            selection = freeze_task(args.output, config, args.task, registry, args.resources)
            for job in selection["jobs"]:
                if args.family and job["family"] != args.family:
                    continue
                run_checkpoint(args, config, registry, job, env)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
