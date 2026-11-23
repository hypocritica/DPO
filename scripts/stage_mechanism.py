#!/usr/bin/env python3
"""Two independent two-GPU submissions share one forward-only stage panel."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import digest, output_lock, read_json, sha256
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.runtime.distributed import subprocess_environment
from appendix_jobs import gpu_tokens, run_probes
from gradient_experiment import checked_json
from gradient_scan import shared_frozen, source_identity

FIELDS = ("pair_count", "pair_seed", "view_seed", "stage_views", "stage_batch_size", "stage_progress_every")


def configuration(path, batch=None):
    config = read_json(path)
    if batch is not None:
        config["stage_batch_size"] = batch
    if (config["schema_version"] != 1 or config["family"] != "sd35_medium" or config["checkpoint_step"] != 1600
            or config["tasks"] != 2 or config["gpus_per_task"] != 2):
        raise ValueError("expected two two-GPU tasks on the archived SD3.5-M Flow-DPO 1600 checkpoint")
    if (any(type(config[k]) is not int or config[k] <= 0 for k in
            ("pair_count", "stage_views", "selection_views", "stage_batch_size", "stage_progress_every",
             "bootstrap_repeats", "k4_draws")) or config["pair_count"] % 4
            or config["selection_views"] < 2 or config["stage_views"] != 2 * config["selection_views"]
            or config["selection_views"] % config["stage_batch_size"] or config["k4_draws"] < 2):
        raise ValueError("require four equal pair shards and two equal banks divisible by forward batch size")
    return config


def task_indices(task, count):
    if task not in (1, 2):
        raise ValueError("stage task must be 1 or 2")
    return list(range(task - 1, count, 2))


def plan(config):
    return dict(model="SD3.5-M Flow-DPO", checkpoint_step=1600, pairs=config["pair_count"],
                views_per_pair_stage=config["stage_views"], selection_views=config["selection_views"],
                weight_estimation_views=config["stage_views"] - config["selection_views"],
                stage_indices=[[0, 500], [500, 1000]], stage_probabilities=[0.5, 0.5],
                total_pair_views=config["pair_count"] * 2 * config["stage_views"],
                generated_images=0, backward_passes=0, forward_batch=config["stage_batch_size"],
                tasks=[dict(task=t, gpus=2, pairs=len(task_indices(t, config["pair_count"])),
                            shard_indices=[t - 1, t + 1], shard_count=4,
                            pair_views=len(task_indices(t, config["pair_count"])) * 2 * config["stage_views"])
                       for t in (1, 2)])


def stage_state(output, config):
    data = output / "sd35-stage"
    expected = set(range(config["pair_count"]))
    result = dict(complete=[], missing=sorted(expected), failed=[], ready=False,
                  tasks={str(t): dict(complete=0, total=len(task_indices(t, config["pair_count"]))) for t in (1, 2)})
    if not (data / "protocol.json").exists():
        return result
    protocol = checked_json(data / "protocol.json", "protocol_id")
    prepared = checked_json(data / "prepared.json", "prepared_id")
    if (protocol["experiment"] != "sd35-stage" or protocol["prepared_id"] != prepared["prepared_id"]
            or protocol["pairs"] != prepared["pairs"] or len(protocol["pairs"]) != config["pair_count"]
            or protocol["checkpoint"]["step"] != config["checkpoint_step"]
            or any(protocol["sampling"][k] != config[k] for k in FIELDS if k != "pair_count")
            or {p["pair_index"] for p in protocol["pairs"]} != expected):
        raise ValueError("stage protocol differs from the frozen shared panel")
    for pair in protocol["pairs"]:
        index = pair["pair_index"]
        path = data / "pairs" / f"{index:04d}.json"
        if not path.exists():
            continue
        try:
            row = checked_json(path, "result_id")
            if ((row["protocol_id"], row["pair_id"], row["pair_index"]) !=
                    (protocol["protocol_id"], pair["pair_id"], index)
                    or len(row["stages"]) != 2
                    or any(len(s["views"]) != config["stage_views"] for s in row["stages"])):
                raise ValueError("record does not match frozen pair/protocol/view count")
            result["complete"].append(index)
        except (ValueError, KeyError, TypeError, OSError) as error:
            result["failed"].append(dict(path=str(path), error=str(error)))
    extra = {p.name for p in (data / "pairs").glob("*.json")} - {f"{i:04d}.json" for i in expected}
    result["failed"].extend(dict(path=name, error="outside shared panel") for name in sorted(extra))
    result["missing"] = sorted(expected - set(result["complete"]))
    result["ready"] = not result["missing"] and not result["failed"]
    result["protocol_id"] = protocol["protocol_id"]
    for t in (1, 2):
        result["tasks"][str(t)]["complete"] = len(set(result["complete"]) & set(task_indices(t, config["pair_count"])))
    return result


def freeze(args, config, registry):
    sources = source_identity()
    for name in ("scripts/stage_mechanism.py", "scripts/appendix_stage_report.py", "scripts/run_stage_mechanism_slai.sh"):
        sources[name] = sha256(ROOT / name)
    identity = dict(config=config, resources=str(args.resources), sources=sources,
                    interpreter=registry["MVPO_FLOW_PROBE_PY"], report_interpreter=sys.executable,
                    training_run=registry["MVPO_APPENDIX_SD35_DPO_RUN"], split_design="prefix A selects negatives; suffix B estimates weights")
    shared_frozen(args.output / "stage-experiment.json", identity)


def maybe_report(args, config, *, required=False):
    """Serialize both finishers; only a complete shared panel produces a report."""
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".stage-report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = stage_state(args.output, config)
        if state["failed"]:
            raise ValueError(json.dumps(state["failed"]))
        if not state["ready"]:
            print(json.dumps(dict(event="report_waiting_for_peer", complete=len(state["complete"]), total=config["pair_count"])), flush=True)
            if required:
                raise ValueError("stage report requires both task panels; rerun the missing task")
            return
        marker = args.output / "report-complete.json"
        identity = dict(protocol_id=state["protocol_id"], config=config,
                        report_source=sha256(ROOT / "scripts/appendix_stage_report.py"))
        products = ("summary.json", "pair-margins.csv", "stage-weights.csv", "stage-margins.png",
                    "stage-margins.pdf", "stage-weights.png", "stage-weights.pdf")
        report = args.output / "sd35-stage/report"
        if marker.exists() and read_json(marker).get("identity") == identity:
            hashes = read_json(marker).get("files", {})
            if all((report / p).is_file() and hashes.get(p) == sha256(report / p) for p in products):
                print(f"Report already complete: {report}", flush=True)
                return
        subprocess.run([sys.executable, str(ROOT / "scripts/appendix_stage_report.py"), "--input", str(report.parent),
                        "--selection-views", str(config["selection_views"]), "--repeats", str(config["bootstrap_repeats"]),
                        "--seed", str(config["bootstrap_seed"]), "--k4-draws", str(config["k4_draws"]),
                        "--k4-seed", str(config["k4_seed"])], check=True, cwd=ROOT)
        atomic_json(marker, dict(identity=identity, files={p: sha256(report / p) for p in products}))
        print(f"Both tasks complete. Figures and CSV: {report}", flush=True)


def run_task(args, config, registry):
    interpreter = registry["MVPO_FLOW_PROBE_PY"]
    env = dict(subprocess_environment(os.environ), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env.setdefault("OMP_NUM_THREADS", "2")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    devices = gpu_tokens(args.gpus, env, interpreter)
    if len(devices) != 2:
        raise ValueError("each stage task requires exactly two allocated GPUs")
    with output_lock(args.output / f"task-{args.task}-execution"):
        freeze(args, config, registry)
        data = args.output / "sd35-stage"
        cmd = [interpreter, str(ROOT / "scripts/mvpo_appendix_probe.py"), "sd35-stage", "--output", str(data),
               "--resources", str(args.resources)]
        for key in FIELDS:
            cmd += ["--" + key.replace("_", "-"), str(config[key])]
        print(f"Task {args.task}: preparing shared {config['pair_count']}-pair input metadata; no numerical pilot", flush=True)
        subprocess.run(cmd + ["--prepare-only"], env=env, cwd=ROOT, check=True)
        prepared = checked_json(data / "prepared.json", "prepared_id")
        if len(prepared["pairs"]) != config["pair_count"]:
            raise ValueError("prepared panel size differs")
        wanted = set(task_indices(args.task, config["pair_count"]))
        state = stage_state(args.output, config)
        if state["failed"]:
            raise ValueError(json.dumps(state["failed"]))
        if not wanted <= set(state["complete"]):
            commands = [(gpu, shard, cmd + ["--shard-index", str(shard), "--shard-count", "4"])
                        for gpu, shard in zip(devices, (args.task - 1, args.task + 1))]
            stopped = threading.Event()
            def heartbeat():
                while not stopped.wait(30):
                    files = {p.name for p in (data / "pairs").glob("*.json")}
                    done = sum(f"{i:04d}.json" in files for i in wanted)
                    print(json.dumps(dict(event="stage_task_progress", task=args.task, completed_pairs=done,
                                          total_pairs=len(wanted), global_pair_files=len(files))), flush=True)
            monitor = threading.Thread(target=heartbeat, daemon=True)
            monitor.start()
            try:
                run_probes(commands, data, env)
            finally:
                stopped.set()
                monitor.join(timeout=1)
        state = stage_state(args.output, config)
        if state["failed"] or not wanted <= set(state["complete"]):
            raise ValueError("task exited without its complete valid pair panel")
        print(f"Task {args.task} complete: {len(wanted)} pairs", flush=True)
        maybe_report(args, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status", "report"))
    parser.add_argument("--task", type=int, choices=(1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=Path(os.environ.get("SCD_RESOURCES", ROOT / "configs/resources.json")))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/appendix-stage-mechanism.json")
    parser.add_argument("--stage-batch-size", type=int)
    parser.add_argument("--gpus")
    args = parser.parse_args()
    args.output, args.resources = args.output.expanduser().resolve(), args.resources.expanduser().resolve()
    try:
        config = configuration(args.config, args.stage_batch_size)
        if args.action == "plan":
            print(json.dumps(plan(config), indent=2))
        elif args.action == "run":
            if args.task is None:
                parser.error("run requires --task 1 or 2")
            run_task(args, config, load_resources(args.resources))
        else:
            config = read_json(args.output / "stage-experiment.json")["config"]
            if args.action == "status":
                print(json.dumps(stage_state(args.output, config), indent=2))
            else:
                maybe_report(args, config, required=True)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
