#!/usr/bin/env python3
"""Human-supervised appendix E.1/E.2 gradient measurements (no automatic all stage)."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.runtime.distributed import subprocess_environment
from appendix_jobs import gpu_tokens, run_probes, shard_group

EXPERIMENTS = {"sd15": "sd15-gradient", "flow": "sd35-gradient"}
SAMPLING_FLAGS = ("pair_count", "pair_seed", "view_seed", "source_blocks", "source_views_per_block",
                  "audit_blocks", "audit_views_per_block")


def settings(args):
    pilot = args.phase == "pilot"
    defaults = dict(pair_count=4 if pilot else 128, pair_seed=20261121, view_seed=17,
                    source_blocks=2 if pilot else 8, source_views_per_block=24 if pilot else 96,
                    audit_blocks=2 if pilot else 8, audit_views_per_block=8 if pilot else 128)
    values = {key: getattr(args, key) if getattr(args, key) is not None else value
              for key, value in defaults.items()}
    ks = [1, 2, 4, 6, 8] if args.experiment == "sd15" else [1, 2, 4]
    if (values["pair_count"] < 1 or values["source_blocks"] < 2 or values["audit_blocks"] < 2
            or values["audit_blocks"] % 2 or values["audit_views_per_block"] < 1
            or values["source_views_per_block"] < max(ks)
            or any(values["source_views_per_block"] % k for k in ks)):
        raise ValueError("require positive pairs/views, >=2 source blocks, even >=2 audit blocks, source views divisible by every K")
    return values, ks


def worker_shards(pair_count, group, devices):
    """A fixed group owns pair_index % groups == part, regardless of GPU count."""
    part, groups = group
    if groups > pair_count:
        raise ValueError("group count cannot exceed pair count")
    active = min(len(devices), pair_count // groups)
    stride = active * groups
    return [(gpu, part + rank * groups, stride, list(range(part + rank * groups, pair_count, stride)))
            for rank, gpu in enumerate(devices[:active])]


def checked_json(path, field):
    data = read_json(path)
    if data.get(field) != digest({key: value for key, value in data.items() if key != field}):
        raise ValueError(f"digest check failed: {path}")
    return data


def inspect(output):
    """Read-only, standard-library progress/identity check; deep Gram checks are in collect."""
    result = {"output": str(output), "prepared": "missing", "protocol": "missing",
              "complete": [], "missing": [], "failed": [], "checks": {}, "deep_validation": "pending collect"}
    try:
        if (output / "prepared.json").exists():
            prepared = checked_json(output / "prepared.json", "prepared_id")
            result["prepared"] = prepared["prepared_id"]
        if not (output / "protocol.json").exists():
            if result["prepared"] != "missing":
                conditions = ["sd15_t499"] if prepared["request"]["experiment"] == "sd15-gradient" else ["sd35_stage0", "sd35_stage1"]
                if "scan" in prepared["request"]:
                    conditions = [row["condition"] for row in prepared["request"]["scan"]["conditions"]]
                result["missing"] = [f"{pair['pair_index']:04d}-{name}.json" for pair in prepared["pairs"] for name in conditions]
            return result
        protocol = checked_json(output / "protocol.json", "protocol_id")
        result["protocol"] = protocol["protocol_id"]
        if (result["prepared"] != protocol["prepared_id"] or prepared["pairs"] != protocol["pairs"]
                or prepared["request"]["sampling"] != protocol["sampling"]
                or prepared["request"]["experiment"] != protocol["experiment"]):
            raise ValueError("prepared pair identity differs from protocol")
        expected = {f"{pair['pair_index']:04d}-{condition['condition']}.json": (pair, condition)
                    for pair in protocol["pairs"] for condition in protocol["conditions"]}
        for name, (pair, condition) in expected.items():
            path = output / "pairs" / name
            if not path.exists():
                result["missing"].append(name)
                continue
            try:
                record = checked_json(path, "result_id")
                identity = (record["protocol_id"], record["pair_id"], record["pair_index"], record["condition"])
                if identity != (protocol["protocol_id"], pair["pair_id"], pair["pair_index"], condition["condition"]):
                    raise ValueError("record identity mismatch")
                result["complete"].append(name)
            except (ValueError, KeyError, TypeError) as error:
                result["failed"].append({"file": name, "error": str(error)})
        for path in (output / "pairs").glob("*.json"):
            if path.name not in expected:
                result["failed"].append({"file": path.name, "error": "outside frozen protocol"})
        check_names = ["init-check", "repeat-view-check"]
        if protocol.get("scan", {}).get("view_batch_size", 1) > 1:
            check_names.append("batch-check")
        for name in check_names:
            path = output / f"{name}.json"
            if not path.exists():
                result["checks"][name] = "missing"
                continue
            check = checked_json(path, "result_id")
            result["checks"][name] = "passed" if check.get("passed") is True and check["protocol_id"] == protocol["protocol_id"] else "failed"
        result["ready_to_collect"] = not result["missing"] and not result["failed"] and all(
            result["checks"].get(key) == "passed" for key in check_names)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failed"].append({"error": str(error)})
    return result


def deep_validate(output):
    state = inspect(output)
    if not state.get("ready_to_collect"):
        raise ValueError(f"complete records and passed initialization/repeat checks required: {json.dumps(state)}")
    # NumPy is only needed after the lightweight readiness check.
    from appendix_gradient_report import read_pair_records
    protocol = checked_json(output / "protocol.json", "protocol_id")
    records = read_pair_records(output, protocol)
    return protocol, records


def require_pilot(parent, workflow):
    pilot = parent / "pilot"
    protocol, _ = deep_validate(pilot)
    for name, expected_sha in protocol["source"]["probe"].items():
        if sha256(ROOT / "src/self_correct_dpo" / name) != expected_sha:
            raise ValueError(f"probe source changed since pilot: {name}; run a new pilot")
    saved = read_json(pilot / "workflow.json")
    for key in ("experiment", "python", "resources", "checkpoint", "pairs", "sources"):
        if saved[key] != workflow[key]:
            raise ValueError(f"pilot {key} differs; run a new pilot with this code/resources/checkpoint")
    prepared = checked_json(parent / "formal" / "prepared.json", "prepared_id")
    pilot_prepared = checked_json(pilot / "prepared.json", "prepared_id")
    if (prepared["request"]["checkpoint"] != protocol["checkpoint"]
            or prepared["dataset"] != pilot_prepared["dataset"]):
        raise ValueError("formal checkpoint or dataset differs from passed pilot")
    return protocol["protocol_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "pilot", "run", "status", "collect", "report"))
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--output", type=Path, required=True, help="new root; data live under ROOT/EXPERIMENT/pilot|formal")
    parser.add_argument("--phase", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--python", help="GPU worker interpreter override; default from resource registry")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--gpus", help="allocated CUDA_VISIBLE_DEVICES tokens, e.g. 0,1 or GPU UUIDs")
    parser.add_argument("--group", type=shard_group, default=(0, 1), help="fixed pair group, 1/N through N/N")
    parser.add_argument("--scratch-dir", type=Path, help="existing local disk for temporary block means")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17, help="report bootstrap seed")
    for key in SAMPLING_FLAGS:
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args()
    if args.action == "pilot":
        args.phase = "pilot"
    if args.action == "run" and args.phase != "formal":
        parser.error("run is formal only; use pilot for the independent engineering check")
    try:
        values, ks = settings(args)
        registry = load_resources(args.resources)
        interpreter = args.python or registry["MVPO_SD15_PROBE_PY" if args.experiment == "sd15" else "MVPO_FLOW_PROBE_PY"]
        parent = args.output.expanduser().resolve() / args.experiment
        output = parent / args.phase
        cmd = [interpreter, str(ROOT / "scripts/mvpo_appendix_probe.py"), EXPERIMENTS[args.experiment],
               "--output", str(output), "--check-initialization"]
        for key, value in values.items():
            cmd += ["--" + key.replace("_", "-"), str(value)]
        for key in ("resources", "checkpoint", "pairs", "scratch_dir"):
            if value := getattr(args, key):
                cmd += ["--" + key.replace("_", "-"), str(value.expanduser().resolve())]
        # Freeze launcher inputs separately: no changes to the legacy probe protocol.
        source_paths = ["scripts/gradient_experiment.py", "scripts/mvpo_appendix_probe.py", "scripts/appendix_jobs.py",
                        "src/self_correct_dpo/evaluation/appendix_probe.py", "src/self_correct_dpo/evaluation/appendix_backend.py",
                        "src/self_correct_dpo/evaluation/appendix_gradients.py"]
        workflow = dict(schema_version=1, experiment=args.experiment, phase=args.phase, parameters=values,
                        python=interpreter, resources=str((args.resources or Path(os.environ.get("SCD_RESOURCES", ROOT / "configs/resources.json"))).expanduser().resolve()),
                        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                        pairs=str(args.pairs.resolve()) if args.pairs else None,
                        sources={path: sha256(ROOT / path) for path in source_paths})
        if args.action == "plan":
            devices = args.gpus.split(",") if args.gpus else ["<allocated-GPU>"]
            shards = worker_shards(values["pair_count"], args.group, devices)
            condition_count = 1 if args.experiment == "sd15" else 2
            per_pair = condition_count * (values["source_blocks"] * values["source_views_per_block"] + values["audit_blocks"] * values["audit_views_per_block"])
            print(json.dumps(dict(experiment=args.experiment, phase=args.phase, output=str(output), ks=ks, **values,
                                  conditions=condition_count, measurement_pair_view_backwards=values["pair_count"] * per_pair,
                                  group_pair_view_backwards=sum(len(row[3]) for row in shards) * per_pair,
                                  check_backwards_extra="2 initialization + 2 per condition, on shard 0",
                                  purpose="ENGINEERING CHECK ONLY: numerics, memory and throughput" if args.phase == "pilot" else
                                  "E.1 raw expected gradients (including flow replication); E.2 weight-gradient decoupling from the same archived Gram",
                                  interpretation="Raw gradients averaged before norms/cosines; finite independent audit reference"), indent=2))
            print("prepare:", shlex.join(cmd + ["--prepare-only"]))
            for gpu, index, count, indices in shards:
                print(f"pairs={indices}; CUDA_VISIBLE_DEVICES={gpu} " + shlex.join(cmd + ["--shard-index", str(index), "--shard-count", str(count)]))
            return
        if args.action == "status":
            state = inspect(output)
            attempts = [read_json(path) for path in sorted((output / "logs").glob("attempt-*.json"))]
            state["worker_attempts"] = attempts
            state["record_counts"] = {key: len(state[key]) for key in ("complete", "missing", "failed")}
            print(json.dumps(state, indent=2))
            return
        if args.action in ("collect", "report"):
            protocol, records = deep_validate(output)
            if args.phase == "formal":
                require_pilot(parent, read_json(output / "workflow.json"))
            manifest = dict(protocol_id=protocol["protocol_id"], records={path.name: sha256(path) for path in sorted((output / "pairs").glob("*.json"))},
                            checks={name: sha256(output / name) for name in ("init-check.json", "repeat-view-check.json")})
            if args.action == "collect":
                atomic_json(output / "collection.json", manifest)
                print(f"Validated {len(records)} pair-condition records; collection: {output / 'collection.json'}")
            else:
                if read_json(output / "collection.json") != manifest:
                    raise ValueError("records/checks changed since collect; collect again")
                subprocess.run([sys.executable, str(ROOT / "scripts/appendix_gradient_report.py"), "--input", str(output),
                                "--repeats", str(args.repeats), "--seed", str(args.seed)], check=True)
            return
        output.mkdir(parents=True, exist_ok=True)
        workflow_path = output / "workflow.json"
        if workflow_path.exists():
            if read_json(workflow_path) != workflow:
                raise ValueError("workflow settings/code changed; use its original settings or a new output root")
        elif args.action != "prepare":
            raise ValueError("prepare this phase first")
        elif any(path.name != "logs" for path in output.iterdir()):
            raise ValueError("refusing to adopt a nonempty legacy output; use a new root")
        else:
            atomic_json(workflow_path, workflow)
        env = dict(subprocess_environment(os.environ), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        env.setdefault("OMP_NUM_THREADS", "2")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
        if args.action == "prepare":
            subprocess.run(cmd + ["--prepare-only"], env=env, cwd=ROOT, check=True)
            return
        if not (output / "prepared.json").exists():
            raise ValueError("prepare did not finish")
        if args.action == "run":
            require_pilot(parent, workflow)
        devices = gpu_tokens(args.gpus, env, interpreter)
        commands = []
        for gpu, index, count, indices in worker_shards(values["pair_count"], args.group, devices):
            shard = cmd + ["--shard-index", str(index), "--shard-count", str(count)]
            print(f"GPU {gpu}: pairs {indices}: {shlex.join(shard)}", flush=True)
            commands.append((gpu, index, shard))
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        attempt_path = logs / f"attempt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{os.getpid()}.json"
        attempt = dict(group=[args.group[0] + 1, args.group[1]], gpus=devices, status="running")
        atomic_json(attempt_path, attempt)
        try:
            run_probes(commands, output, env)
        except BaseException as error:
            atomic_json(attempt_path, {**attempt, "status": "failed", "error": str(error)})
            raise
        atomic_json(attempt_path, {**attempt, "status": "complete"})
        print("This group finished. Inspect status, then collect and report as separate steps.")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
