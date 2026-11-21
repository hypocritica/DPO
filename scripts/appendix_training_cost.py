"""Export the appendix training configuration and measured costs from a run inventory."""

import argparse
import csv
import math
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import digest, read_json, sha256


PAPER_RUNS = {
    "sd15-dpo-p384-cu130-v1": ("SD1.5 DPO K=1", "11c4dbedab00"),
    "sd15-dpo-k6-stage-pv384-2gpu-10000-final-v1": ("SD1.5 Stage-wise MVPO K=6", None),
    "sdxl-dpo-p192-cu130-v1": ("SDXL DPO K=1", None),
    "sdxl-dpo-k6-stage-h800-2gpu-b2-pv192-10000-v1": ("SDXL Stage-wise MVPO K=6", None),
    "sd35_medium-dpo-k1-pair_views128-45090-b4-3200-C-bfloat16-trainonly-paper-v2-flow-b4-stat-v1": (
        "SD3.5-M Flow-DPO K=1", "3d6cc23e4e80",
    ),
    "sd35_medium-mvpo-k4-pair_views128-45090-b1-3200-C-bfloat16-trainonly-paper-v2-flow-microviews4-modelreport-v2": (
        "SD3.5-M Global MVPO K=4", "9d2fba88a824",
    ),
    "sd35_medium-mvpo-k4-pair_views128-45090-b1-3200-C-bfloat16-trainonly-stage500-paper-v2-flow-microviews4-modelreport-v2": (
        "SD3.5-M Stage-wise MVPO K=4", "830ec96c3b40",
    ),
    "zimage-dpo-k1-pair_views128-2h100-b4-3200-C-float32-trainonly-paper-v2-flow-b4-stat-v1": (
        "Z-Image Flow-DPO K=1", None,
    ),
    "zimage-mvpo-k4-pair_views128-2h100-b1-3200-C-float32-trainonly-stage500-paper-v2-flow-microviews4-v1": (
        "Z-Image Stage-wise MVPO K=4", None,
    ),
}


def _verified_manifests(run, run_id):
    verified = []
    for checkpoint in run.get("checkpoints", []):
        receipt = checkpoint.get("manifest") or {}
        recorded_id = receipt.get("checkpoint_id")
        if (receipt.get("run_id") == run_id and recorded_id
                and digest({k: v for k, v in receipt.items() if k != "checkpoint_id"}) == recorded_id):
            verified.append(receipt)
    return verified


def _observed_cuda_peak(run, run_id, step):
    entries = []
    for attempt in run.get("metric_attempts", []):
        for entry in attempt.get("last_records", []):
            peak = entry.get("peak_cuda_bytes")
            if (entry.get("run_id") == run_id and 0 < entry["step"] <= step
                    and type(peak) is int and peak > 0):
                entries.append((peak, entry["step"], attempt["name"]))
    return max(entries, default=(None, None, None)), len(entries)


def _elapsed_optimization(run, terminal, steps):
    if not steps:
        return None, None
    cost = terminal.get("cost") or {}
    seconds = cost.get("training_seconds")
    if seconds is not None and math.isfinite(seconds) and seconds > 0:
        return seconds, "complete.json cost.training_seconds"
    receipts = []
    for entry in run.get("latest_rank0_sessions", []):
        data = entry["data"]
        seconds = data.get("optimization_seconds_cumulative")
        if data.get("end_step") == steps and seconds is not None and math.isfinite(seconds) and seconds > 0:
            receipts.append((seconds, PurePosixPath(entry["path"]).name))
    if receipts and len({item[0] for item in receipts}) == 1:
        return receipts[0]
    return None, None


def _cost_row(run, inventory_id, inventory_end):
    name = run["name"]
    display, expected_checkpoint = PAPER_RUNS.get(name, (name, None))
    training = run["metadata"].get("training.json")
    if not training:
        raise ValueError(f"no archived training.json for {name}")
    config = training["config"]
    settings = config["training"]
    run_id = training["run_id"]
    identity_fields = ("config", "pairs_id", "base_identity", "code", "validation_pairs_id",
                       "monitor_inputs", "diagnostic_suite_sha256", "parent")
    if digest({key: training[key] for key in identity_fields if key in training}) != run_id:
        raise ValueError(f"training metadata failed its run ID check for {name}")
    manifests = _verified_manifests(run, run_id)
    last_step = max((m["step"] for m in manifests), default=0)
    expected_steps = settings["max_steps"]
    terminal = run["metadata"].get("complete.json") or {}
    if not terminal.get("complete") or terminal.get("run_id") != run_id or terminal.get("step") != expected_steps:
        terminal = {}
    completed = expected_steps if terminal or last_step == expected_steps else None
    matching = [m for m in manifests if expected_checkpoint and m["checkpoint_id"].startswith(expected_checkpoint)]
    if expected_checkpoint and len(matching) != 1:
        raise ValueError(f"cannot verify expected checkpoint {expected_checkpoint} for {name}")
    checkpoint = matching[0] if matching else {}
    seconds, cost_source = _elapsed_optimization(run, terminal, completed)
    (gpu_peak, gpu_peak_step, gpu_peak_attempt), memory_rows = _observed_cuda_peak(run, run_id, completed or last_step)
    world = training.get("world_size")
    pairs = training.get("effective_batch_size_pairs")
    views = training.get("effective_batch_size_pair_views")
    k = training.get("noise_views", settings.get("noise_views"))
    if all(value is not None for value in (k, pairs, views)) and pairs * k != views:
        raise ValueError(f"inconsistent pair-view budget for {name}")
    gpu = training.get("gpu")
    device_count = world if gpu and gpu != "cpu-test" else None
    terminal_cost = terminal.get("cost") or {}
    stage = settings.get("stagewise") or {}
    return {
        "label": display,
        "run_name": name,
        "run_id": run_id,
        "run_path": run["path"],
        "inventory_sha256": inventory_id,
        "inventory_as_of_utc": inventory_end,
        "family": config["model"]["family"],
        "objective": settings.get("objective"),
        "method": config.get("method"),
        "formal_arm": settings.get("formal_arm"),
        "stage_boundary": stage.get("boundary"),
        "training_seed": config.get("seed"),
        "training_runs_in_row": 1,
        "training_split": config["data"].get("split"),
        "training_pairs": training.get("training_pairs"),
        "training_pairs_id": training.get("pairs_id"),
        "training_manifest": config["data"].get("pairs"),
        "validation_pairs_id": training.get("validation_pairs_id"),
        "gpu": gpu,
        "gpu_count": device_count,
        "trainable_parameters": training.get("trainable_parameters"),
        "k": k,
        "local_pair_batch": settings.get("batch_size"),
        "accumulation_steps": settings.get("gradient_accumulation_steps"),
        "b_pairs_per_update": pairs,
        "n_pair_views_per_update": views,
        "configured_steps": expected_steps,
        "latest_archived_checkpoint_step": last_step or None,
        "completed_steps": completed,
        "completion_evidence": "complete.json" if terminal else "archived final checkpoint manifest" if completed else None,
        "appendix_checkpoint_step": checkpoint.get("step"),
        "appendix_checkpoint_id": checkpoint.get("checkpoint_id"),
        "gradient_checkpointing": settings.get("gradient_checkpointing"),
        "batch_views": settings.get("batch_views"),
        "training_dtype": settings.get("dtype"),
        "transformer_dtype_override": config["model"].get("transformer_dtype"),
        "lora_rank": settings.get("lora_rank"),
        "lora_alpha": settings.get("lora_alpha"),
        "learning_rate": settings.get("learning_rate"),
        "beta": settings.get("beta"),
        "configured_rollout": bool(settings.get("independent_rollout")),
        "rollout_model_calls": terminal_cost.get("rollout_model_calls"),
        "optimization_seconds": seconds,
        "optimization_seconds_per_update": seconds / completed if seconds else None,
        "optimization_gpu_hours": seconds * device_count / 3600 if seconds and device_count else None,
        "optimization_time_source": cost_source,
        "optimization_cost_scope": "Cumulative completed optimizer-loop time times the training GPU count",
        "allocated_gpu_hours": None,
        "observed_single_process_peak_cuda_gib": gpu_peak / 1024**3 if gpu_peak else None,
        "observed_peak_step": gpu_peak_step,
        "observed_peak_attempt_file": gpu_peak_attempt,
        "observed_peak_metric_rows": memory_rows,
        "peak_cuda_all_ranks_gib": None,
        "memory_scope": "Max allocated CUDA bytes from available matching metrics tail rows per emitting process",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="append", help="Exact archived run name, repeat to select multiple rows")
    args = parser.parse_args()
    inventory = read_json(args.inventory)
    if inventory.get("schema_version") != 1:
        raise ValueError("unsupported run inventory")
    inventory_id = sha256(args.inventory)
    runs = {run["name"]: run for run in inventory["runs"]}
    names = args.run or list(PAPER_RUNS)
    if len(set(names)) != len(names) or not set(names) <= runs.keys():
        raise ValueError("select distinct run names present in the inventory")
    rows = [_cost_row(runs[name], inventory_id, inventory.get("finished_at_utc")) for name in names]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
