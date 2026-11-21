"""Derive the matched B-arm configuration from the archived Global K4 training identity."""
import hashlib
import json
from pathlib import Path
import sys


def derive(selection, runs, checkout):
    ref = json.loads(selection.read_text())["runs"]["global_k4"]
    info = json.loads((runs / ref["run"] / "training.json").read_text())
    cfg = info["config"]
    t = cfg["training"]
    if (info["run_id"] != ref["run_id"] or cfg["method"] != "multiview-C-gaussian"
            or t["formal_arm"] != "C" or t.get("stagewise") or t["noise_views"] != 4
            or info["effective_batch_size_pair_views"] != 128 or cfg["seed"] != 17):
        raise ValueError("Unexpected Global K4 training identity")
    src = checkout / "src/self_correct_dpo"
    for relative in ("training.py", "models.py", "algorithms/formal.py", "algorithms/losses.py"):
        path = src / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["code"][path.name]:
            raise ValueError(f"Historical training source differs: {path}")
    for section in ("data", "model"):
        if not Path(cfg[section]["validation_report"]).is_file():
            raise FileNotFoundError(cfg[section]["validation_report"])
    cfg["method"] = "multiview-B-gaussian"
    t["formal_arm"] = "B"
    t["lora_checkpoint_steps"] = sorted(set(t["lora_checkpoint_steps"]) | {1600})
    t["wandb"]["tags"] = ["lossavg" if tag == "mvpo" else tag for tag in t["wandb"]["tags"]]
    cfg["experiment_note"] = "Matched Global K4 control, per-view logistic loss averaged over four Gaussian views."
    return cfg


if __name__ == "__main__":
    selection, runs, checkout, target = map(Path, sys.argv[1:])
    target.write_text(json.dumps(derive(selection, runs, checkout), indent=2))
