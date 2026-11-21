"""Freeze the main-table SD3.5-M Stage-wise K4 adapter for the existing qualitative runner."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import read_json
from self_correct_dpo.evaluation.suite import freeze_json
from self_correct_dpo.resources import load_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new qualitative output root")
    parser.add_argument("--gpus", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    spec = read_json(ROOT / "qualitative/config.json")["backbones"]["sd35_medium"]
    runs = read_json(ROOT / "configs/appendix-generation.json")["runs"]
    run_root = Path(load_resources()["FLOW_RUNS_ROOT"])
    for label, key in (("DPO", "dpo"), ("MVPO", "stagewise_k4")):
        run = runs[key]
        spec["methods"][label] = dict(run=str(run_root / run["run"]), step=1600,
                                      run_id=run["run_id"], checkpoint_id=run["checkpoints"]["1600"],
                                      source_method=key)
    spec["workers"] = args.gpus
    conf = {"seeds": spec["sampling"]["seeds"], "backbones": {"sd35_medium": spec}}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "stagewise-config.json"
    freeze_json(path, conf)
    print(path)


if __name__ == "__main__":
    main()
