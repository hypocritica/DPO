"""Two-task forward collection, independent-bank selection, and final reporting."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import digest
from self_correct_dpo.evaluation.appendix_probe import stage_measurement
import stage_mechanism as runner
from appendix_stage_report import pair_rows, summarize


def fixture(root, config, completed=()):
    data = root / "sd35-stage"
    (data / "pairs").mkdir(parents=True, exist_ok=True)
    pairs = [dict(pair_index=i, pair_id=f"p{i}") for i in range(config["pair_count"])]
    sampling = {k: config[k] for k in runner.FIELDS if k != "pair_count"}
    prepared = dict(pairs=pairs, request=dict(experiment="sd35-stage", sampling=sampling))
    prepared["prepared_id"] = digest(prepared)
    protocol = dict(experiment="sd35-stage", pairs=pairs, sampling=sampling, checkpoint={"step": 1600},
                    prepared_id=prepared["prepared_id"])
    protocol["protocol_id"] = digest(protocol)
    for name, value in (("prepared", prepared), ("protocol", protocol)):
        (data / f"{name}.json").write_text(json.dumps(value))
    records = []
    for pair in pairs:
        banks = []
        for stage in (0, 1):
            banks.append(dict(stage_index=stage, index_start=stage*500, index_stop=(stage+1)*500, probability=.5,
                              views=[dict(view_index=v, scheduler_index=stage*500+v,
                                          seed=10000*pair["pair_index"]+1000*stage+v,
                                          margin=([-2., 3.] if v < config["selection_views"] else [1., -1.])[stage])
                                     for v in range(config["stage_views"])]))
        record = dict(**pair, protocol_id=protocol["protocol_id"], stages=banks)
        record["result_id"] = digest(record)
        records.append(record)
        if pair["pair_index"] in completed:
            (data / "pairs" / f"{pair['pair_index']:04d}.json").write_text(json.dumps(record))
    return protocol, records


def small_config():
    config = runner.configuration(ROOT / "configs/appendix-stage-mechanism.json")
    config.update(pair_count=4, stage_views=8, selection_views=4, bootstrap_repeats=32, k4_draws=64)
    return config


class StageMechanismTests(unittest.TestCase):
    def test_four_gpu_shards_cover_exactly_one_shared_panel(self):
        config = runner.configuration(ROOT / "configs/appendix-stage-mechanism.json")
        plan = runner.plan(config)
        self.assertEqual(plan["total_pair_views"], 49152)
        self.assertEqual([t["pair_views"] for t in plan["tasks"]], [24576, 24576])
        self.assertEqual(plan["backward_passes"], 0)
        sets = [set(runner.task_indices(t, 128)) for t in (1, 2)]
        self.assertFalse(sets[0] & sets[1])
        self.assertEqual(sets[0] | sets[1], set(range(128)))
        for t in plan["tasks"]:
            observed = {i for shard in t["shard_indices"] for i in range(shard, 128, 4)}
            self.assertEqual(observed, sets[t["task"]-1])

    def test_disjoint_bank_selects_negative_stage_but_keeps_negative_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, _ = fixture(root, small_config(), range(4))
            rows, stages = pair_rows(root / "sd35-stage", protocol, 64, 17, selection_views=4)
            self.assertTrue(all(r["opposite_sign"] and r["global_positive_with_negative"] for r in rows))
            for s in stages:
                self.assertEqual(s["negative_stage"], int(s["stage"] == 0))
                self.assertEqual(s["bank_sign_agrees"], 0)
                if s["negative_stage"]:
                    self.assertGreater(s["stage_margin"], 0)
                    self.assertLess(s["gate_difference"], 0)
                    self.assertLess(s["k4_delta_mean"], 0)
            summary = summarize(rows, stages, 32, 17)
            self.assertEqual(summary["n_negative_stages"], 4)
            self.assertEqual(summary["n_stage_bank_sign_disagreements"], 8)
            self.assertLess(summary["negative_stage_mean_gate_difference"]["estimate"], 0)
            legacy, legacy_stages = pair_rows(root / "sd35-stage", protocol, 64, 17)
            self.assertGreater(legacy_stages[0]["gate_difference"], 0)

    def test_stage_collector_is_forward_only_and_saves_disjoint_seed_prefixes(self):
        import torch
        class Probe:
            def margins(self, condition, seeds):
                self_outer.assertFalse(torch.is_grad_enabled())
                return torch.tensor([s % 11 / 10 for s in seeds]), torch.full((len(seeds),), condition["index_start"])
        self_outer = self
        conditions = [dict(condition=f"sd35_stage{j}", stage_index=j, index_start=j*500, index_stop=(j+1)*500)
                      for j in (0, 1)]
        schedule = dict(native_timesteps=list(range(1000)), native_sigmas=list(np.linspace(1, 0, 1000)))
        sampling = dict(stage_views=8, stage_batch_size=2, stage_progress_every=4, view_seed=17)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            record = stage_measurement(Probe(), conditions, sampling, "cp", "pair", schedule)
        seeds = [v["seed"] for b in record["stages"] for v in b["views"]]
        self.assertEqual(len(set(seeds)), 16)
        self.assertEqual(log.getvalue().count('"event": "stage_progress"'), 4)
        with contextlib.redirect_stdout(io.StringIO()):
            prefix = stage_measurement(Probe(), conditions, {**sampling, "stage_views": 4}, "cp", "pair", schedule)
        for whole, old in zip(record["stages"], prefix["stages"]):
            self.assertEqual(whole["views"][:4], old["views"])

    def test_task_finish_report_and_resume_do_not_duplicate_gpu_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = small_config()
            _, records = fixture(root, config)
            args = SimpleNamespace(output=root, resources=ROOT / "configs/resources.json", task=1, gpus="0,1")
            registry = {"MVPO_FLOW_PROBE_PY": "/fake/python", "MVPO_APPENDIX_SD35_DPO_RUN": "/fake/run"}
            spawned, reports = [], []
            def workers(commands, data, env):
                for gpu, shard, cmd in commands:
                    spawned.append(shard)
                    self.assertNotIn("--check-initialization", cmd)
                    self.assertIn("192" if config["stage_views"] == 192 else "8", cmd)
                    for i in range(shard, config["pair_count"], 4):
                        (data / "pairs" / f"{i:04d}.json").write_text(json.dumps(records[i]))
            def subprocess_run(cmd, **kwargs):
                if "--prepare-only" in cmd:
                    return
                self.assertEqual(cmd[cmd.index("--selection-views")+1], "4")
                reports.append(cmd)
                report = root / "sd35-stage/report"
                report.mkdir(exist_ok=True)
                for name in ("summary.json", "pair-margins.csv", "stage-weights.csv", "stage-margins.png", "stage-margins.pdf",
                             "stage-weights.png", "stage-weights.pdf"):
                    (report / name).write_text("synthetic dispatch test")
            with patch.object(runner, "run_probes", side_effect=workers), \
                 patch.object(runner.subprocess, "run", side_effect=subprocess_run):
                runner.run_task(args, config, registry)
                self.assertEqual(reports, [])
                self.assertEqual(runner.stage_state(root, config)["tasks"]["1"]["complete"], 2)
                args.task = 2
                runner.run_task(args, config, registry)
                self.assertEqual(len(reports), 1)
                # Two finishing jobs must not publish the same report concurrently.
                from concurrent.futures import ThreadPoolExecutor
                (root / "report-complete.json").unlink()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda _: runner.maybe_report(args, config), (0, 1)))
                self.assertEqual(len(reports), 2)
                self.assertTrue(runner.stage_state(root, config)["ready"])
                args.task = 1
                runner.run_task(args, config, registry)
                self.assertEqual(len(reports), 2)
            self.assertEqual(spawned, [0, 2, 1, 3])
            with self.assertRaises(ValueError):
                runner.freeze(args, {**config, "stage_batch_size": 4}, registry)
            (root / "sd35-stage/pairs/0000.json").unlink()
            with self.assertRaises(ValueError):
                runner.maybe_report(args, config, required=True)

    def test_launcher_forwards_task_batch_and_existing_resource_path(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            driver, capture = root / "python", root / "args.json"
            driver.write_text(f"#!{sys.executable}\nimport sys,json,os\nfrom pathlib import Path\n"
                              "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
            driver.chmod(0o755)
            env = {**os.environ, "SCD_SD15_PY": str(driver), "MVPO_STAGE_ROOT": str(root / "out"),
                   "MVPO_SCAN_RESOURCES": "/old/configs/resources.json", "MVPO_STAGE_BATCH": "2", "CAPTURE": str(capture)}
            for task in ("1", "2"):
                subprocess.run(["bash", str(ROOT / "scripts/run_stage_mechanism_slai.sh"), task], env=env, check=True)
                args = json.loads(capture.read_text())
                self.assertEqual(args[args.index("--task")+1], task)
                self.assertEqual(args[args.index("--resources")+1], "/old/configs/resources.json")
                self.assertEqual(args[args.index("--stage-batch-size")+1], "2")


if __name__ == "__main__":
    unittest.main()
