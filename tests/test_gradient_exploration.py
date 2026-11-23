"""Opt-in unchecked scans still collect complete, traceable, labelled results."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from self_correct_dpo.common import digest
from self_correct_dpo.evaluation.appendix_probe import gradient_measurement
from self_correct_dpo.evaluation.appendix_scan import fixed_conditions, validate_job
from gradient_experiment import deep_validate, inspect
import gradient_scan
import appendix_gradient_report


def fixture(root):
    class Toy:
        def gradient(self, condition, seed):
            return float(seed % 3), torch.tensor([float(seed % 5), 1.]), condition["index_start"]
    condition = fixed_conditions("sd35_medium", [99])[0]
    scan = dict(schema_version=1, family="sd35_medium", research_phase="exploration",
                conditions=[condition], checkpoint={"step": 800}, rng_scope={"family": "sd35_medium"},
                view_batch_size=1, skip_numerical_checks=True)
    scan["job_id"] = digest(scan)
    sampling = dict(view_seed=23, ks=[1, 2, 4], source_blocks=2, source_views_per_block=4,
                    audit_blocks=2, audit_views_per_block=2)
    pairs = [dict(pair_index=0, pair_id="p")]
    prepared = dict(request=dict(experiment="sd35-gradient", sampling=sampling, scan=scan), pairs=pairs)
    prepared["prepared_id"] = digest(prepared)
    protocol = dict(sampling=sampling, conditions=[condition], pairs=pairs, scan=scan,
                    experiment="sd35-gradient", prepared_id=prepared["prepared_id"], numerical_validation="skipped",
                    checkpoint=scan["checkpoint"], model={"config": {"family": "sd35_medium"}, "lora_parameter_count": 2})
    protocol["protocol_id"] = digest(protocol)
    measurement = gradient_measurement(Toy(), condition, sampling, "sd35-gradient", "cp", "p", None, scan)
    record = dict(protocol_id=protocol["protocol_id"], pair_index=0, pair_id="p", condition=condition["condition"], **measurement)
    record["result_id"] = digest(record)
    (root / "pairs").mkdir(parents=True)
    for path, value in (("prepared.json", prepared), ("protocol.json", protocol),
                        ("pairs/0000-sd35_medium_index99.json", record)):
        (root / path).write_text(json.dumps(value))
    return protocol


class ExplorationTests(unittest.TestCase):
    def test_both_shell_launchers_forward_exploration_switch(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            interpreter = root / "driver"
            interpreter.write_text(f"#!{sys.executable}\nimport sys,os,json\nfrom pathlib import Path\n"
                                   "if sys.argv[1]=='-c': print('/unused')\n"
                                   "else: Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
            interpreter.chmod(0o755)
            for script, task in (("run_gradient_scan_slai.sh", "1"), ("run_objective_scan_slai.sh", "4")):
                for enabled in ("0", "1"):
                    capture = root / "arguments.json"
                    env = {**os.environ, "SCD_SD15_PY": str(interpreter), "CAPTURE": str(capture),
                           "MVPO_SKIP_NUMERICAL_CHECKS": enabled}
                    subprocess.run(["bash", str(ROOT / "scripts" / script), task], env=env, check=True)
                    argv = json.loads(capture.read_text())
                    self.assertEqual("--skip-numerical-checks" in argv, enabled == "1")

    def test_unchecked_collect_and_report_do_not_require_passed_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            self.assertTrue(inspect(root)["ready_to_collect"])
            self.assertEqual(inspect(root)["checks"], {"numerical-validation": "skipped"})
            protocol, records = deep_validate(root)
            self.assertEqual(len(records), 1)
            with patch.object(sys, "argv", ["report", "--input", temp, "--repeats", "8"]), \
                 patch.object(appendix_gradient_report, "plot_gradient_statistics") as plot, \
                 patch.object(appendix_gradient_report, "plot_pair_diagnostics"):
                appendix_gradient_report.main()
            self.assertEqual(plot.call_args.args[0]["numerical_validation"], "skipped")
            summary = json.loads((root / "report/summary.json").read_text())
            self.assertEqual(summary["numerical_validation"], "skipped")
            for name in ("aggregate", "pair-metrics", "paired-changes", "audit-reliability"):
                import csv
                with (root / f"report/{name}.csv").open() as f:
                    rows = list(csv.DictReader(f))
                self.assertTrue(rows)
                self.assertTrue(all(r["numerical_validation"] == "skipped" for r in rows))
            (root / "pairs/0000-sd35_medium_index99.json").unlink()
            self.assertFalse(inspect(root)["ready_to_collect"])
            with self.assertRaises(ValueError):
                deep_validate(root)

    def test_unchecked_flag_cannot_be_applied_to_confirmation_or_unfrozen_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = fixture(root)
            job = {**protocol["scan"], "research_phase": "confirmation", "excluded_pair_ids": ["old"]}
            job["job_id"] = digest({k: v for k, v in job.items() if k != "job_id"})
            with self.assertRaisesRegex(ValueError, "exploration only"):
                validate_job(job, "sd35_medium")
            del protocol["numerical_validation"]
            protocol["protocol_id"] = digest({k: v for k, v in protocol.items() if k != "protocol_id"})
            (root / "protocol.json").write_text(json.dumps(protocol))
            self.assertFalse(inspect(root).get("ready_to_collect", False))

    def test_shared_runner_skips_pilot_and_initial_checks_for_all_objectives(self):
        config = gradient_scan.configuration(ROOT / "configs/appendix-gradient-scan.json", 2, 2)
        config["skip_numerical_checks"] = True
        for family, method, task in (("sd15", None, 1), ("sd35_medium", None, 1),
                                      ("sd15", "dspo", 4), ("sd15", "dmpo", 5)):
            with tempfile.TemporaryDirectory() as temp:
                args = SimpleNamespace(output=Path(temp), resources=Path("/resources"), task=task,
                                       gpus="0,1", scratch_dir=Path(temp), repeats=8, pilot_only=False)
                job = dict(family=family, checkpoint={"step": 1000}, skip_numerical_checks=True)
                if method:
                    job["method"] = method
                registry = {key: "/python" for _, key in gradient_scan.FAMILIES.values()}
                with patch.object(gradient_scan.subprocess, "run") as run, \
                     patch.object(gradient_scan, "checked_json", return_value={}), \
                     patch.object(gradient_scan, "check_panel"), \
                     patch.object(gradient_scan, "inspect", return_value={}), \
                     patch.object(gradient_scan, "run_probes") as probes, \
                     patch.object(gradient_scan, "assert_checks", return_value=({"source": {"probe": {}}}, [])) as checks:
                    gradient_scan.run_checkpoint(args, config, registry, job, {})
                self.assertEqual(checks.call_count, 1)
                self.assertEqual(checks.call_args.args[0].name, "formal")
                commands = probes.call_args.args[0]
                self.assertEqual(len(commands), 2)
                self.assertTrue(all("--check-initialization" not in c[2] for c in commands))
                self.assertEqual(run.call_count, 2)  # prepare followed by the actual report
                self.assertTrue(any("appendix_gradient_report.py" in arg for arg in run.call_args.args[0]))
                self.assertFalse(list(Path(temp).rglob("pilot")))


if __name__ == "__main__":
    unittest.main()
