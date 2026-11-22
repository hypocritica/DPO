"""Small CPU checks for the supervised gradient launcher; no model or GPU required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gradient_experiment import inspect, worker_shards
from self_correct_dpo.common import digest


class GradientWorkflowTests(unittest.TestCase):
    def test_fixed_groups_cover_every_pair_once_with_different_gpu_counts(self):
        for pairs, groups in ((4, 3), (16, 4), (128, 3), (128, 4), (3, 3)):
            all_pairs = []
            for group in range(groups):
                devices = [str(i) for i in range(group % 4 + 1)]
                for gpu, index, stride, assigned in worker_shards(pairs, (group, groups), devices):
                    self.assertLessEqual(stride, pairs)
                    self.assertTrue(all(pair % groups == group for pair in assigned))
                    self.assertEqual(assigned, list(range(index, pairs, stride)))
                    all_pairs.extend(assigned)
            self.assertEqual(sorted(all_pairs), list(range(pairs)))

    def test_plan_is_pure_standard_library_and_counts_both_flow_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "uncreated"
            for experiment, count in (("sd15", 229376), ("flow", 458752)):
                result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/gradient_experiment.py"),
                                         "plan", experiment, "--output", str(output), "--gpus", "0,1,2,3", "--group", "1/4"],
                                        check=True, capture_output=True, text=True)
                plan, _ = json.JSONDecoder().raw_decode(result.stdout)
                self.assertEqual(plan["measurement_pair_view_backwards"], count)
                self.assertEqual(plan["group_pair_view_backwards"], count // 4)
                self.assertIn("/formal", plan["output"])
                self.assertFalse(output.exists())
            result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/gradient_experiment.py"),
                                     "plan", "sd15", "--phase", "pilot", "--output", str(output)],
                                    check=True, capture_output=True, text=True)
            plan, _ = json.JSONDecoder().raw_decode(result.stdout)
            self.assertEqual(plan["measurement_pair_view_backwards"], 256)
            self.assertIn("ENGINEERING CHECK ONLY", plan["purpose"])
            self.assertIn("/pilot", plan["output"])

    def test_formal_run_refuses_missing_pilot_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, "-S", str(ROOT / "scripts/gradient_experiment.py")]
            arguments = ["sd15", "--output", temp, "--python", "/missing-gradient-worker"]
            # This failed prepare creates only the frozen launcher settings.
            prepare = subprocess.run(command + ["prepare"] + arguments, capture_output=True, text=True)
            self.assertNotEqual(prepare.returncode, 0)
            formal = Path(temp) / "sd15/formal"
            (formal / "prepared.json").write_text("{}")
            run = subprocess.run(command + ["run"] + arguments + ["--gpus", "0"], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("complete records and passed initialization/repeat checks required", run.stderr)
            self.assertFalse((formal / "logs").exists())

    def test_prepared_status_lists_missing_records_and_rejects_corrupt_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            prepared = dict(request={"experiment": "sd15-gradient"}, pairs=[{"pair_index": 0}])
            prepared["prepared_id"] = digest(prepared)
            path = output / "prepared.json"
            path.write_text(json.dumps(prepared))
            self.assertEqual(inspect(output)["missing"], ["0000-sd15_t499.json"])
            prepared["pairs"].append({"pair_index": 1})
            path.write_text(json.dumps(prepared))
            self.assertTrue(inspect(output)["failed"])


if __name__ == "__main__":
    unittest.main()
