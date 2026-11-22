"""Scan isolation and per-view gradient checks; no downloaded models required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gradient_scan import configuration, frozen, shared_frozen, task_plan
from gradient_experiment import inspect
from self_correct_dpo.common import digest
from self_correct_dpo.evaluation.appendix_probe import PairProbe, batch_check, gradient_measurement, load_checkpoint
from self_correct_dpo.evaluation.appendix_scan import fixed_conditions, scan_seed, validate_job


class ScanPlanTests(unittest.TestCase):
    def test_three_tasks_cover_thirty_conditions_once(self):
        config = configuration(ROOT / "configs/appendix-gradient-scan.json")
        plans = [task_plan(config, i) for i in (1, 2, 3)]
        self.assertEqual(sum(p["formal_pair_view_observations"] for p in plans), 430080)
        cells = [(c["family"], c["step"], i) for p in plans for c in p["checkpoints"] for i in c["indices"]]
        self.assertEqual(len(cells), 30)
        self.assertEqual(len(set(cells)), 30)
        self.assertTrue(all(p["allocated_gpus"] == 2 for p in plans))

    def test_rng_scope_separates_phase_role_and_pair_but_not_checkpoint(self):
        scope = dict(phase="exploration", family="sd15", panel_seed=17)
        source = scan_seed(scope, "view", 23, "p", "sd15_index499", "source", 0, 0)
        audit = scan_seed(scope, "view", 23, "p", "sd15_index499", "audit", 0, 0)
        confirm = scan_seed({**scope, "phase": "confirmation"}, "view", 23, "p", "sd15_index499", "source", 0, 0)
        self.assertEqual(len({source, audit, confirm}), 3)
        self.assertNotIn("checkpoint", scope)

    def test_invalid_conditions_and_changed_frozen_inputs_fail(self):
        for indices in ([499, 499], [-1], [1000], [499.0]):
            with self.assertRaises(ValueError):
                fixed_conditions("sd15", indices)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frozen.json"
            frozen(path, {"batch": 2})
            frozen(path, {"batch": 2})
            with self.assertRaises(ValueError):
                frozen(path, {"batch": 4})

    def test_simultaneous_tasks_share_metadata_without_lock_failure(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scan.json"
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda _: shared_frozen(path, {"panel": 17}), range(12)))
            self.assertEqual(json.loads(path.read_text()), {"panel": 17})

    def test_batch_check_is_required_before_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pairs").mkdir()
            pairs = [dict(pair_id="p", pair_index=0)]
            prepared = dict(request={"experiment": "sd15-gradient", "sampling": {}}, pairs=pairs)
            prepared["prepared_id"] = digest(prepared)
            protocol = dict(prepared_id=prepared["prepared_id"], pairs=pairs, experiment="sd15-gradient",
                            sampling={}, conditions=fixed_conditions("sd15", [99]), scan={"view_batch_size": 2})
            protocol["protocol_id"] = digest(protocol)
            (root / "prepared.json").write_text(json.dumps(prepared))
            (root / "protocol.json").write_text(json.dumps(protocol))
            record = dict(protocol_id=protocol["protocol_id"], pair_id="p", pair_index=0, condition="sd15_index99")
            record["result_id"] = digest(record)
            (root / "pairs/0000-sd15_index99.json").write_text(json.dumps(record))
            check = dict(protocol_id=protocol["protocol_id"], passed=True)
            check["result_id"] = digest(check)
            for name in ("init-check", "repeat-view-check"):
                (root / f"{name}.json").write_text(json.dumps(check))
            self.assertFalse(inspect(root)["ready_to_collect"])
            (root / "batch-check.json").write_text(json.dumps(check))
            self.assertTrue(inspect(root)["ready_to_collect"])

    def test_new_checkpoint_still_checks_expected_identity(self):
        identity = dict(checkpoint_id="wrong", run_id="run", step=4000)
        with patch("self_correct_dpo.evaluation.appendix_probe.resolve_checkpoint", return_value=({"family": "sd15"}, None, identity)):
            with self.assertRaisesRegex(ValueError, "Wrong checkpoint"):
                load_checkpoint("sd15", "/fake/checkpoints/step-0004000", expected={**identity, "checkpoint_id": "correct"})

    def test_status_enumerates_all_frozen_indices_before_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            prepared = dict(request={"experiment": "sd15-gradient", "scan": {"conditions": fixed_conditions("sd15", [99, 499])}},
                            pairs=[{"pair_index": 0}, {"pair_index": 1}])
            prepared["prepared_id"] = digest(prepared)
            (path / "prepared.json").write_text(json.dumps(prepared))
            state = inspect(path)
            self.assertEqual(len(state["missing"]), 4)
            self.assertIn("0001-sd15_index499.json", state["missing"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch required")
class ScanGradientTests(unittest.TestCase):
    def test_batch_preserves_each_view_and_all_archived_statistics(self):
        import torch
        import numpy as np

        class Toy(PairProbe):
            def __init__(self):
                self.parameters = (torch.tensor([0.2, -0.3], requires_grad=True),)
            def margins(self, condition, seeds):
                slopes = torch.tensor([[s % 7 - 3., s % 11 - 5.] for s in seeds])
                offsets = torch.tensor([(s % 13 - 6.) / 2 for s in seeds])
                return slopes @ self.parameters[0] + offsets, torch.full((len(seeds),), condition["index_start"])

        probe = Toy()
        condition = fixed_conditions("sd15", [499])[0]
        seeds = [3, 5, 9, 11]
        single = [probe.gradient(condition, s) for s in seeds]
        batched = list(probe.gradients(condition, seeds))
        for a, b in zip(single, batched):
            self.assertEqual(a[0], b[0])
            torch.testing.assert_close(a[1], b[1])
            self.assertEqual(a[2], b[2])
        self.assertTrue(batch_check(probe, [condition], 4, 17)["passed"])
        sampling = dict(view_seed=23, ks=[1, 2, 4, 6, 8], source_blocks=2, source_views_per_block=24,
                        audit_blocks=2, audit_views_per_block=8)
        scan = dict(rng_scope={"phase": "exploration", "family": "sd15"}, view_batch_size=1)
        reference = gradient_measurement(probe, condition, sampling, "sd15-gradient", "checkpoint-A", "pair", None, scan)
        for size in (2, 4):
            measured = gradient_measurement(probe, condition, sampling, "sd15-gradient", "checkpoint-B", "pair", None,
                                            {**scan, "view_batch_size": size})
            self.assertEqual(reference["source_views"], measured["source_views"])
            self.assertEqual(reference["audit_views"], measured["audit_views"])
            np.testing.assert_allclose(reference["statistics"]["block_mean_gram"], measured["statistics"]["block_mean_gram"], rtol=1e-6)
            self.assertEqual(reference["statistics"]["source_blocks"], measured["statistics"]["source_blocks"])

    def test_flow_native_time_mapping_is_shifted_and_descending(self):
        import torch
        from types import SimpleNamespace
        from self_correct_dpo.evaluation.appendix_probe import schedule_identity
        linear = 1 - torch.arange(1000, dtype=torch.float64) / 1000
        sigmas = 3 * linear / (1 + 2 * linear)
        backend = SimpleNamespace(family="sd35_medium", dtype=torch.bfloat16,
                    scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000, shift=3),
                                              sigmas=sigmas, timesteps=1000 * sigmas))
        schedule = schedule_identity(backend)
        self.assertAlmostEqual(schedule["native_sigmas"][500], 0.75)
        self.assertAlmostEqual(schedule["native_timesteps"][500], 750)
        self.assertGreater(schedule["native_sigmas"][99], schedule["native_sigmas"][899])
        backend.scheduler.config.shift = 1
        with self.assertRaisesRegex(ValueError, "schedule"):
            schedule_identity(backend)

    def test_scan_report_accepts_new_indices_and_rejects_missing_records(self):
        import torch
        from appendix_gradient_report import read_pair_records
        class Toy:
            def gradient(self, condition, seed):
                return float(seed % 3), torch.tensor([float(seed % 5), 1.]), condition["index_start"]
        condition = fixed_conditions("sd35_medium", [99])[0]
        scan = dict(schema_version=1, family="sd35_medium", research_phase="exploration",
                    conditions=[condition], checkpoint={"step": 800}, rng_scope={"family": "sd35_medium"}, view_batch_size=1)
        scan["job_id"] = digest(scan)
        validate_job(scan, "sd35_medium")
        sampling = dict(view_seed=23, ks=[1, 2, 4], source_blocks=2, source_views_per_block=4,
                        audit_blocks=2, audit_views_per_block=2)
        measurement = gradient_measurement(Toy(), condition, sampling, "sd35-gradient", "cp", "p", None, scan)
        protocol = dict(sampling=sampling, conditions=[condition], pairs=[dict(pair_index=0, pair_id="p")], scan=scan,
                        checkpoint=scan["checkpoint"], model={"config": {"family": "sd35_medium"}, "lora_parameter_count": 2})
        protocol["protocol_id"] = digest(protocol)
        record = dict(protocol_id=protocol["protocol_id"], pair_index=0, pair_id="p", condition=condition["condition"], **measurement)
        record["result_id"] = digest(record)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pairs").mkdir()
            path = root / "pairs/0000-sd35_medium_index99.json"
            path.write_text(json.dumps(record))
            self.assertEqual(len(read_pair_records(root, protocol)), 1)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                read_pair_records(root, protocol)


if __name__ == "__main__":
    unittest.main()
