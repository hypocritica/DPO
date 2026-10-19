"""Check relocation, archived config validity and copied-source CLI execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.common import sha256, write_jsonl
from self_correct_dpo.config import load_config
from self_correct_dpo.data import load_pairs
from self_correct_dpo.train import training_lock
from self_correct_dpo.training import validate_config


def make_pairs(root):
    from PIL import Image
    rows = []
    for i, split in enumerate(("train", "validation")):
        names = []
        hashes = []
        for j in range(2):
            name = f"{i}-{j}.png"
            Image.new("RGB", (4, 4), (i * 80, j * 80, 10)).save(root / name)
            names.append(name)
            hashes.append(sha256(root / name))
        caption = f"test scene {i}"
        rows.append({"pair_id":str(i), "scene_id":str(i), "split":split,
            "winner":names[0], "loser":names[1],
            "winner_sha256":hashes[0], "loser_sha256":hashes[1],
            "prompts":[{"variant_id":"canonical", "prompt":caption,
                        "template_family":"canonical", "transform_type":"canonical"}],
            "human_preference":{"source":"pickapic", "source_revision":"test-only",
                "labels":[1,0], "winner_index":0, "image_sha256":hashes,
                "original_prompt":caption}})
    write_jsonl(root / "pairs.jsonl", rows)
    return rows


class PortabilityTests(unittest.TestCase):
    def test_archived_configs_and_invalid_combinations(self):
        configs = list((ROOT / "configs/historical").glob("*.json"))
        self.assertEqual(len(configs), 9)
        for path in configs:
            with self.subTest(config=path.name):
                validate_config(json.loads(path.read_text()), files=False)
        c = json.loads((ROOT / "configs/model-K2-C.json").read_text())
        c["training"]["formal_arm"] = "D"
        with self.assertRaisesRegex(ValueError, "only B/C"):
            validate_config(c, files=False)
        c["training"]["formal_arm"] = "C"
        c["training"]["world_size"] = 2
        validate_config(c,files=False)

    def test_config_paths_and_missing_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "config.json"
            p.write_text(json.dumps({"model":{"path":"${SCD_TEST_PATH}"}, "data":{"pairs":"pairs.jsonl"}}))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "environment"):
                    load_config(p)
            with patch.dict(os.environ, {"SCD_TEST_PATH":str(root / "model")}):
                c = load_config(p)
            self.assertEqual(c["model"]["path"], str(root / "model"))
            self.assertEqual(c["data"]["pairs"], str(root / "pairs.jsonl"))

    def test_export_preserves_bytes_labels_splits_and_relocates(self):
        spec = importlib.util.spec_from_file_location("export_data", ROOT / "scripts/export_data.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.mkdir()
            original = make_pairs(source)
            exported = root / "exported"
            module.export_data(source / "pairs.jsonl", exported)
            with self.assertRaises(FileExistsError):
                module.export_data(source / "pairs.jsonl", exported)
            moved = root / "moved"
            exported.rename(moved)
            shutil.rmtree(source)
            rows = load_pairs(moved / "pairs.jsonl")
            for a, b in zip(original, rows):
                self.assertEqual(a["split"], b["split"])
                self.assertEqual(a["human_preference"], b["human_preference"])
                self.assertEqual(a["winner_sha256"], b["winner_sha256"])
            Path(rows[0]["winner"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_pairs(moved / "pairs.jsonl")

    def test_copy_only_this_directory_then_run_cli_without_parent_imports(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkout = root / "relocated"
            shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist", "runs", "logs", "models", "data", ".venv*"))
            data = root / "data"
            data.mkdir()
            make_pairs(data)
            model = root / "base"
            (model / "unet").mkdir(parents=True)
            (model / "model_index.json").write_text('{"_class_name":"StableDiffusionPipeline"}')
            (model / "unet/config.json").write_text('{"cross_attention_dim":768,"in_channels":4}')
            config = json.loads((checkout / "configs/model-K2-C.json").read_text())
            config["evaluation"]={"enabled":False}
            config["model"]["path"] = str(model)
            config["data"]["pairs"] = str(data / "pairs.jsonl")
            p = root / "config.json"
            p.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, "-I", str(checkout / "train.py"),
                "--config", str(p), "--out", str(root / "out"), "--check-only"],
                cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"training_pairs": 1', result.stdout)
            self.assertFalse((root / "out").exists())

    def test_single_writer_lock(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            with training_lock(out):
                with self.assertRaisesRegex(RuntimeError, "Another training process"):
                    with training_lock(out):
                        pass
            with training_lock(out):
                pass


if __name__ == "__main__":
    unittest.main()
