"""Diagnostic reports remain bounded, non-destructive and restore loader methods."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_flow_startup import Trace, instrument_loaders, inventory, probe_reads


class StartupProfileTests(unittest.TestCase):
    def test_metadata_cli_and_bounded_read_preserve_model_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            (model / 'transformer').mkdir(parents=True)
            (model / 'model_index.json').write_text(json.dumps({'_class_name': 'ZImagePipeline'}))
            header = json.dumps({'w': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
            weights = model / 'transformer/model.safetensors'
            payload = struct.pack('<Q', len(header)) + header + b'\0\0'
            weights.write_bytes(payload)
            config = root / 'config.json'
            config.write_text(json.dumps({'model': {'family': 'zimage', 'path': str(model)}}))
            command = [sys.executable, str(ROOT / 'scripts/profile_flow_startup.py'),
                       '--config', str(config), '--out', str(root / 'report')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = root / 'report/rank-0.jsonl'
            before = report.read_bytes()
            rows = [json.loads(line) for line in before.splitlines()]
            self.assertIn('no_torch_import_or_gpu_load', rows[-1]['scope'])
            self.assertEqual(rows[-2]['weights'][0]['tensor_dtypes'], {'BF16': 1})
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            self.assertEqual(before, report.read_bytes())
            output = io.StringIO()
            with redirect_stdout(io.StringIO()):
                probe_reads(model, inventory(model), 1, Trace(output))
            self.assertEqual(json.loads(output.getvalue().splitlines()[-1])['bytes'], len(payload))
            self.assertEqual(weights.read_bytes(), payload)

    def test_loader_instrumentation_restores_descriptors_even_when_load_fails(self):
        class Component:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                if path == 'broken':
                    raise RuntimeError('fixture failure')
                return (cls, path, kwargs)
        class Pipeline:
            def to(self, device):
                return self
        module = SimpleNamespace(Component=Component, Pipeline=Pipeline)
        descriptor = Component.__dict__['from_pretrained']
        move = Pipeline.to
        output = io.StringIO()
        index = {'_class_name': 'Pipeline', 'transformer': ['diffusers', 'Component']}
        with patch('profile_flow_startup.importlib.import_module', return_value=module), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                with instrument_loaders(index, Trace(output)):
                    self.assertEqual(Component.from_pretrained('model', torch_dtype='float32'),
                                     (Component, 'model', {'torch_dtype': 'float32'}))
                    pipeline = Pipeline()
                    self.assertIs(pipeline.to('cpu'), pipeline)
                    Component.from_pretrained('broken')
        self.assertIs(Component.__dict__['from_pretrained'], descriptor)
        self.assertIs(Pipeline.to, move)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(rows[-1]['event'], 'error')
        self.assertEqual(rows[-1]['stage'], 'load.transformer')
        self.assertIn('pipeline.to_device', [row.get('stage') for row in rows])


if __name__ == '__main__':
    unittest.main()
