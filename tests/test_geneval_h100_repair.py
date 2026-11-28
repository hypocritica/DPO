"""Checks for isolation of repaired scores and the actual operator dispatch."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('geneval_h100_repair', Path(__file__).resolve().parents[1] / 'scripts/reuse_geneval_h100.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class RepairTests(unittest.TestCase):
    def test_patch_replaces_shared_native_class_without_moving_values_to_cpu(self):
        class Native:
            @staticmethod
            def apply(*args):
                raise AssertionError('Broken native extension must not run')
        class Tensor:
            def __init__(self, name): self.name = name
            def cpu(self): return 'CPU_SHAPES'
            def type_as(self, other): return (self.name, other)
        calls = []
        def reference(*args):
            calls.append(args)
            return 'reference-result'
        ops = SimpleNamespace(MultiScaleDeformableAttnFunction=Native,
                              multi_scale_deformable_attn_pytorch=reference)
        mmcv = ModuleType('mmcv'); mmcv.__version__ = '1.7.2'
        mmcv_ops = ModuleType('mmcv.ops'); mmcv_ops.multi_scale_deform_attn = ops
        alias = Native  # MMDet may have imported this class before the patch.
        with patch.dict(sys.modules, {'mmcv': mmcv, 'mmcv.ops': mmcv_ops}):
            counter = repair.install_backend()
            value = Tensor('cuda_value')
            result = alias.apply(value, Tensor('shapes'), object(), Tensor('locations'), Tensor('weights'), 64)
        self.assertEqual(result, 'reference-result')
        self.assertEqual(counter['count'], 1)
        self.assertIs(calls[0][0], value)
        self.assertEqual(calls[0][1], 'CPU_SHAPES')
        self.assertEqual(calls[0][2], ('locations', value))

    def test_new_scores_have_separate_path_and_backend_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            old = folder / 'task/scores/geneval/summary.json'
            old.parent.mkdir(parents=True); old.write_text('old-score')
            task = {'output': 'task', 'family': 'sdxl', 'method': 'base'}
            def score(render, output, metric, settings, **kwargs):
                self.assertEqual(render, folder / 'task/render')
                self.assertEqual(output, folder / 'task/scores' / repair.SCORE_NAME)
                self.assertEqual(settings['runtime_patch'], repair.repair_identity())
                with patch.object(repair.subprocess, 'run') as run:
                    kwargs['command_runner'](['python', '-m', 'original.worker', 'request.json'])
                self.assertEqual(run.call_args.args[0][-3:], ['worker', '--request', 'request.json'])
                return {'mean': .5}
            with patch.object(repair, 'jobs', return_value=(folder, [({}, folder, task)])), \
                 patch.object(repair, 'validate_task_render'), \
                 patch.object(repair.comp, 'load_settings', return_value={}), \
                 patch.object(repair.comp, 'score', side_effect=score):
                repair.score_one(0)
            self.assertEqual(old.read_text(), 'old-score')

    def test_gpu_assignment_stays_within_allocation(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b,GPU-c,GPU-d',
                                     'SCD_EVAL_GPUS': 'GPU-a,GPU-b,GPU-c,GPU-d'}):
            self.assertEqual(repair.gpu_tokens(), ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'])
            os.environ['SCD_EVAL_GPUS'] = '0,1,2,3'
            with self.assertRaises(ValueError): repair.gpu_tokens()
            os.environ['SCD_EVAL_GPUS'] = 'GPU-a,GPU-b,GPU-c,GPU-c'
            with self.assertRaises(ValueError): repair.gpu_tokens()

    def test_worker_rejects_foreign_code_before_model_import(self):
        with patch.object(repair, 'read_json', return_value={'benchmark': 'geneval', 'settings': {'runtime_patch': {}}}):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                repair.worker('unused.json')


if __name__ == '__main__':
    unittest.main()
