"""Real CPU worker dispatch checks for resumable checkpoint-level GPU assignment."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from self_correct_dpo.evaluation import suite


class CurveWorkerTests(unittest.TestCase):
    def test_two_then_four_workers_cover_each_checkpoint_once_with_allocated_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            (root / 'evaluate_suite.py').write_text('''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('command')
p.add_argument('--out')
p.add_argument('--shard-index',type=int)
p.add_argument('--shard-count',type=int)
a=p.parse_args()
Path(a.out).mkdir(exist_ok=True)
Path(a.out,f'{a.shard_index}.json').write_text(json.dumps({
 'device':os.environ['CUDA_VISIBLE_DEVICES'],
 'tasks':list(range(17))[a.shard_index::a.shard_count]}))
''')
            for devices in (['3', '5'], ['3', '5', '7', '9']):
                out = root / f'workers-{len(devices)}'
                args = argparse.Namespace(gpus=','.join(devices))
                with patch.object(suite, 'CONFIG_DIR', root / 'configs'), patch.dict(
                        os.environ, {'CUDA_VISIBLE_DEVICES': '3,5,7,9'}):
                    code = suite.launch_workers(args, ['generate', '--out', str(out), '--gpus', args.gpus])
                self.assertEqual(code, 0)
                results = [json.loads((out / f'{i}.json').read_text()) for i in range(len(devices))]
                self.assertEqual([r['device'] for r in results], devices)
                self.assertEqual(sorted(t for r in results for t in r['tasks']), list(range(17)))

    def test_actual_prepare_plan_cli_freezes_all_steps_and_replays(self):
        from test_curve_prepare import CurvePrepareTest
        fixture = CurvePrepareTest(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        script = Path(__file__).resolve().parents[1] / 'evaluate_curves.py'
        command = [sys.executable, '-S', str(script), 'prepare', '--runs-root', str(fixture.runs),
                   '--config', str(fixture.config), '--out', fixture.args.prepare_dir]
        subprocess.run(command, check=True, capture_output=True, text=True)
        out = fixture.root / 'evaluation'
        plan_command = [sys.executable, '-S', str(script), 'plan', '--prepared', fixture.args.prepare_dir,
                        '--out', str(out)]
        subprocess.run(plan_command + ['--dry-run'], check=True, capture_output=True, text=True)
        self.assertFalse(out.exists())
        subprocess.run(plan_command, check=True, capture_output=True, text=True)
        original = (out / 'plan.json').read_bytes()
        subprocess.run(plan_command, check=True, capture_output=True, text=True)
        self.assertEqual(original, (out / 'plan.json').read_bytes())
        plan = json.loads(original)
        self.assertEqual(len(plan['tasks']), 15)
        self.assertEqual(sum(t['images'] for t in plan['tasks']), 1920)
        self.assertEqual(sum(t['method'] == 'base' for t in plan['tasks']), 1)
        self.assertEqual(plan['panels']['pickv2']['mode'], 'development-curve')

    def test_devices_outside_allocation_rejected_before_worker_start(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,5'}):
            with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
                suite.launch_workers(argparse.Namespace(gpus='0,1'), ['generate', '--gpus', '0,1'])


if __name__ == '__main__':
    unittest.main()
