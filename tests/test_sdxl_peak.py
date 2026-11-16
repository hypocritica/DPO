import copy
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_sdxl_peak as peak
from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.evaluation import suite
import test_evaluation_suite as fixtures


class PeakSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.template = self.root / 'template.json'
        self.catalog = read_json(ROOT / 'configs/evaluation-sdxl-parti-supplement.json')
        self.catalog['runs_roots']['sdxl'] = str(self.root / 'runs')
        model = self.catalog['models']['sdxl']
        for e in self.catalog['experiments']:
            run = self.root / 'runs' / e['run']
            run.mkdir(parents=True)
            stage = e['method'].startswith('stagewise')
            training = {'objective': 'dspo' if 'dspo' in e['method'] else 'dmpo', 'noise_views': 6 if stage else 1, 'formal_arm': 'C'}
            if stage:
                training['stagewise'] = {'objective': 'dpo'}
            self.write(run / 'training.json', {'run_id': e['expected_run_id'], 'config': {'model': model,
                'method': 'stagewise-C-gaussian' if stage else 'dpo', 'training': training}})
            info = self.checkpoint(run, 3000)
            e['expected_checkpoint_ids'] = {'3000': info['checkpoint_id']}
            if e['method'] == 'dmpo':
                self.checkpoint(run, 10000)
            if e['method'] == 'stagewise_dmpo_k6':
                self.run = run
                self.checkpoint(run, 9000)
                self.checkpoint(run, 9525, rolling=True)
        self.write(self.template, self.catalog)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def checkpoint(self, run, step, rolling=False):
        folder = run / 'checkpoints' / (f'resume-{step:07d}-fixture' if rolling else f'step-{step:07d}')
        (folder / 'adapter').mkdir(parents=True)
        weight = folder / 'adapter/weights.bin'
        weight.write_bytes(b'weights')
        info = {'run_id': read_json(run / 'training.json')['run_id'], 'step': step, 'full_state': rolling,
                'files': {'adapter/weights.bin': sha256(weight)}}
        if rolling:
            state = folder / 'state.pt'
            state.write_bytes(b'optimizer')
            info['files']['state.pt'] = sha256(state)
        info['checkpoint_id'] = digest(info)
        self.write(folder / 'checkpoint.json', info)
        if rolling:
            self.write(run / 'checkpoints/latest.json', {'directory': folder.name, 'checkpoint_id': info['checkpoint_id']})
        return info

    def test_latest_permanent9000_frozen_and_later10000_does_not_change_selection(self):
        path = peak.prepare(self.root / 'out', self.template)
        catalog = read_json(path)
        steps = {e['method']: e['steps'] for e in catalog['experiments']}
        self.assertEqual(steps, {'dmpo': [10000], 'stagewise_dmpo_k6': [9000], 'dspo': [3000], 'stagewise_dspo_k6': [3000]})
        self.assertFalse((self.root / 'out/snapshots').exists())
        self.checkpoint(self.run, 10000)
        shutil.rmtree(self.run / 'checkpoints/resume-0009525-fixture')
        peak.prepare(self.root / 'out', self.template)
        self.assertEqual(read_json(path), catalog)

    def test_10000_is_preferred_when_present_before_initialization(self):
        self.checkpoint(self.run, 10000)
        catalog = read_json(peak.prepare(self.root / 'out', self.template))
        row = next(e for e in catalog['experiments'] if e['method'] == 'stagewise_dmpo_k6')
        self.assertEqual(row['steps'], [10000])
        self.assertEqual(row['root'], 'sdxl')

    def test_corrupt_latest_permanent_does_not_fall_back(self):
        (self.run / 'checkpoints/step-0009000/adapter/weights.bin').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            peak.prepare(self.root / 'out', self.template)
        self.assertFalse((self.root / 'out/catalog.json').exists())

    def test_latest_pointer_is_ignored(self):
        pointer = self.run / 'checkpoints/latest.json'
        self.write(pointer, {'directory': '../other', 'checkpoint_id': 'bad'})
        catalog = read_json(peak.prepare(self.root / 'out', self.template))
        row = next(e for e in catalog['experiments'] if e['method'] == 'stagewise_dmpo_k6')
        self.assertEqual(row['steps'], [9000])


class PeakReuseTest(unittest.TestCase):
    def test_completed_render_reused_but_changed_sampling_not_imported(self):
        f = fixtures.EvaluationSuiteTest(methodName='runTest')
        f.setUp()
        self.addCleanup(f.doCleanups)
        plan, task = f.generated()
        from self_correct_dpo.evaluation.metrics import score_panel
        score_panel(f.out / task['output'] / 'render', f.out / task['output'] / 'scores/pickscore', 'pickscore', {},
            scorer_factory=lambda *a: lambda images, prompts: [.25] * len(images),
            asset_id={'metric': 'pickscore', 'scale': 'cosine', 'version': 'same'})
        panel = task['benchmark']
        previous = f.out.parent / 'previous'
        (previous / 'supplement').mkdir(parents=True)
        shutil.copytree(f.out, previous / 'supplement' / panel)
        out = f.out.parent / 'new'
        shutil.copytree(f.out, out)
        shutil.rmtree(out / task['output'] / 'render')
        shutil.rmtree(out / task['output'] / 'scores')
        peak.reuse(out, previous)
        suite.validate_task_render(task, plan, out)
        self.assertEqual(sha256(out / task['output'] / 'render/manifest.jsonl'), sha256(f.out / task['output'] / 'render/manifest.jsonl'))
        self.assertEqual(sha256(out / task['output'] / 'scores/pickscore/complete.json'), sha256(f.out / task['output'] / 'scores/pickscore/complete.json'))
        shutil.rmtree(out / task['output'] / 'render')
        changed = copy.deepcopy(plan)
        changed['tasks'][0]['sampling']['guidance_scale'] += 1
        changed.pop('plan_id')
        changed['plan_id'] = digest(changed)
        (out / 'plan.json').write_text(json.dumps(changed))
        peak.reuse(out, previous)
        self.assertFalse((out / task['output'] / 'render').exists())


class PeakLauncherTest(unittest.TestCase):
    def test_two_groups_order_gpu_routing_and_generation_failure(self):
        for group, fail in [('A', False), ('B', False), ('B', True)]:
            with self.subTest(group=group, fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fake = root / 'python'
                fake.write_text('''#!/usr/bin/env python3
import os,json,sys
with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ['FAIL'] == '1' and sys.argv[1:3] == ['evaluate_suite.py','generate']: sys.exit(7)
''')
                fake.chmod(0o700)
                env = {k:v for k,v in os.environ.items() if not k.startswith('SCD_') and k != 'CUDA_VISIBLE_DEVICES'}
                env.update(SCD_EVAL_PY=str(fake), SCD_SD15_PY=str(fake), SCD_SDXL_PEAK_ROOT=str(root/'output'),
                           CALLS=str(root/'calls'), FAIL=str(int(fail)))
                result = subprocess.run(['bash', str(ROOT/'scripts/run_sdxl_peak.sh'), group, 'all'], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 7 if fail else 0, result.stderr)
                calls = [json.loads(s) for s in (root/'calls').read_text().splitlines()]
                actions = [r[1] for r in calls if r[0] == 'evaluate_suite.py']
                if fail:
                    self.assertNotIn('score', actions)
                else:
                    count = 1 if group == 'A' else 2
                    self.assertEqual(actions, ['generate', 'index']*count + ['score']*count + ['summarize']*count)
                for row in calls:
                    if '--gpus' in row:
                        self.assertEqual(row[row.index('--gpus')+1], '0,1,2,3')


if __name__ == '__main__':
    unittest.main()
