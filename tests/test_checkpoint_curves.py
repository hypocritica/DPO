"""Real tiny-image lifecycle for fixed-panel incremental checkpoint curves."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.evaluation.curves import summarize_curves, validate_curve_plan
from self_correct_dpo.evaluation.metrics import score_panel
from self_correct_dpo.evaluation.suite import generate_task, make_plan, parse_args
import test_evaluation_suite as fixtures


class _CurveFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EvaluationSuiteTest(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.model['family'] = 'sd35_medium'
        f.write(f.run / 'training.json', {'run_id': 'trial', 'config': {'model': f.model}})
        for step in (2000, 3000):
            adapter = f.run / 'checkpoints' / f'step-{step:07d}' / 'adapter'
            adapter.mkdir(parents=True)
            (adapter / 'adapter_model.bin').write_bytes(b'test-only-adapter')
            checkpoint = {'run_id': 'trial', 'step': step,
                          'files': {'adapter/adapter_model.bin': sha256(adapter / 'adapter_model.bin')}}
            checkpoint['checkpoint_id'] = digest(checkpoint)
            f.write(adapter.parent / 'checkpoint.json', checkpoint)
        rows = [{'scene_id': f'p{i}', 'variant_id': 'canonical', 'prompt': f'test image {i}', 'tag': 'pickv2',
                 'metadata': {'prompt': f'test image {i}', 'tag': 'pickv2'}, 'suite_mode': 'canonical-only-v1',
                 'template_family': 'canonical', 'transform_type': 'canonical'} for i in range(128)]
        f.prompts.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        f.write(f.prompts.parent / 'selection.json', {'mode': 'development-curve', 'panels': {
            'pickv2': {'prompts': 128, 'sha256': sha256(f.prompts)}}})
        config = read_json(f.config)
        config['protocols'] = {'sd35_medium': dict(f.sampling, seeds=[17])}
        config['metric_groups']['modern'] = ['pickscore']
        f.write(f.config, config)
        f.plan_args.steps = [0, 1000, 2000, 3000]
        self.plan = make_plan(f.plan_args)
        for task in self.plan['tasks']:
            task.update(method='base' if task['step'] == 0 else 'dpo',
                        method_label='Base' if task['step'] == 0 else 'DPO')
            task['task_id'] = digest({k: v for k, v in task.items() if k != 'task_id'})
        self.plan['plan_id'] = digest({k: v for k, v in self.plan.items() if k != 'plan_id'})
        f.write(f.out / 'plan.json', self.plan)

    def score(self, task, version='fixed'):
        f = self.fixture
        generate_task(task, self.plan, f.out, parse_args(['generate', '--out', str(f.out)]),
                      command_runner=f.command_render)
        target = f.out / task['output']
        return score_panel(target / 'render', target / 'scores/pickscore', 'pickscore', {},
            scorer_factory=lambda *a: lambda images, prompts: [.1 + .001 * int(p.split()[-1]) for p in prompts],
            asset_id={'metric': 'pickscore', 'scale': 'test-only cosine fixture', 'version': version})


class CurvePlanTests(_CurveFixture):
    def test_curve_preflight_rejects_wrong_seed_and_other_metrics_without_plotting(self):
        validate_curve_plan(self.plan, self.fixture.out)
        bad = copy.deepcopy(self.plan)
        bad['tasks'][0]['sampling']['seeds'] = [29]
        with self.assertRaisesRegex(ValueError, 'seed 17'):
            validate_curve_plan(bad, self.fixture.out)
        bad = copy.deepcopy(self.plan)
        bad['metric_groups']['modern'] = ['pickscore', 'hpsv3']
        with self.assertRaisesRegex(ValueError, 'only PickScore'):
            validate_curve_plan(bad, self.fixture.out)


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional matplotlib is not installed')
class CurveOutputTests(_CurveFixture):
    def test_partial_to_complete_preserves_missing_gap_base_line_and_hashes(self):
        for task in self.plan['tasks']:
            if task['step'] != 2000:
                self.score(task)
        base = self.fixture.out / self.plan['tasks'][0]['output'] / 'render/sessions'
        base.mkdir()
        (base / 'first.json').write_text(json.dumps({'elapsed_seconds': 9.5}))
        import matplotlib.axes
        original = matplotlib.axes.Axes.plot
        plotted, baselines = [], []
        original_baseline = matplotlib.axes.Axes.axhline
        def record_baseline(axis, y, *args, **kwargs):
            baselines.append(y)
            return original_baseline(axis, y, *args, **kwargs)
        def record(axis, xs, ys, *args, **kwargs):
            if len(xs) == 3:
                plotted.append(list(ys))
            return original(axis, xs, ys, *args, **kwargs)
        with patch.object(matplotlib.axes.Axes, 'plot', record), \
                patch.object(matplotlib.axes.Axes, 'axhline', record_baseline):
            partial = summarize_curves(self.fixture.out)
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['status_counts'], {'complete': 3, 'missing_render': 1})
        self.assertTrue(any(math.isnan(values[1]) for values in plotted))
        self.assertEqual(len(baselines), 1)
        out = Path(partial['output'])
        with (out / 'points.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r['step'] for r in rows], ['0', '1000', '2000', '3000'])
        self.assertEqual(rows[2]['mean'], '')
        self.assertEqual(rows[0]['method'], 'base')
        self.assertEqual(float(rows[0]['generation_session_seconds']), 9.5)
        self.assertTrue((out / 'pickscore.png').read_bytes().startswith(b'\x89PNG'))
        self.assertTrue((out / 'pickscore.pdf').read_bytes().startswith(b'%PDF'))
        before = partial['summary_id']
        self.score(next(t for t in self.plan['tasks'] if t['step'] == 2000))
        complete = summarize_curves(self.fixture.out)
        self.assertTrue(complete['complete'])
        self.assertNotEqual(complete['summary_id'], before)
        self.assertEqual(complete['scored_images'], 512)
        for name, expected in read_json(out / 'status.json')['files_sha256'].items():
            self.assertEqual(sha256(out / name), expected)
        self.assertNotIn('confidence interval', read_json(out / 'points.json')['selection'])

    def test_corrupt_scores_and_pixels_remain_invalid_without_zero(self):
        for task in self.plan['tasks']:
            self.score(task)
        target = self.fixture.out / self.plan['tasks'][1]['output']
        (target / 'scores/pickscore/summary.json').write_text('{}')
        image_target = self.fixture.out / self.plan['tasks'][2]['output']
        next((image_target / 'render/images').glob('*.png')).write_bytes(b'corrupted image')
        result = summarize_curves(self.fixture.out)
        self.assertFalse(result['complete'])
        self.assertEqual(result['status_counts']['invalid'], 2)
        rows = read_json(Path(result['output']) / 'points.json')['rows']
        self.assertTrue(all(r['mean'] is None for r in rows if r['status'] == 'invalid'))

    def test_incompatible_score_protocols_cannot_be_joined(self):
        self.score(self.plan['tasks'][0])
        self.score(self.plan['tasks'][1], version='different-scorer')
        with self.assertRaisesRegex(ValueError, 'protocols'):
            summarize_curves(self.fixture.out)
        self.assertFalse((self.fixture.out / 'curves/status.json').exists())

    def test_missing_matplotlib_is_an_explicit_error(self):
        with patch.dict(sys.modules, {'matplotlib': None}):
            with self.assertRaisesRegex(RuntimeError, 'install matplotlib'):
                summarize_curves(self.fixture.out)


if __name__ == '__main__':
    unittest.main()
