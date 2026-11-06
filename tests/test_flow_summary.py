"""Paired effect reporting on completed artifacts, without GPU generation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from self_correct_dpo.common import sha256, write_json, write_jsonl
from summarize_flow_runs import collect


class FlowSummaryTests(unittest.TestCase):
    def fixture(self, root, algorithm, steps=(0, 200), shift=0.):
        run = root/algorithm
        run.mkdir()
        objective = 'dpo' if algorithm == 'mvpo' else algorithm
        write_json(run/'training.json', {'pairs_id': 'same-data', 'config': {
            'seed': 17, 'model': {'family': 'zimage'}, 'method': algorithm,
            'training': {'objective': objective, 'noise_views': 2 if algorithm == 'mvpo' else 1,
                         'batch_size': 2, 'world_size': 2, 'gradient_accumulation_steps': 32,
                         'beta': 500, 'learning_rate': 5e-6}}})
        for step in steps:
            path = run/'evaluations'/f'step-{step:07d}'
            (path/'scores/pickscore').mkdir(parents=True)
            scores = [{'scene_id': str(i), 'variant_id': 'canonical', 'seed': 17,
                       'prompt': f'prompt {i}', 'score': .2 + i*.01 + (shift if step else 0.)}
                      for i in range(4)]
            score_file = path/'scores/pickscore/scores.jsonl'
            write_jsonl(score_file, scores)
            write_json(path/'evaluation.json', {'identity': {'base_identity': 'same-base',
                       'inputs': {'prompts': 'same-panel', 'scorer': 'same-scorer'},
                       'config': {'sampling': {'seeds': [17]}}}})
            write_json(path/'summary.json', {'step': step, 'metrics': {'pickscore': {
                       'mean': sum(r['score'] for r in scores)/len(scores)}}})
            write_json(path/'complete.json', {'summary_sha256': sha256(path/'summary.json'),
                       'scores_sha256': {'scores/pickscore/scores.jsonl': sha256(score_file)}})
        return run

    def test_sft_budget_and_paired_comparison_with_missing_reference_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dpo = self.fixture(root, 'dpo', shift=.01)
            sft = self.fixture(root, 'sft', steps=(0, 200, 400), shift=-.02)
            mvpo = self.fixture(root, 'mvpo', shift=.03)
            rows = collect([sft, mvpo], reference_run=dpo)
            s = next(r for r in rows if r['algorithm'] == 'sft' and r['step'] == 200)
            self.assertEqual(s['winner_images_per_update'], 128)
            self.assertEqual(s['policy_image_views_per_update'], 128)
            self.assertEqual(s['pair_views_per_update'], 0)
            self.assertIsNone(s['beta'])
            self.assertAlmostEqual(s['delta_base'], -.02)
            self.assertAlmostEqual(s['delta_reference'], -.03)
            self.assertAlmostEqual(s['delta_reference_ci_high'], -.03)
            m = next(r for r in rows if r['algorithm'] == 'mvpo' and r['step'] == 200)
            self.assertAlmostEqual(m['delta_reference'], .02)
            self.assertEqual(m['policy_image_views_per_update'], 512)
            missing = next(r for r in rows if r['step'] == 400)
            self.assertIsNone(missing['delta_reference'])
            self.assertEqual(missing['reference_status'], 'not_evaluated_at_this_step')

    def test_reference_rejects_different_training_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dpo = self.fixture(root, 'dpo')
            sft = self.fixture(root, 'sft')
            info = json.loads((sft/'training.json').read_text())
            info['config']['seed'] = 42
            (sft/'training.json').write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, 'seed differ'):
                collect([sft], reference_run=dpo)


if __name__ == '__main__':
    unittest.main()
