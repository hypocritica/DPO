"""CPU checks for cross-panel selection, protocols and immutable combined publication."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.evaluation import suite
from self_correct_dpo.evaluation.selected import identity

spec = importlib.util.spec_from_file_location('ztable', Path(__file__).resolve().parents[1] / 'export_zimage_table.py')
ztable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ztable)


class ZImageTableTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selection = {'family': 'zimage', 'methods': list(ztable.METHODS), 'choices': [
            {'method': method, 'selected_step': 1000, 'training_sha256': method,
             'run_path': '/unavailable/' + method, 'checkpoints': [{'step': 1000, 'checkpoint_id': method}]}
            for method in ztable.METHODS[1:]]}
        self.selection['identity_sha256'] = identity(self.selection)
        self.panels = {}
        for panel in ('partiprompts', 'geneval'):
            root = self.root / panel
            root.mkdir()
            (root / 'prompts.jsonl').write_text('frozen prompts\n')
            tasks, rows = [], []
            for method in ztable.METHODS:
                checkpoint = {'checkpoint_id': 'base' if method == 'base' else method}
                if method != 'base':
                    checkpoint['training_sha256'] = method
                task = {'method': method, 'family': 'zimage', 'benchmark': panel, 'run': method,
                    'run_path': None if method == 'base' else '/unavailable/' + method,
                    'step': 0 if method == 'base' else 1000, 'model': {'family': 'zimage', 'path': '/unavailable/base', 'validation_report': '/audit/' + method},
                    'checkpoint': checkpoint, 'prompts': 1632 if panel == 'partiprompts' else 553,
                    'sampling': {'seeds': [17] if panel == 'partiprompts' else [17, 29, 43, 71]}}
                tasks.append(task)
                for name, metric, _ in ztable.COLUMNS:
                    if name == panel:
                        rows.append({'method': method, 'step': task['step'], 'checkpoint_id': checkpoint['checkpoint_id'],
                            'benchmark': panel, 'metric': metric, 'status': 'complete', 'mean': .25,
                            'sampling_id': panel, 'prompt_sha256': panel, 'base_model_id': 'same',
                            'panel_mode': 'full-benchmark', 'scorer_protocol_id': metric})
            plan = {'tasks': tasks, 'panels': {panel: {'snapshot': 'prompts.jsonl', 'sha256': sha256(root/'prompts.jsonl')}}}
            plan['report_selection'] = {'selection': self.selection, 'selection_id': digest(self.selection)}
            plan['plan_id'] = digest(plan)
            suite.freeze_json(root/'plan.json', plan)
            suite.freeze_json(root/'selection.json', self.selection)
            suite.freeze_json(root/'selected-checkpoints.json', {'plan_id': plan['plan_id'], 'benchmarks': [panel],
                'selection': self.selection, 'selection_sha256': sha256(root/'selection.json')})
            self.panels[panel] = rows

    def collect(self, plan, root, args):
        return copy.deepcopy(self.panels[args.benchmarks[0]]), {}

    def test_combines_three_rows_and_hashes_without_training_assets(self):
        with patch.object(ztable.paper, 'collect', side_effect=self.collect):
            destination = self.root/'report'
            result = ztable.export(self.root/'partiprompts', self.root/'geneval', destination)
            saved = read_json(destination/'results.json')
            self.assertEqual(len(saved['rows']), 3)
            self.assertEqual(saved['rows'][2]['GenEval macro accuracy [0,1]'], .25)
            self.assertEqual(saved['export_id'], result['export_id'])
            receipt = read_json(destination/'export.json')
            for name, checksum in receipt['files_sha256'].items():
                self.assertEqual(sha256(destination/name), checksum)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                ztable.export(self.root/'partiprompts', self.root/'geneval', destination)

    def test_rejects_incomplete_or_incoherent_scorer_results(self):
        with patch.object(ztable.paper, 'collect', side_effect=self.collect):
            self.panels['partiprompts'][0]['status'] = 'invalid'
            with self.assertRaisesRegex(ValueError, 'complete and valid'):
                ztable.export(self.root/'partiprompts', self.root/'geneval', self.root/'report')
            self.panels['partiprompts'][0]['status'] = 'complete'
            self.panels['partiprompts'][0]['scorer_protocol_id'] = 'different'
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                ztable.export(self.root/'partiprompts', self.root/'geneval', self.root/'report')
            self.assertFalse((self.root/'report').exists())

    def test_rejects_different_selection_provenance_and_corrupted_receipt(self):
        root = self.root/'geneval'
        receipt = read_json(root/'selected-checkpoints.json')
        receipt['selection']['new_provenance'] = 'another corpus'
        receipt['selection']['identity_sha256'] = identity(receipt['selection'])
        (root/'selection.json').unlink()
        suite.freeze_json(root/'selection.json', receipt['selection'])
        receipt['selection_sha256'] = sha256(root/'selection.json')
        plan = read_json(root/'plan.json')
        plan['report_selection'] = {'selection': receipt['selection'], 'selection_id': digest(receipt['selection'])}
        plan['plan_id'] = digest({k: v for k, v in plan.items() if k != 'plan_id'})
        (root/'plan.json').unlink()
        suite.freeze_json(root/'plan.json', plan)
        receipt['plan_id'] = plan['plan_id']
        (root/'selected-checkpoints.json').unlink()
        suite.freeze_json(root/'selected-checkpoints.json', receipt)
        with patch.object(ztable.paper, 'collect', side_effect=self.collect):
            with self.assertRaisesRegex(ValueError, 'same chosen selection'):
                ztable.export(self.root/'partiprompts', root, self.root/'report')
        receipt['selection_sha256'] = 'corrupt'
        (root/'selected-checkpoints.json').unlink()
        suite.freeze_json(root/'selected-checkpoints.json', receipt)
        with self.assertRaisesRegex(ValueError, 'receipt'):
            ztable.load_panel(root, 'geneval')


if __name__ == '__main__':
    unittest.main()
