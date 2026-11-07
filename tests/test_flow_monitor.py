"""Generated flow panels must be accepted by the real evaluation suite loader."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from self_correct_dpo.common import load_suite


class FlowMonitorTest(unittest.TestCase):
    def test_prepared_panel_has_valid_canonical_metadata_and_stable_selection(self):
        script = Path(__file__).resolve().parents[1]/'scripts/prepare_flow_monitor.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root/'pairs.jsonl'
            rows = [{'split': split, 'prompts': [{'variant_id': 'canonical', 'prompt': text}]}
                    for split, text in [('train', 'training only'),
                        *[('validation', f'heldout scene {i}') for i in range(6)]]]
            pairs.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            outputs = [root/'panel1.jsonl', root/'panel2.jsonl']
            for output in outputs:
                subprocess.run([sys.executable, str(script), '--pairs', str(pairs),
                    '--out', str(output), '--count', '4', '--seed', '17'], check=True)
                panel = load_suite(output)
                self.assertEqual(len(panel), 4)
                self.assertTrue(all(row['prompt'].startswith('heldout scene') for row in panel))
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())

    def test_old_panel_repair_preserves_prompts_and_source(self):
        script = Path(__file__).resolve().parents[1]/'scripts/repair_flow_monitor.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root/'old.jsonl', root/'fixed.jsonl'
            rows = [{'scene_id': f'scene-{i}', 'variant_id': 'canonical',
                     'prompt': f'caption {i}', 'tag': 'hpdv3-validation'} for i in range(3)]
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            original = source.read_bytes()
            command = [sys.executable, str(script), '--source', str(source), '--out', str(out)]
            subprocess.run(command, check=True)
            self.assertEqual([r['prompt'] for r in load_suite(out)], [r['prompt'] for r in rows])
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(json.loads(out.with_suffix('.audit.json').read_text())['prompt_text_and_order_preserved'])
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
