"""Prompt preservation and provenance checks for the published evaluation panels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_eval_prompts.py'
spec = importlib.util.spec_from_file_location('prepare_eval_prompts', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PromptTests(unittest.TestCase):
    def test_duplicate_text_and_whitespace_are_preserved_and_resume_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = [{'prompt': ' same text '}, {'prompt': ' same text '}, {'prompt': 'Same text'}]
            payload = json.dumps(data).encode()
            (root / 'test.json').write_bytes(payload)
            sources = {'hpsv2': ('test.json', 3, hashlib.sha256(payload).hexdigest())}
            result = module.prepare(root / 'out', source_dir=root, sources=sources)
            rows = module.load_suite(root / 'out/hpsv2.jsonl')
            self.assertEqual([r['prompt'] for r in rows], [r['prompt'] for r in data])
            self.assertEqual(len({r['scene_id'] for r in rows}), 3)
            self.assertEqual(result['panels']['hpsv2']['unique_exact_texts'], 2)
            self.assertEqual(module.prepare(root / 'out', source_dir=root, sources=sources), result)
            smoke = module.prepare(root / 'smoke', source_dir=root, sources=sources, smoke_prompts=1)
            self.assertEqual(smoke['mode'], 'smoke-only')
            self.assertEqual(smoke['panels']['hpsv2']['prompts'], 1)
            self.assertEqual(smoke['panels']['hpsv2']['source_prompts'], 3)
            with (root / 'out/hpsv2.jsonl').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'modified'):
                module.prepare(root / 'out', source_dir=root, sources=sources)

    def test_wrong_source_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.json').write_text('[{"prompt":"hello"}]')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                module.prepare(root / 'out', source_dir=root,
                    sources={'pickv2': ('test.json', 1, '0' * 64)})
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
