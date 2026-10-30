"""Metadata-only conversion, identity protection and interrupted publication."""
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_sdxl_from_sd15 as conversion
from self_correct_dpo.common import read_json, read_jsonl, sha256
from self_correct_dpo.data.pairs import load_pairs


class WordTokenizer:
    model_max_length = 8

    def __init__(self, extra=None):
        self.extra = extra

    def __call__(self, texts, **options):
        assert options == dict(padding=False, truncation=False, add_special_tokens=True)
        return {'input_ids': [[0] * (len(t.split()) + 2 + (8 if self.extra in t else 0))
                              for t in texts]}


class SDXLConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'sd15' / 'pairs.jsonl'
        self.source.parent.mkdir()
        self.out, self.panels, self.model = [self.root / name for name in ('sdxl', 'panels', 'model')]
        for name in ('tokenizer', 'tokenizer_2'):
            (self.model / name).mkdir(parents=True)
            (self.model / name / 'tokenizer_config.json').write_text('{"model_max_length":8}')
        # Images intentionally do not exist: conversion must never stat/read them.
        self.rows = [dict(pair_id=f'p{i}', scene_id=f'scene{i}',
            split=('train', 'validation', 'test')[i % 3],
            winner=f'../images/w{i}.img', loser=str(self.root / 'images' / f'l{i}.img'),
            winner_sha256=f'{i+1:064x}', loser_sha256=f'{i+101:064x}',
            human_preference={'source_revision': 'fixture@abc'},
            prompts=[dict(variant_id='canonical', prompt=f'caption {i}',
                          template_family='canonical', transform_type='canonical')])
                     for i in range(12)]
        self.rows[3]['prompts'][0]['prompt'] = 'second_only_reject'
        self.rows[4]['prompts'][0]['prompt'] = ' '.join(['too_long'] * 10)
        self.save_source()
        official = self.source.parent / 'official-test-unique.jsonl'
        official.write_text(''.join(json.dumps({'caption': p}) + '\n' for p in
            ['caption 0', 'second_only_reject', ' '.join(['too_long'] * 10)] +
            [f'heldout {i}' for i in range(10)]))
        (self.source.parent / 'audit.json').write_text('{"family":"sd15"}')

    def save_source(self):
        self.source.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        (self.source.parent / 'ready.json').write_text(json.dumps({
            'pairs_sha256': sha256(self.source), 'source_revision': 'fixture@abc',
            'splits': dict(Counter(r['split'] for r in self.rows))}))

    def run_conversion(self, **options):
        with redirect_stdout(StringIO()):
            return conversion.convert(self.source, self.out, self.model, self.panels,
                count=4, batch_size=2, tokenizers=[WordTokenizer(extra='never'),
                                                WordTokenizer(extra='second_only')], **options)

    def test_filter_preserves_splits_references_hashes_and_panels_without_image_io(self):
        source_bytes = self.source.read_bytes()
        original_open, original_stat = Path.open, Path.stat

        def guard_open(path, *args, **kwargs):
            if path.suffix in ('.img', '.parquet'):
                raise AssertionError(f'Unexpected image/raw read: {path}')
            return original_open(path, *args, **kwargs)

        def guard_stat(path, *args, **kwargs):
            if path.suffix in ('.img', '.parquet'):
                raise AssertionError(f'Unexpected image/raw stat: {path}')
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, 'open', guard_open), patch.object(Path, 'stat', guard_stat):
            audit = self.run_conversion()
        result = load_pairs(self.out / 'pairs.jsonl', verify_files=False)
        expected = {r['pair_id']: r for i, r in enumerate(self.rows) if i not in (3, 4)}
        self.assertEqual({r['pair_id'] for r in result}, set(expected))
        for row in result:
            old = expected[row['pair_id']]
            self.assertEqual(row['split'], old['split'])
            self.assertEqual(row['prompts'], old['prompts'])
            self.assertEqual(row['human_preference'], old['human_preference'])
            for side in ('winner', 'loser'):
                self.assertEqual(row[side], str((self.source.parent / old[side]).resolve()))
                self.assertEqual(row[side + '_sha256'], old[side + '_sha256'])
        panel = read_jsonl(self.panels / 'pickapic-monitor.jsonl')
        self.assertEqual(len(panel), 4)
        self.assertTrue(all(r['prompt'].startswith('heldout') for r in panel))
        self.assertEqual(audit['splits'], {'train': 3, 'validation': 3, 'test': 4})
        ready = read_json(self.out / 'ready.json')
        self.assertEqual(ready['pairs_sha256'], sha256(self.out / 'pairs.jsonl'))
        self.assertEqual(ready['source_revision'], 'fixture@abc')
        self.assertEqual(source_bytes, self.source.read_bytes())
        self.assertFalse((self.root / 'images').exists())
        # A completed retry checks identity/hashes and does not rewrite outputs.
        stamp = (self.out / 'pairs.jsonl').stat().st_mtime_ns
        self.run_conversion()
        self.assertEqual(stamp, (self.out / 'pairs.jsonl').stat().st_mtime_ns)
        (self.model / 'tokenizer_2/tokenizer_config.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.run_conversion()

    def test_incomplete_changed_or_foreign_inputs_do_not_publish(self):
        (self.source.parent / 'ready.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_conversion()
        self.save_source()
        self.source.write_text(self.source.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'ready.json'):
            self.run_conversion()
        self.save_source()
        self.out.mkdir()
        (self.out / 'preparing.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'no conversion marker'):
            self.run_conversion()
        self.assertFalse((self.out / 'ready.json').exists())
        self.assertFalse(self.panels.exists())

    def test_cli_uses_resource_defaults_and_loads_both_local_tokenizers(self):
        resources = self.root / 'resources.json'
        resources.write_text(json.dumps({'PICKAPIC_SD15_PREPARED': str(self.source.parent),
            'PICKAPIC_SDXL_PREPARED': str(self.out), 'SDXL_MODEL': str(self.model),
            'MONITOR_PROMPTS_SDXL': str(self.panels / 'pickapic-monitor.jsonl')}))
        loaded = []

        def load(path, *, local_files_only):
            self.assertTrue(local_files_only)
            loaded.append(Path(path).name)
            return WordTokenizer(extra='second_only' if len(loaded) == 2 else 'never')

        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load))
        with patch.dict(sys.modules, transformers=transformers), \
                patch.object(sys, 'argv', ['prepare_sdxl_from_sd15.py', '--resources', str(resources),
                                          '--monitor-count', '4']), redirect_stdout(StringIO()):
            conversion.main()
        self.assertEqual(loaded, ['tokenizer', 'tokenizer_2'])
        self.assertEqual(len(read_jsonl(self.out / 'pairs.jsonl')), 10)
        self.assertTrue((self.panels / 'pickapic-monitor.jsonl').is_file())

    def test_failure_and_interruption_are_retryable_without_partial_ready(self):
        with patch.object(conversion, 'select_monitor', side_effect=ValueError('not enough prompts')):
            with self.assertRaisesRegex(ValueError, 'not enough'):
                self.run_conversion()
        self.assertFalse(self.out.exists())
        self.assertFalse(self.panels.exists())
        rename = conversion.os.rename

        def interrupt(src, dst, *args, **kwargs):
            if Path(dst) == self.out.resolve():
                raise KeyboardInterrupt('interrupted after panel publication')
            return rename(src, dst, *args, **kwargs)

        with patch.object(conversion.os, 'rename', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_conversion()
        self.assertTrue((self.panels / 'selection.json').is_file())
        self.assertFalse(self.out.exists())
        self.run_conversion()
        self.assertTrue((self.out / 'ready.json').is_file())
        (self.out / 'pairs.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            self.run_conversion()


if __name__ == '__main__':
    unittest.main()
