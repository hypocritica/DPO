"""CPU checks for paired ownership, persistent noise, resume and plot artifacts."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from PIL import Image
import run


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='qualitative-test-')
        self.root = Path(self.tmp.name)
        self.rows = run.prompts(Path(__file__).with_name('MVPO_qualitative_prompts.jsonl'))
        self.config = run.read_json(Path(__file__).with_name('config.json'))

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixed_batch_coverage(self):
        for family, spec in self.config['backbones'].items():
            plan = {'prompts': self.rows, 'spec': spec}
            batches = run.batches(plan, 'smoke', 0, spec['workers'])
            for rank in range(spec['workers']):
                batches += run.batches(plan, 'full', rank, spec['workers'])
            keys = [(p['id'], s) for batch in batches for p, s in batch]
            self.assertEqual(len(keys), 120)
            self.assertEqual(len(set(keys)), 120)
            self.assertEqual(keys[:2], [('P02', 0), ('P26', 0)])
            self.assertTrue(all(len(b) <= spec['sampling']['generation_batch_size'] for b in batches))

    def test_noise_cache_and_corruption(self):
        import torch
        for family in ('sd15', 'zimage'):
            sampling = self.config['backbones'][family]['sampling']
            pipe = SimpleNamespace(unet=SimpleNamespace(config=SimpleNamespace(in_channels=4)), vae_scale_factor=8)
            a, meta = run.noise_for(pipe, self.root / family, self.rows[1], 0, sampling, torch)
            original = a.clone()
            a.add_(123)
            b, meta2 = run.noise_for(pipe, self.root / family, self.rows[1], 0, sampling, torch)
            self.assertTrue(torch.equal(b, original))
            self.assertEqual(meta, meta2)
            self.assertEqual(b.dtype, torch.float32 if family == 'zimage' else torch.bfloat16)
            Path(meta['noise_path']).with_suffix('.json').write_text('{"sha256":"wrong"}')
            with self.assertRaises(ValueError):
                run.noise_for(pipe, self.root / family, self.rows[1], 0, sampling, torch)

    def test_resume_image_integrity(self):
        path = self.root / 'image.png'
        Image.new('RGB', (64, 64), 'red').save(path)
        run.atomic_json(path.with_suffix('.json'), {'plan_id': 'a', 'image_sha256': run.sha256(path)})
        self.assertTrue(run.valid_image(path, 'a', (64, 64)))
        self.assertFalse(run.valid_image(path, 'a', (32, 32)))
        with self.assertRaises(ValueError): run.valid_image(path, 'b')
        path.write_bytes(path.read_bytes()[:20])
        self.assertFalse(run.valid_image(path, 'a'))

    def test_layout_fixture(self):
        import layout
        # Explicit synthetic fixture only; never used as model results.
        selected, gallery = [], []
        for family in layout.FAMILIES:
            plan = {'plan_id': family, 'prompts': self.rows, 'spec': self.config['backbones'][family]}
            run.atomic_json(self.root / family / 'run_config.json', plan)
            for i, p in enumerate(self.rows[:5]):
                images = {}
                for method, color in zip(layout.METHODS, ['steelblue', 'coral', 'seagreen']):
                    path = run.image_path(self.root / family, method, p['id'], 0)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (64, 64), color).save(path)
                    run.atomic_json(path.with_suffix('.json'), {'plan_id': family, 'image_sha256': run.sha256(path), 'initial_noise_sha256': p['id']})
                    images[method] = str(path)
                if i < 3:
                    item = layout.case(self.root, family, p, 0, plan)
                    item['position'] = i
                    selected.append(item)
                gallery.append({'backbone': family, 'prompt_id': p['id'], 'seed': 0, 'position': i,
                                'image_path': images['MVPO'], 'reviewed': False})
        run.atomic_json(self.root / 'comparison_selection.json', {'items': selected})
        run.atomic_json(self.root / 'gallery_selection.json', {'items': gallery})
        layout.export(self.root)
        for name in ['qualitative_comparison', 'qualitative_comparison_compact', 'qualitative_comparison_supplement', 'mvpo_gallery']:
            with Image.open(self.root / 'figures' / (name + '.png')) as image:
                self.assertAlmostEqual(image.width / 300 * 25.4, 174, delta=.1)
                self.assertAlmostEqual(image.info['dpi'][0], 300, delta=.1)
            self.assertTrue((self.root / 'figures' / (name + '.pdf')).stat().st_size > 1000)
        from unittest.mock import patch
        plan_path = self.root / 'sd15/run_config.json'
        contact_plan = run.read_json(plan_path)
        contact_plan['prompts'] = contact_plan['prompts'][:1]
        run.atomic_json(plan_path, contact_plan)
        with patch.object(layout, 'audit', return_value=True):
            layout.contacts(self.root, ['sd15'])
        with Image.open(self.root / 'contact_sheets/sd15/P01.png') as sheet:
            self.assertEqual(sheet.size, (1600, 2240))
        contact_plan['prompts'] = self.rows
        run.atomic_json(plan_path, contact_plan)
        bad = copy.deepcopy(selected)
        bad[0]['images']['Base'] = bad[0]['images']['MVPO']
        with self.assertRaises(ValueError): layout.check_selection(self.root, bad)
        # Optional fixture preview destination, explicitly named to prevent confusion.
        import os, shutil
        if os.environ.get('QUAL_LAYOUT_FIXTURE'):
            shutil.copytree(self.root / 'figures', os.environ['QUAL_LAYOUT_FIXTURE'], dirs_exist_ok=True)


if __name__ == '__main__': unittest.main()
