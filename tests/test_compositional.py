"""CPU checks for official layouts, provenance, completeness and resumed scoring."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from self_correct_dpo.common import digest, read_json, sha256, write_json, write_jsonl
from self_correct_dpo.evaluation import compositional as comp


def fixture_repo(root, benchmark):
    repo = root / 'official'
    repo.mkdir()
    if benchmark == 'geneval':
        (repo / 'prompts').mkdir()
        write_jsonl(repo / 'prompts/evaluation_metadata.jsonl', [
            {'prompt': 'a red cat', 'tag': 'colors', 'include': [{'class': 'cat', 'count': 1, 'color': 'red'}]},
            {'prompt': 'a cat left of a dog', 'tag': 'position',
             'include': [{'class': 'dog', 'count': 1}, {'class': 'cat', 'count': 1, 'position': ['left of', 0]}]}])
    else:
        directory = repo / 'dpg_bench'
        (directory / 'prompts').mkdir(parents=True)
        (directory / 'prompts/scene1.txt').write_text(' a red cat\n', encoding='utf-8')
        (directory / 'prompts/scene2.txt').write_text('a dog', encoding='utf-8')
        (directory / 'dpg_bench.csv').write_text(
            'item_id,text,keywords,proposition_id,dependency,category_broad,category_detailed,tuple,question_natural_language\n'
            'scene1,a red cat,,1,0,entity,whole,entity - whole (cat),Is there a cat?\n'
            'scene1,a red cat,,2,1,attribute,color,attribute - color (cat red),Is the cat red?\n'
            'scene2,a dog,,1,0,entity,whole,entity - whole (dog),Is there a dog?\n', encoding='utf-8')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=CPU test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
    spec = {**comp.SOURCES[benchmark], 'revision': revision, 'count': 2}
    return repo, spec


def render_fixture(root, suite):
    render = root / 'render'
    (render / 'images').mkdir(parents=True)
    run = {'suite': suite, 'suite_id': digest(suite), 'expected_images': 4 * len(suite),
           'sampling': {'seeds': [3, 7, 11, 19]}}
    run['run_id'] = digest(run)
    write_json(render / 'run.json', run)
    rows = []
    for i, prompt in enumerate(suite):
        for j, seed in enumerate(run['sampling']['seeds']):
            path = render / 'images' / f'{i}-{seed}.png'
            Image.new('RGB', (4, 4), (j * 50, i * 40, 3)).save(path)
            row = {'scene_id': prompt['scene_id'], 'variant_id': 'canonical', 'seed': seed,
                   'prompt': prompt['prompt'], 'tag': prompt['tag'],
                   'image': str(path.relative_to(render)), 'image_sha256': sha256(path)}
            row['image_id'] = digest({'run_id': run['run_id'], **row})
            rows.append(row)
    write_jsonl(render / 'manifest.jsonl', rows)
    write_json(render / 'complete.json', {'images': len(rows), 'manifest_sha256': sha256(render / 'manifest.jsonl')})
    return render


class CompositionalTests(unittest.TestCase):
    def test_official_preparation_preserves_first_question_ids_and_whitespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, source = fixture_repo(root, 'dpg')
            with patch.dict(comp.SOURCES, {'dpg': source}):
                result = comp.prepare(repo, 'dpg', root / 'prepared')
                self.assertEqual(result['prompts'], 2)
                rows = comp.load_suite(root / 'prepared/suite.jsonl')
                self.assertEqual(rows[0]['prompt'], ' a red cat\n')
                self.assertEqual(rows[0]['source_id'], 'scene1')
                self.assertEqual(rows[0]['metadata']['questions'][0]['proposition_id'], '1')
                self.assertEqual(rows[0]['metadata']['questions'][1]['dependency'], '1')
                self.assertEqual(comp.prepare(repo, 'dpg', root / 'prepared'), result)
                (repo / 'dpg_bench/prompts/scene1.txt').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'local modifications'):
                    comp.prepare(repo, 'dpg', root / 'new')
                self.assertFalse((root / 'new').exists())

    def test_real_grid_layout_and_missing_image_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, source = fixture_repo(root, 'dpg')
            with patch.dict(comp.SOURCES, {'dpg': source}):
                suite = comp.benchmark_rows(repo, 'dpg')
                render = render_fixture(root, suite)
                _, _, groups = comp.validate_render(render, 'dpg', repo)
                exported = comp.export_images(render, 'dpg', root / 'export', suite, groups)
                self.assertEqual(exported['resolution'], 4)
                with Image.open(root / 'export/scene1.png') as grid:
                    self.assertEqual(grid.size, (8, 8))
                    self.assertEqual([grid.getpixel(p)[0] for p in [(0, 0), (4, 0), (0, 4), (4, 4)]], [0, 50, 100, 150])
                (render / 'images/0-7.png').unlink()
                with self.assertRaises(FileNotFoundError):
                    comp.validate_render(render, 'dpg', repo)

    def test_aggregation_uses_task_macro_and_dpg_prompt_mean(self):
        chunks = [{'results': [{'tag': 'colors', 'correct': True}] * 4},
                  {'results': [{'tag': 'position', 'correct': False}] * 4},
                  {'results': [{'tag': 'position', 'correct': False}] * 4}]
        result = comp.aggregate('geneval', chunks)
        self.assertEqual(result['mean'], 0.5)
        self.assertAlmostEqual(result['micro_image_accuracy'], 1 / 3)
        dpg = comp.aggregate('dpg', [
            {'score': .25, 'qid2tuple': {'1': 'entity - whole (cat)'}, 'qid2scores_last_crop': {'1': 1.0}},
            {'score': .75, 'qid2tuple': {'1': 'entity - whole (dog)'}, 'qid2scores_last_crop': {'1': 0.0}}])
        self.assertEqual(dpg['mean'], 50)
        self.assertEqual(dpg['by_category']['entity'], 50)
        self.assertIn('fourth', dpg['category_definition'])

    def test_interrupted_score_resumes_committed_chunks_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, source = fixture_repo(root, 'geneval')
            assets = root / 'models'
            assets.mkdir()
            (assets / 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth').write_bytes(b'fixture-only')
            (assets / 'clip.pt').write_bytes(b'fixture-only')
            (assets / 'detector.py').write_text('model = {}\n')
            settings = {'repo': str(repo), 'model_dir': str(assets),
                        'clip_checkpoint': str(assets / 'clip.pt'), 'model_config': str(assets / 'detector.py')}
            out = root / 'score'
            writes = []
            def fake_worker(command, interrupt=False):
                for item in read_json(out / 'export.json')['records']:
                    directory = out / 'raw' / f"{item['index']:05d}"
                    if directory.exists():
                        continue
                    directory.mkdir()
                    rows = [{'filename': p, 'correct': True, 'tag': 'colors' if item['index'] == 0 else 'position'}
                            for p in item['images']]
                    write_json(directory / 'result.json', {'source_id': item['source_id'], 'results': rows})
                    write_json(directory / 'complete.json', {'files_sha256': {'result.json': sha256(directory / 'result.json')}})
                    writes.append(item['index'])
                    if interrupt:
                        raise RuntimeError('simulated interruption')
            with patch.dict(comp.SOURCES, {'geneval': source}), patch.object(comp, 'require_gpu_allocation'):
                suite = comp.benchmark_rows(repo, 'geneval')
                render = render_fixture(root, suite)
                check = comp.score(render, out, 'geneval', settings, check_only=True)
                self.assertEqual(check['images'], 8)
                self.assertFalse(out.exists())
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    comp.score(render, out, 'geneval', settings, command_runner=lambda cmd: fake_worker(cmd, True))
                self.assertFalse((out / 'complete.json').exists())
                first_checksum = sha256(out / 'raw/00000/result.json')
                result = comp.score(render, out, 'geneval', settings, command_runner=fake_worker)
                self.assertEqual(writes, [0, 1])
                self.assertEqual(first_checksum, sha256(out / 'raw/00000/result.json'))
                self.assertEqual(result['mean'], 1)
                self.assertEqual(comp.verify_result(out, render), result)
                self.assertEqual(comp.score(render, out, 'geneval', settings,
                    command_runner=lambda cmd: self.fail('Completed results must not reload scorers')), result)
                (assets / 'clip.pt').write_bytes(b'changed asset')
                with self.assertRaisesRegex(ValueError, 'changed inputs'):
                    comp.score(render, out, 'geneval', settings, check_only=True)
                (out / 'raw/00000/result.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    comp.verify_result(out, render)


if __name__ == '__main__':
    unittest.main()
