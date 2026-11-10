"""Run pinned GenEval / DPG evaluators on a completed render, without a generator."""
import argparse
from collections import defaultdict
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from self_correct_dpo.common import (artifact_identity, digest, load_suite, output_lock,
    read_json, read_jsonl, require_gpu_allocation, sha256, verify_images, write_json, write_jsonl)
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json


SOURCES = {
    'geneval': {'repository': 'https://github.com/djghosh13/geneval',
        'revision': 'af4902f24d3ca90ebbb446dd9891a59e0f82725f', 'count': 553,
        'paths': ['prompts', 'evaluation']},
    'dpg': {'repository': 'https://github.com/TencentQQGYLab/ELLA',
        'revision': '3c228f1dc6c4d3cad0a47493816151a419f14db3', 'count': 1065,
        'paths': ['dpg_bench']},
}
PROTOCOLS = {'geneval': 'geneval-official-four-samples-v1',
             'dpg': 'dpg-complete-csv-four-samples-v1'}
DPG_NOTE = ('Uses all CSV rows: fixes upstream prepare_dpg_data skipping the first real row. '
    'Uses the unmodified upstream compute_dpg_one_sample with dependency gating. '
    'No failed prompts are omitted. Not an exact reproduction of historical runs that skipped '
    'partiprompts97. Category breakdown follows upstream: ungated answers from the fourth image only.')


def source_identity(repo, benchmark):
    """Verify both the pinned commit and actual bytes of all benchmark code/data."""
    repo = Path(repo).resolve(strict=True)
    source = SOURCES[benchmark]
    def git(*args, **kwargs):
        return subprocess.check_output(['git', '-C', str(repo), *args], **kwargs)
    if git('rev-parse', 'HEAD').decode().strip() != source['revision']:
        raise ValueError(f"{benchmark} requires official revision {source['revision']}")
    entries = git('ls-tree', '-rz', 'HEAD', '--', *source['paths']).split(b'\0')
    blobs = {}
    for entry in filter(None, entries):
        info, name = entry.decode().split('\t', 1)
        mode, kind, blob = info.split()
        if kind != 'blob' or mode not in ('100644', '100755') or '\n' in name:
            raise ValueError(f'Unsupported benchmark source entry: {name}')
        target = repo / name
        if target.is_symlink() or not target.is_file():
            raise ValueError(f'Missing or linked benchmark source: {target}')
        blobs[name] = blob
    if not blobs:
        raise ValueError('No tracked official benchmark sources found')
    # One git invocation for all 1,065 DPG prompt files; no network access.
    names = list(blobs)
    actual = git('hash-object', '--no-filters', '--stdin-paths',
                 input=''.join(str(repo / name) + '\n' for name in names).encode()).decode().splitlines()
    if actual != [blobs[name] for name in names]:
        raise ValueError('Official benchmark code/data has local modifications')
    return {**source, 'path': str(repo),
            'files_sha256': {name: sha256(repo / name) for name in names}}


def benchmark_rows(repo, benchmark):
    """Preserve original IDs, exact prompt strings and structured question metadata."""
    repo = Path(repo)
    if benchmark == 'geneval':
        originals = read_jsonl(repo / 'prompts/evaluation_metadata.jsonl')
        records = [(str(i), metadata) for i, metadata in enumerate(originals)]
    else:
        questions = defaultdict(list)
        with (repo / 'dpg_bench/dpg_bench.csv').open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                questions[row['item_id']].append(dict(row))
        files = sorted((repo / 'dpg_bench/prompts').glob('*.txt'))
        if {p.stem for p in files} != set(questions):
            raise ValueError('DPG prompt IDs and CSV question IDs differ')
        records = []
        for path in files:
            qrows = questions[path.stem]
            ids = [int(q['proposition_id']) for q in qrows]
            if len(set(ids)) != len(ids):
                raise ValueError(f'Duplicate DPG question ID: {path.stem}')
            # Upstream asks all questions before applying dependencies; CSV row
            # order is not topological and must not be reordered by the adapter.
            available = {0, *ids}
            for q in qrows:
                dependencies = [int(v.strip()) for v in q['dependency'].split(',')]
                if not set(dependencies) <= available:
                    raise ValueError(f'DPG dependency refers to a missing question: {path.stem}')
            records.append((path.stem, {'prompt': path.read_text(encoding='utf-8'),
                'tag': 'dpg', 'questions': qrows}))
    rows = []
    for index, (source_id, metadata) in enumerate(records):
        rows.append({'scene_id': f'{benchmark}-{source_id}', 'variant_id': 'canonical',
            'prompt': metadata['prompt'], 'tag': metadata['tag'], 'metadata': metadata,
            'suite_mode': 'canonical-only-v1', 'template_family': 'canonical',
            'transform_type': 'canonical', 'benchmark': benchmark, 'source_id': source_id,
            'source_index': index, 'source_revision': SOURCES[benchmark]['revision']})
    if len(rows) != SOURCES[benchmark]['count']:
        raise ValueError(f"Expected {SOURCES[benchmark]['count']} {benchmark} prompts, got {len(rows)}")
    return rows


def prepare(repo, benchmark, out, *, smoke_prompts=None):
    identity = source_identity(repo, benchmark)
    rows = benchmark_rows(repo, benchmark)
    if smoke_prompts is not None:
        if type(smoke_prompts) is not int or not 0 < smoke_prompts < len(rows):
            raise ValueError('smoke_prompts must be positive and smaller than the full benchmark')
        rows = rows[:smoke_prompts]
    selection = {'benchmark': benchmark, 'source': identity, 'prompts': len(rows),
        'mode': 'smoke-only' if smoke_prompts is not None else 'full-benchmark',
        'smoke_prompts': smoke_prompts, 'protocol': PROTOCOLS[benchmark],
        'policy': 'Exact official prompt text and IDs; original metadata and all CSV rows retained.'}
    out = Path(out).expanduser().resolve()
    with output_lock(str(out) + '.prepare'):
        with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
            stage = Path(temporary) / 'suite'
            stage.mkdir()
            write_jsonl(stage / 'suite.jsonl', rows)
            load_suite(stage / 'suite.jsonl')
            selection['suite_sha256'] = sha256(stage / 'suite.jsonl')
            write_json(stage / 'selection.json', selection)
            if out.exists():
                if read_json(out / 'selection.json') != selection:
                    raise ValueError('Prepared suite provenance changed; choose a new output directory')
                if sha256(out / 'suite.jsonl') != selection['suite_sha256']:
                    raise ValueError('Prepared suite was modified')
            else:
                stage.rename(out)
    return {'out': str(out), **selection}


def load_settings(path, benchmark, resources=None):
    config = read_json(path)
    if benchmark not in config:
        raise ValueError(f'Missing {benchmark} config')
    registry = load_resources(resources)
    settings = dict(config[benchmark])
    for key in ('repo', 'model_dir', 'clip_checkpoint', 'model_config', 'python'):
        if key not in settings:
            continue
        value = settings[key]
        if value.startswith('${') and value.endswith('}'):
            name = value[2:-1]
            if name not in registry:
                raise ValueError(f'Resource {name} is not configured')
            value = registry[name]
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = Path(path).resolve().parent / candidate
        settings[key] = str(candidate.resolve())
    if not settings.get('python'):
        settings['python'] = sys.executable
    return settings


def asset_identity(settings, benchmark):
    identity = {'model': artifact_identity(settings['model_dir'])}
    if benchmark == 'geneval':
        checkpoint = Path(settings['model_dir']) / 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        clip = Path(settings['clip_checkpoint'])
        stat = clip.stat()
        identity['clip'] = {'path': str(clip), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        config = Path(settings['model_config'])
        if not config.is_file():
            raise FileNotFoundError(config)
        # MMDetection configs import _base_ files: include every config under configs/.
        root = next((p for p in config.parents if p.name == 'configs'), config.parent)
        identity['detector_configs'] = {str(p.relative_to(root)): sha256(p)
                                      for p in sorted(root.rglob('*.py'))}
    elif not (Path(settings['model_dir']) / 'configuration.json').is_file():
        raise ValueError('DPG_MODEL_DIR must contain the complete local ModelScope mPLUG snapshot')
    return identity


def environment_identity(python, benchmark):
    packages = ['torch', 'torchvision', 'numpy', 'Pillow', 'pandas']
    packages += (['mmdet', 'mmcv-full', 'open-clip-torch', 'clip-benchmark'] if benchmark == 'geneval'
                 else ['modelscope', 'transformers', 'timm', 'accelerate'])
    # Query the selected evaluator's distribution metadata without importing torch/CUDA.
    script = ('import importlib.metadata as m, json, sys\n'
              'versions = {}\n'
              'for name in json.loads(sys.argv[1]):\n'
              ' try: versions[name] = m.version(name)\n'
              ' except m.PackageNotFoundError: versions[name] = "missing"\n'
              'print(json.dumps({"python":sys.executable,"version":sys.version,"packages":versions}))\n')
    return json.loads(subprocess.check_output([python, '-c', script, json.dumps(packages)], text=True))


def validate_render(run_dir, benchmark, repo):
    run, images = verify_images(run_dir)
    seeds = run['sampling']['seeds']
    if (len(seeds) != 4 or len(set(seeds)) != 4
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError('Official compositional evaluation requires four distinct seeds per prompt')
    expected = benchmark_rows(repo, benchmark)
    suite = run['suite']
    # A prefix is accepted only as an explicitly labelled smoke run in the score config.
    if suite != expected[:len(suite)] or not suite:
        raise ValueError('Render suite differs from pinned official benchmark metadata or order')
    mapping = {(r['scene_id'], r['seed']): r for r in images}
    return run, images, [[mapping[(row['scene_id'], seed)] for seed in seeds] for row in suite]


def export_images(run_dir, benchmark, out, suite, groups):
    """Materialize official layout. Never resize images or change seed ordering."""
    from PIL import Image
    target = Path(out)
    target.mkdir()
    records, resolution = [], None
    for index, (row, images) in enumerate(zip(suite, groups)):
        sample_paths = []
        if benchmark == 'geneval':
            directory = target / f'{index:05d}'
            (directory / 'samples').mkdir(parents=True)
            write_json(directory / 'metadata.jsonl', row['metadata'])
            for seed_index, image in enumerate(images):
                path = directory / 'samples' / f'{seed_index:05d}.png'
                shutil.copyfile(Path(run_dir) / image['image'], path)
                with Image.open(path) as decoded:
                    if decoded.format != 'PNG':
                        raise ValueError('GenEval export requires PNG render images')
                    decoded.verify()
                sample_paths.append(str(path))
            records.append({'source_id': row['source_id'], 'index': index, 'images': sample_paths})
        else:
            grid = None
            for seed_index, image in enumerate(images):
                with Image.open(Path(run_dir) / image['image']) as decoded:
                    decoded.load()
                    width, height = decoded.size
                    if width != height or (resolution is not None and width != resolution):
                        raise ValueError('DPG requires equal square images; resizing is forbidden')
                    resolution = width
                    if grid is None:
                        grid = Image.new('RGB', (2 * width, 2 * width))
                    grid.paste(decoded.convert('RGB'), ((seed_index % 2) * width, (seed_index // 2) * width))
            path = target / (row['source_id'] + '.png')
            grid.save(path)
            records.append({'source_id': row['source_id'], 'index': index, 'image': str(path)})
    return {'records': records, 'resolution': resolution,
            'files_sha256': {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob('*')) if p.is_file()}}


def verify_files(root, files):
    root = Path(root).resolve()
    for name, checksum in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != checksum:
            raise ValueError(f'Evaluation artifact changed or missing: {name}')


def aggregate(benchmark, chunks, *, smoke=False):
    by_category = defaultdict(list)
    if benchmark == 'geneval':
        results = [row for chunk in chunks for row in chunk['results']]
        for row in results:
            if type(row['correct']) is not bool:
                raise ValueError('GenEval correct must be a boolean')
            by_category[row['tag']].append(float(row['correct']))
        categories = {k: sum(v) / len(v) for k, v in by_category.items()}
        mean = sum(categories.values()) / len(categories)
        detail = {'aggregation': 'Arithmetic mean of category-wise image success rates (official).',
                  'micro_image_accuracy': sum(r['correct'] for r in results) / len(results)}
    else:
        scores = [chunk['score'] for chunk in chunks]
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError('Invalid DPG score')
        for chunk in chunks:
            for qid, category_tuple in chunk['qid2tuple'].items():
                category = category_tuple.split('(')[0].strip().split('-')[0].strip()
                by_category[category].append(chunk['qid2scores_last_crop'][str(qid)])
        categories = {k: 100 * sum(v) / len(v) for k, v in by_category.items()}
        mean = 100 * sum(scores) / len(scores)
        detail = {'aggregation': 'Mean across prompts of four dependency-gated image scores, multiplied by 100.',
                  'category_definition': 'Official legacy L1: ungated answers of fourth seed only.',
                  'protocol_note': DPG_NOTE}
    return {'metric': benchmark, 'mean': mean, 'prompts': len(chunks), 'images': 4 * len(chunks),
            'by_category': categories, 'protocol': PROTOCOLS[benchmark],
            'mode': 'smoke-only' if smoke else 'full-benchmark', **detail}


def load_official(path):
    spec = importlib.util.spec_from_file_location('official_compositional_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk_result(directory):
    complete = read_json(Path(directory) / 'complete.json')
    verify_files(directory, complete['files_sha256'])
    return read_json(Path(directory) / 'result.json')


def verify_result(out_dir, render_dir):
    """Verify persisted results and their completed render, without importing models."""
    out, render = Path(out_dir).resolve(), Path(render_dir).resolve()
    request = read_json(out / 'evaluation.json')
    complete = read_json(out / 'complete.json')
    if request['evaluation_id'] != digest(request['identity']):
        raise ValueError('Compositional evaluation identity checksum mismatch')
    if sha256(out / 'summary.json') != complete['summary_sha256']:
        raise ValueError('Compositional summary checksum mismatch')
    verify_files(out, complete['files_sha256'])
    run, images = verify_images(render)
    identity = request['identity']
    if (identity['run_id'] != run['run_id'] or
            identity['manifest_sha256'] != sha256(render / 'manifest.jsonl')):
        raise ValueError('Compositional evaluation belongs to a different render')
    summary = read_json(out / 'summary.json')
    if (summary['evaluation_id'] != request['evaluation_id'] or summary['images'] != len(images)
            or summary['prompts'] != len(run['suite'])):
        raise ValueError('Compositional summary does not cover this render')
    return summary


def worker(request_path):
    """Runs inside the isolated evaluator environment; imports no generation code."""
    request = read_json(request_path)
    benchmark, settings = request['benchmark'], request['settings']
    require_gpu_allocation(request['execution'])
    source_identity(settings['repo'], benchmark)
    out = Path(request_path).parent
    run = read_json(Path(request['run']) / 'run.json')
    exported = read_json(out / 'export.json')
    pending = [item for item in exported['records'] if not (out / 'raw' / f"{item['index']:05d}").exists()]
    if not pending:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Official compositional evaluation requires CUDA')
    runtime = {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
               'environment': environment_identity(sys.executable, benchmark)}
    if (out / 'runtime.json').exists():
        if read_json(out / 'runtime.json') != runtime:
            raise ValueError('Evaluator runtime changed while resuming')
    else:
        atomic_json(out / 'runtime.json', runtime)
    if benchmark == 'geneval':
        from importlib.metadata import version
        if version('open-clip-torch') != '2.20.0':
            raise ValueError('GenEval local OpenAI CLIP hook requires open-clip-torch==2.20.0')
        import open_clip.openai
        def local_clip(url, **kwargs):
            if not url.endswith('/ViT-L-14.pt'):
                raise ValueError(f'Unexpected OpenAI CLIP asset: {url}')
            return settings['clip_checkpoint']
        open_clip.openai.download_pretrained_from_url = local_clip
        official = load_official(Path(settings['repo']) / 'evaluation/evaluate_images.py')
        args = SimpleNamespace(model_path=settings['model_dir'], model_config=settings['model_config'], options={})
        official.args = args
        official.object_detector, (official.clip_model, official.transform, official.tokenizer), official.classnames = official.load_models(args)
        official.THRESHOLD, official.COUNTING_THRESHOLD = 0.3, 0.9
        official.MAX_OBJECTS, official.NMS_THRESHOLD, official.POSITION_THRESHOLD = 16, 1.0, 0.1
    else:
        official = load_official(Path(settings['repo']) / 'dpg_bench/compute_dpg_bench.py')
        vqa = official.MPLUG(ckpt=settings['model_dir'], device='cuda')
    for item in pending:
        metadata = run['suite'][item['index']]['metadata']
        with tempfile.TemporaryDirectory(prefix='.pending-', dir=out / 'raw') as temporary:
            stage = Path(temporary) / 'result'
            stage.mkdir()
            if benchmark == 'geneval':
                results = [official.evaluate_image(path, metadata) for path in item['images']]
                for result in results:
                    result['correct'] = bool(result['correct'])
                payload = {'source_id': item['source_id'], 'results': results}
                write_jsonl(stage / 'results.jsonl', results)
            else:
                # Preserve all rows. Upstream prepare_dpg_data drops the first real CSV row.
                qrows = metadata['questions']
                questions = {'qid2tuple': {}, 'qid2question': {}, 'qid2dependency': {}}
                for q in qrows:
                    qid = int(q['proposition_id'])
                    questions['qid2tuple'][qid] = q['tuple']
                    questions['qid2question'][qid] = q['question_natural_language']
                    questions['qid2dependency'][qid] = [int(p.strip()) for p in q['dependency'].split(',')]
                args = SimpleNamespace(pic_num=4, res_path=str(stage / 'results.txt'))
                score, tuples, last_scores = official.compute_dpg_one_sample(args,
                    {item['source_id']: questions}, item['image'], vqa, exported['resolution'])
                payload = {'source_id': item['source_id'], 'score': score,
                           'qid2tuple': tuples, 'qid2scores_last_crop': last_scores}
            write_json(stage / 'result.json', payload)
            write_json(stage / 'complete.json', {'files_sha256':
                {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}})
            stage.rename(out / 'raw' / f"{item['index']:05d}")
        print(f"{benchmark}: committed {item['index'] + 1}/{len(exported['records'])}", flush=True)


def score(run_dir, out, benchmark, settings, *, execution=None, check_only=False, command_runner=None):
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    source = source_identity(settings['repo'], benchmark)
    run, images, groups = validate_render(run_dir, benchmark, settings['repo'])
    smoke = len(run['suite']) != SOURCES[benchmark]['count']
    if smoke and not settings.get('allow_smoke', False):
        raise ValueError('Incomplete official prompt suite; allow_smoke=true is required for smoke-only results')
    identity = {'benchmark': benchmark, 'protocol': PROTOCOLS[benchmark], 'source': source,
        'settings': settings, 'assets': asset_identity(settings, benchmark), 'run_id': run['run_id'],
        'environment': environment_identity(settings.get('python') or sys.executable, benchmark),
        'manifest_sha256': sha256(run_dir / 'manifest.jsonl'), 'code_sha256': sha256(__file__),
        'common_sha256': sha256(Path(__file__).resolve().parents[1] / 'common.py')}
    request = {'benchmark': benchmark, 'settings': settings, 'run': str(run_dir),
        'identity': identity, 'evaluation_id': digest(identity),
        'execution': execution or {'mode': 'slurm'}}
    if out.exists():
        previous = read_json(out / 'evaluation.json')
        if previous['identity'] != identity or previous['evaluation_id'] != digest(identity):
            raise ValueError('Cannot resume compositional evaluation with changed inputs, assets or code')
        if (out / 'complete.json').exists():
            return verify_result(out, run_dir)
    if check_only:
        return {'metric': benchmark, 'prompts': len(groups), 'images': len(images),
                'protocol': PROTOCOLS[benchmark], 'evaluation_id': request['evaluation_id'],
                'mode': 'smoke-only' if smoke else 'full-benchmark', 'check_only': True}
    require_gpu_allocation(execution)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
            stage = Path(temporary) / 'score'
            stage.mkdir()
            write_json(stage / 'evaluation.json', request)
            stage.rename(out)
    if not (out / 'export.json').exists():
        with tempfile.TemporaryDirectory(prefix='.export-', dir=out) as temporary:
            stage = Path(temporary) / 'images'
            exported = export_images(run_dir, benchmark, stage, run['suite'], groups)
            # Rewrite staged paths to their final published location.
            for item in exported['records']:
                if benchmark == 'geneval':
                    item['images'] = [str(out / 'official-images' / Path(p).relative_to(stage)) for p in item['images']]
                else:
                    item['image'] = str(out / 'official-images' / Path(item['image']).relative_to(stage))
            if (out / 'official-images').exists():
                # Recover the narrow crash window between directory publication and index commit.
                existing = {str(p.relative_to(out / 'official-images')): sha256(p)
                            for p in (out / 'official-images').rglob('*') if p.is_file()}
                if existing != exported['files_sha256']:
                    raise ValueError('Uncommitted official export differs from the verified render')
            else:
                stage.rename(out / 'official-images')
            atomic_json(out / 'export.json', exported)
    exported = read_json(out / 'export.json')
    verify_files(out / 'official-images', exported['files_sha256'])
    (out / 'raw').mkdir(exist_ok=True)
    for item in exported['records']:
        directory = out / 'raw' / f"{item['index']:05d}"
        if directory.exists():
            chunk_result(directory)
    command = [settings.get('python') or sys.executable, '-m',
               'self_correct_dpo.evaluation.compositional', '--worker', str(out / 'evaluation.json')]
    if command_runner is None:
        from self_correct_dpo.evaluation.runner import run_command
        command_runner = run_command
    command_runner(command)
    chunks = [chunk_result(out / 'raw' / f"{item['index']:05d}") for item in exported['records']]
    if [c['source_id'] for c in chunks] != [r['source_id'] for r in run['suite']]:
        raise ValueError('Official score rows do not cover the complete prompt suite')
    if benchmark == 'geneval' and any(len(c['results']) != 4 for c in chunks):
        raise ValueError('Missing GenEval per-image results')
    summary = {**aggregate(benchmark, chunks, smoke=smoke), 'evaluation_id': request['evaluation_id']}
    atomic_json(out / 'summary.json', summary)
    checksums = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                 if p.is_file() and p.name != 'complete.json' and not any(part.startswith('.') for part in p.relative_to(out).parts)}
    atomic_json(out / 'complete.json', {'summary_sha256': sha256(out / 'summary.json'), 'files_sha256': checksums})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=SOURCES)
    parser.add_argument('--run')
    parser.add_argument('--out')
    parser.add_argument('--config')
    parser.add_argument('--resources')
    parser.add_argument('--python', help='Override the isolated evaluator worker interpreter')
    parser.add_argument('--execution', choices=['slurm', 'standalone'], default='slurm')
    parser.add_argument('--run-label')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.worker)
        return
    if not all((args.benchmark, args.run, args.out, args.config)):
        parser.error('--benchmark, --run, --out and --config are required')
    settings = load_settings(args.config, args.benchmark, args.resources)
    if args.python:
        settings['python'] = str(Path(args.python).expanduser().resolve(strict=True))
    with output_lock(str(args.out) + '.compositional'):
        result = score(args.run, args.out, args.benchmark, settings,
            execution={'mode': args.execution, 'run_label': args.run_label}, check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
