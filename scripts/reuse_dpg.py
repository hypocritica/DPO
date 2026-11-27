"""Score frozen SD3.5-M DPG images in independent, resumable prompt blocks."""
import argparse
import fcntl
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.common import digest, output_lock, read_json, read_jsonl, require_gpu_allocation, sha256
from self_correct_dpo.evaluation import compositional as comp
from self_correct_dpo.evaluation.suite import read_plan, safe_target, validate_task_render
from self_correct_dpo.resources import CONFIG_DIR, load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json


METHODS = ('base', 'linear_dpo', 'global_k4', 'dpo', 'sft', 'global_k2', 'stagewise_k2', 'stagewise_k4')
SEEDS = [17, 29, 43, 71]
BLOCK = 32
COUNT = 1065


def check_block(path, identity, prompts):
    saved = read_json(path)
    rows = saved['rows']
    ids = [r['source_id'] for r in rows]
    if (saved['identity'] != identity or saved['sha256'] != digest(rows)
            or ids != [p['source_id'] for p in prompts]):
        raise ValueError(f'Invalid saved block: {path}')
    return rows


def score_prompt(official, vqa, render, prompt, images):
    by_id, dependencies, types = {}, {}, {}
    for q in prompt['metadata']['questions']:
        number = int(q['proposition_id'])
        by_id[number] = q['question_natural_language']
        types[number] = q['tuple']
        dependencies[number] = [int(p.strip()) for p in q['dependency'].split(',')]
    key = prompt['source_id']
    questions = {key: {'qid2question': by_id, 'qid2tuple': types, 'qid2dependency': dependencies}}
    for row in images:
        path = safe_target(render, row['image'])
        if sha256(path) != row['image_sha256']:
            raise ValueError(f'PNG checksum mismatch: {path}')
    with tempfile.TemporaryDirectory(prefix='mvpo-dpg-') as tmp:
        tmp = Path(tmp)
        export = comp.export_images(render, 'dpg', tmp / 'grid', [prompt], [images])
        args = SimpleNamespace(pic_num=4, res_path=str(tmp / 'answers.txt'))
        score, tuples, last = official.compute_dpg_one_sample(
            args, questions, export['records'][0]['image'], vqa, export['resolution'])
        # The official detail log records one answer per image/question.
        details = (tmp / 'answers_detail.txt').read_text().splitlines()
        if len(details) != 4 * len(by_id) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Incomplete or invalid DPG answer set: {key}')
        return {'source_id': key, 'score': float(score), 'qid2tuple': {str(k): v for k, v in tuples.items()},
                'qid2scores_last_crop': {str(k): v for k, v in last.items()},
                'detail': [line.split(', ', 1)[1] for line in details]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'work', 'report'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world:
        parser.error('Expected 0 <= rank < world')
    registry = load_resources()
    root = Path(os.environ.get('SCD_REVIEW_ROOT', registry['EVALUATIONS_ROOT']))
    output = root / 'composition-review-v1' / 'sd35-dpg'
    plan, folder = read_plan(root / 'sd35_medium/dpg-5090-four-seed-v1/dpg')
    tasks = {t['method']: t for t in plan['tasks'] if t['benchmark'] == 'dpg'}
    if set(tasks) != set(METHODS) or sum(t['benchmark'] == 'dpg' for t in plan['tasks']) != len(METHODS):
        raise ValueError('Expected the frozen eight-method SD3.5-M DPG plan')
    model = None
    runtime_start = time.monotonic()
    if args.mode != 'report':
        require_gpu_allocation({'mode': 'standalone', 'run_label': 'sd35-dpg'})
        settings = comp.load_settings(CONFIG_DIR / 'compositional.json', 'dpg')
        source = comp.source_identity(settings['repo'], 'dpg')
        expected = comp.benchmark_rows(settings['repo'], 'dpg')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('DPG requires CUDA in the selected SCD_DPG_PY environment')
        scorer = {'source': source['revision'], 'model': comp.asset_identity(settings, 'dpg'),
                  'env': comp.environment_identity(sys.executable, 'dpg'), 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0), 'protocol': comp.PROTOCOLS['dpg']}
        print(f"DPG device={scorer['gpu']} torch={torch.__version__} CUDA={torch.version.cuda}", flush=True)
    summaries = []
    scorer_id = None
    blocks = (COUNT + BLOCK - 1) // BLOCK
    for method_index, method in enumerate(METHODS):
        task = tasks[method]
        target = safe_target(folder, task['output'])
        render = target / 'render'
        native = target / 'scores/dpg'
        if args.mode != 'smoke' and (native / 'complete.json').is_file():
            if args.mode == 'report':
                validate_task_render(task, plan, folder)
                native_summary = comp.verify_result(native, render)
                if (native_summary['mode'], native_summary['prompts'], native_summary['images'],
                    native_summary['protocol']) != ('full-benchmark', COUNT, COUNT*4, comp.PROTOCOLS['dpg']):
                    raise ValueError(f'Unexpected native DPG coverage: {native}')
                summaries.append({'method': method, 'step': task['step'], 'source': str(native), **native_summary})
            continue
        indices = range(blocks)
        if args.mode == 'smoke':
            indices = [0]
        elif args.mode == 'work':
            indices = [i for i in indices if (method_index * blocks + i) % args.world == args.rank]
        if not indices:
            continue
        run, complete = validate_task_render(task, plan, folder)
        suite = run['suite']
        if (len(suite) != COUNT or run['sampling']['seeds'] != SEEDS
                or task['family'] != 'sd35_medium' or task['step'] != (0 if method == 'base' else 1600)):
            raise ValueError(f'Unexpected frozen DPG protocol: {method}')
        dest = output / method
        identity = {'plan': plan['plan_id'], 'run': run['run_id'], 'manifest': complete['manifest_sha256'],
                    'script': sha256(__file__), 'adapter': sha256(comp.__file__)}
        if args.mode != 'report':
            if suite != expected:
                raise ValueError('Render prompts differ from the pinned DPG source')
            rows = read_jsonl(render / 'manifest.jsonl')
            manifest = {(r['scene_id'], r['variant_id'], r['seed']): r for r in rows}
            keys = {(p['scene_id'], p['variant_id'], s) for p in suite for s in SEEDS}
            if len(rows) != COUNT * 4 or manifest.keys() != keys:
                raise ValueError('Manifest does not cover the full four-seed panel')
            identity['scorer'] = digest(scorer)
        else:
            stored = read_json(dest / 'identity.json')
            identity['scorer'] = digest(stored['scorer'])
            if stored['identity'] != identity or (scorer_id and scorer_id != identity['scorer']):
                raise ValueError(f'Changed render, code or scoring environment: {method}')
            scorer_id = identity['scorer']
        if args.mode == 'work':
            # Multiple nodes publish the same identity. Atomic creation under a short shared lock.
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / '.identity.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                record = {'identity': identity, 'scorer': scorer}
                stamp = dest / 'identity.json'
                if stamp.exists():
                    if read_json(stamp) != record:
                        raise ValueError(f'DPG inputs or runtime changed: {stamp}')
                else:
                    atomic_json(stamp, record)
        all_rows = []
        for i in indices:
            prompts = suite[i*BLOCK:(i+1)*BLOCK]
            path = dest / f'block-{i:03}.json'
            if args.mode == 'report':
                all_rows.extend(check_block(path, identity, prompts))
                continue
            if args.mode == 'work' and path.exists():
                check_block(path, identity, prompts)
                continue
            if model is None:
                official = comp.load_official(Path(settings['repo']) / 'dpg_bench/compute_dpg_bench.py')
                model = official.MPLUG(ckpt=settings['model_dir'], device='cuda')
            if args.mode == 'smoke':
                prompts = prompts[:1]
            results = []
            for p in prompts:
                images = [manifest[p['scene_id'], p['variant_id'], s] for s in SEEDS]
                results.append(score_prompt(official, model, render, p, images))
            if args.mode == 'smoke':
                print(f"DPG_SMOKE_OK method={method} id={p['source_id']} images=4 answers={len(results[0]['detail'])} elapsed={time.monotonic()-runtime_start:.1f}s", flush=True)
                return
            with output_lock(path):
                if path.exists():
                    check_block(path, identity, prompts)
                else:
                    atomic_json(path, {'identity': identity, 'rows': results, 'sha256': digest(results)})
            print(f'{method} block {i+1}/{blocks} complete', flush=True)
        if args.mode == 'report':
            summaries.append({'method': method, 'step': task['step'], 'identity': identity,
                              **comp.aggregate('dpg', all_rows)})
    if args.mode == 'report':
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'summary.json', summaries)
        for row in summaries:
            print(row['method'], row['step'], row['mean'], row['prompts'], row['images'])
        print(output / 'summary.json')


if __name__ == '__main__':
    main()
