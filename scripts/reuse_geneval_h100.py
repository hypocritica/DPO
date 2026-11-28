"""Rescore frozen GenEval PNGs with MMCV's PyTorch deformable attention on CUDA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.common import digest, output_lock, read_json, sha256
from self_correct_dpo.evaluation import compositional as comp
from self_correct_dpo.evaluation.suite import read_plan, safe_target, validate_task_render
from self_correct_dpo.resources import CONFIG_DIR, load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json

SCRIPT = Path(__file__).resolve()
SCORE_NAME = 'geneval-pytorch-h100-v1'
PLANS = ('sdxl/paper-2000-5090-v1/main/geneval', 'sd35_medium/compact-h100-v3/core')


def repair_identity():
    return {'backend': 'mmcv-1.7.2-pytorch-on-cuda-v1', 'script_sha256': sha256(SCRIPT)}


def install_backend():
    """Use the upstream reference operator, keeping values and sampling on GPU.

    MMCV 1.7.2's native CUDA kernel can printf an unsupported-architecture error
    and return its zero-initialized output without raising a Python exception.
    Patching the shared Function class also covers aliases imported by MMDet.
    """
    import mmcv
    from mmcv.ops import multi_scale_deform_attn as ops
    if mmcv.__version__ != '1.7.2':
        raise RuntimeError('This repair is validated against mmcv-full==1.7.2')
    calls = {'count': 0}

    def reference_apply(value, shapes, starts, locations, weights, im2col_step):
        calls['count'] += 1
        return ops.multi_scale_deformable_attn_pytorch(
            value, shapes.cpu(), locations.type_as(value), weights.type_as(value))

    ops.MultiScaleDeformableAttnFunction.apply = staticmethod(reference_apply)
    return calls


def fixture(device):
    import torch
    # Two feature levels, with analytically known bilinear interpolation.
    value = torch.tensor([1., 2., 3., 4., 10., 11., 12., 13.], device=device).view(1, 8, 1, 1)
    shapes = torch.tensor([[2, 2], [2, 2]], dtype=torch.long, device=device)
    starts = torch.tensor([0, 4], dtype=torch.long, device=device)
    locations = torch.tensor([.25, .25, .75, .75, .25, .25, .75, .75], device=device).view(1, 1, 1, 2, 2, 2)
    weights = torch.tensor([.1, .2, .3, .4], device=device).view(1, 1, 1, 2, 2)
    return value, shapes, starts, locations, weights, 64


def operator_check(device, patched):
    import torch
    from mmcv.ops import multi_scale_deform_attn as ops
    counter = install_backend() if patched else None
    with torch.no_grad():
        output = ops.MultiScaleDeformableAttnFunction.apply(*fixture(device))
        expected = torch.full_like(output, 9.1)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
        # Exercise realistic head width, non-square levels, boundary sampling,
        # and multiple batches/queries against the upstream CPU reference.
        gen = torch.Generator().manual_seed(17029)
        value = torch.randn(2, 14, 2, 32, generator=gen)
        shapes = torch.tensor([[3, 4], [1, 2]], dtype=torch.long)
        starts = torch.tensor([0, 12], dtype=torch.long)
        locations = torch.rand(2, 7, 2, 2, 4, 2, generator=gen) * 1.4 - .2
        weights = torch.rand(2, 7, 2, 2, 4, generator=gen)
        weights /= weights.sum(dim=(-1, -2), keepdim=True)
        expected = ops.multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
        actual = ops.MultiScaleDeformableAttnFunction.apply(
            *[x.to(device) for x in (value, shapes, starts, locations, weights)], 64)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
    if counter is not None and counter['count'] != 2:
        raise RuntimeError('Reference operator was not called by the active dispatch')
    return {'backend': 'pytorch' if patched else 'native', 'known_answer': output.item(),
            'max_abs_error': (actual.cpu() - expected).abs().max().item()}


def probe(native=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run the probe on an allocated GPU')
    result = operator_check('cuda', patched=not native)
    print(('NATIVE_MSDA_OK ' if native else 'GENEVAL_BACKEND_OK ') + json.dumps(
        dict(result, gpu=torch.cuda.get_device_name(0), torch=torch.__version__)), flush=True)


def jobs():
    registry = load_resources()
    root = Path(os.environ.get('SCD_REVIEW_ROOT', registry['EVALUATIONS_ROOT']))
    rows = []
    for relative, expected in zip(PLANS, (10, 8)):
        plan, folder = read_plan(root / relative)
        selected = [t for t in plan['tasks'] if t['benchmark'] == 'geneval']
        if len(selected) != expected or len({t['method'] for t in selected}) != expected:
            raise ValueError(f'Unexpected frozen GenEval panel: {folder}')
        rows += [(plan, folder, t) for t in selected]
    return root, rows


def score_one(index):
    _, rows = jobs()
    plan, folder, task = rows[index]
    validate_task_render(task, plan, folder)
    target = safe_target(folder, task['output'])
    config = os.environ.get('SCD_COMPOSITIONAL_CONFIG', str(CONFIG_DIR / 'compositional.json'))
    settings = comp.load_settings(config, 'geneval')
    settings.update(python=str(Path(sys.executable).resolve()), runtime_patch=repair_identity())

    def worker_command(command):
        request = command[-1]
        subprocess.run([sys.executable, str(SCRIPT), 'worker', '--request', request], check=True)

    output = target / 'scores' / SCORE_NAME
    with output_lock(str(output) + '.compositional'):
        result = comp.score(target / 'render', output, 'geneval', settings,
                            execution={'mode': 'standalone', 'run_label': SCORE_NAME},
                            command_runner=worker_command)
    print('GENEVAL_VERSION_OK', task['family'], task['method'], result['mean'], flush=True)


def worker(request_path):
    request = read_json(request_path)
    if request['benchmark'] != 'geneval' or request['settings'].get('runtime_patch') != repair_identity():
        raise ValueError('Worker code/backend does not match frozen scoring identity')
    probe()
    calls = install_backend()
    out = Path(request_path).parent
    exported = read_json(out / 'export.json')
    pending = any(not (out / 'raw' / f"{item['index']:05d}").exists() for item in exported['records'])
    comp.worker(request_path)
    if pending and calls['count'] == 0:
        raise RuntimeError('Detector bypassed the repaired deformable-attention operator')
    atomic_json(out / 'backend-check.json', dict(repair_identity(), detector_operator_calls=calls['count']))


def report():
    root, rows = jobs()
    result = []
    for plan, folder, task in rows:
        validate_task_render(task, plan, folder)
        target = safe_target(folder, task['output'])
        out = target / 'scores' / SCORE_NAME
        request = read_json(out / 'evaluation.json')
        if request['settings'].get('runtime_patch') != repair_identity():
            raise ValueError('Report contains a different scoring backend/code')
        summary = comp.verify_result(out, target / 'render')
        if (summary['prompts'], summary['images'], summary['mode']) != (553, 2212, 'full-benchmark'):
            raise ValueError('Incomplete GenEval coverage')
        result.append(dict(family=task['family'], method=task['method'], step=task['step'], **summary))
    output = root / 'composition-review-v1' / SCORE_NAME
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'summary.json', result)
    fields = ['family', 'method', 'step', 'mean', 'prompts', 'images']
    categories = list(result[0]['by_category'])
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + categories)
        writer.writeheader()
        writer.writerows({**{k: row[k] for k in fields}, **row['by_category']} for row in result)
    print('GENEVAL_ALL_OK', output / 'summary.csv', flush=True)


def gpu_tokens():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    tokens = os.environ.get('SCD_EVAL_GPUS', visible or '0,1,2,3').split(',')
    if len(tokens) != 4 or len(set(tokens)) != 4 or any(not t or t == '-1' or any(c.isspace() for c in t) for t in tokens):
        raise ValueError('Exactly four distinct allocated GPU tokens are required')
    if visible and not set(tokens) <= set(visible.split(',')):
        raise ValueError('Selected GPUs are outside CUDA_VISIBLE_DEVICES')
    return tokens


def run():
    root, rows = jobs()
    tokens = gpu_tokens()
    logs = root / 'composition-review-v1' / SCORE_NAME / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    def slot(slot_id):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=tokens[slot_id])
        log_path = logs / f'gpu-{slot_id}.log'
        with log_path.open('a', buffering=1) as log:
            # Separate process: an unsupported native kernel must not poison the
            # CUDA context used for the reference implementation or scoring.
            try:
                native = subprocess.run([sys.executable, str(SCRIPT), 'probe-native'],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
                log.write(native.stdout)
                print('NATIVE_MSDA_STATUS', slot_id, native.returncode, log_path, flush=True)
            except subprocess.TimeoutExpired:
                log.write('Native operator probe timed out; using reference backend.\n')
            subprocess.run([sys.executable, str(SCRIPT), 'probe'], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            print('GENEVAL_BACKEND_OK', 'slot', slot_id, log_path, flush=True)
            for index in range(slot_id, len(rows), 4):
                task = rows[index][2]
                print('GENEVAL_START', slot_id, task['family'], task['method'], flush=True)
                subprocess.run([sys.executable, str(SCRIPT), 'score', '--index', str(index)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                print('GENEVAL_DONE', slot_id, task['family'], task['method'], flush=True)

    with output_lock(logs.parent / 'run'):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(slot, range(4)))
        report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'report', 'probe', 'probe-native', 'score', 'worker'])
    parser.add_argument('--index', type=int)
    parser.add_argument('--request')
    args = parser.parse_args()
    if args.mode == 'run': run()
    elif args.mode == 'report': report()
    elif args.mode.startswith('probe'): probe(native=args.mode == 'probe-native')
    elif args.mode == 'score':
        if args.index is None or not 0 <= args.index < 18: parser.error('--index must be 0..17')
        score_one(args.index)
    elif args.mode == 'worker':
        if not args.request: parser.error('--request required')
        worker(args.request)


if __name__ == '__main__':
    main()
