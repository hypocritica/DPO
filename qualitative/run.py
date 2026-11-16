#!/usr/bin/env python3
"""SLAI paired qualitative rendering. No downloads or resource submission."""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import (artifact_identity, digest, output_lock,
    read_json, require_gpu_allocation, sha256)
from self_correct_dpo.evaluation.checkpoint_catalog import resolve_checkpoint
from self_correct_dpo.evaluation.render import (FAMILIES, effective_sampling,
    load_pipeline, validate_config)

METHODS = ('Base', 'DPO', 'MVPO')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen inputs changed; use a new output directory: {path}')
    else:
        atomic_json(path, value)


def prompts(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if [r['id'] for r in rows] != [f'P{i:02}' for i in range(1, 31)]:
        raise ValueError('Expected exactly P01–P30 in order')
    if any(not isinstance(r['prompt'], str) or not r['prompt'].strip() for r in rows):
        raise ValueError('Empty or non-string prompt')
    return rows


def source_identity():
    paths = [*sorted((ROOT / 'src/self_correct_dpo').rglob('*.py')), Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def resolve_spec(spec):
    """Import Z-Image identities only from the existing frozen curve inventory."""
    spec = json.loads(json.dumps(spec))
    if 'prepared' in spec:
        from self_correct_dpo.evaluation.curve_prepare import load_prepared
        folder = Path(spec['prepared'])
        receipt = load_prepared(folder)
        catalog = read_json(folder / 'catalog.json')
        spec['model'] = catalog['models']['zimage']
        for method, selection in spec['methods'].items():
            matches = [e for e in catalog['experiments'] if e['method'] == selection['source_method']]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous Z-Image run: {method}')
            e = matches[0]
            run = (Path(catalog['runs_roots'][e['root']]) / e['run']).resolve()
            source = next(x for x in receipt['training_sources'] if Path(x['path']).resolve() == run)
            ckpt = next((x for x in source['checkpoints'] if x['step'] == selection['step']), None)
            if ckpt is None:
                # Let normal missing-checkpoint reporting preserve Base / other methods.
                selection['unavailable'] = f'Prepared inventory lacks {method} step {selection["step"]}'
                continue
            training = read_json(run / 'training.json')
            if sha256(run / 'training.json') != source['training_sha256']:
                raise ValueError(f'Training identity changed: {run}')
            selection.update(run=str(run), run_id=training['run_id'], checkpoint_id=ckpt['checkpoint_id'])
    return spec


def prepare(args):
    config = read_json(args.config)
    rows = prompts(args.prompts)
    root = Path(args.out).resolve() / args.backbone
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root / 'plan'):
        spec = config['backbones'][args.backbone]
        if spec['sampling']['seeds'] != [0, 1, 2, 3]:
            raise ValueError('This panel fixes seeds to [0, 1, 2, 3]')
        try:
            spec = resolve_spec(spec)
        except FileNotFoundError as exc:
            spec = json.loads(json.dumps(spec))
            for m in spec['methods'].values():
                m['unavailable'] = str(exc)
        model, sampling = spec['model'], effective_sampling({'model': spec['model'], 'sampling': spec['sampling']})
        available, missing = {}, {}
        for method in METHODS:
            cfg = {'model': model, 'sampling': sampling}
            try:
                identity = {'checkpoint_id': 'base', 'step': 0}
                if not (Path(model['path']) / 'model_index.json').is_file():
                    raise FileNotFoundError(f'Missing base pipeline: {model["path"]}')
                if method != 'Base':
                    selected = spec['methods'][method]
                    if selected.get('unavailable'):
                        raise FileNotFoundError(selected['unavailable'])
                    trained, adapter, identity = resolve_checkpoint(selected['run'], selected['step'])
                    for key in ('family', 'revision'):
                        if trained.get(key) != model.get(key):
                            raise ValueError(f'{method}: training/base {key} differs')
                    if identity['run_id'] != selected['run_id'] or identity['checkpoint_id'] != selected['checkpoint_id']:
                        raise ValueError(f'{method}: requested checkpoint identity differs')
                    cfg['aligned'] = {'kind': 'peft', 'path': str(adapter), 'revision': identity['checkpoint_id']}
                    identity['source_method'] = selected['source_method']
                validate_config(cfg, 'base' if method == 'Base' else 'aligned')
                available[method] = {'config': cfg, 'identity': identity}
            except FileNotFoundError as exc:
                missing[method] = str(exc)
        plan = {'backbone': args.backbone, 'spec': spec, 'prompts': rows, 'methods': available,
                'missing': missing, 'sources': source_identity(),
                'model_inventory': artifact_identity(model['path']) if available else None,
                'noise_policy': 'CPU standard-normal raw latents cached once per case; cloned per method',
                'workers': spec['workers']}
        plan['plan_id'] = digest(plan)
        freeze(root / 'run_config.json', plan)
        freeze(root / 'prompt_list.json', rows)
        prompt_copy = root / 'MVPO_qualitative_prompts.jsonl'
        if not prompt_copy.exists():
            prompt_copy.write_bytes(Path(args.prompts).read_bytes())
        print(json.dumps({'backbone': args.backbone, 'available': list(available), 'missing': missing,
                          'planned_images': 120 * len(available)}, indent=2))
    return plan


def batches(plan, phase, rank, workers):
    rows = plan['prompts']
    seeds = plan['spec']['sampling']['seeds']
    smoke = [(p, seeds[0]) for p in rows if p['id'] in ('P02', 'P26')]
    if phase == 'smoke':
        return [smoke] if rank == 0 else []
    # Smoke batch is immutable; full batches exclude it. Resume never repacks missing items.
    rest = [(p, s) for p in rows for s in seeds if (p['id'], s) not in {('P02', seeds[0]), ('P26', seeds[0])}]
    size = plan['spec']['sampling']['generation_batch_size']
    return [rest[i:i + size] for j, i in enumerate(range(0, len(rest), size)) if j % workers == rank]


def image_path(root, method, prompt_id, seed):
    return root / method / prompt_id / f'seed_{seed}.png'


def valid_image(path, plan_id, size=None):
    from PIL import Image
    sidecar = path.with_suffix('.json')
    if not path.exists() or not sidecar.exists():
        return False
    record = read_json(sidecar)
    if record['plan_id'] != plan_id:
        raise ValueError(f'Image belongs to another plan: {path}')
    if record['image_sha256'] != sha256(path):
        return False
    try:
        with Image.open(path) as im:
            im.load()
            return im.format == 'PNG' and (size is None or im.size == size)
    except (OSError, ValueError):
        return False


def noise_for(pipe, root, row, seed, sampling, torch):
    component = getattr(pipe, 'unet', None)
    if component is None:
        component = pipe.transformer
    shape = (1, component.config.in_channels,
             sampling['height'] // pipe.vae_scale_factor,
             sampling['width'] // pipe.vae_scale_factor)
    path = root / 'noise' / row['id'] / f'seed_{seed}.pt'
    dtype = getattr(torch, sampling.get('latent_dtype', sampling['dtype']))
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=True)
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f'Noise shape/dtype changed: {path}')
        if sha256(path) != read_json(path.with_suffix('.json'))['sha256']:
            raise ValueError(f'Corrupt noise cache: {path}')
    else:
        value = torch.randn(shape, generator=torch.Generator('cpu').manual_seed(seed), dtype=dtype)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        torch.save(value, tmp)
        os.replace(tmp, path)
        atomic_json(path.with_suffix('.json'), {'sha256': sha256(path), 'shape': list(shape), 'dtype': str(dtype)})
    noise_hash = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return value, {'initial_noise_sha256': noise_hash, 'noise_path': str(path), 'latent_shape': list(shape)}


def worker(args):
    import torch
    require_gpu_allocation({'mode': 'standalone', 'run_label': 'SLAI-qualitative-30'})
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required in an allocated SLAI container')
    root = Path(args.out).resolve() / args.backbone
    plan = read_json(root / 'run_config.json')
    if plan['sources'] != source_identity():
        raise ValueError('Source changed after planning')
    if args.workers != plan['workers']:
        raise ValueError('Worker count differs from frozen plan')
    sampling = plan['spec']['sampling']
    size = (sampling['width'], sampling['height'])
    with output_lock(root / f'worker-{args.rank}'):
        packages = {n: importlib.metadata.version(n) for n in ('torch', 'diffusers', 'transformers', 'peft', 'Pillow')}
        freeze(root / f'environment-{args.rank}.json', {'packages': packages, 'gpu': torch.cuda.get_device_name(0),
               'cuda': torch.version.cuda, 'python': sys.version, 'executable': sys.executable})
        todo = batches(plan, args.phase, args.rank, args.workers)
        for method, entry in plan['methods'].items():
            needed = [batch for batch in todo if not all(valid_image(image_path(root, method, p['id'], s), plan['plan_id'], size) for p, s in batch)]
            if not needed:
                continue
            # Fresh base per method, reused across batches. Never fuse or stack adapters.
            pipe = load_pipeline(entry['config'], 'base' if method == 'Base' else 'aligned', torch)
            if 'latents' not in inspect.signature(pipe.__call__).parameters:
                raise RuntimeError('Pipeline must accept explicit paired latents')
            scheduler_class, scheduler_config = type(pipe.scheduler), dict(pipe.scheduler.config)
            call_defaults = {k: repr(v.default) for k, v in inspect.signature(pipe.__call__).parameters.items()
                             if v.default is not inspect.Parameter.empty}
            freeze(root / f'pipeline-defaults-{method}-rank{args.rank}.json', call_defaults)
            freeze(root / f'scheduler-{method}-rank{args.rank}.json', json.loads(json.dumps(scheduler_config)))
            component = getattr(pipe, FAMILIES[args.backbone][2])
            active = getattr(component, 'active_adapters', [])
            if callable(active):
                active = active()
            if method == 'Base' and getattr(component, 'peft_config', None):
                raise RuntimeError('Base has PEFT adapters')
            if method != 'Base' and len(active) != 1:
                raise RuntimeError(f'Expected exactly one active adapter, found {active}')
            print(f'{method}: active adapters={active}; sampling={sampling}', flush=True)
            for batch in needed:
                noise = [noise_for(pipe, root, p, s, sampling, torch) for p, s in batch]
                # Independent RNG per sample/method; global RNG reset as well.
                torch.manual_seed(batch[0][1]); torch.cuda.manual_seed_all(batch[0][1])
                pipe.scheduler = scheduler_class.from_config(scheduler_config)
                kwargs = {k: sampling[k] for k in ('height', 'width', 'num_inference_steps', 'guidance_scale')}
                if 'max_sequence_length' in sampling:
                    kwargs['max_sequence_length'] = sampling['max_sequence_length']
                kwargs.update(prompt=[p['prompt'] for p, s in batch],
                              negative_prompt=[sampling['negative_prompt']] * len(batch),
                              latents=torch.cat([v for v, meta in noise]).to('cuda').clone(),
                              generator=[torch.Generator('cuda').manual_seed(s) for p, s in batch])
                with torch.inference_mode():
                    images = pipe(**kwargs).images
                if len(images) != len(batch):
                    raise ValueError('Wrong pipeline output count')
                for (p, seed), im, (_, meta) in zip(batch, images, noise):
                    path = image_path(root, method, p['id'], seed)
                    if valid_image(path, plan['plan_id'], size):
                        continue
                    if im.size != size:
                        raise ValueError('Wrong pipeline output resolution')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix('.tmp.png'); im.save(tmp, dpi=(300, 300)); os.replace(tmp, path)
                    record = dict(plan_id=plan['plan_id'], backbone=args.backbone, method=method,
                                  checkpoint=entry['identity'], adapter=entry['config'].get('aligned'),
                                  model=entry['config']['model'], prompt_id=p['id'], prompt=p['prompt'], seed=seed,
                                  sampling=sampling, scheduler_config=scheduler_config, pipeline_defaults=call_defaults, active_adapters=list(active),
                                  image_path=str(path), image_sha256=sha256(path), **meta)
                    atomic_json(path.with_suffix('.json'), record)
                    print(f'saved {path}', flush=True)
            del component, pipe
            gc.collect(); torch.cuda.empty_cache()


def audit(root, smoke=False):
    plan = read_json(root / 'run_config.json')
    records, missing = [], []
    for method in plan['methods']:
        for p in plan['prompts']:
            for seed in plan['spec']['sampling']['seeds']:
                if smoke and (p['id'] not in ('P02', 'P26') or seed != plan['spec']['sampling']['seeds'][0]):
                    continue
                path = image_path(root, method, p['id'], seed)
                sampling = plan['spec']['sampling']
                if valid_image(path, plan['plan_id'], (sampling['width'], sampling['height'])):
                    records.append(read_json(path.with_suffix('.json')))
                else:
                    missing.append(str(path))
    grouped = {}
    for row in records:
        key = row['prompt_id'], row['seed']
        pair = {k: row[k] for k in ('prompt', 'seed', 'sampling', 'initial_noise_sha256', 'scheduler_config', 'pipeline_defaults', 'model')}
        if key in grouped and grouped[key] != pair:
            raise ValueError(f'Pairing mismatch: {key}')
        grouped[key] = pair
    tmp = root / 'manifest.jsonl.tmp'
    # A smoke audit must not erase a previously assembled full manifest.
    if not smoke:
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        os.replace(tmp, root / 'manifest.jsonl')
    report = {'generated': len(records), 'expected_available': len(records) + len(missing),
              'missing_images': missing, 'unavailable_methods': plan['missing'], 'plan_id': plan['plan_id']}
    atomic_json(root / ('smoke_report.json' if smoke else 'report.json'), report)
    print(json.dumps(report, indent=2))
    return not missing and bool(records)


def launch(args):
    root = Path(args.out).resolve() / args.backbone
    plan = prepare(args)
    tokens = args.gpus.split(',')
    if len(tokens) != plan['workers'] or len(set(tokens)) != len(tokens) or any(not t.strip() or t.strip() == '-1' for t in tokens):
        raise ValueError(f'Expected {plan["workers"]} distinct allocated GPU tokens')
    with output_lock(root / 'launcher'):
        (root / 'logs').mkdir(exist_ok=True)
        stamp = time.strftime('%Y%m%dT%H%M%S') + f'-{os.getpid()}'
        revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        freeze(root / 'logs' / f'command-{stamp}.json', {'argv': sys.argv, 'command': shlex.join([sys.executable, *sys.argv]),
               'cwd': os.getcwd(), 'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
               'source_identity': plan['sources']})
        if args.phase == 'full' and not audit(root, smoke=True):
            raise RuntimeError('Run smoke before full generation')
        processes = []
        try:
            for rank, token in enumerate(tokens):
                if args.phase == 'smoke' and rank:
                    continue
                cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--out', args.out,
                       '--backbone', args.backbone, '--phase', args.phase, '--rank', str(rank), '--workers', str(len(tokens))]
                log = open(root / 'logs' / f'{args.phase}-{stamp}-rank{rank}.log', 'w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=token, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
                processes.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
            codes = [p.wait() for p, log in processes]
            if any(codes):
                raise RuntimeError(f'Worker failures {codes}; see {root / "logs"}')
        finally:
            for p, log in processes:
                if p.poll() is None:
                    p.terminate(); p.wait()
                log.close()
        if not audit(root, smoke=args.phase == 'smoke'):
            raise RuntimeError('Output audit incomplete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'launch', 'worker', 'audit'])
    parser.add_argument('--config', default=str(ROOT / 'qualitative/config.json'))
    parser.add_argument('--prompts', default=str(ROOT / 'qualitative/MVPO_qualitative_prompts.jsonl'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--backbone', choices=list(FAMILIES), required=True)
    parser.add_argument('--phase', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'plan': prepare(args)
    elif args.action == 'launch': launch(args)
    elif args.action == 'worker': worker(args)
    elif not audit(Path(args.out).resolve() / args.backbone): sys.exit(1)

if __name__ == '__main__':
    main()
