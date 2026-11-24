#!/usr/bin/env python3
"""Two-phase appendix evaluation, importing frozen A/B pixels without changing them.

This launcher deliberately lives outside src/: the original A/B collectors hash
all src files, including unrelated modules, when resuming their frozen plans.
"""
import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from self_correct_dpo.common import digest, read_json, read_jsonl, sha256, output_lock, load_suite
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.evaluation.appendix_generation import (
    resolve_adapter, aligned_config, payload_jsonl, prompt_key, image_key,
    checked_render, checked_scores, noise_runtime)
from self_correct_dpo.evaluation.multiseed_generation import (
    load_plan as legacy_plan, package_versions, sources as legacy_sources)
from self_correct_dpo.evaluation.multiseed_history import read_source, choose_slots, import_pixels, protocol
from self_correct_dpo.evaluation.experiment_catalog import validate_method, model_spec
from self_correct_dpo.evaluation.generation_inventory import discover
from self_correct_dpo.evaluation.suite import freeze_json, render_header
from self_correct_dpo.evaluation.appendix_progress import run_stage, run_workers
from appendix_jobs import gpu_tokens

CONFIG = ROOT / 'configs/appendix-flexible.json'
FAMILIES = ('sd15', 'sdxl', 'sd35_medium', 'zimage')
OLD_PACKS = {'A': {'A1', 'A2', 'F1', 'D1', 'D2', 'F2', 'E1', 'E2', 'E3'},
             'B': {'B', 'L', 'C', 'E4', 'E5'},
             'F2': {'F2'}, 'L': {'L'}, 'C': {'C'}}


def code_identity():
    return dict(src=legacy_sources(), launcher=sha256(__file__))


def task_jobs(plans, spec, tasks):
    names = list(spec['tasks']) if 'core' in tasks else list(tasks)
    names += [t for t in tasks if t != 'core' and t not in names]
    unknown = set(names) - set(spec['tasks']) - set(OLD_PACKS)
    if unknown:
        raise ValueError(f'Unknown task: {sorted(unknown)}')
    result = {}
    for name in names:
        count = 0
        for family, plan in plans.items():
            for job in plan['jobs']:
                if name in OLD_PACKS:
                    match = job.get('old_pack') in OLD_PACKS[name]
                else:
                    rule = spec['tasks'][name]
                    match = (family == rule['family'] and job['method'] in rule['methods']
                        and job['seed'] in rule['seeds'] and job['step'] in rule['steps']
                        and ('panel' not in rule or job['panel'] == rule['panel'])
                        and ('old_pack' not in rule or job.get('old_pack') == rule['old_pack'])
                        and not (rule.get('omit_table_main') and job['step'] == 1600))
                if match:
                    result[(family, job['name'])] = (family, job)
                    count += 1
        if not count:
            raise ValueError(f'Task {name} has no prepared jobs; prepare its family first')
    return list(result.values())


def extra_jobs(spec):
    return [dict(name=f"{e['method']}-{s}-s0", method=e['method'], step=s, seed=0,
                 panel='curve', metrics=['pickscore', 'hpsv2', 'image_reward'], old_pack=None)
            for e in spec['sd15_extra'] for s in e['curve_steps']]


def offline_plan(spec, tasks):
    from self_correct_dpo.evaluation.multiseed_generation import jobs_for
    old = read_json(ROOT / 'configs/appendix-multiseed.json')
    plans = {}
    for family in FAMILIES:
        jobs = [dict(j, old_pack=j['pack']) for j in jobs_for(old, family, old['families'][family]['seeds'])]
        if family == 'sd15':
            jobs += extra_jobs(spec)
        if family == 'sd35_medium':
            jobs += [dict(name=f'{m}-{s}-s17', method=m, step=s, seed=17, panel='full', old_pack=None)
                     for m, s in [('base', 0), ('sft', 1600)]]
        plans[family] = dict(jobs=jobs)
    selected = task_jobs(plans, spec, tasks)
    return dict(target_slots=sum(spec[j['panel'] + '_count'] for _, j in selected),
                jobs=[dict(family=f, **j) for f, j in selected],
                note='Offline target slots include historical reuse; no remaining-image or runtime estimate.')


def prepare(output, legacy, spec, registry, families):
    for family in families:
        old, panels = legacy_plan(legacy / family)
        if len(panels['full']) != spec['full_count'] or len(panels['curve']) != spec['curve_count']:
            raise ValueError('Legacy full/curve panel sizes differ from the reviewed design')
        if old['seeds'][0] != (0 if family == 'sd15' else 17):
            raise ValueError('Legacy primary seed differs from the reviewed task selection')
        out = output / family
        with output_lock(output / ('prepare-' + family)):
            if (out / 'plan.json').exists():
                plan = load_plan(output, family)
                if plan['legacy_plan_id'] != old['plan_id'] or plan['spec'] != spec:
                    raise ValueError('Preparation differs; use a new flexible output root')
                continue
            experiments = [copy.deepcopy(e) for e in old['config']['experiments'] if e['family'] == family]
            jobs = [{k: copy.deepcopy(j[k]) for k in ('name', 'method', 'step', 'seed', 'panel', 'metrics')}
                    | {'old_pack': j['pack']} for j in old['jobs']]
            if family == 'sd15':
                experiments += spec['sd15_extra']
                jobs += extra_jobs(spec)
            if family == 'sd35_medium':
                figures = read_json(ROOT / 'configs/appendix-figures.json')
                experiments.append(figures['methods'][family]['sft'])
                for method, step in [('base', 0), ('sft', 1600)]:
                    jobs.append(dict(name=f'{method}-{step}-s17', method=method, step=step, seed=17,
                        panel='full', metrics=list(old['scorers']), old_pack=None))
            by_method = {e['method']: e for e in experiments}
            adapters = {}
            failures = {}
            old_adapters = {(j['method'], j['step']): j.get('adapter') for j in old['jobs']}
            training = {}
            for job in jobs:
                method, step = job['method'], job['step']
                key = (method, step)
                if method == 'base':
                    adapters[key] = None
                elif key not in adapters and key not in failures:
                    e = by_method[method]
                    run = Path(registry[e['run_resource']]) / e['run']
                    try:
                        info = read_json(run / 'training.json')
                        validate_method(info['config'], method, str(run))
                        if model_spec(info['config']['model']) != old['model'] or info['base_identity'] != old['base_identity']:
                            raise ValueError('Training model/base differs from the legacy protocol')
                        training[method] = info
                        expected = (old_adapters.get(key) or {}).get('checkpoint_id') or e['checkpoints'].get(str(step))
                        adapters[key] = resolve_adapter(run, step, e['run_id'], expected)
                    except (OSError, ValueError, KeyError) as error:
                        # A frozen adapter identity still allows old scores/pixels to be used.
                        if old_adapters.get(key):
                            adapters[key] = old_adapters[key]
                        else:
                            failures[key] = str(error)
                job.update(adapter=adapters.get(key), blocked=failures.get(key), shards=spec['shards'])
            # Retain the reviewed matched-control check for newly resolved LossAvg.
            for e in experiments:
                if not e.get('matched_reference') or e['method'] not in training:
                    continue
                info = training[e['method']]
                ref_e = by_method[e['matched_reference']]
                ref = read_json(Path(registry[ref_e['run_resource']]) / ref_e['run'] / 'training.json')
                keys = ('objective', 'noise_views', 'beta', 'comparison_budget', 'resolution', 'dtype', 'sampling',
                        'prompt_mode', 'timestep_sampling', 'reference_mode', 'learning_rate', 'lr_schedule',
                        'warmup_steps', 'weight_decay', 'max_grad_norm', 'lora_rank', 'lora_alpha')
                if (any(info.get(k) != ref.get(k) for k in ('pairs_id', 'base_identity', 'effective_batch_size_pair_views'))
                        or info['config'].get('seed') != ref['config'].get('seed')
                        or any(info['config']['training'].get(k) != ref['config']['training'].get(k) for k in keys)):
                    raise ValueError(f"LossAvg matched-control settings differ: {e['method']}")
            out.mkdir(parents=True, exist_ok=True)
            for name, rows in panels.items():
                payload_jsonl(out / f'{name}.jsonl', rows)
            freeze_json(out / 'scorers.resolved.json', read_json(legacy / family / 'scorers.resolved.json'))
            plan = {k: copy.deepcopy(old[k]) for k in ('model', 'base_identity', 'sampling', 'interpreters',
                    'render_versions', 'scorers', 'score_versions', 'seeds')}
            plan.update(version=1, family=family, legacy_root=str(legacy), legacy_plan_id=old['plan_id'],
                spec=spec, jobs=jobs, code=code_identity(), experiments=experiments,
                panels={name: sha256(out / f'{name}.jsonl') for name in panels},
                scorers_sha256=sha256(out / 'scorers.resolved.json'))
            plan['plan_id'] = digest(plan)
            freeze_json(out / 'plan.json', plan)
            print(f'{family}: {len(jobs)} jobs prepared; {len(failures)} unresolved checkpoints', flush=True)


def load_plan(output, family, check_code=True):
    out = output / family
    plan = read_json(out / 'plan.json')
    if plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('Flexible plan identity changed')
    if check_code and plan['code'] != code_identity():
        raise ValueError('Flexible collector code changed; keep the pinned checkout or prepare a new root')
    if sha256(out / 'scorers.resolved.json') != plan['scorers_sha256']:
        raise ValueError('Scorer configuration changed')
    for name, checksum in plan['panels'].items():
        if sha256(out / f'{name}.jsonl') != checksum:
            raise ValueError('Frozen panel changed')
    return plan


def compatible_source(run, plan, adapter):
    if (run['run_id'] != digest({k: v for k, v in run.items() if k != 'run_id'})
            or model_spec(run['model']) != plan['model'] or run['base_identity'] != plan['base_identity']
            or protocol(run['config']) != protocol(dict(model=plan['model'], sampling=plan['sampling']))
            or run['versions'] != plan['render_versions']):
        return False
    noise_runtime(run)  # Existing exact renderer compatibility policy remains authoritative.
    if adapter is None:
        return run['arm'] == 'base' and run['model_identity'] == plan['base_identity']
    return (run['arm'] == 'aligned' and run['model_identity'] == adapter['inventory']
            and run['config'].get('aligned') == aligned_config(plan['model'], plan['sampling'], adapter)['aligned'])


def refresh_history(output, plans, roots):
    """Only index stopped source jobs; workers later verify all recorded metadata hashes."""
    files, issues = discover(roots, exclude=output)
    sources = {f: {} for f in plans}
    adapters = {f: {j['adapter']['checkpoint_id']: j['adapter'] for j in p['jobs'] if j.get('adapter')}
                for f, p in plans.items()}
    for path in files:
        if path.name != 'run.json' or not (path.parent / 'manifest.jsonl').exists():
            continue
        try:
            run = read_json(path)
            family = run['model']['family']
            if family not in plans:
                continue
            cp = run.get('config', {}).get('aligned', {}).get('revision', 'base') if run['arm'] != 'base' else 'base'
            if cp != 'base' and cp not in adapters[family]:
                continue
            if not compatible_source(run, plans[family], adapters[family].get(cp)):
                raise ValueError('model/checkpoint/sampling/package mismatch')
            tracked = [path, path.parent / 'manifest.jsonl']
            tracked += [p for p in [path.parent / 'complete.json'] if p.exists()]
            for metric in plans[family]['scorers']:
                tracked += [p for p in (path.parent.parent / 'scores' / metric).glob('*.json*') if p.is_file()]
            sources[family][str(path.parent)] = dict(checkpoint=cp, files={str(p): sha256(p) for p in tracked})
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(path=str(path), reason=str(error)))
    value = dict(plan_ids={f: p['plan_id'] for f, p in plans.items()}, sources=sources, issues=issues)
    value['history_id'] = digest(value)
    # Append a snapshot. Already-materialized shards never depend on this changing pointer.
    with output_lock(output / 'history'):
        freeze_json(output / 'history' / (value['history_id'] + '.json'), value)
        atomic_json(output / 'history.json', value)
    return {f: len(s) for f, s in sources.items()}


def shard_root(output, family, job, index):
    return output / family / 'jobs' / job['name'] / f'shard-{index}'


def wanted_rows(output, plan, job, index):
    rows = load_suite(output / plan['family'] / (job['panel'] + '.jsonl'))
    return sorted(rows, key=lambda p: tuple(map(str, (*prompt_key(p), job['seed']))))[index::job['shards']]


def receipt_rows(output, plan, job, index, phase):
    root = shard_root(output, plan['family'], job, index)
    mark, filename = ('generated.json', 'images.jsonl') if phase == 'generate' else ('complete.json', 'scores.jsonl')
    if not (root / mark).exists():
        return None
    if read_json(root / mark) != dict(plan_id=plan['plan_id'], job=digest(job), sha256=sha256(root / filename)):
        raise ValueError(f'Changed {phase} receipt: {root}')
    rows = read_jsonl(root / filename)
    expected = {(*prompt_key(p), job['seed']) for p in wanted_rows(output, plan, job, index)}
    if len(rows) != len(expected) or {tuple(r['key']) for r in rows} != expected:
        raise ValueError('Shard coverage differs')
    if phase == 'score':
        import math
        if any(set(r['scores']) != set(job['metrics']) or any(not math.isfinite(v) for v in r['scores'].values()) for r in rows):
            raise ValueError('Missing/nonfinite final scores')
    return rows


def publish(root, plan, job, rows, phase):
    mark, filename = ('generated.json', 'images.jsonl') if phase == 'generate' else ('complete.json', 'scores.jsonl')
    payload_jsonl(root / filename, rows)
    atomic_json(root / mark, dict(plan_id=plan['plan_id'], job=digest(job), sha256=sha256(root / filename)))


def candidate_sources(output, plan, job, root):
    history = read_json(output / 'history.json')
    if history['history_id'] != digest({k: v for k, v in history.items() if k != 'history_id'}):
        raise ValueError('History snapshot changed')
    if history['plan_ids'].get(plan['family']) != plan['plan_id']:
        raise ValueError('Refresh history after preparing this family')
    cp = job['adapter']['checkpoint_id'] if job.get('adapter') else 'base'
    candidates = {}
    for render, record in history['sources'][plan['family']].items():
        if record['checkpoint'] != cp:
            continue
        if any(sha256(path) != h for path, h in record['files'].items()):
            raise ValueError(f'Historical files changed: {render}; stop the old task and refresh-history')
        source = read_source(render, plan['scorers'], plan['score_versions'])
        if any(sha256(path) != h for path, h in record['files'].items()):
            raise ValueError(f'Historical source changed while reading: {render}')
        candidates[render] = source
    # Salvage committed images from an interrupted generation on either GPU model.
    for path in sorted(root.glob('**/render/run.json')):
        if not (path.parent / 'manifest.jsonl').exists():
            continue
        source = read_source(path.parent, plan['scorers'], plan['score_versions'])
        if not compatible_source(source['run'], plan, job.get('adapter')):
            raise ValueError('Local partial render identity differs')
        candidates[str(path.parent)] = source
    return candidates


def checked_runtime(plan, devices):
    probe = ('import json,torch,importlib.metadata as m;'
             'print(json.dumps({"gpu":torch.cuda.get_device_name(0),"torch":m.version("torch"),"diffusers":m.version("diffusers")}))')
    results = [json.loads(subprocess.check_output([plan['interpreters']['render'], '-c', probe], text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=device))) for device in devices]
    if any(r != results[0] for r in results):
        raise ValueError('Use a homogeneous H100 or 5090 allocation for one launch')
    runtime = results[0]
    if not any(name in runtime['gpu'] for name in ('H100', '5090')):
        raise ValueError(f"Unsupported generation GPU: {runtime['gpu']}")
    if package_versions(plan['interpreters']['render'], list(plan['render_versions'])) != plan['render_versions']:
        raise ValueError('Rendering packages differ from the frozen protocol')
    return runtime


def generate_shard(output, plan, job, index, runtime, batch_size):
    root = shard_root(output, plan['family'], job, index)
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root):
        if receipt_rows(output, plan, job, index, 'generate') is not None:
            return
        if job.get('blocked'):
            raise ValueError(job['blocked'])
        wanted = wanted_rows(output, plan, job, index)
        candidates = candidate_sources(output, plan, job, root)
        slots, _ = choose_slots(wanted, job['seed'], job['metrics'], candidates)
        lookup = {prompt_key(p): p for p in wanted}
        records, imports, missing = {}, defaultdict(list), []
        for slot in slots:
            key = tuple(slot['key'])
            if slot['source'] is None:
                missing.append(lookup[key[:2]])
                continue
            source = candidates[slot['source']]
            pixel = source['images'][key]
            scores = {m: float(source['scores'][m][key]['score']) for m in slot['metrics']}
            records[key] = dict(key=list(key), image_sha256=pixel['image_sha256'],
                generation_gpu=source['runtime']['gpu'], cached_scores=scores,
                provenance={m: dict(source=source['render'], record_id=source['scores'][m][key]['record_id'],
                                    image_sha256=pixel['image_sha256']) for m in slot['metrics']},
                origin=slot['state'], render=source['render'], image=pixel['image'])
            if slot['state'] == 'reuse_images':
                imports[slot['source']].append(key)
        for sid, keys in imports.items():
            target = root / 'inputs' / digest([sid, keys])[:20] / 'render'
            _, pixels = import_pixels(candidates[sid], keys, target)
            for pixel in pixels:
                records[image_key(pixel)].update(render=str(target), image=pixel['image'])
        if missing:
            adapter = job['adapter']
            if adapter is None:
                raise ValueError('Base references are import-only; no matching historical pixels/scores')
            if resolve_adapter(Path(adapter['run']), job['step'], adapter['run_id'], adapter['checkpoint_id']) != adapter:
                raise ValueError('Checkpoint changed after preparation')
            sampling = dict(plan['sampling'], seeds=[job['seed']], generation_batch_size=batch_size, cpu_offload=False)
            config = aligned_config(plan['model'], sampling, adapter)
            attempt = root / 'attempts' / digest([runtime, batch_size, missing])[:24]
            attempt.mkdir(parents=True, exist_ok=True)
            freeze_json(attempt / 'config.json', config)
            freeze_json(attempt / 'runtime.json', runtime)
            payload_jsonl(attempt / 'prompts.jsonl', missing)
            render = attempt / 'render'
            if not (render / 'complete.json').exists():
                command = [plan['interpreters']['render'], '-m', 'self_correct_dpo.evaluation.render',
                    '--config', str(attempt / 'config.json'), '--prompts', str(attempt / 'prompts.jsonl'),
                    '--out', str(render), '--arm', 'aligned', '--execution', 'standalone', '--run-label', 'appendix-flex-generate']
                if render.exists():
                    command.append('--resume')
                run_stage(command, log=attempt / 'generation.log', journal=render / 'manifest.jsonl',
                          total=len(missing), label='generate only')
            run, _, pixels = checked_render(render, config, plan['base_identity'], adapter, [missing])
            if noise_runtime(run) != runtime or run['versions'] != plan['render_versions']:
                raise ValueError('Generated runtime differs')
            for key, pixel in pixels.items():
                records[key] = dict(key=list(key), image_sha256=pixel['image_sha256'], generation_gpu=runtime['gpu'],
                    cached_scores={}, provenance={}, origin='generated', render=str(render), image=pixel['image'])
        expected = {(*prompt_key(p), job['seed']) for p in wanted}
        if set(records) != expected:
            raise ValueError('Generation has missing slots')
        publish(root, plan, job, [records[k] for k in sorted(records)], 'generate')


def score_shard(output, plan, job, index):
    root = shard_root(output, plan['family'], job, index)
    with output_lock(root):
        if receipt_rows(output, plan, job, index, 'score') is not None:
            return
        rows = receipt_rows(output, plan, job, index, 'generate')
        if rows is None:
            raise ValueError('Generate first; scoring never creates images')
        scores = {tuple(r['key']): dict(r['cached_scores']) for r in rows}
        provenance = {tuple(r['key']): dict(r['provenance']) for r in rows}
        for render in sorted({r['render'] for r in rows if set(r['cached_scores']) != set(job['metrics'])}):
            selected = [r for r in rows if r['render'] == render]
            run, receipt = render_header(Path(render).parent)
            images = {image_key(p): p for p in read_jsonl(Path(render) / 'manifest.jsonl')}
            for row in selected:
                path = Path(render) / row['image']
                if not path.is_file() or sha256(path) != row['image_sha256']:
                    raise ValueError(f'Missing/changed generated pixel: {path}')
            for metric in job['metrics']:
                if all(metric in scores[tuple(r['key'])] for r in selected):
                    continue
                target = Path(render).parent / 'scores' / metric
                python = plan['interpreters']['hpsv3' if metric == 'hpsv3' else 'legacy']
                if not (target / 'complete.json').exists():
                    run_stage([python, '-m', 'self_correct_dpo.evaluation.metrics', '--metric', metric,
                        '--run', render, '--out', str(target), '--scorers-config', str(output / plan['family'] / 'scorers.resolved.json'),
                        '--execution', 'standalone', '--run-label', 'appendix-flex-score'],
                        log=target.parent / (metric + '.log'), journal=target / 'scores.jsonl', total=len(images), label='score ' + metric)
                values, identity, _ = checked_scores(Path(render).parent, metric, run, receipt, images, plan['scorers'][metric])
                if identity['identity']['versions'] != plan['score_versions'][metric]:
                    raise ValueError('Scorer package versions changed')
                for value in values:
                    key = image_key(value)
                    if key in scores:
                        scores[key][metric] = float(value['score'])
                        provenance[key][metric] = dict(source=render, record_id=value['record_id'], image_sha256=value['image_sha256'])
        result = [dict(key=r['key'], scores=scores[tuple(r['key'])], provenance=provenance[tuple(r['key'])],
                       generation_gpu=r['generation_gpu'], image_sha256=r['image_sha256']) for r in rows]
        if any(set(r['scores']) != set(job['metrics']) for r in result):
            raise ValueError('Scoring did not cover all metrics')
        publish(root, plan, job, result, 'score')


def launch(args, plans, selected):
    # Barrier for the entire requested scope, before scoring any shard or starting GPUs.
    if args.action == 'score':
        missing = [f'{f}/{j["name"]}/{i}' for f, j in selected for i in range(j['shards'])
                   if receipt_rows(args.output, plans[f], j, i, 'generate') is None]
        if missing:
            raise ValueError(f'Generation barrier: {len(missing)} shards not ready; first: {missing[:5]}')
    todo = [(f, j, i) for f, j in selected for i in range(j['shards'])
            if receipt_rows(args.output, plans[f], j, i, args.action) is None]
    if not todo:
        print('Selected phase is already complete')
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    devices = gpu_tokens(args.gpus or env.get('SCD_EVAL_GPUS'), env, plans[todo[0][0]]['interpreters']['render'])
    if len(devices) not in (2, 4, 8):
        raise ValueError('Use 2/4/8 allocated devices')
    runtimes = {}
    if args.action == 'generate':
        for family in {f for f, _, _ in todo}:
            runtimes[family] = checked_runtime(plans[family], devices)
    session = args.output / 'sessions' / digest([args.action, args.tasks, os.getpid(), __import__('time').time_ns()])[:20]
    session.mkdir(parents=True)
    atomic_json(session / 'session.json', dict(phase=args.action, tasks=args.tasks, devices=devices, runtimes=runtimes))
    for start in range(0, len(todo), len(devices)):
        commands, progress = [], {}
        for gpu, (family, job, index) in zip(devices, todo[start:start + len(devices)]):
            label = f'{family}-{job["name"]}-{index}'
            cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(args.output),
                   '--legacy-root', plans[family]['legacy_root'], '--config', str(args.config.resolve()), '--family', family,
                   '--job', job['name'], '--index', str(index), '--phase', args.action]
            if args.action == 'generate':
                gpu_key = 'h100' if 'H100' in runtimes[family]['gpu'] else '5090'
                batch = args.batch_size or plans[family]['spec']['batch_sizes'][gpu_key][family]
                cmd += ['--runtime', json.dumps(runtimes[family]), '--batch-size', str(batch)]
            commands.append((gpu, label, cmd))
            progress[label] = shard_root(args.output, family, job, index) / 'progress.json'
        run_workers(commands, session, env, progress)


def status(output, plans, selected):
    result = []
    for family, job in selected:
        sizes = [len(wanted_rows(output, plans[family], job, i)) for i in range(job['shards'])]
        generated = [receipt_rows(output, plans[family], job, i, 'generate') for i in range(job['shards'])]
        scored = [receipt_rows(output, plans[family], job, i, 'score') for i in range(job['shards'])]
        result.append(dict(family=family, job=job['name'], blocked=job.get('blocked'), target_slots=sum(sizes),
            generated_shards=sum(r is not None for r in generated), scored_shards=sum(r is not None for r in scored),
            shards=job['shards'], generation_gpus=sorted({r['generation_gpu'] for rows in generated if rows for r in rows}),
            materialized_origins={origin: sum(r['origin'] == origin for rows in generated if rows for r in rows)
                                  for origin in ('reuse_scores', 'reuse_images', 'generated')}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'refresh-history', 'plan', 'status', 'generate', 'score', 'report', 'worker'])
    p.add_argument('--output', type=Path)
    p.add_argument('--legacy-root', type=Path)
    p.add_argument('--resources', type=Path)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--family', choices=FAMILIES)
    p.add_argument('--tasks', nargs='+', default=['core'])
    p.add_argument('--history-root', action='append', type=Path, default=[])
    p.add_argument('--gpus')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--job')
    p.add_argument('--index', type=int)
    p.add_argument('--phase', choices=['generate', 'score'])
    p.add_argument('--runtime')
    args = p.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        p.error('batch-size must be positive')
    if args.resources:
        os.environ['SCD_RESOURCES'] = str(args.resources.resolve())
    try:
        registry = load_resources(args.resources)
        args.output = (args.output or Path(os.environ.get('MVPO_FLEX_ROOT') or registry['MVPO_FLEX_ROOT'])).expanduser().resolve()
        legacy = (args.legacy_root or Path(os.environ.get('MVPO_FLEX_LEGACY_ROOT') or registry['MVPO_FLEX_LEGACY_ROOT'])).expanduser().resolve()
        if args.output == legacy or args.output in legacy.parents or legacy in args.output.parents:
            raise ValueError('Use a separate flexible root; never overwrite/nest inside the legacy A/B root')
        spec = read_json(args.config)
        if args.action == 'plan' and not any((args.output / f / 'plan.json').exists() for f in FAMILIES):
            print(json.dumps(offline_plan(spec, args.tasks), indent=2))
            return
        families = [args.family] if args.family else list(FAMILIES)
        if args.action == 'prepare':
            prepare(args.output, legacy, spec, registry, families)
        plans = {f: load_plan(args.output, f, check_code=args.action not in ('status', 'report', 'plan'))
                 for f in FAMILIES if (args.output / f / 'plan.json').exists()}
        if not plans:
            raise ValueError('Prepare the flexible plans first')
        if any(plan['spec'] != spec for plan in plans.values()):
            raise ValueError('Configuration differs from the frozen flexible plan')
        if args.action in ('prepare', 'refresh-history'):
            roots = args.history_root or [Path(registry[k]) for k in ('EVALUATIONS_ROOT', 'RUNS_ROOT', 'FLOW_RUNS_ROOT')]
            if legacy not in roots:
                roots.append(legacy)
            print(json.dumps(refresh_history(args.output, plans, roots), indent=2))
        elif args.action == 'worker':
            if args.family not in plans or args.index is None or args.phase is None:
                p.error('worker requires family/job/index/phase')
            plan = plans[args.family]
            job = next(j for j in plan['jobs'] if j['name'] == args.job)
            if not 0 <= args.index < job['shards']:
                p.error('invalid shard index')
            if args.phase == 'generate':
                if not args.runtime or not args.batch_size:
                    p.error('generation worker requires runtime and batch-size')
                generate_shard(args.output, plan, job, args.index, json.loads(args.runtime), args.batch_size)
            else:
                score_shard(args.output, plan, job, args.index)
        else:
            selected = task_jobs(plans, spec, args.tasks)
            if args.family:
                selected = [(f, j) for f, j in selected if f == args.family]
                if not selected:
                    raise ValueError('No selected jobs for this family')
            if args.action in ('status', 'plan'):
                print(json.dumps(status(args.output, plans, selected), indent=2))
            elif args.action == 'report':
                from appendix_flexible_report import report
                report(args.output, plans, selected)
            else:
                # Prevent two coordinators with the same selection; shard locks protect overlaps.
                with output_lock(args.output / ('launch-' + digest(sorted(args.tasks))[:16])):
                    launch(args, plans, selected)
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration, subprocess.CalledProcessError) as error:
        p.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
