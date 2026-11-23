"""Freeze, execute and collect the SD3.5-M PartiPrompts comparisons."""
import argparse
from collections import defaultdict
import fcntl
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from self_correct_dpo.common import (
    artifact_identity, digest, load_suite, model_artifact_identity, output_lock,
    read_json, read_jsonl, require_gpu_allocation, sha256, verify_images,
)
from self_correct_dpo.config import load_config
from self_correct_dpo.evaluation.checkpoint_catalog import _manifest
from self_correct_dpo.evaluation.experiment_catalog import model_spec, validate_method
from self_correct_dpo.evaluation import render as native_render
from self_correct_dpo.evaluation.render import effective_sampling, select_shard
from self_correct_dpo.evaluation.runner import run_command
from self_correct_dpo.evaluation.scorers import scorer_identity
from self_correct_dpo.evaluation.suite import freeze_json, render_header, safe_target, score_summary
from self_correct_dpo.resources import CONFIG_DIR, load_resources
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.runtime.storage_compat import compatible_sources
from self_correct_dpo.runtime.zimage_precision_compat import LEGACY_RENDER_SHA256, BF16_RENDER_SHA256

FAMILY = 'sd35_medium'
PANEL = 'partiprompts'
METHODS = {
    'dpo': 'Flow-DPO K=1', 'lossavg_k4': 'LossAvg K=4',
    'global_k4': 'Global MVPO K=4', 'stagewise_k4': 'Stage-wise MVPO K=4',
}
PAIR = ('dpo', 'stagewise_k4')
METRICS = ('pickscore', 'hpsv3', 'image_reward', 'aesthetic')
STEPS = (800, 1600, 2400, 3200)
SEEDS = (17, 23, 42)
GROUPS = ('direct', 'seed23', 'seed42', 'step800', 'step2400', 'step3200')
SECTIONS = ('full', 'direct', 'seeds', 'curve')
FULL_COUNT, SUBSET_COUNT = 1632, 512
SUBSET_SALT = 'MVPO appendix SD3.5-M PartiPrompts subset v1'


def payload_jsonl(path, rows):
    content = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    temporary = path.with_name(f'.{path.name}-{os.getpid()}.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def prompt_key(row):
    return row['scene_id'], row['variant_id']


def image_key(row):
    return (*prompt_key(row), row['seed'])


def fixed_subset(rows):
    selected = sorted(rows, key=lambda row: digest([SUBSET_SALT, *prompt_key(row)]))[:SUBSET_COUNT]
    keys = {prompt_key(row) for row in selected}
    return [row for row in rows if prompt_key(row) in keys]


def resolve_adapter(run, step, expected_run, expected_checkpoint):
    info = read_json(run / 'training.json')
    identity_keys = ('config', 'pairs_id', 'base_identity', 'code', 'validation_pairs_id',
                     'monitor_inputs', 'diagnostic_suite_sha256', 'parent')
    recorded_id = digest({k: info[k] for k in identity_keys if k in info})
    if recorded_id != expected_run or info['run_id'] != expected_run:
        raise ValueError(f'Training run ID mismatch: {run}')
    checkpoint, manifest = _manifest(run, step, expected_run, verify_files=False)
    if expected_checkpoint is not None and manifest['checkpoint_id'] != expected_checkpoint:
        raise ValueError(f'Checkpoint ID mismatch: {checkpoint}')
    adapter = checkpoint / 'adapter'
    registered = {name: value for name, value in manifest['files'].items() if name.startswith('adapter/')}
    actual = {p.relative_to(checkpoint).as_posix(): sha256(p) for p in adapter.rglob('*') if p.is_file()}
    if actual != registered or 'adapter/adapter_model.safetensors' not in registered:
        raise ValueError(f'Adapter files do not match the checkpoint hashes: {checkpoint}')
    return {'run': str(run), 'run_id': expected_run, 'step': step, 'path': str(adapter),
            'checkpoint_id': manifest['checkpoint_id'], 'files': registered,
            'checkpoint_sha256': sha256(checkpoint / 'checkpoint.json'),
            'training_sha256': sha256(run / 'training.json'),
            'inventory': artifact_identity(adapter)}


def lossavg_matches(info, reference):
    config = info.get('config', {})
    training = config.get('training', {})
    ref = reference['config']['training']
    return (config.get('method') == 'multiview-B-gaussian'
            and config.get('seed') == reference['config']['seed']
            and model_spec(config.get('model', {})) == model_spec(reference['config']['model'])
            and info.get('base_identity') == reference['base_identity']
            and info.get('pairs_id') == reference['pairs_id']
            and info.get('effective_batch_size_pair_views') == reference['effective_batch_size_pair_views']
            and training.get('formal_arm') == 'B' and training.get('noise_views') == 4
            and not training.get('independent_rollout') and not training.get('stagewise')
            and all(training.get(k) == ref.get(k) for k in (
                'objective', 'beta', 'comparison_budget', 'resolution', 'dtype', 'sampling', 'prompt_mode',
                'timestep_sampling', 'reference_mode', 'learning_rate', 'lr_schedule', 'warmup_steps',
                'weight_decay', 'max_grad_norm', 'lora_rank', 'lora_alpha')))


def checkpoint_plan(root, config, explicit_lossavg):
    records, training = {}, {}
    for method, target in config['runs'].items():
        run = (root / target['run']).resolve()
        info = read_json(run / 'training.json')
        training[method] = info
        validate_method(info['config'], method, run)
        if method == 'stagewise_k4' and info['config']['training']['stagewise']['boundary'] != 500:
            raise ValueError('Stage-wise K=4 checkpoint must use stage boundary 500')
        for step, identity in target['checkpoints'].items():
            records[method, int(step)] = resolve_adapter(run, int(step), target['run_id'], identity)
    reference = training['global_k4']
    model = model_spec(reference['config']['model'])
    base = model_artifact_identity(model)
    for method, info in training.items():
        if (model_spec(info['config']['model']) != model or info['base_identity'] != base
                or info['pairs_id'] != reference['pairs_id']
                or info['config']['training']['beta'] != reference['config']['training']['beta']
                or info['effective_batch_size_pair_views'] != reference['effective_batch_size_pair_views']):
            raise ValueError(f'Unmatched base, training pairs, beta or pair-view budget: {method}')

    candidates = [Path(explicit_lossavg).expanduser().resolve()] if explicit_lossavg else [
        p.parent for p in sorted(root.glob('*/training.json'))
        if (p.parent / 'checkpoints/step-0001600/checkpoint.json').is_file()
    ]
    matches = []
    for run in candidates:
        info = read_json(run / 'training.json')
        if lossavg_matches(info, reference):
            matches.append((run, info['run_id']))
    if explicit_lossavg and not matches:
        raise ValueError('Requested LossAvg run does not match the Global MVPO sampling and training identity')
    lossavg = {'status': 'missing' if not matches else 'ambiguous',
               'searched': str(root), 'candidates': [str(run) for run, _ in matches]}
    if len(matches) == 1:
        run, identity = matches[0]
        records['lossavg_k4', 1600] = resolve_adapter(run, 1600, identity, None)
        training['lossavg_k4'] = read_json(run / 'training.json')
        lossavg['status'] = 'available'
    return model, base, records, lossavg, {
        name: {'seed': i['config']['seed'], 'pairs_id': i['pairs_id'],
               'pair_views_per_update': i['effective_batch_size_pair_views'],
               'training_method': i['config']['method'], 'training': i['config']['training']}
        for name, i in training.items()
    }


def historical_tasks(roots, checkpoint_ids, skip):
    history = defaultdict(list)
    audit, seen = [], set()
    prune = {'images', 'render', 'scores', 'logs', 'paper', 'gallery', 'sessions', 'checkpoints'}
    for root in roots:
        if not root.is_dir():
            audit.append({'root': str(root), 'status': 'missing'})
            continue
        for folder, dirs, files in os.walk(root):
            folder = Path(folder)
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in prune
                             and (folder / d).resolve() != skip)
            if 'plan.json' not in files or folder in seen:
                continue
            seen.add(folder)
            path = folder / 'plan.json'
            try:
                plan = read_json(path)
                if plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
                    raise ValueError('Historical evaluation plan checksum differs')
                panel = plan['panels'].get(PANEL)
                if not panel:
                    continue
                snapshot = safe_target(folder, panel['snapshot'])
                if sha256(snapshot) != panel['sha256']:
                    raise ValueError('Historical PartiPrompts checksum differs')
                for task in plan['tasks']:
                    if task['family'] != FAMILY or task['benchmark'] != PANEL:
                        continue
                    checkpoint_id = task['checkpoint']['checkpoint_id']
                    if checkpoint_id in checkpoint_ids:
                        target = safe_target(folder, task['output'])
                        history[checkpoint_id].append((path, snapshot, target, task))
            except (OSError, ValueError, KeyError, TypeError) as error:
                audit.append({'path': str(path), 'status': 'rejected', 'reason': str(error)})
    return history, audit


def full_prompts(path, historical):
    if path.is_file():
        return path, load_suite(path)
    candidates = {}
    for tasks in historical.values():
        for _, snapshot, _, task in tasks:
            if task['prompts'] == FULL_COUNT:
                candidates.setdefault(sha256(snapshot), snapshot)
    if len(candidates) == 1:
        source = next(iter(candidates.values()))
        return source, load_suite(source)
    raise ValueError(f'Cannot locate a unique full PartiPrompts manifest; use --prompts: {path}')


def aligned_config(model, sampling, adapter):
    return {'model': model, 'sampling': sampling, 'aligned': {
        'kind': 'peft', 'path': adapter['path'], 'revision': adapter['checkpoint_id'],
    }}


def sampling_protocol(config):
    # Packing and offload can differ across allocations. Preserve their actual
    # values on each source while matching the sampler and conditioning settings.
    return {k: v for k, v in effective_sampling(config).items()
            if k not in ('generation_batch_size', 'cpu_offload')}


@lru_cache(maxsize=1)
def renderer_hash():
    return sha256(native_render.__file__)


def noise_runtime(run):
    source = run['render_sha256']
    current = renderer_hash()
    # e6ee941 -> d3a539c changed Z-Image precision and session metadata.
    # The diff only changes Z-Image compute and session metadata; SD1.5,
    # SDXL and SD3.5-M keep identical sampler/noise/denoiser calls.
    same_non_zimage = (run['model']['family'] in ('sd15', 'sdxl', FAMILY) and source == LEGACY_RENDER_SHA256
                 and current == BF16_RENDER_SHA256)
    if source != current and not same_non_zimage and not compatible_sources({'render.py': source}, {'render.py': current}):
        raise ValueError('Cannot verify the historical CUDA noise implementation against this renderer')
    return {'gpu': run['gpu'], **{name: run['versions'][name] for name in ('torch', 'diffusers')}}


def checked_render(render, config, base, adapter, expected_suites):
    run, receipt = render_header(render.parent)
    if (run['arm'] != 'aligned' or model_spec(run['model']) != model_spec(config['model'])
            or run['config']['aligned'] != config['aligned']
            or run['sampling'] != effective_sampling(config)
            or run['base_identity'] != base or run['model_identity'] != adapter['inventory']
            or run['scheduler_config'].get('_class_name', run['sampling']['scheduler']) != config['sampling']['scheduler']
            or all(run['suite'] != suite for suite in expected_suites)):
        raise ValueError('Rendered checkpoint, adapter inventory, suite or sampling differs')
    manifests = read_jsonl(render / 'manifest.jsonl')
    expected = {(*prompt_key(row), seed): row for row in run['suite'] for seed in run['sampling']['seeds']}
    indexed = {}
    for row in manifests:
        key = image_key(row)
        payload = {k: v for k, v in row.items() if k != 'image_id'}
        if (key in indexed or key not in expected
                or any(row[k] != expected[key][k] for k in ('prompt', 'tag'))
                or row['image_id'] != digest({'run_id': run['run_id'], **payload})):
            raise ValueError('Image manifest contains a duplicate or unmatched prompt/seed identity')
        safe_target(render, row['image'])
        indexed[key] = row
    if indexed.keys() != expected.keys():
        raise ValueError('Image manifest does not cover the complete prompt/seed design')
    return run, receipt, indexed


def checked_scores(target, metric, run, receipt, images, scorer):
    score_summary(target, metric, run, receipt['manifest_sha256'])
    score_out = target / 'scores' / metric
    identity = read_json(score_out / 'identity.json')
    if identity['identity']['scorer'] != scorer:
        raise ValueError(f'{metric} scorer assets, source or convention differ')
    rows = read_jsonl(score_out / 'scores.jsonl')
    seen = set()
    for row in rows:
        key = image_key(row)
        image = images.get(key)
        if (not image or key in seen or row['metric'] != metric
                or row['evaluation_id'] != identity['evaluation_id']
                or any(row[k] != image[k] for k in ('prompt', 'tag', 'image_sha256'))
                or row['record_id'] != digest({k: v for k, v in row.items() if k != 'record_id'})
                or not math.isfinite(float(row['score']))):
            raise ValueError(f'Invalid or unpaired {metric} score for {key}')
        seen.add(key)
    if seen != images.keys():
        raise ValueError(f'Unscored {metric} image slots')
    return rows, identity, read_json(score_out / 'complete.json')


def verified_history(job, candidates, config, base, adapter, panels, scorers, audit):
    preferred = []
    for plan, snapshot, target, task in candidates:
        try:
            if (task['checkpoint']['run_id'] != adapter['run_id'] or task['step'] != adapter['step']
                    or task['checkpoint']['checkpoint_sha256'] != adapter['checkpoint_sha256']
                    or task['checkpoint']['training_sha256'] != adapter['training_sha256']):
                raise ValueError('Plan points to a different training/checkpoint manifest')
            suite = load_suite(snapshot)
            if suite != panels['full'] and suite != panels[job['panel']]:
                raise ValueError('Plan is for a different PartiPrompts panel')
            actual_config = {**config, 'sampling': task['sampling']}
            if sampling_protocol(actual_config) != sampling_protocol(config):
                raise ValueError('Plan is for a different sampling protocol or generation seed')
            render = target / 'render'
            run, receipt, images = checked_render(render, actual_config, base, adapter, [suite])
            runtime = noise_runtime(run)
            scores = {}
            for metric in job['metrics']:
                try:
                    _, identity, complete = checked_scores(target, metric, run, receipt, images, scorers[metric])
                    scores[metric] = {'evaluation_id': identity['evaluation_id'], **complete}
                except (OSError, ValueError, KeyError, TypeError) as error:
                    audit.append({'job': job['name'], 'source': str(target), 'metric': metric,
                                  'status': 'score-unusable', 'reason': str(error)})
            if len(scores) < len(job['metrics']):
                # New scores must refer to the same pixels as the reusable scores.
                verify_images(render)
            preferred.append({'target': str(target), 'plan': str(plan), 'task_id': task['task_id'],
                              'source_label': task.get('method_label'), 'source_method': task.get('method'),
                              'suite_id': run['suite_id'], 'run_id': run['run_id'],
                              'sampling': task['sampling'], 'runtime': runtime,
                              'render_receipt': receipt, 'scores': scores})
        except (OSError, ValueError, KeyError, TypeError) as error:
            audit.append({'job': job['name'], 'source': str(target), 'status': 'rejected', 'reason': str(error)})
    return sorted(preferred, key=lambda source: (len(source['scores']),
                                                source['sampling'] == config['sampling']), reverse=True)


def jobs_for(adapters):
    jobs = []
    for method, step in adapters:
        seeds = SEEDS if method in PAIR and step == 1600 else (17,)
        for seed in seeds:
            full = step == 1600 and seed == 17
            group = 'direct' if full else (f'seed{seed}' if seed != 17 else f'step{step}')
            jobs.append({'name': f'{method}-{step}-seed{seed}', 'method': method,
                         'step': step, 'seed': seed, 'panel': 'full' if full else 'subset',
                         'group': group, 'metrics': list(METRICS if step == 1600 else METRICS[:2]),
                         'adapter': adapters[method, step]})
    return sorted(jobs, key=lambda j: (GROUPS.index(j['group']), list(METHODS).index(j['method'])))


def prepare(args, out):
    option_names = ('prompts', 'runs_root', 'history_root', 'lossavg_run', 'generation_batch_size',
                    'render_shards', 'scorers_config', 'fresh')
    options = {name: getattr(args, name) for name in option_names if getattr(args, name)}
    if args.cpu_offload is not None:
        options['cpu_offload'] = args.cpu_offload
    if (out / 'plan.json').exists():
        plan, _ = load_plan(out)
        if any(plan['prepare_options'].get(key) != value for key, value in options.items()):
            raise ValueError('This plan is frozen; use an independent output to change options')
        print(json.dumps({'plan_id': plan['plan_id'], 'jobs': len(plan['jobs']), 'status': 'already prepared',
                          'noise_runtime': plan['noise_runtime']}))
        return
    registry = load_resources()
    setup = read_json(CONFIG_DIR / 'appendix-generation.json')
    suite_config = load_config(CONFIG_DIR / setup['suite_config'], benchmarks=[PANEL])
    scorers_config = Path(args.scorers_config or suite_config['scorers_config']).expanduser().resolve()
    specs = load_config(scorers_config, scorers=METRICS)['scorers']
    scorers = {metric: scorer_identity(metric, specs[metric]) for metric in METRICS}
    runs_root = Path(args.runs_root or registry.get('FLOW_RUNS_ROOT')
                     or Path(registry['WORKSPACE']) / 'self_correct/flow-paper-v2/runs').expanduser().resolve()
    if out == runs_root or out.is_relative_to(runs_root) or runs_root.is_relative_to(out):
        raise ValueError('Appendix output must be separate from the training runs tree')
    historical_base = Path(registry['EVALUATIONS_ROOT']) / FAMILY
    roots = [historical_base / path for path in setup['history_roots']]
    roots = [*(Path(p).expanduser().resolve() for p in args.history_root), *roots]
    model, base, adapters, lossavg, training = checkpoint_plan(runs_root, setup, args.lossavg_run)
    history, audit = historical_tasks(roots, {a['checkpoint_id'] for a in adapters.values()}, out)
    prompt_path = Path(args.prompts or suite_config['benchmarks'][PANEL]).expanduser().resolve()
    if args.prompts and not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    source, full = full_prompts(prompt_path, history)
    if len(full) != FULL_COUNT or any(r['variant_id'] != 'canonical' for r in full):
        raise ValueError(f'The full PartiPrompts panel must have {FULL_COUNT} canonical prompts')
    panels = {'full': full, 'subset': fixed_subset(full)}
    sampling = suite_config['protocols'][FAMILY]
    if args.generation_batch_size:
        sampling['generation_batch_size'] = args.generation_batch_size
    if args.cpu_offload is not None:
        sampling['cpu_offload'] = args.cpu_offload
    jobs = jobs_for(adapters)
    candidates = {}
    for job in jobs:
        adapter = job['adapter']
        sample = dict(sampling, seeds=[job['seed']])
        invocation = aligned_config(model, sample, adapter)
        candidates[job['name']] = [] if args.fresh else verified_history(
            job, history[adapter['checkpoint_id']], invocation, base, adapter, panels, scorers, audit)
    runtimes = {digest(source['runtime']): source['runtime']
                for found in candidates.values() for source in found}

    def reuse_weight(runtime):
        best = {name: next((s for s in found if s['runtime'] == runtime), None)
                for name, found in candidates.items()}
        return (sum(len(best[j['name']]['scores']) for j in jobs if j['group'] == 'direct' and best[j['name']]),
                sum(len(s['scores']) for s in best.values() if s), sum(bool(s) for s in best.values()))

    matched_runtime = max(runtimes.values(), key=reuse_weight, default=None)
    for job in jobs:
        selected = next((s for s in candidates[job['name']] if s['runtime'] == matched_runtime), None)
        job['history'] = selected
        job['shards'] = 1 if selected else (args.render_shards or 2)
    out.mkdir(parents=True, exist_ok=True)
    for panel, rows in panels.items():
        payload_jsonl(out / f'{panel}.jsonl', rows)
    resolved_scorers = out / 'scorers.resolved.json'
    freeze_json(resolved_scorers, {'scorers': specs})
    plan = {'version': 1, 'family': FAMILY, 'benchmark': PANEL, 'model': model, 'base_identity': base,
            'prepare_options': options,
            'sampling': sampling, 'generation_seeds': list(SEEDS), 'steps': list(STEPS),
            'noise_pairing': ('Native render draws one latent per image from a CUDA generator reset to its seed. '
                              'Every source must use one GPU model and matching Torch, Diffusers, and renderer semantics.'),
            'noise_runtime': matched_runtime,
            'prompt_source': {'path': str(source), 'sha256': sha256(source)},
            'panels': {name: {'suite_id': digest(rows), 'prompts': len(rows)} for name, rows in panels.items()},
            'subset_algorithm': f'SHA-256 of JSON [salt, scene_id, variant_id], first {SUBSET_COUNT}, native order',
            'subset_salt': SUBSET_SALT, 'training': training, 'lossavg': lossavg,
            'scorers_config': str(scorers_config), 'scorers_config_sha256': sha256(scorers_config),
            'resolved_scorers_sha256': sha256(resolved_scorers),
            'scorers': scorers, 'jobs': jobs, 'history_audit': audit}
    plan['plan_id'] = digest(plan)
    freeze_json(out / 'plan.json', plan)
    print(json.dumps({'plan_id': plan['plan_id'], 'jobs': len(jobs), 'lossavg': lossavg, 'noise_runtime': matched_runtime,
                      'history_jobs': sum(bool(j['history']) for j in jobs),
                      'history_metrics': sum(len((j['history'] or {}).get('scores', {})) for j in jobs)}))


def load_plan(out):
    plan = read_json(out / 'plan.json')
    if plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('Appendix generation plan checksum differs')
    panels = {panel: load_suite(out / f'{panel}.jsonl') for panel in plan['panels']}
    if any(digest(rows) != plan['panels'][panel]['suite_id'] for panel, rows in panels.items()):
        raise ValueError('Frozen PartiPrompts prompt snapshot checksum differs')
    return plan, panels


def artifact_target(out, job, index):
    return safe_target(out, f'jobs/{job["name"]}/shard-{index}')


def render_for(out, job, index):
    if job['history']:
        return Path(job['history']['target']) / 'render'
    return artifact_target(out, job, index) / 'render'


def score_target(out, job, index, metric):
    if job['history'] and metric in job['history']['scores']:
        return Path(job['history']['target'])
    return artifact_target(out, job, index)


def validate_artifact(out, plan, panels, job, index):
    history = job['history']
    sampling = history['sampling'] if history else dict(plan['sampling'], seeds=[job['seed']])
    invocation = aligned_config(plan['model'], sampling, job['adapter'])
    render = render_for(out, job, index)
    if job['history']:
        suites = [panels[job['panel']], panels['full']]
    else:
        suites = [select_shard(panels[job['panel']], index, job['shards'])]
    run, receipt, images = checked_render(render, invocation, plan['base_identity'], job['adapter'], suites)
    expected_runtime = plan['noise_runtime'] or read_json(out / 'noise-runtime.json')['runtime']
    if noise_runtime(run) != expected_runtime:
        raise ValueError(f'Render used an incompatible CUDA noise runtime: {render}')
    if job['history'] and (run['run_id'] != job['history']['run_id'] or receipt != job['history']['render_receipt']):
        raise ValueError(f'Frozen historical render changed: {render}')
    if not job['history'] and run['shard'] != {'index': index, 'count': job['shards'],
                                               'assignment': 'sorted-scene-round-robin-v1'}:
        raise ValueError(f'Render used a different prompt shard: {render}')
    return run, receipt, images


def worker(args, out, plan, panels, job, index):
    target = artifact_target(out, job, index)
    render = render_for(out, job, index)
    execution = ['--execution', args.execution, '--run-label', args.run_label]
    with output_lock(target):
        if not (render / 'complete.json').is_file():
            if job['history']:
                raise ValueError(f'Frozen historical render missing: {render}')
            adapter = job['adapter']
            actual = resolve_adapter(Path(adapter['run']), job['step'], adapter['run_id'], adapter['checkpoint_id'])
            if actual != adapter:
                raise ValueError(f'Frozen training/checkpoint identity changed: {adapter["path"]}')
            config = aligned_config(plan['model'], dict(plan['sampling'], seeds=[job['seed']]), adapter)
            invocation = target / 'render-config.json'
            freeze_json(invocation, config)
            command = [args.flow_python, '-m', 'self_correct_dpo.evaluation.render',
                       '--config', str(invocation), '--prompts', str(out / f'{job["panel"]}.jsonl'),
                       '--out', str(render), '--arm', 'aligned', '--shard-index', str(index),
                       '--shard-count', str(job['shards']), *execution]
            if render.exists():
                command.append('--resume')
            run_command(command)
        run, receipt, images = validate_artifact(out, plan, panels, job, index)
        for metric in job['metrics']:
            source = score_target(out, job, index, metric)
            score_out = source / 'scores' / metric
            if not (score_out / 'complete.json').is_file():
                if source != target:
                    raise ValueError(f'Frozen historical score export missing: {score_out}')
                python = args.hpsv3_python if metric == 'hpsv3' else args.legacy_python
                run_command([python, '-m', 'self_correct_dpo.evaluation.metrics', '--metric', metric,
                             '--run', str(render), '--out', str(score_out),
                             '--scorers-config', str(out / 'scorers.resolved.json'), *execution])
            _, identity, complete = checked_scores(source, metric, run, receipt, images, plan['scorers'][metric])
            previous = (job['history'] or {}).get('scores', {}).get(metric)
            if previous and previous != {'evaluation_id': identity['evaluation_id'], **complete}:
                raise ValueError(f'Frozen historical scores changed: {score_out}')


def preflight_noise_runtime(python, devices, out, expected):
    probe = ('import json, importlib.metadata as m, torch; '
             'p=torch.cuda.get_device_properties(0); '
             'print(json.dumps({"gpu":p.name, "torch":m.version("torch"), "diffusers":m.version("diffusers"),'
             '"cuda":torch.version.cuda, "sms":p.multi_processor_count}))')
    observed = []
    for token in devices:
        result = subprocess.check_output([python, '-c', probe], text=True,
                                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=token))
        observed.append(json.loads(result))
    if any(runtime != observed[0] for runtime in observed[1:]):
        raise ValueError('The render workers must use one GPU model and CUDA noise runtime')
    runtime = {k: observed[0][k] for k in ('gpu', 'torch', 'diffusers')}
    if expected and runtime != expected:
        raise ValueError(f'Historical noise pairing requires {expected}; allocated runtime is {runtime}. '
                         'Use the matching allocation or prepare a new output with --fresh.')
    path = out / 'noise-runtime.json'
    # Independently launched sections can register the same runtime concurrently.
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        freeze_json(path, {'runtime': runtime, 'fresh_cuda_details': observed[0]})


def launch_workers(args, out, plan, panels):
    jobs = [j for j in plan['jobs'] if args.group == 'all' or j['group'] == args.group]
    if sha256(out / 'scorers.resolved.json') != plan['resolved_scorers_sha256']:
        raise ValueError('Frozen scorer configuration changed after prepare')
    work = []
    for job in jobs:
        for index in range(job['shards']):
            render_complete = (render_for(out, job, index) / 'complete.json').is_file()
            if render_complete and all((score_target(out, job, index, m) / 'scores' / m / 'complete.json').is_file()
                                       for m in job['metrics']):
                # The worker verifies the render and every score before accepting completed shards.
                worker(args, out, plan, panels, job, index)
            else:
                work.append((job, index))
    if not work:
        print('All selected render and score records verified.')
        return 0
    require_gpu_allocation({'mode': args.execution, 'run_label': args.run_label})
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus = args.gpus or os.environ.get('SCD_EVAL_GPUS') or visible or ''
    devices = gpus.split(',')
    if (len(devices) not in (2, 4) or len(set(devices)) != len(devices)
            or any(not gpu or gpu == '-1' or any(c.isspace() for c in gpu) for gpu in devices)
            or (visible is not None and not set(devices) <= set(visible.split(',')))):
        raise ValueError('Select two or four distinct allocated GPU tokens with --gpus or SCD_EVAL_GPUS')
    incomplete_metrics = {m for job, index in work for m in job['metrics']
                          if not (score_target(out, job, index, m) / 'scores' / m / 'complete.json').is_file()}
    dependencies = []
    needs_rendering = any(not (render_for(out, job, index) / 'complete.json').is_file() for job, index in work)
    if needs_rendering:
        dependencies.append('flow_python')
    if 'hpsv3' in incomplete_metrics:
        dependencies.append('hpsv3_python')
    if incomplete_metrics - {'hpsv3'}:
        dependencies.append('legacy_python')
    for name in dependencies:
        python = getattr(args, name)
        if not shutil.which(python):
            raise FileNotFoundError(f'{name} executable does not exist: {python}')
    if needs_rendering:
        preflight_noise_runtime(args.flow_python, devices, out, plan['noise_runtime'])
    children, interrupted = [], False

    def forward(signum, frame):
        nonlocal interrupted
        interrupted = True
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)

    saved = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    script = CONFIG_DIR.parent / 'scripts/appendix_generation.py'
    try:
        for gpu, offset in zip(devices, range(len(work))):
            assigned = [f'{job["name"]}/{index}' for job, index in work[offset::len(devices)]]
            command = [sys.executable, str(script), 'worker', '--out', str(out),
                       '--flow-python', args.flow_python, '--legacy-python', args.legacy_python,
                       '--hpsv3-python', args.hpsv3_python, '--execution', args.execution,
                       '--run-label', args.run_label, '--jobs', *assigned]
            if interrupted:
                break
            children.append(subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
        statuses = [child.wait() for child in children]
        return 130 if interrupted else int(any(statuses))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for signum, handler in saved.items():
            signal.signal(signum, handler)


def section_jobs(plan, section):
    if section == 'full':
        return plan['jobs']
    if section == 'direct':
        return [j for j in plan['jobs'] if j['group'] == 'direct']
    groups = ('direct', 'seed23', 'seed42') if section == 'seeds' else ('direct', 'step800', 'step2400', 'step3200')
    return [{**j, 'metrics': j['metrics'] if section == 'seeds' else list(METRICS[:2])}
            for j in plan['jobs'] if j['group'] in groups and j['method'] in PAIR]


def report_path(out, section):
    return out / 'report' if section == 'full' else out / 'report' / section


def collect(out, plan, panels, section):
    bundle_dir = report_path(out, section) / 'data'
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(bundle_dir):
        if (bundle_dir / 'bundle.json').is_file():
            bundle, _ = read_bundle(bundle_dir)
            if bundle['plan']['plan_id'] != plan['plan_id']:
                raise ValueError('Existing score bundle belongs to a different plan')
            return bundle_dir

        rows, sources = [], {}
        for job in section_jobs(plan, section):
            keep = {prompt_key(row) for row in panels[job['panel']]}
            for index in range(job['shards']):
                source_id = f'{job["name"]}/shard-{index}'
                run, receipt, images = validate_artifact(out, plan, panels, job, index)
                render = render_for(out, job, index)
                source = {'render': str(render), 'run_id': run['run_id'], 'receipt': receipt,
                          'suite_id': run['suite_id'], 'shard': run['shard'],
                          'versions': run['versions'], 'gpu': run['gpu'],
                          'sampling': run['sampling'],
                          'noise_runtime': noise_runtime(run),
                          'render_sha256': run['render_sha256'],
                          'scheduler_config': run['scheduler_config'], 'scores': {}}
                for metric in job['metrics']:
                    target = score_target(out, job, index, metric)
                    results, identity, complete = checked_scores(target, metric, run, receipt, images, plan['scorers'][metric])
                    previous = (job['history'] or {}).get('scores', {}).get(metric)
                    if previous and previous != {'evaluation_id': identity['evaluation_id'], **complete}:
                        raise ValueError(f'Frozen historical scores changed: {target}')
                    source['scores'][metric] = {'target': str(target), 'identity': identity, 'receipt': complete}
                    for row in results:
                        if prompt_key(row) in keep:
                            rows.append({'job': job['name'], 'source': source_id, 'image_id': images[image_key(row)]['image_id'],
                                         **row})
                sources[source_id] = source
        score_file = bundle_dir / 'scores.jsonl'
        payload_jsonl(score_file, rows)
        for name in panels:
            shutil.copyfile(out / f'{name}.jsonl', bundle_dir / f'{name}.jsonl')
        manifest = {'version': 1, 'section': section, 'plan': plan, 'sources': sources,
                    'scores_sha256': sha256(score_file)}
        manifest['bundle_id'] = digest(manifest)
        atomic_json(bundle_dir / 'bundle.json', manifest)
    return bundle_dir


def read_bundle(bundle_dir):
    bundle = read_json(bundle_dir / 'bundle.json')
    if bundle['bundle_id'] != digest({k: v for k, v in bundle.items() if k != 'bundle_id'}):
        raise ValueError('Score bundle checksum differs')
    plan = bundle['plan']
    panels = {name: load_suite(bundle_dir / f'{name}.jsonl') for name in plan['panels']}
    if any(digest(rows) != plan['panels'][name]['suite_id'] for name, rows in panels.items()):
        raise ValueError('Score bundle prompt snapshot checksum differs')
    score_file = bundle_dir / 'scores.jsonl'
    if bundle['scores_sha256'] != sha256(score_file):
        raise ValueError('Bundled score journal checksum differs')
    rows = read_jsonl(score_file)
    expected = {}
    jobs = {job['name']: job for job in section_jobs(plan, bundle['section'])}
    for name, job in jobs.items():
        expected[name] = {(metric, *prompt_key(row)): row for metric in job['metrics'] for row in panels[job['panel']]}
    indexed = defaultdict(dict)
    for row in rows:
        name, source_id, metric = row['job'], row['source'], row['metric']
        job, source = jobs[name], bundle['sources'][source_id]
        key = (metric, *prompt_key(row))
        native = {k: v for k, v in row.items() if k not in ('job', 'source', 'image_id', 'record_id')}
        if (key in indexed[name] or key not in expected[name] or row['seed'] != job['seed']
                or not source_id.startswith(name + '/')
                or row['evaluation_id'] != source['scores'][metric]['identity']['evaluation_id']
                or row['record_id'] != digest(native)
                or any(row[k] != expected[name][key][k] for k in ('prompt', 'tag'))
                or not math.isfinite(float(row['score']))):
            raise ValueError(f'Duplicate or mispaired score in the bundle: {name}/{key}')
        indexed[name][key] = float(row['score'])
    for name, job in jobs.items():
        if indexed[name].keys() != expected[name].keys():
            raise ValueError(f'Score bundle is missing scheduled rows: {name}')
    scheduler_keys = {
        digest({k: v for k, v in src['scheduler_config'].items()
                if k not in ('_diffusers_version', '_use_default_values')})
        for src in bundle['sources'].values()
    }
    if len(scheduler_keys) != 1:
        raise ValueError('Incompatible scheduler configuration across appendix renders')
    noise_keys = {digest(src['noise_runtime']) for src in bundle['sources'].values()}
    if len(noise_keys) != 1 or (plan['noise_runtime'] and digest(plan['noise_runtime']) not in noise_keys):
        raise ValueError('Incompatible CUDA noise runtime across appendix renders')
    return bundle, (panels, indexed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('prepare', 'run', 'worker', 'collect', 'summarize'):
        p = actions.add_parser(action)
        p.add_argument('--out', required=True, help='Independent appendix generation output directory')
        p.add_argument('--resources', help='Override configs/resources.json')
        if action == 'prepare':
            p.add_argument('--prompts', help='Original full PartiPrompts JSONL')
            p.add_argument('--runs-root', help='Flow training run root')
            p.add_argument('--history-root', action='append', default=[], help='Additional frozen evaluate_suite result root')
            p.add_argument('--lossavg-run', help='Explicit matching LossAvg K=4 run when discovery is ambiguous')
            p.add_argument('--scorers-config')
            p.add_argument('--generation-batch-size', type=int, metavar='N')
            p.add_argument('--cpu-offload', action=argparse.BooleanOptionalAction, default=None)
            p.add_argument('--fresh', action='store_true', help='Rerender all images with a single chosen CUDA runtime')
            p.add_argument('--render-shards', type=int, choices=(1, 2, 4))
        if action in ('run', 'worker'):
            p.add_argument('--flow-python')
            p.add_argument('--legacy-python')
            p.add_argument('--hpsv3-python')
            p.add_argument('--execution', choices=('standalone', 'slurm'), default='standalone')
            p.add_argument('--run-label', default='mvpo-appendix-generation')
        if action == 'run':
            p.add_argument('--gpus', help='Two or four actual allocated GPU tokens')
            p.add_argument('--group', choices=('all', *GROUPS), default='all')
        if action == 'worker':
            p.add_argument('--jobs', nargs='+', required=True)
        if action in ('collect', 'summarize'):
            p.add_argument('--section', choices=SECTIONS, help='Allow early export of a complete subsection')
        if action == 'summarize':
            p.add_argument('--bundle', help='Locally copied report/data directory for CPU-only reanalysis')
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    if args.resources:
        os.environ['SCD_RESOURCES'] = str(Path(args.resources).expanduser().resolve())
    if args.action == 'prepare':
        if args.generation_batch_size is not None and args.generation_batch_size < 1:
            parser.error('--generation-batch-size must be positive')
        with output_lock(out):
            prepare(args, out)
        return 0
    if args.action == 'summarize':
        from self_correct_dpo.evaluation.appendix_generation_report import summarize
        if args.bundle:
            bundle_dir = Path(args.bundle).expanduser().resolve()
        else:
            plan, panels = load_plan(out)
            bundle_dir = collect(out, plan, panels, args.section or 'full')
        bundle, data = read_bundle(bundle_dir)
        if args.section and args.section != bundle['section']:
            raise ValueError('The selected section differs from the score bundle')
        report = report_path(out, bundle['section'])
        with output_lock(report / 'summary.json'):
            summarize(report, bundle, *data)
        return 0

    plan, panels = load_plan(out)
    if args.action == 'collect':
        print(collect(out, plan, panels, args.section or 'full'))
        return 0
    if args.action == 'run':
        resources = load_resources()
        args.flow_python = args.flow_python or resources.get('SCD_FLOW_PY') or os.environ.get('SCD_FLOW_PY') or sys.executable
        args.legacy_python = args.legacy_python or resources.get('SCD_SD15_PY') or os.environ.get('SCD_SD15_PY') or sys.executable
        args.hpsv3_python = args.hpsv3_python or resources.get('SCD_HPSV3_PY') or os.environ.get('SCD_HPSV3_PY') or sys.executable
        return launch_workers(args, out, plan, panels)

    by_name = {j['name']: j for j in plan['jobs']}
    failures = []
    for name in args.jobs:
        job_name, index = name.split('/')
        print(f'GPU {os.environ["CUDA_VISIBLE_DEVICES"]}: {name}', flush=True)
        try:
            worker(args, out, plan, panels, by_name[job_name], int(index))
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            if isinstance(error, subprocess.CalledProcessError) and (error.returncode < 0 or error.returncode in (75, 130, 143)):
                return 130
            failures.append(name)
            print(f'{name}: {error}', file=sys.stderr, flush=True)
    return int(bool(failures))
