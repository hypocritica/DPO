"""Single-GPU SD3.5 DPO K1 probe: resident frozen modules, B2, 1024px.

Loads once and uses the production trainer, including AdamW and checkpointing.
This measures single-GPU feasibility; four-rank DDP still needs its own margin.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def make_config(steps, resources, pairs, model, model_report):
    from run_flow import build_config
    config = build_config('sd35_medium', 'dpo', world=1, profile='5090',
                          resources=resources, smoke=True, smoke_steps=steps,
                          smoke_batch_size=2, smoke_accumulation_steps=1)
    report = json.loads(model_report.read_text())
    if report.get('status') != 'passed':
        raise ValueError('Expected the existing passed SD3.5 model validation report')
    config['model'].update(path=str(model), revision=report['revision'],
                           validation_report=str(model_report))
    config['data']['pairs'] = str(pairs)
    config['training'].update(offload_frozen=False, vae_batch_size=1, learning_rate=5e-6,
                              beta=500, logging='jsonl', checkpoint_every=steps)
    return config


def summarize_updates(run):
    rows = []
    for path in run.glob('metrics-*.jsonl'):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows.sort(key=lambda row: row['step'])
    tail = rows[2:]  # Exclude two initial updates; default six gives four samples.
    return {'updates': len(rows), 'last_step': rows[-1]['step'] if rows else None,
            'timed_updates': len(tail),
            'median_seconds_per_update': statistics.median(r['seconds_per_update'] for r in tail) if tail else None,
            'median_pairs_per_second': statistics.median(r['pairs_per_second'] for r in tail) if tail else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=6, help='3..20 optimizer updates; default 6')
    parser.add_argument('--tag', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:6])
    parser.add_argument('--resources', default=os.environ.get('SCD_RESOURCES', str(ROOT / 'configs/resources-slai-flow.json')))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--pairs', type=Path, default=Path(os.environ.get('FLOW_DEBUG_PAIRS', '/mnt/aoss/xietianyu/datasets/hpdv3/flow-debug-v1/prepared/pairs.jsonl')))
    parser.add_argument('--model-report', type=Path, default=Path(os.environ.get('SCD_FLOW_REPORTS_ROOT', str(ROOT / 'reports/slai'))) / 'sd35_medium-model-validation.json')
    parser.add_argument('--report-root', type=Path, default=ROOT / 'logs/sd35-b2-probe')
    parser.add_argument('--runs-root', type=Path)
    parser.add_argument('--config-only', action='store_true', help='CPU-only configuration check; no model/data loading')
    args = parser.parse_args()
    if not 3 <= args.steps <= 20:
        parser.error('--steps must be 3..20')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', args.tag):
        parser.error('Invalid tag')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Run once with python/bash, without torchrun')
    from self_correct_dpo.resources import load_resources
    resources = load_resources(args.resources)
    model = (args.model or Path(resources['SD35_MODEL'])).expanduser().resolve()
    config = make_config(args.steps, args.resources, args.pairs.expanduser().resolve(),
                         model, args.model_report.expanduser().resolve())
    report_dir = args.report_root.expanduser().resolve() / args.tag
    run = (args.runs_root or Path(resources['RUNS_ROOT'])).expanduser().resolve() / ('sd35-b2-resident-1gpu-' + args.tag)
    if run.exists():
        raise FileExistsError(f'Existing run; use a new tag: {run}')
    report_dir.mkdir(parents=True, exist_ok=False)
    config_path = report_dir / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Probe report directory: {report_dir}\nTraining output: {run}', flush=True)
    print('Settings: one logical GPU 0; SD3.5 DPO K1; 1024px; BF16; B2; accumulation 1; offload OFF; VAE chunk 1.', flush=True)
    if args.config_only:
        return 0
    from self_correct_dpo.common import require_gpu_allocation
    require_gpu_allocation({'mode': 'standalone', 'run_label': 'sd35-b2-resident-probe'})
    import torch
    summary = {'status': 'failed', 'config': str(config_path), 'run': str(run),
               'scope': 'single_gpu_no_ddp', 'steps_requested': args.steps}
    start = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('An allocated CUDA GPU is required')
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info(0)
        summary.update(gpu=torch.cuda.get_device_name(0), visible_gpu_count=torch.cuda.device_count(),
                       torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                       initial_free_GiB=free / 1024**3, total_GiB=total / 1024**3)
        torch.cuda.reset_peak_memory_stats(0)
        from self_correct_dpo.train import main as train_main
        original_args = sys.argv
        try:
            sys.argv = ['train.py', '--config', str(config_path), '--out', str(run),
                        '--execution', 'standalone', '--run-label', 'sd35-b2-resident-probe']
            train_main()
        finally:
            sys.argv = original_args
        summary.update(summarize_updates(run))
        summary['status'] = 'passed' if summary['last_step'] == args.steps and (run / 'complete.json').is_file() else 'incomplete'
        return 0 if summary['status'] == 'passed' else 1
    except BaseException as error:
        summary.update(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed',
                       error_type=type(error).__name__, traceback=traceback.format_exc())
        raise
    finally:
        summary['wall_seconds_including_load_and_save'] = time.monotonic() - start
        if torch.cuda.is_initialized():
            summary.update(peak_allocated_GiB=torch.cuda.max_memory_allocated(0) / 1024**3,
                           peak_reserved_GiB=torch.cuda.max_memory_reserved(0) / 1024**3)
        (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        # Plain text survives the platform export which dropped numeric JSON fields.
        print(f'PROBE RESULT: {summary["status"]}', flush=True)
        for key in ('gpu', 'updates', 'median_seconds_per_update', 'median_pairs_per_second',
                    'peak_allocated_GiB', 'peak_reserved_GiB', 'total_GiB', 'wall_seconds_including_load_and_save'):
            if key in summary:
                print(f'PROBE {key}: {summary[key]}', flush=True)
        print(f'PROBE summary: {report_dir / "summary.json"}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
