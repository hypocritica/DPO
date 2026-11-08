"""Sequential Z-Image loading and B1/B2 measurements on two allocated GPUs.

Use the same tag/arguments to skip successful stages and retry failed stages in
new directories. --part load/k1/k4 splits the same plan across task submissions.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARTS = {'load': ['load-first', 'load-repeat'],
         'k1': ['k1-b1', 'k1-b2'], 'k4': ['k4-b1', 'k4-b2']}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def run_child(command, log, environment):
    """Keep full console output, forwarding task cancellation to descendants."""
    print('Running:', ' '.join(map(str, command)), flush=True)
    with log.open('x') as handle:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        previous, stopped = {}, [False]
        def forward(signum, frame):
            stopped[0] = True
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f'  still running; log: {log}', flush=True)
            return child.returncode, stopped[0]
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def read_rows(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # An interrupted final write is not a completed measurement.
    return rows


def output_path(log):
    for line in log.read_text().splitlines():
        if line.startswith('Output: '):
            return Path(line[len('Output: '):].strip())
    return None


def training_summary(run, steps):
    """Use real per-rank training journals, excluding startup and checkpoints."""
    ranks = {}
    for rank in (0, 1):
        pattern = 'metrics-*.jsonl' if rank == 0 else 'rank-1-*.jsonl'
        by_step = {}
        for path in sorted(run.glob(pattern)):
            for row in read_rows(path):
                if 'seconds_per_update' in row and 'step' in row:
                    by_step[row['step']] = row
        ordered = [by_step[key] for key in sorted(by_step)]
        tail = ordered[-10:]
        ranks[str(rank)] = {
            'measured_updates': len(ordered),
            'last_step': ordered[-1]['step'] if ordered else None,
            'tail_updates': len(tail),
            'median_seconds_per_update': statistics.median(r['seconds_per_update'] for r in tail) if tail else None,
            'median_pairs_per_second': statistics.median(r['pairs_per_second'] for r in tail) if tail else None,
            'peak_allocated_GiB': max((r.get('peak_cuda_bytes', 0) / 1024**3 for r in ordered), default=None),
        }
        if tail:
            ranks[str(rank)]['pair_views_per_update'] = tail[-1]['effective_batch_size_pair_views']
            ranks[str(rank)]['median_pair_views_per_second'] = statistics.median(
                r['pairs_per_second'] * r['noise_views'] for r in tail)
    complete = all(r['last_step'] == steps and r['measured_updates'] == steps for r in ranks.values())
    return {'run': str(run), 'measurement_complete': complete, 'ranks': ranks,
            'memory_scope': 'per-rank PyTorch peak allocated; not total device memory'}


def load_summary(directory):
    ranks = {}
    for rank in (0, 1):
        rows = read_rows(directory / f'rank-{rank}.jsonl')
        ranks[str(rank)] = {
            'complete': any(r['event'] == 'complete' for r in rows),
            'stages_seconds': {r['stage']: r['seconds'] for r in rows if r['event'] == 'end'},
            'runtime': next((r for r in rows if r['event'] == 'runtime'), None),
        }
    return {'measurement_complete': all(r['complete'] for r in ranks.values()), 'ranks': ranks,
            'cache_state': 'uncontrolled; first/repeat describe order, not cold/warm guarantees'}


def write_summary(directory, summary):
    write_json(directory / 'summary.json', summary)
    lines = ['# ' + summary['tag'], '',
             '| Stage | Status | Total seconds | Rank 0 / 1 update seconds | Rank 0 / 1 allocated GiB |',
             '|---|---|---:|---|---|']
    for name, result in summary['stages'].items():
        ranks = result['measurement'].get('ranks', {})
        def values(key):
            return ' / '.join(f'{ranks[str(rank)][key]:.3f}' if ranks.get(str(rank), {}).get(key) is not None
                              else '-' for rank in (0, 1))
        lines.append(f'| {name} | {result["status"]} | {result["wall_seconds"]:.1f} | '
                     f'{values("median_seconds_per_update")} | {values("peak_allocated_GiB")} |')
    lines += ['', 'Update times are medians of the last 10 updates. Total seconds include startup and saving.',
              'Loading stage details and throughput are in summary.json; complete resource snapshots are in profile/rank-*.jsonl.',
              'First/repeat are execution order, not guarantees of cold/warm cache. GPU memory is PyTorch allocated only.', '']
    (directory / 'summary.md').write_text('\n'.join(lines))


def debug_command(args, attempt, name, config_only=False):
    views = 4 if name.startswith('k4') else 1
    batch = 2 if name.endswith('b2') else 1
    command = ['bash', str(ROOT / 'launchers/flow/debug_train.sh'), 'zimage',
               'mvpo' if views == 4 else 'dpo', '--smoke-batch-size', str(batch),
               '--smoke-accumulation-steps', str(2 // batch),
               '--run-tag', f'{args.tag}-{name}-a{attempt.name}',
               '--config-dir', str(attempt / 'configs'), '--report-to', 'jsonl']
    if views == 4:
        command += ['--views', '4']
    if args.runs_root:
        command += ['--runs-root', str(args.runs_root)]
    if config_only:
        command += ['--config-only']
    return command


def execute_stage(args, name, attempt, environment):
    start = time.monotonic()
    if name.startswith('load-'):
        code, stopped = run_child(debug_command(args, attempt, name, config_only=True),
                                  attempt / 'config.log', environment)
        if not code and not stopped:
            configs = list((attempt / 'configs').glob('local-*.json'))
            if len(configs) != 1:
                raise ValueError('Expected exactly one diagnostic config')
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nnodes=1', '--nproc-per-node=2',
                       str(ROOT / 'scripts/profile_flow_startup.py'), '--config', str(configs[0]),
                       '--out', str(attempt / 'profile'), '--load', '--execution', 'standalone',
                       '--cpu-threads', str(args.threads)]
            code, stopped = run_child(command, attempt / 'console.log', environment)
        measurement = load_summary(attempt / 'profile')
    else:
        code, stopped = run_child(debug_command(args, attempt, name), attempt / 'console.log', environment)
        run = output_path(attempt / 'console.log')
        measurement = training_summary(run, args.steps) if run else {'measurement_complete': False}
    return {'status': 'passed' if code == 0 and not stopped and measurement['measurement_complete'] else 'failed',
            'returncode': code, 'interrupted': stopped, 'wall_seconds': time.monotonic() - start,
            'attempt': str(attempt), 'measurement': measurement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True, help='Distinct machine/resource label, e.g. cpu8-ram128-v1')
    parser.add_argument('--part', choices=('all', *PARTS), default='all')
    parser.add_argument('--threads', type=int, default=4, help='OMP/MKL threads per rank; same value on both machines')
    parser.add_argument('--steps', type=int, default=20, help='12..200 optimizer updates per training stage')
    parser.add_argument('--out-root', type=Path, default=ROOT / 'logs/flow-benchmarks')
    parser.add_argument('--runs-root', type=Path, help='Optional checkpoint root; default uses existing SLAI resources')
    parser.add_argument('--plan-only', action='store_true', help='Print plan without importing torch or touching GPUs/files')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', args.tag):
        parser.error('tag must be 1..80 letters/digits/dots/underscores/hyphens')
    if args.threads < 1 or not 12 <= args.steps <= 200:
        parser.error('threads must be positive; steps must be 12..200')
    selected = sum(PARTS.values(), []) if args.part == 'all' else PARTS[args.part]
    if args.plan_only:
        print(json.dumps({'tag': args.tag, 'stages': selected, 'threads_per_rank': args.threads,
                          'steps': args.steps, 'world_size': 2, 'training': 'K1/K4 C; B1A2 versus B2A1'}, indent=2))
        return 0
    directory = args.out_root.expanduser().resolve() / args.tag
    directory.mkdir(parents=True, exist_ok=True)
    # One task per tag; parts can be submitted sequentially with the same tag.
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                           OPENBLAS_NUM_THREADS='1', FLOW_DEBUG_STEPS=str(args.steps),
                           SCD_FLOW_PYTHON=sys.executable, PYTHONUNBUFFERED='1')
        sources = ['scripts/benchmark_flow.py', 'scripts/profile_flow_startup.py', 'scripts/run_flow.py',
                   'launchers/flow/debug_train.sh', 'launchers/flow/slai_common.sh']
        identity = {'tag': args.tag, 'steps': args.steps, 'threads': args.threads,
                    'runs_root': str(args.runs_root.resolve()) if args.runs_root else None,
                    'python': sys.executable,
                    'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                    'input_environment': {k: os.environ.get(k) for k in
                                          ('SCD_RESOURCES', 'FLOW_DEBUG_PAIRS', 'SCD_FLOW_REPORTS_ROOT', 'ZIMAGE_MODEL')}}
        manifest = directory / 'plan.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['identity'] != identity:
                raise ValueError('Benchmark plan changed; choose a new tag')
        else:
            revision = subprocess.run(['git', '-c', f'safe.directory={ROOT}', 'rev-parse', 'HEAD'],
                                      cwd=ROOT, capture_output=True, text=True)
            from profile_flow_startup import snapshot
            write_json(manifest, {'identity': identity, 'git_revision': revision.stdout.strip(),
                                 'started_utc': datetime.now(timezone.utc).isoformat(), 'parts': PARTS,
                                 'hostname': os.uname().nodename, 'cpu_affinity_count': len(os.sched_getaffinity(0)),
                                 'resources': snapshot()})
        summary_file = directory / 'summary.json'
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {'tag': args.tag, 'stages': {}}
        for name in selected:
            previous = summary['stages'].get(name)
            if previous and previous['status'] == 'passed':
                print(f'Skipping completed stage: {name}', flush=True)
                continue
            stage_dir = directory / name
            stage_dir.mkdir(exist_ok=True)
            attempt = stage_dir / str(1 + max((int(p.name) for p in stage_dir.iterdir() if p.name.isdigit()), default=0))
            attempt.mkdir()
            print(f'[{name}] starting; results: {attempt}', flush=True)
            result = execute_stage(args, name, attempt, environment)
            write_json(attempt / 'result.json', result)
            summary['stages'][name] = result
            write_summary(directory, summary)
            print(f'[{name}] {result["status"]}; wall={result["wall_seconds"]:.1f}s', flush=True)
            if result['interrupted']:
                return 75
        print(f'Summary: {summary_file}', flush=True)
        return int(any(summary['stages'][name]['status'] != 'passed' for name in selected))


if __name__ == '__main__':
    sys.exit(main())
