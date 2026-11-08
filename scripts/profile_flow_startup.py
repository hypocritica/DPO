"""Profile the existing Flow backend without changing training or checkpoints.

Default: metadata only. --load uses an already allocated GPU; torchrun supports
the same concurrent two-rank load as training. Each rank writes its own JSONL.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone, timedelta
import importlib
import json
import os
from pathlib import Path
import resource
import socket
import struct
import sys
import time
from unittest.mock import patch


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_snapshot():
    """Read container-visible limits/counters, including tighter ancestor limits."""
    root = Path('/sys/fs/cgroup')
    membership = read_text('/proc/self/cgroup') or ''
    relative = next((s[3:] for s in membership.splitlines() if s.startswith('0::')), '')
    current = root / relative.lstrip('/')
    if not current.is_dir():
        current = root
    result = {}
    for directory in [current, *current.parents]:
        if directory != root and root not in directory.parents:
            continue
        fields = {name: read_text(directory / name) for name in
                  ('cpu.max', 'cpu.stat', 'cpuset.cpus.effective', 'memory.max',
                   'memory.current', 'memory.peak', 'memory.events', 'memory.swap.current')}
        if any(v is not None for v in fields.values()):
            result[str(directory)] = {k: v for k, v in fields.items() if v is not None}
    if not result:
        result['v1'] = {name: read_text(root / name) for name in
                        ('cpu/cpu.cfs_quota_us', 'cpu/cpu.cfs_period_us', 'cpu/cpu.stat',
                         'memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes',
                         'memory/memory.max_usage_in_bytes', 'memory/memory.failcnt')}
    return result


def snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status = read_text('/proc/self/status') or ''
    proc = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
            for line in status.splitlines() if line.startswith(('VmRSS:', 'VmHWM:', 'Threads:'))}
    io = {}
    for line in (read_text('/proc/self/io') or '').splitlines():
        key, value = line.split(':', 1)
        if key in ('rchar', 'read_bytes', 'syscr'):
            io[key] = int(value)
    result = {'process': proc, 'io': io, 'cpu_user_s': usage.ru_utime,
              'cpu_system_s': usage.ru_stime, 'major_faults': usage.ru_majflt,
              'minor_faults': usage.ru_minflt, 'cgroups': cgroup_snapshot()}
    torch = sys.modules.get('torch')
    if torch is not None:
        result['torch_threads'] = torch.get_num_threads()
        if torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            result['cuda'] = {'device': torch.cuda.current_device(), 'free_bytes': free,
                             'total_bytes': total, 'allocated_bytes': torch.cuda.memory_allocated(),
                             'reserved_bytes': torch.cuda.memory_reserved(),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    return result


class Trace:
    def __init__(self, handle):
        self.handle = handle
        self.started = time.perf_counter()
        self.rank = int(os.environ.get('RANK', '0'))

    def emit(self, event, **fields):
        row = {'event': event, 'rank': self.rank, 'pid': os.getpid(),
               'utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_s': time.perf_counter() - self.started, **fields}
        line = json.dumps(row, ensure_ascii=False)
        self.handle.write(line + '\n')
        self.handle.flush()
        print(line, flush=True)

    @contextmanager
    def span(self, name, **fields):
        before = snapshot()
        self.emit('begin', stage=name, snapshot=before, **fields)
        start = time.perf_counter()
        try:
            yield
        except BaseException as error:
            self.emit('error', stage=name, seconds=time.perf_counter()-start,
                      error_type=type(error).__name__, snapshot=snapshot())
            raise
        else:
            after = snapshot()
            deltas = {key: after[key]-before[key] for key in
                      ('cpu_user_s', 'cpu_system_s', 'major_faults', 'minor_faults')}
            self.emit('end', stage=name, seconds=time.perf_counter()-start,
                      delta=deltas, snapshot=after)


def inventory(model):
    """Read safetensors headers only; never touch the large tensor payloads."""
    result = []
    for file in sorted(model.rglob('*.safetensors')):
        with file.open('rb') as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f'Truncated safetensors header: {file}')
            length = struct.unpack('<Q', length_bytes)[0]
            if length > 16 * 1024**2:
                raise ValueError(f'Unexpectedly large safetensors header: {file}')
            header = json.loads(handle.read(length))
        dtypes = Counter(t['dtype'] for name, t in header.items() if name != '__metadata__')
        result.append({'file': str(file.relative_to(model)), 'bytes': file.stat().st_size,
                       'tensor_dtypes': dict(dtypes)})
    if not result:
        raise ValueError(f'No safetensors weights found: {model}')
    return result


def model_mount(model):
    matches = []
    for line in (read_text('/proc/self/mountinfo') or '').splitlines():
        left, right = line.split(' - ', 1)
        mount = Path(left.split()[4].replace('\\040', ' '))
        if model == mount or mount in model.parents:
            # Do not expose mount sources/options, which may contain credentials.
            matches.append({'mount_point': str(mount), 'fs_type': right.split()[0]})
    return max(matches, key=lambda item: len(item['mount_point'])) if matches else None


def probe_reads(model, weights, mib, trace):
    """Bounded read-only probe. Run separately AFTER the initial load measurement."""
    buffer = bytearray(1024**2)
    for entry in weights:
        target = min(entry['bytes'], mib * 1024**2)
        start = time.perf_counter()
        done = 0
        with trace.span('read_probe', file=entry['file'], requested_bytes=target):
            with (model / entry['file']).open('rb', buffering=0) as handle:
                while done < target:
                    count = handle.readinto(memoryview(buffer)[:min(len(buffer), target-done)])
                    if not count:
                        break
                    done += count
        seconds = time.perf_counter() - start
        trace.emit('read_probe_result', file=entry['file'], bytes=done, seconds=seconds,
                   MiB_per_second=done/1024**2/max(seconds, 1e-9),
                   caveat='prefix_read_including_open; may_hit_cache; not_full_checkpoint_cold_bandwidth')


@contextmanager
def instrument_loaders(index, trace):
    """Wrap exactly the public classes selected by this pipeline's model_index."""
    with ExitStack() as stack:
        for component, spec in index.items():
            if not isinstance(spec, list) or len(spec) != 2 or spec[0] not in ('diffusers', 'transformers'):
                continue
            cls = getattr(importlib.import_module(spec[0]), spec[1])
            original = cls.from_pretrained
            def wrapped(_cls, *args, _original=original, _component=component, **kwargs):
                dtype = kwargs.get('torch_dtype', kwargs.get('dtype'))
                with trace.span('load.' + _component, requested_dtype=str(dtype),
                                subfolder=kwargs.get('subfolder')):
                    return _original(*args, **kwargs)
            stack.enter_context(patch.object(cls, 'from_pretrained', classmethod(wrapped)))
        # Separate CPU component loading/pipeline assembly from GPU placement.
        pipeline_cls = getattr(importlib.import_module('diffusers'), index['_class_name'])
        original_to = pipeline_cls.to
        def move_pipeline(self, *args, **kwargs):
            with trace.span('pipeline.to_device'):
                result = original_to(self, *args, **kwargs)
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                return result
        stack.enter_context(patch.object(pipeline_cls, 'to', move_pipeline))
        yield


def load_backend(config, args, trace, index):
    world, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    with trace.span('import_torch_and_backend'):
        import torch
        import torch.distributed as dist
        from self_correct_dpo.common import require_gpu_allocation
        from self_correct_dpo.models import PreferenceBackend
    require_gpu_allocation({'mode': args.execution, 'run_label': 'flow-startup-profile'})
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    local = int(os.environ.get('LOCAL_RANK', '0'))
    with trace.span('cuda_and_process_group'):
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
        if world > 1:
            dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
        torch.cuda.synchronize()
    trace.emit('runtime', gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
               torch_threads=torch.get_num_threads(), cuda_runtime=torch.version.cuda,
               world_size=world, serialized_load=args.serial_load)
    try:
        with trace.span('import_component_classes'):
            for spec in index.values():
                if isinstance(spec, list) and len(spec) == 2 and spec[0] in ('diffusers', 'transformers'):
                    getattr(importlib.import_module(spec[0]), spec[1])
        with instrument_loaders(index, trace):
            def load():
                with trace.span('backend.total', family=config['model']['family'],
                                model_dtype=config['model'].get('transformer_dtype'),
                                activation_dtype=config['training']['dtype'],
                                offload_frozen=config['training'].get('offload_frozen', False)):
                    value = PreferenceBackend(config['model'], config['training'], device)
                    torch.cuda.synchronize()
                    return value
            if args.serial_load and world > 1:
                for turn in range(world):
                    if rank == turn:
                        backend = load()
                    with trace.span('wait_for_load_rank_' + str(turn)):
                        dist.barrier(device_ids=[local])
            else:
                backend = load()
            if world > 1:
                with trace.span('wait_for_other_ranks'):
                    dist.barrier(device_ids=[local])
                with trace.span('ddp_wrap_and_initial_sync'):
                    backend.enable_distributed()
                    torch.cuda.synchronize()
        trace.emit('complete', scope='model_load_and_ddp_only_no_training_or_resume', snapshot=snapshot())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Existing frozen training JSON, read-only')
    parser.add_argument('--out', type=Path, required=True, help='New report directory; each rank refuses to overwrite its file')
    parser.add_argument('--load', action='store_true', help='Actually load the backend on already allocated GPUs')
    parser.add_argument('--execution', choices=('slurm', 'standalone'), default='slurm')
    parser.add_argument('--cpu-threads', type=int, help='Optional controlled CPU-thread comparison; default unchanged')
    parser.add_argument('--serial-load', action='store_true', help='Load ranks sequentially to diagnose concurrent load contention')
    parser.add_argument('--read-mib-per-shard', type=int, default=0,
                        help='Separate read-only storage probe, 0..64 MiB per shard; warms cache')
    args = parser.parse_args()
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if not args.load and (args.cpu_threads is not None or args.serial_load):
        parser.error('--cpu-threads/--serial-load require --load')
    if not 0 <= args.read_mib_per_shard <= 64 or (args.load and args.read_mib_per_shard):
        parser.error('--read-mib-per-shard must be 0..64 and cannot be combined with --load')
    if args.read_mib_per_shard and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('Run the bounded storage probe as a single process')
    # The script can be copied alone to the existing checkout; no production files are patched.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    config = json.loads(args.config.read_text())
    model = Path(config['model']['path']).expanduser().resolve(strict=True)
    index = json.loads((model / 'model_index.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    with (args.out / f'rank-{rank}.jsonl').open('x') as handle:
        trace = Trace(handle)
        trace.emit('environment', hostname=socket.gethostname(), python=sys.version.split()[0],
                   config=str(args.config.resolve()), model=str(model), mount=model_mount(model),
                   cpu_affinity_count=len(os.sched_getaffinity(0)),
                   thread_environment={key: os.environ.get(key) for key in
                                       ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                   snapshot=snapshot())
        with trace.span('model_header_inventory'):
            weights = inventory(model)
        trace.emit('model_inventory', components=index, weights=weights,
                   disk_weight_GiB=sum(w['bytes'] for w in weights) / 1024**3)
        if args.read_mib_per_shard:
            probe_reads(model, weights, args.read_mib_per_shard, trace)
        if args.load:
            load_backend(config, args, trace, index)
        else:
            trace.emit('complete', scope='metadata_and_optional_read_probe_no_torch_import_or_gpu_load')


if __name__ == '__main__':
    main()
