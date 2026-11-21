"""Run one appendix task inside its allocated SLAI container."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.resources import load_resources
from self_correct_dpo.runtime.distributed import subprocess_environment

GENERATION = ("direct", "seed23", "seed42", "step800", "step2400", "step3200")
PROBES = ("sd15-gradient", "sd35-gradient", "sd35-stage")


def shard_group(value):
    try:
        part, total = map(int, value.split("/"))
        if 1 <= part <= total:
            return part - 1, total
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected 1/N through N/N, e.g. 2/4")


def gpu_tokens(requested, environ, interpreter):
    mask = environ.get("CUDA_VISIBLE_DEVICES")
    if requested is None and mask is None:
        count = int(subprocess.check_output(
            [interpreter, "-c", "import torch; print(torch.cuda.device_count())"], env=environ, text=True))
        return gpu_tokens(",".join(map(str, range(count))), environ, interpreter)
    ids = (requested if requested is not None else mask).split(",")
    allowed = set(mask.split(",")) if mask is not None else set(ids)
    if not all(ids) or "-1" in ids or len(set(ids)) != len(ids) or not set(ids) <= allowed:
        raise ValueError(f"Invalid GPU list {ids} for CUDA_VISIBLE_DEVICES={mask!r}")
    return ids


def run_probes(commands, output, env):
    """A process per allocated device, with separate logs and failure propagation."""
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers = []
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for gpu, index, cmd in commands:
            path = logs / f"shard-{index}-{time.time_ns()}-{os.getpid()}.log"
            print(f"GPU {gpu}, shard {index}: {path}", flush=True)
            with path.open("x") as log:
                proc = subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log,
                                        stderr=subprocess.STDOUT, cwd=ROOT)
            workers.append((proc, path))
        while workers:
            for proc, path in workers[:]:
                status = proc.poll()
                if status is not None:
                    if status:
                        raise SystemExit(f"Worker exited {status}. See {path}")
                    workers.remove((proc, path))
            if workers:
                time.sleep(1)
    finally:
        for proc, _ in workers:
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in workers:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", choices=("prepare-generation", *GENERATION, *PROBES))
    p.add_argument("--out", type=Path, help="appendix root (default MVPO_APPENDIX_ROOT)")
    p.add_argument("--gpus", help="comma-separated tokens from CUDA_VISIBLE_DEVICES")
    p.add_argument("--group", type=shard_group, help="probe job number 1/N through N/N")
    p.add_argument("--smoke", action="store_true", help="small independent protocol for a probe job")
    p.add_argument("--prepare-only", action="store_true", help="verify probe inputs and freeze the pair manifest on CPU")
    p.add_argument("--stage-batch-size", type=int, help="views per forward pass for sd35-stage")
    p.add_argument("--scratch-dir", type=Path, help="existing fast local disk directory for gradient block means")
    p.add_argument("--dry-run", action="store_true", help="print commands; pass --gpus when running off-cluster")
    args = p.parse_args()
    if args.stage_batch_size is not None and (args.job != "sd35-stage" or args.stage_batch_size < 1):
        p.error("--stage-batch-size requires sd35-stage and a positive value")
    if args.scratch_dir and args.job not in ("sd15-gradient", "sd35-gradient"):
        p.error("--scratch-dir requires a gradient job")
    registry = load_resources()
    out = (args.out or Path(registry["MVPO_APPENDIX_ROOT"])).expanduser().resolve()
    env = dict(subprocess_environment(os.environ), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    env.setdefault("OMP_NUM_THREADS", "2")
    for key in ("SCD_SD15_PY", "SCD_FLOW_PY", "SCD_HPSV3_PY"):
        env[key] = registry[key]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    if args.job not in PROBES:
        if args.group or args.smoke or args.prepare_only:
            p.error("--group, --smoke and --prepare-only apply to probe jobs")
        prepare = args.job == "prepare-generation"
        cmd = [registry["SCD_SD15_PY"], str(ROOT / "scripts/appendix_generation.py"),
               "prepare" if prepare else "run", "--out", str(out / "generation")]
        if not prepare:
            ids = gpu_tokens(args.gpus, env, registry["SCD_FLOW_PY"])
            env["SCD_EVAL_GPUS"] = ",".join(ids)
            cmd.extend(["--group", args.job])
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            os.execve(cmd[0], cmd, env)
        return
    py = registry["MVPO_SD15_PROBE_PY" if args.job == "sd15-gradient" else "MVPO_FLOW_PROBE_PY"]
    output = (out / "smoke" if args.smoke else out) / args.job
    cmd = [py, str(ROOT / "scripts/mvpo_appendix_probe.py"), args.job, "--output", str(output)]
    if args.stage_batch_size:
        cmd += ["--stage-batch-size", str(args.stage_batch_size)]
    if args.scratch_dir:
        cmd += ["--scratch-dir", str(args.scratch_dir.expanduser().resolve())]
    if args.smoke:
        cmd += ["--pair-count", "4"]
        if args.job == "sd35-stage":
            cmd += ["--stage-views", "4"]
        else:
            cmd += ["--check-initialization", "--source-blocks", "2", "--source-views-per-block", "24",
                    "--audit-blocks", "2", "--audit-views-per-block", "8"]
    if args.prepare_only:
        cmd += ["--prepare-only"]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            os.execve(py, cmd, env)
        return
    part, groups = args.group or (0, 1)
    devices = gpu_tokens(args.gpus, env, py)
    pair_count = 4 if args.smoke else {"sd15-gradient": 16, "sd35-gradient": 8, "sd35-stage": 128}[args.job]
    if groups > pair_count:
        p.error(f"--group specifies {groups} jobs for only {pair_count} preference pairs")
    stride = min(len(devices) * groups, pair_count)
    commands = []
    for rank, gpu in enumerate(devices):
        index = rank * groups + part
        if index >= pair_count:
            continue
        shard = cmd + ["--shard-index", str(index), "--shard-count", str(stride)]
        print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(shard)}", flush=True)
        commands.append((gpu, index, shard))
    if not args.dry_run:
        run_probes(commands, output, env)


if __name__ == "__main__":
    main()
