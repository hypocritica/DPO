"""Train one portable SD1.5/SDXL preference experiment on allocated GPUs."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from self_correct_dpo.config import add_training_arguments, training_config
from self_correct_dpo.resources import CONFIG_DIR
from self_correct_dpo.common import read_json, require_gpu_allocation
from self_correct_dpo.data.pairs import load_pairs
from self_correct_dpo.training import check_rows, train, validate_config


@contextmanager
def training_lock(out):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # All DDP ranks belong to the rank-zero owner of this run.
    if int(os.environ.get("RANK", "0")) != 0:
        yield
        return
    with open(out.parent / ("." + out.name + ".lock"), "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another training process owns {out}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / 'multiview-C.json'))
    parser.add_argument("--out", '--output_dir', dest='out', required=True)
    parser.add_argument("--resume", '--resume_from_checkpoint', dest='resume', help="latest, or a complete checkpoint from the same output run")
    parser.add_argument("--extend-from", help="Complete rolling checkpoint from a parent run; requires a fresh --out")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execution", choices=["slurm", "standalone"], default="slurm")
    parser.add_argument("--run-label")
    parser.add_argument("--stop-after-step", type=int, help="Pause after this absolute optimizer step without changing the configured training budget")
    parser.add_argument("--internal-stage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument('--stage-id',help=argparse.SUPPRESS)
    parser.add_argument('--print-config', action='store_true', help='Print resolved settings without loading models or data')
    add_training_arguments(parser)
    args = parser.parse_args()
    config = training_config(args)
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    config = validate_config(config)
    if args.resume == 'latest':
        args.resume = str(Path(args.out).resolve() / 'checkpoints/latest.json')
    if args.stop_after_step is not None and not 1 <= args.stop_after_step <= config["training"]["max_steps"]:
        parser.error('--stop-after-step must be between 1 and configured max_steps')
    all_rows = load_pairs(config["data"]["pairs"], verify_files=True)
    rows = [r for r in all_rows if r["split"] == "train"]
    validation_rows = [r for r in all_rows if r["split"] == "validation"]
    if config["data"].get("require_human_preference") and not validation_rows:
        raise ValueError("Human baseline needs a nonempty independent validation split")
    check_rows(rows, config)
    monitored = config.get('evaluation',{}).get('enabled',False)
    world=config['training'].get('world_size',1)
    runtime_world=int(os.environ.get('WORLD_SIZE','1'))
    if runtime_world>1 and monitored and not args.internal_stage:
        parser.error('For monitoring run python train.py --launcher accelerate once; direct accelerate launch uses --no-evaluation')
    if runtime_world>1 and runtime_world!=world or args.internal_stage and runtime_world!=world:
        parser.error('Internal worker WORLD_SIZE differs from training.world_size')
    monitor_identity=None
    if monitored and not args.internal_stage:
        from self_correct_dpo.runtime.pipeline import monitor_preflight
        monitor_identity=monitor_preflight(config,rows)
    if Path(args.out).exists() and not args.resume:
        raise FileExistsError(args.out)
    print(json.dumps({"family": config["model"]["family"], "objective": config["training"]["objective"],
                     "training_pairs": len(rows), "steps": config["training"]["max_steps"], "seed": config["seed"]}))
    if args.check_only:
        return
    execution = {"mode": args.execution, "run_label": args.run_label}
    require_gpu_allocation(execution)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for research training")
    if runtime_world==1 and world>torch.cuda.device_count():
        raise RuntimeError(f'training.world_size={world} requires that many visible allocated GPUs; found {torch.cuda.device_count()}')
    if monitored and not args.internal_stage:
        from self_correct_dpo.runtime.pipeline import run_pipeline
        with training_lock(str(args.out)+'.workflow'):
            from self_correct_dpo.runtime.checkpoints import atomic_json
            if args.resume:
                from self_correct_dpo.runtime.checkpoints import recover_run
                recover_run(args.out)
            config_path = Path(args.out).resolve().parent / ('.' + Path(args.out).name + '.config.json')
            atomic_json(config_path, config)
            result=run_pipeline(config,config_path,args.out,execution=execution,
                resume=args.resume,extend_from=args.extend_from,stop_after_step=args.stop_after_step,
                monitor_identity=monitor_identity)
        print(json.dumps(result))
        if not result['complete'] and not result.get('planned_pause',False): raise SystemExit(75)
        return
    if world>1 and runtime_world==1 and not args.internal_stage:
        from self_correct_dpo.runtime.pipeline import training_command, stage_interrupted
        from self_correct_dpo.evaluation.runner import run_command
        with training_lock(str(args.out)+'.workflow'):
            from self_correct_dpo.runtime.checkpoints import atomic_json
            config_path = Path(args.out).resolve().parent / ('.' + Path(args.out).name + '.config.json')
            atomic_json(config_path, config)
            command=training_command(config,config_path,args.out,execution=execution,
                stop_after_step=args.stop_after_step,resume=args.resume,extend_from=args.extend_from)
            run_command(command)
            if stage_interrupted(command,args.out):raise SystemExit(75)
        return
    from self_correct_dpo.runtime.distributed import cleanup
    try:
        with training_lock(args.out):
            result = train(config, rows, args.out, resume=args.resume, extend_from=args.extend_from, execution=execution, validation_rows=validation_rows, stop_after_step=args.stop_after_step)
    finally:
        cleanup()
    if int(os.environ.get("RANK", "0")) == 0:
        if args.stage_id:
            from self_correct_dpo.runtime.checkpoints import atomic_json
            atomic_json(Path(args.out)/'stage-result.json',{'stage_id':args.stage_id,
                **{key:result[key] for key in ('run_id','step','complete','planned_pause','interrupted')}})
        print(json.dumps({k:v for k,v in result.items() if k != "cost"}))
    if not result["complete"] and not result.get("planned_pause", False):
        # A graceful distributed stop must let all workers finish before the
        # coordinator converts the committed stage result into exit status 75.
        if runtime_world==1:raise SystemExit(75)


if __name__ == "__main__":
    main()
