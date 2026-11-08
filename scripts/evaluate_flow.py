"""Evaluate committed Flow snapshots on another GPU, reusing the base pipeline.

Training does not stop or acquire this job's locks. --watch waits for permanent
step-N snapshots; rolling/latest and pending checkpoint directories are ignored.
"""
import argparse
import copy
import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


class PipelineView:
    def __init__(self, pipe, component, base):
        self.pipe, self.component, self.base = pipe, component, base

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def __call__(self, **kwargs):
        from peft import PeftModel
        model = getattr(self.pipe, self.component)
        if self.base and isinstance(model, PeftModel):
            with model.disable_adapter():
                return self.pipe(**kwargs)
        return self.pipe(**kwargs)


class EvaluationSession:
    """One base pipeline and one scorer per job; only small adapters change."""
    def __init__(self, scorer_device='cuda'):
        self.pipe = None
        self.key = self.adapter = None
        self.counter = 0
        self.scorers = {}
        self.scorer_device = scorer_device
        self.pipeline_loads = 0

    def load_pipeline(self, config, arm, torch):
        from peft import PeftModel
        from self_correct_dpo.common import digest
        from self_correct_dpo.evaluation import render
        key = digest({'model': render.model_spec(config), 'sampling': render.effective_sampling(config)})
        component = render.FAMILIES[config['model']['family']][2]
        adapter = config['aligned']['path'] if arm == 'aligned' else None
        if self.pipe is None:
            self.pipe = render.load_pipeline(config, arm, torch)
            self.key, self.adapter = key, adapter
            self.pipeline_loads += 1
        elif self.key != key:
            raise ValueError('A persistent evaluation session cannot change base model or sampling')
        elif adapter is not None and adapter != self.adapter:
            # Reinstall placement hooks after changing the PEFT wrapper. Base
            # weights stay in this process; no from_pretrained(base) call here.
            self.pipe.remove_all_hooks()
            model = getattr(self.pipe, component)
            name = f'offline_{self.counter}'
            self.counter += 1
            if isinstance(model, PeftModel):
                previous = list(model.peft_config)
                model.load_adapter(adapter, adapter_name=name, is_trainable=False, local_files_only=True)
                model.set_adapter(name)
                for old in previous:
                    model.delete_adapter(old)
            else:
                model = PeftModel.from_pretrained(model, adapter, adapter_name=name,
                                                  is_trainable=False, local_files_only=True)
                setattr(self.pipe, component, model)
            model.requires_grad_(False).eval()
            if config['sampling'].get('cpu_offload', False):
                self.pipe.enable_model_cpu_offload(device='cuda')
            else:
                self.pipe.to('cuda')
            self.adapter = adapter
        # Schedulers may retain their previous sampling cursor.
        self.pipe.scheduler = type(self.pipe.scheduler).from_config(self.pipe.scheduler.config)
        return PipelineView(self.pipe, component, arm == 'base')

    def load_scorer(self, metric, spec):
        from self_correct_dpo.common import digest
        from self_correct_dpo.evaluation.scorers import load_scorer
        key = digest({'metric': metric, 'spec': spec})
        if key not in self.scorers:
            self.scorers[key] = load_scorer(metric, spec, device=self.scorer_device)
        return self.scorers[key]

    def run_command(self, command):
        module = command[command.index('-m') + 1]
        argv = command[command.index(module) + 1:]
        if module == 'self_correct_dpo.evaluation.render':
            from self_correct_dpo.evaluation.render import main
            main(argv, pipeline_loader=self.load_pipeline)
        elif module == 'self_correct_dpo.evaluation.metrics':
            if Path(command[0]).resolve() != Path(sys.executable).resolve():
                raise ValueError('Persistent Flow evaluation requires the scorer in the same Python environment')
            from self_correct_dpo.evaluation.metrics import main
            main(argv, scorer_factory=self.load_scorer)
        else:
            raise ValueError(f'Unexpected evaluation subprocess: {module}')


def committed_adapter(run, step, run_id):
    """Read immutable published snapshots, never the trainer's rolling slot."""
    if step == 0:
        return None, None
    from self_correct_dpo.training import checkpoint_manifest
    path = Path(run) / 'checkpoints' / f'step-{step:07d}'
    if not (path / 'checkpoint.json').is_file():
        return None
    saved = checkpoint_manifest(path)
    if saved['run_id'] != run_id or saved['step'] != step:
        raise ValueError('Checkpoint run/step differs from requested training run')
    if not (path / 'adapter').is_dir():
        raise ValueError('Published checkpoint has no adapter')
    return path / 'adapter', saved['checkpoint_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Training output directory, not a config path')
    parser.add_argument('--out', type=Path, help='Independent evaluation output; default RUN/offline-eval-v1')
    parser.add_argument('--steps', type=int, nargs='+', help='Default uses the frozen panel schedule including base step 0')
    parser.add_argument('--watch', action='store_true', help='Wait for future permanent snapshots without restarting the model')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--check-only', action='store_true', help='CPU-only metadata and availability plan; no output writes')
    parser.add_argument('--scorer-device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--report-to', choices=('none', 'wandb', 'tensorboard', 'all'), default='none')
    parser.add_argument('--execution', choices=('standalone', 'slurm'), default='standalone')
    parser.add_argument('--run-label', default='flow-independent-evaluation')
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        parser.error('--poll-seconds must be in (0, 60]')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one process on a separate GPU, without torchrun')
    from self_correct_dpo.common import digest, output_lock, read_json, require_gpu_allocation, sha256, source_file, model_artifact_identity
    from self_correct_dpo.runtime.checkpoints import atomic_json
    from self_correct_dpo.runtime.pipeline import evaluation_steps, update_curve
    from self_correct_dpo.evaluation.runner import evaluate_checkpoint, validate_evaluation, verify_complete
    run = args.run.expanduser().resolve()
    while not (run / 'training.json').is_file():
        if not args.watch or args.check_only:
            raise FileNotFoundError(f'Training metadata is not published yet: {run / "training.json"}')
        print(f'Waiting for training metadata: {run}', flush=True)
        time.sleep(args.poll_seconds)
    training = read_json(run / 'training.json')
    config, run_id = training['config'], training['run_id']
    if config['model']['family'] not in ('sd35_medium', 'zimage'):
        raise ValueError('Expected a Flow training run')
    settings = config['evaluation']
    if not all(k in settings for k in ('prompts', 'sampling', 'scorers_config')):
        raise ValueError('Run has no frozen evaluation recipe; use the new train-only launcher for new experiments')
    planned = sorted(set(args.steps if args.steps is not None else evaluation_steps(config['training']['max_steps'], settings)))
    if not planned or any(step < 0 or step > config['training']['max_steps'] for step in planned):
        parser.error('Evaluation steps must be within the training budget')
    available = {step: committed_adapter(run, step, run_id) for step in planned}
    if args.check_only:
        print(json.dumps({'run': str(run), 'steps': planned,
                          'ready_steps': [s for s in planned if available[s] is not None],
                          'pending_steps': [s for s in planned if available[s] is None],
                          'training_evaluation_enabled': settings.get('enabled', False)}, indent=2))
        return 0
    execution = {'mode': args.execution, 'run_label': args.run_label}
    require_gpu_allocation(execution)
    out = (args.out or run / 'offline-eval-v1').expanduser().resolve()
    if out == run or out == run / 'evaluations':
        raise ValueError('Use an independent evaluation directory')
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'worker'):
        inputs = validate_evaluation(settings['prompts'], settings['scorers_config'], ['pickscore'],
                                     expected_prompts=settings.get('expected_prompts', 64))
        base_identity = model_artifact_identity(config['model'])
        if base_identity != training['base_identity']:
            raise ValueError('Evaluation base model differs from the recorded training base')
        identity = {'training_run_id': run_id, 'steps': planned, 'settings': settings,
                    'inputs': inputs, 'base_identity': base_identity,
                    'scorer_device': args.scorer_device, 'report_to': args.report_to,
                    'versions': {name: importlib.metadata.version(name) for name in
                                 ('torch', 'diffusers', 'transformers', 'peft', 'Pillow')},
                    'source_sha256': {name: sha256(source_file(name)) for name in
                                      ('render.py', 'evaluation.py', 'metrics.py', 'scorers.py',
                                       'common.py', 'zimage_compat.py', 'tracking.py', 'pipeline.py')}}
        identity['source_sha256']['evaluate_flow.py'] = sha256(__file__)
        metadata = out / 'evaluation-job.json'
        if metadata.exists() and read_json(metadata) != identity:
            raise ValueError('Independent evaluation identity changed; choose a new --out')
        atomic_json(metadata, identity)
        # Validation is independent of the trainer; a held-out panel is still required.
        from self_correct_dpo.training import load_preflight
        preflight_path = run.with_name('.' + run.name + '.preflight.pkl')
        if preflight_path.is_file():
            saved = load_preflight(config, run)
            train_rows = saved['rows']
        else:
            from self_correct_dpo.data.validation import load_training_pairs
            all_rows = load_training_pairs(config['data']['pairs'], mode='none')
            train_rows = [row for row in all_rows if row['split'] == 'train']
            if config['data'].get('validation_report'):
                from self_correct_dpo.flow_receipts import data_receipt
                receipt = data_receipt(config['data']['validation_report'], config['data']['pairs'])
                pairs_id = digest({'manifest_sha256': receipt['dataset_identity']['pairs']['sha256'], 'split': 'train'})
            else:
                pairs_id = digest(train_rows)
            if pairs_id != training['pairs_id']:
                raise ValueError('Training captions changed since the training identity was recorded')
        from self_correct_dpo.data.benchmarks import prompt_key
        from self_correct_dpo.common import load_suite
        train_keys = {prompt_key(p['prompt']) for row in train_rows for p in row['prompts']}
        if any(prompt_key(row['prompt']) in train_keys for row in load_suite(settings['prompts'])):
            raise ValueError('Evaluation panel overlaps a training caption')
        render_preflight = {'inputs': inputs, 'base_identity': base_identity,
                            'code_sha256': {name: sha256(source_file(name)) for name in
                                            ('evaluation.py', 'render.py', 'metrics.py', 'scorers.py',
                                             'common.py', 'zimage_compat.py')}}
        tracker = None
        session = EvaluationSession(args.scorer_device)
        bindings_path = out / 'checkpoint-bindings.json'
        bindings = read_json(bindings_path) if bindings_path.exists() else {}
        try:
            while True:
                pending = []
                for step in planned:
                    target = out / 'evaluations' / f'step-{step:07d}'
                    if (target / 'complete.json').is_file():
                        verify_complete(target)
                        continue
                    if pending:
                        pending.append(step)
                        continue
                    snapshot = committed_adapter(run, step, run_id)
                    if snapshot is None:
                        pending.append(step)
                        continue
                    adapter, checkpoint_id = snapshot
                    if str(step) in bindings and bindings[str(step)] != checkpoint_id:
                        raise ValueError('A previously bound checkpoint changed')
                    bindings[str(step)] = checkpoint_id
                    atomic_json(bindings_path, bindings)
                    print(f'Evaluating step {step}; base pipeline loads so far: {session.pipeline_loads}', flush=True)
                    summary = evaluate_checkpoint(config['model'], adapter, settings['prompts'],
                        settings['scorers_config'], ['pickscore'], settings['sampling'], target,
                        step=step, keep_images=True, execution=execution,
                        expected_prompts=settings.get('expected_prompts', 64), command_runner=session.run_command,
                        preflight=render_preflight)
                    update_curve(out)
                    if args.report_to != 'none':
                        from self_correct_dpo.runtime.tracking import ExperimentTracker
                        if tracker is None:
                            tracking = copy.deepcopy(config)
                            tracking['training']['logging'] = args.report_to
                            wandb = tracking['training'].setdefault('wandb', {})
                            wandb['name'] = (wandb.get('name') or run.name) + '-evaluation'
                            tracker = ExperimentTracker(tracking, out, digest(identity), start_step=step,
                                                        resume=True, evaluation=True)
                        tracker.add_scalar('evaluation/pickscore', summary['metrics']['pickscore']['mean'], step)
                        if tracker.run:
                            from self_correct_dpo.common import read_jsonl
                            render = target / 'render'
                            images = []
                            for row in read_jsonl(render / 'manifest.jsonl')[:config['training'].get('wandb', {}).get('image_count', 8)]:
                                path = (render / row['image']).resolve()
                                if not path.is_relative_to(render.resolve() / 'images'):
                                    raise ValueError('Evaluation image escapes its render directory')
                                images.append((path, f'{row["scene_id"]} | seed={row["seed"]} | {row["prompt"]}'))
                            tracker.add_images('evaluation/images', images, step)
                        tracker.flush()
                    print(f'EVAL step={step} PickScore={summary["metrics"]["pickscore"]["mean"]} base_loads={session.pipeline_loads}', flush=True)
                atomic_json(out / 'status.json', {'training_run_id': run_id, 'pending_steps': pending,
                            'pipeline_loads_this_process': session.pipeline_loads, 'complete': not pending})
                if not pending:
                    return 0
                if (run / 'complete.json').is_file():
                    raise ValueError(f'Training finished but permanent snapshots are missing: {pending}')
                if not args.watch:
                    print(f'Pending snapshots: {pending}; rerun with --watch to wait.', flush=True)
                    return 0
                print(f'Waiting for snapshots: {pending}; keeping loaded models in this process.', flush=True)
                time.sleep(args.poll_seconds)
        finally:
            if tracker is not None:
                tracker.close()


if __name__ == '__main__':
    sys.exit(main())
