"""Rank-zero TensorBoard/W&B logging with the optimizer step as the x axis."""
from pathlib import Path


class ExperimentTracker:
    def __init__(self, config, out, run_id, *, start_step=0, resume=False, evaluation=False):
        self.tb = self.run = None
        self.pending = {}
        self.step = start_step
        self.evaluation = evaluation
        self.offline = False
        out = Path(out)
        training = config['training']
        report = training.get('logging', 'tensorboard')
        if report in ('tensorboard', 'all'):
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(out / 'tensorboard'),
                purge_step=start_step + 1 if resume and not evaluation else None)
        if report in ('wandb', 'all'):
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError('W&B logging requires: pip install wandb') from error
            settings = training.get('wandb', {})
            self.offline = settings.get('mode', 'offline') == 'offline'
            self.run = wandb.init(project=settings.get('project', 'self-correct-dpo'),
                name=settings.get('name') or out.name, id=run_id[:32],
                resume=None if self.offline else 'allow', config=config,
                group=settings.get('group'), tags=settings.get('tags'),
                dir=str(out), mode=settings.get('mode', 'offline'))
            self.run.define_metric('optimizer_step')
            self.run.define_metric('*', step_metric='optimizer_step')

    def add_text(self, name, text, step):
        if self.tb:
            self.tb.add_text(name, text, step)

    def add_scalar(self, name, value, step):
        if self.pending and self.step != step:
            self.flush()
        self.step = step
        if self.tb:
            self.tb.add_scalar(name, value, step)
        if self.run:
            self.pending[name] = float(value)

    def add_images(self, name, images, step):
        """Queue images with captions in the SAME history row as evaluation metrics."""
        if not self.run or not images:
            return
        if self.pending and self.step != step:
            self.flush()
        self.step = step
        import wandb
        self.pending[name] = [wandb.Image(str(path), caption=caption) for path, caption in images]

    def flush(self):
        if self.tb:
            self.tb.flush()
        if self.run and self.pending:
            # Offline training/evaluation segments share one run ID. Give each
            # record a global history step so sequential sync --append does not
            # restart the W&B history counter at every evaluation boundary.
            options = {'step': 2 * self.step + int(self.evaluation)} if self.offline else {}
            self.run.log({'optimizer_step': self.step, **self.pending}, **options)
            self.pending.clear()

    def close(self):
        self.flush()
        if self.tb:
            self.tb.close()
        if self.run:
            self.run.finish()


def log_evaluation(config, out, step, summary):
    from self_correct_dpo.common import read_json, read_jsonl
    run_id = read_json(Path(out) / 'training.json')['run_id']
    tracker = ExperimentTracker(config, out, run_id, start_step=step, resume=True, evaluation=True)
    try:
        for metric, values in summary['metrics'].items():
            tracker.add_scalar('evaluation/' + metric, values['mean'], step)
        # Stable first scenes/seeds allow visual comparison across servers and steps.
        # No extra generation, image upload or model load happens here.
        count = config['training'].get('wandb', {}).get('image_count', 8)
        render = Path(out) / 'evaluations' / f'step-{step:07d}' / 'render'
        manifest = render / 'manifest.jsonl'
        if tracker.run and count and summary.get('images_retained', True) and manifest.is_file():
            images = []
            for row in read_jsonl(manifest)[:count]:
                path = (render / row['image']).resolve()
                if not path.is_relative_to(render.resolve() / 'images'):
                    raise ValueError('Evaluation image escapes render/images')
                if path.is_file():
                    caption = f'{row["scene_id"]} | seed={row["seed"]} | {row["prompt"]}'
                    images.append((path, caption))
            tracker.add_images('evaluation/images', images, step)
    finally:
        tracker.close()
