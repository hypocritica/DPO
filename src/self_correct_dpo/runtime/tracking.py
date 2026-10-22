"""Rank-zero TensorBoard/W&B logging with the optimizer step as the x axis."""
from pathlib import Path


class ExperimentTracker:
    def __init__(self, config, out, run_id, *, start_step=0, resume=False, evaluation=False):
        self.tb = self.run = None
        self.pending = {}
        self.step = start_step
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
            self.run = wandb.init(project=settings.get('project', 'self-correct-dpo'),
                name=out.name, id=run_id[:32], resume='allow', config=config,
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

    def flush(self):
        if self.tb:
            self.tb.flush()
        if self.run and self.pending:
            self.run.log({'optimizer_step': self.step, **self.pending})
            self.pending.clear()

    def close(self):
        self.flush()
        if self.tb:
            self.tb.close()
        if self.run:
            self.run.finish()


def log_evaluation(config, out, step, summary):
    from self_correct_dpo.common import read_json
    run_id = read_json(Path(out) / 'training.json')['run_id']
    tracker = ExperimentTracker(config, out, run_id, start_step=step, resume=True, evaluation=True)
    try:
        for metric, values in summary['metrics'].items():
            tracker.add_scalar('evaluation/' + metric, values['mean'], step)
    finally:
        tracker.close()
