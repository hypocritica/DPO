"""Incremental SD3.5 checkpoint curves on the fixed 128-prompt PickScore panel."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

from self_correct_dpo.common import (digest, load_suite, output_lock, read_json, sha256,
                                     verify_images)
from self_correct_dpo.evaluation.experiment_catalog import model_spec
from self_correct_dpo.evaluation.paper import collect, csv_text
from self_correct_dpo.evaluation.suite import metrics_for, read_plan, safe_target


def _selection():
    return SimpleNamespace(families=None, methods=None, run_glob=None, steps=None,
                           benchmarks=None, metrics=None, metric_group=None)


def validate_curve_plan(plan, out):
    out = Path(out)
    tasks = plan['tasks']
    if not tasks or len(plan['panels']) != 1:
        raise ValueError('Checkpoint curves require one fixed development panel')
    name, panel = next(iter(plan['panels'].items()))
    suite = load_suite(safe_target(out, panel['snapshot']))
    if (panel.get('mode') != 'development-curve' or len(suite) != 128
            or len({r['prompt'] for r in suite}) != 128):
        raise ValueError('Checkpoint curves require 128 unique development-curve prompts')
    family = tasks[0]['family']
    frozen_metric = plan['metric_groups']['modern']
    allowed = ['pickscore'] if family == 'sd35_medium' else ['pickscore', 'hpsv3']
    if family == 'sd35_medium' and frozen_metric != ['pickscore']:
        raise ValueError('Checkpoint curves require only PickScore in the frozen plan')
    if family not in ('sd35_medium', 'zimage') or len(frozen_metric) != 1 or frozen_metric[0] not in allowed:
        raise ValueError('Curve family/selection metric is unsupported')
    seen = set()
    for task in tasks:
        if (task['family'] != family or task['benchmark'] != name
                or task['prompts'] != 128 or task['images'] != 128
                or task['sampling']['seeds'] != [17]):
            raise ValueError('Checkpoint curves require a single family, 128 images and seed 17')
        if metrics_for(task, plan, _selection()) != frozen_metric:
            raise ValueError('Checkpoint curves require one frozen selection metric')
        key = (task['method'], task['step'])
        if key in seen:
            raise ValueError(f'Duplicate method/step in checkpoint curve: {key}')
        if (task['step'] == 0) != (task['method'] == 'base'):
            raise ValueError('Only the Base method may have checkpoint step zero')
        seen.add(key)
    if sum(t['method'] == 'base' for t in tasks) != 1:
        raise ValueError('Checkpoint curves require exactly one Base reference')
    if len({digest(t['sampling']) for t in tasks}) != 1:
        raise ValueError('Incompatible checkpoint curve identities: sampling_id')
    if len({digest(model_spec(t['model'])) for t in tasks}) != 1:
        raise ValueError('Incompatible checkpoint curve identities: base_model_id')


def _timing(records, render):
    """Informational generation cost; score values never depend on timing logs."""
    def total(values):
        if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            return None
        return sum(values)
    result = {'generation_seconds': total([r.get('generation_seconds') for r in records]),
              'generation_session_seconds': None}
    sessions = sorted((render / 'sessions').glob('*.json'))
    if sessions:
        try:
            result['generation_session_seconds'] = total([read_json(p).get('elapsed_seconds') for p in sessions])
        except (OSError, ValueError, TypeError):
            result['timing_note'] = 'Generation session timing unavailable or invalid'
    return result


def _plot(rows, path, plt, metric='pickscore', family='sd35_medium'):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods = defaultdict(list)
    for row in rows:
        methods[row['method']].append(row)
    xmax = max(1, max(r['step'] for r in rows))
    try:
        for method, points in sorted(methods.items(), key=lambda item: (item[0] != 'base', item[0])):
            label = points[0]['method_label']
            if method == 'base':
                point = points[0]
                if point['status'] == 'complete':
                    ax.axhline(point['mean'], linestyle='--', color='black', label=label)
                    ax.fill_between([0, xmax], point['mean'] - point['prompt_standard_error'],
                                    point['mean'] + point['prompt_standard_error'], color='black', alpha=.1)
                continue
            xs = [p['step'] for p in points]
            ys = [p['mean'] if p['status'] == 'complete' else math.nan for p in points]
            errors = [p['prompt_standard_error'] if p['status'] == 'complete' else math.nan for p in points]
            line, = ax.plot(xs, ys, marker='o', label=label)
            ax.fill_between(xs, [v - e for v, e in zip(ys, errors)],
                            [v + e for v, e in zip(ys, errors)], color=line.get_color(), alpha=.15)
        done = sum(r['status'] == 'complete' for r in rows)
        label_family = 'SD3.5 Medium' if family == 'sd35_medium' else 'Z-Image Base'
        ax.set(title=f'{label_family} · 128 development prompts · seed 17 ({done}/{len(rows)} points)',
               xlabel='Training step', ylabel='PickScore (raw paired cosine)' if metric == 'pickscore' else 'HPSv3')
        ax.text(.01, .01, 'Mean ± prompt standard error; missing points remain gaps.\nDevelopment panel; no best-checkpoint selection.',
                transform=ax.transAxes, fontsize=8, va='bottom')
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(path / f'{metric}.png', dpi=160)
        fig.savefig(path / f'{metric}.pdf')
    finally:
        plt.close(fig)


def summarize_curves(out, metric=None):
    """Publish current verified points and figures; completion is recorded last.

    Consumers should validate status.json's hashes before using this mutable
    snapshot. An interrupted publication leaves no status marker for mixed files.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Checkpoint curve plots require matplotlib; install matplotlib in the evaluation environment') from error
    plan, out = read_plan(out)
    validate_curve_plan(plan, out)
    frozen_metric = plan['metric_groups']['modern'][0]
    metric = metric or frozen_metric
    family = plan['tasks'][0]['family']
    if metric not in (['pickscore'] if family == 'sd35_medium' else ['pickscore', 'hpsv3']):
        raise ValueError('Unsupported curve metric for this family')
    destination = safe_target(out, 'curves' if metric == frozen_metric else f'curves-{metric}')
    with output_lock(destination):
        selection = _selection()
        selection.metrics = [metric]
        rows, provenance = collect(plan, out, selection)
        for key in ('family', 'benchmark', 'base_model_id', 'prompt_sha256', 'sampling_id'):
            if len({r[key] for r in rows}) != 1:
                raise ValueError(f'Incompatible checkpoint curve identities: {key}')
        protocols = {r['scorer_protocol_id'] for r in rows if r['status'] == 'complete'}
        if len(protocols) > 1:
            raise ValueError('Incompatible scorer protocols in checkpoint curve')
        tasks = {t['task_id']: t for t in plan['tasks']}
        for row in rows:
            row.update(generation_seconds=None, generation_session_seconds=None)
            if row['status'] != 'complete':
                continue
            try:
                error = row['prompt_standard_error']
                if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
                    raise ValueError('Checkpoint curve requires a finite prompt standard error')
                render = safe_target(out, tasks[row['task_id']]['output']) / 'render'
                _, records = verify_images(render)
                row.update(_timing(records, render))
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status='invalid', mean=None, prompt_standard_error=None, error=str(error))
                row.pop('prompt_mean', None)
                provenance.pop(row['task_id'] + '/' + metric, None)
        rows.sort(key=lambda r: (r['method'] != 'base', r['method'], r['step']))
        complete = all(r['status'] == 'complete' for r in rows)
        counts = dict(Counter(r['status'] for r in rows))
        result = {'schema_version': 1, 'plan_id': plan['plan_id'], 'complete': complete,
                  'metric': metric, 'scope': 'development-curve', 'points': len(rows),
                  'status_counts': counts, 'planned_images': sum(r['images'] for r in rows),
                  'scored_images': sum(r['images'] for r in rows if r['status'] == 'complete'),
                  'rows': rows, 'panels': plan['panels'], 'provenance': provenance,
                  'uncertainty': 'Mean plus/minus prompt standard error, not a confidence interval',
                  'selection': 'All frozen checkpoints; no best-checkpoint selection'}
        if family == 'zimage':
            result.update(selection_metric=frozen_metric, auxiliary=(metric != frozen_metric),
                          metric_policy='Selection metric was frozen before generation; auxiliary scores do not change checkpoint choices.')
        result['summary_id'] = digest(result)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.current-', dir=destination) as temporary:
            stage = Path(temporary)
            (stage / 'points.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            (stage / 'points.csv').write_text(csv_text(rows))
            _plot(rows, stage, plt, metric=metric, family=family)
            names = ['points.csv', 'points.json', f'{metric}.png', f'{metric}.pdf']
            status = {k: result[k] for k in ('plan_id', 'summary_id', 'complete', 'points', 'status_counts',
                                             'planned_images', 'scored_images')}
            status['files_sha256'] = {name: sha256(stage / name) for name in names}
            (stage / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
            # Remove the old marker only after every new artifact has been prepared.
            (destination / 'status.json').unlink(missing_ok=True)
            for name in [*names, 'status.json']:
                (stage / name).replace(destination / name)
    return {'output': str(destination), **status}
