"""CPU-only tables, paired win ratios and raw checkpoint curves for the flexible queue."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from self_correct_dpo.common import read_json, sha256, output_lock, load_suite
from self_correct_dpo.evaluation.multiseed_history import read_source, choose_slots
from self_correct_dpo.evaluation.multiseed_report import interval, compare, csv_write, plot_wins, LABELS, METRIC_LABELS
from self_correct_dpo.runtime.checkpoints import atomic_json
from appendix_flexible import receipt_rows, prompt_key, compatible_source

LABELS = dict(LABELS, global_k6='Global MVPO K6')


def collect(output, family, plan):
    """Read prior A/B scores even when their jobs were not requested in the new queue."""
    history = read_json(output / 'history.json')
    adapters = {j['adapter']['checkpoint_id']: j['adapter'] for j in plan['jobs'] if j.get('adapter')}
    sources = defaultdict(dict)
    for path, record in history['sources'][family].items():
        if any(sha256(p) != h for p, h in record['files'].items()):
            raise ValueError('Historical scores changed; refresh-history after stopping the old jobs')
        source = read_source(path, plan['scorers'], plan['score_versions'])
        if not compatible_source(source['run'], plan, adapters.get(record['checkpoint'])):
            raise ValueError('Historical report protocol differs')
        sources[record['checkpoint']][path] = source
    panels = {p: load_suite(output / family / (p + '.jsonl')) for p in plan['panels']}
    data = {}
    for job in plan['jobs']:
        if job.get('blocked'):
            continue
        cp = job['adapter']['checkpoint_id'] if job.get('adapter') else 'base'
        slots, _ = choose_slots(panels[job['panel']], job['seed'], job['metrics'], sources[cp])
        for slot in slots:
            if slot['state'] != 'reuse_scores':
                continue
            source = sources[cp][slot['source']]
            key = tuple(slot['key'])
            data[(job['method'], job['step'], *key)] = dict(
                scores={m: float(source['scores'][m][key]['score']) for m in job['metrics']},
                generation_gpu=source['runtime']['gpu'], image_sha256=source['images'][key]['image_sha256'])
        for index in range(job['shards']):
            rows = receipt_rows(output, plan, job, index, 'score')
            for row in rows or []:
                data[(job['method'], job['step'], *row['key'])] = row
    return panels, data


def summarize(plan, panels, data, selected, repeats=10000):
    means, pairs = [], []
    seed0 = plan['seeds'][0]
    def matrix(method, step, seeds, panel, metric):
        try:
            rows = [[data[(method, step, *prompt_key(p), s)] for s in seeds] for p in panels[panel]]
            return np.array([[r['scores'][metric] for r in row] for row in rows]), [r['generation_gpu'] for row in rows for r in row]
        except KeyError:
            return None, []
    def mean(method, step, seeds, panel, metric, scope):
        values, gpu = matrix(method, step, seeds, panel, metric)
        row = dict(scope=scope, method=method, step=step, seeds=','.join(map(str, seeds)),
                   metric=metric, prompts=len(panels[panel]), status='missing', mean=None, low=None, high=None)
        if values is not None:
            m, lo, hi = interval(values.mean(axis=1), repeats)
            row.update(status='complete', mean=float(m[0]), low=float(lo[0]), high=float(hi[0]),
                       generation_gpus=' | '.join(sorted(set(gpu))),
                       seed_mean_sd=float(values.mean(axis=0).std(ddof=1)) if len(seeds) > 1 else None)
        means.append(row)
    def pair(a, b, sa, sb, seeds, panel, metric, scope):
        av, ag = matrix(a, sa, seeds, panel, metric)
        bv, bg = matrix(b, sb, seeds, panel, metric)
        row = dict(scope=scope, method=a, baseline=b, step=sa, baseline_step=sb,
                   seeds=','.join(map(str, seeds)), metric=metric, prompts=len(panels[panel]), status='missing')
        if av is not None and bv is not None:
            row.update(status='complete', **compare(av, bv, repeats),
                generation_gpus=' | '.join(sorted(set(ag + bg))), paired_same_generation_gpu=ag == bg)
        pairs.append(row)
    full = sorted({(j['method'], j['step']) for j in selected if j['panel'] == 'full'})
    mains = dict(full)
    seeds_sets = [[s] for s in plan['seeds']] + [plan['seeds']]
    for method, step in full:
        for metric in plan['scorers']:
            for seeds in seeds_sets:
                mean(method, step, seeds, 'full', metric, 'main')
    comparisons = [(m, 'dpo') for m in mains if m != 'dpo' and 'dpo' in mains]
    comparisons += [(a, b) for a, b in [('global_k2', 'lossavg_k2'), ('global_k4', 'lossavg_k4'),
                    ('stagewise_k2', 'global_k2'), ('stagewise_k4', 'global_k4')]
                    if a in mains and b in mains]
    for a, b in comparisons:
        for metric in plan['scorers']:
            for seeds in seeds_sets:
                pair(a, b, mains[a], mains[b], seeds, 'full', metric, 'main')
    grids = defaultdict(set)
    selected_names = {j['name'] for j in selected}
    for task, rule in plan['spec']['tasks'].items():
        if rule['family'] != plan['family'] or rule.get('panel') == 'full':
            continue
        if not any(j['name'] in selected_names and j['method'] in rule['methods'] and j['panel'] == 'curve' for j in plan['jobs']):
            continue
        for method in rule['methods']:
            grids[method].update(rule.get('report_steps', rule['steps']))
    metrics = ['pickscore', 'hpsv2' if plan['family'] == 'sd15' else 'hpsv3', 'image_reward']
    for method, steps in grids.items():
        for step in sorted(steps):
            for metric in metrics:
                mean(method, step, [seed0], 'curve', metric, 'curve')
    curve_pairs = [(m, 'dpo') for m in grids if m != 'dpo' and 'dpo' in grids]
    curve_pairs += [(a, b) for a, b in [('global_k2', 'lossavg_k2'), ('global_k4', 'lossavg_k4'),
                    ('stagewise_k2', 'global_k2'), ('stagewise_k4', 'global_k4'), ('stagewise_k6', 'global_k6'),
                    ('stagewise_dmpo_k6', 'dmpo'), ('stagewise_dspo_k6', 'dspo')]
                    if a in grids and b in grids]
    for a, b in curve_pairs:
        for step in sorted(grids[a] & grids[b]):
            for metric in metrics:
                pair(a, b, step, step, [seed0], 'curve', metric, 'curve')
    return means, pairs


def plot_curves(dest, family, means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ({'views': ['dpo', 'stagewise_k2', 'stagewise_k4', 'stagewise_k6'],
               'aggregation-k2': ['dpo', 'lossavg_k2', 'global_k2', 'stagewise_k2'],
               'global-stage-k6': ['dpo', 'global_k6', 'stagewise_k6'],
               'dmpo': ['dmpo', 'stagewise_dmpo_k6'], 'dspo': ['dspo', 'stagewise_dspo_k6']} if family == 'sd15' else
              {'k4': ['dpo', 'lossavg_k4', 'global_k4', 'stagewise_k4'],
               'k2': ['dpo', 'lossavg_k2', 'global_k2', 'stagewise_k2']})
    metrics = ['pickscore', 'hpsv2' if family == 'sd15' else 'hpsv3', 'image_reward']
    for name, methods in groups.items():
        rows = [r for r in means if r['scope'] == 'curve' and r['method'] in methods]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
        for ax, metric in zip(axes, metrics):
            for method in methods:
                points = sorted([r for r in rows if r['method'] == method and r['metric'] == metric], key=lambda r: r['step'])
                if not points:
                    continue
                x = [r['step'] for r in points]
                ax.plot(x, [r['mean'] if r['mean'] is not None else np.nan for r in points], 'o-',
                        label=LABELS.get(method, method), markersize=3)
                ax.fill_between(x, [r['low'] if r['low'] is not None else np.nan for r in points],
                    [r['high'] if r['high'] is not None else np.nan for r in points], alpha=.10)
            ax.set(xlabel='Optimizer update', title=METRIC_LABELS[metric])
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle(f'{family}: fixed 512 prompts, one generation seed; pointwise 95% prompt-bootstrap CI\n'
                     'Raw checkpoint means; H100/5090 provenance retained; not training-seed repetitions')
        for ext in ('png', 'pdf'):
            fig.savefig(dest / f'curves-{name}.{ext}', dpi=180)
        plt.close(fig)


def report(output, plans, selected):
    for family in sorted({f for f, _ in selected}):
        plan = plans[family]
        panels, data = collect(output, family, plan)
        means, paired = summarize(plan, panels, data, [j for f, j in selected if f == family])
        dest = output / 'report' / family
        dest.mkdir(parents=True, exist_ok=True)
        with output_lock(dest / 'report'):
            csv_write(dest / 'means.csv', means)
            csv_write(dest / 'paired-differences.csv', paired)
            wins = [r for r in paired if r['scope'] == 'main']
            csv_write(dest / 'win-ratios.csv', wins)
            provenance = [dict(method=k[0], step=k[1], scene_id=k[2], variant_id=k[3], seed=k[4],
                              generation_gpu=r['generation_gpu'], image_sha256=r['image_sha256']) for k, r in data.items()]
            csv_write(dest / 'pixel-provenance.csv', provenance)
            atomic_json(dest / 'protocol.json', dict(plan_id=plan['plan_id'], report_sha256=sha256(__file__),
                seeds=plan['seeds'], sampling=plan['sampling'], scorers=plan['scorers'],
                note='Fixed trained checkpoints; prompt-bootstrap intervals. Cross-GPU variation includes hardware effects.'))
            plot_curves(dest, family, means)
            for seeds in [str(plan['seeds'][0]), ','.join(map(str, plan['seeds']))]:
                target = dest / ('seeds-' + seeds.replace(',', '-'))
                target.mkdir(exist_ok=True)
                plot_wins(target, family, wins, seeds, 'H100/5090 sources retained; hardware effects not separated')
        print(f'{family}: tables, curves and win ratios -> {dest}', flush=True)
