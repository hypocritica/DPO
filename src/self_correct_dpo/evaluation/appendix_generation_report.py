"""Prompt-cluster bootstrap and draft appendix plots from a portable score bundle."""
import csv

from self_correct_dpo.evaluation.appendix_generation import METRICS, METHODS, PAIR, SEEDS, STEPS, prompt_key
from self_correct_dpo.runtime.checkpoints import atomic_json

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261121
LABELS = {'pickscore': 'PickScore', 'hpsv3': 'HPSv3', 'image_reward': 'ImageReward', 'aesthetic': 'Aesthetic'}


def bootstrap(np, difference):
    # Rows are prompts and columns are the fixed generation seeds. Resampling
    # prompt means keeps every seed of a sampled prompt together.
    delta = difference.mean(axis=1)
    wins = ((difference > 0) + 0.5 * (difference == 0)).mean(axis=1)
    per_prompt = np.column_stack([delta, wins])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    batches = []
    for offset in range(0, BOOTSTRAP_REPLICATES, 256):
        size = min(256, BOOTSTRAP_REPLICATES - offset)
        draws = rng.integers(len(delta), size=(size, len(delta)))
        batches.append(per_prompt[draws].mean(axis=1))
    low, high = np.quantile(np.concatenate(batches), [0.025, 0.975], axis=0)
    return {'mean_delta': float(delta.mean()), 'ci_low': float(low[0]), 'ci_high': float(high[0]),
            'win_rate': float(wins.mean()), 'win_ci_low': float(low[1]), 'win_ci_high': float(high[1])}


def csv_write(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def comparisons(has_lossavg):
    direct = [('global_k4', 'dpo'), ('stagewise_k4', 'global_k4'), ('stagewise_k4', 'dpo')]
    if has_lossavg:
        direct.append(('global_k4', 'lossavg_k4'))
    for method, baseline in direct:
        yield 'direct', 'full', method, baseline, 1600, (17,), METRICS
    for seeds in ((17,), (23,), (42,), SEEDS):
        yield 'generation_seeds', 'subset', PAIR[1], PAIR[0], 1600, seeds, METRICS
    for step in STEPS:
        yield 'checkpoint_curve', 'subset', PAIR[1], PAIR[0], step, (17,), METRICS[:2]


def make_plots(out, rows, section, n_prompts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def draw(axes, selected, metric, horizontal, ticks, *, connect=False):
        y = [r['mean_delta'] for r in selected]
        low = [r['ci_low'] for r in selected]
        high = [r['ci_high'] for r in selected]
        axes.axhline(0, color='0.55', linestyle='--', linewidth=0.75)
        axes.vlines(horizontal, low, high, color='#176760', linewidth=1.4)
        axes.plot(horizontal, low, '_', color='#176760', markersize=7)
        axes.plot(horizontal, high, '_', color='#176760', markersize=7)
        axes.plot(horizontal, y, 'o-' if connect else 'o', color='#176760', linewidth=1.1, markersize=4)
        axes.set_xticks(horizontal, ticks)
        axes.set_title(LABELS[metric], loc='left')
        axes.grid(axis='y', alpha=0.2)
        axes.spines[['top', 'right']].set_visible(False)

    def save(fig, name):
        for suffix in ('png', 'pdf'):
            path = out / f'{name}.{suffix}'
            temporary = path.with_suffix('.tmp')
            fig.savefig(temporary, format=suffix, dpi=180)
            temporary.replace(path)
        plt.close(fig)

    interval_note = 'Paired 95% prompt-bootstrap intervals, one training run per method'
    if section in ('full', 'curve'):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout='constrained')
        for ax, metric in zip(axes, METRICS[:2]):
            selected = [r for r in rows if r['analysis'] == 'checkpoint_curve' and r['metric'] == metric]
            draw(ax, selected, metric, STEPS, STEPS, connect=True)
            ax.set_xlabel('Optimizer update')
            ax.set_ylabel('Stage-wise MVPO − Flow-DPO')
        fig.suptitle(f'{n_prompts} PartiPrompts, generation seed 17\n{interval_note}')
        save(fig, 'checkpoint-differences')

    if section in ('full', 'seeds'):
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.8), layout='constrained')
        for ax, metric in zip(axes.flat, METRICS):
            selected = [r for r in rows if r['analysis'] == 'generation_seeds' and r['metric'] == metric]
            draw(ax, selected, metric, range(len(selected)), ['17', '23', '42', 'Mean'])
            ax.set_xlabel('Generation seed')
            ax.set_ylabel('Stage-wise MVPO − Flow-DPO')
        fig.suptitle(f'{n_prompts} PartiPrompts at update 1,600\n{interval_note}')
        save(fig, 'generation-seed-differences')


def summarize(out, bundle, panels, indexed):
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    plan = bundle['plan']
    section = bundle['section']
    jobs = {(j['method'], j['step'], j['seed']): j for j in plan['jobs']}
    has_lossavg = ('lossavg_k4', 1600, 17) in jobs

    def values(method, step, seed, metric, keys):
        scores = indexed[jobs[method, step, seed]['name']]
        return np.array([scores[(metric, *key)] for key in keys])

    direct_rows, paired_rows, prompt_rows = [], [], []
    full_keys = [prompt_key(r) for r in panels['full']]
    if section in ('full', 'direct'):
        for method, label in METHODS.items():
            job = jobs.get((method, 1600, 17))
            direct_rows.append({'method': method, 'label': label,
                                'status': 'complete' if job else plan['lossavg']['status'],
                                'checkpoint_id': job['adapter']['checkpoint_id'] if job else '',
                                'adapter_sha256': job['adapter']['files']['adapter/adapter_model.safetensors'] if job else '',
                                'updates': 1600, 'prompts': len(full_keys), 'generation_seed': 17, 'training_runs': 1 if job else 0,
                                **{m: float(values(method, 1600, 17, m, full_keys).mean()) if job else '' for m in METRICS}})
        csv_write(out / 'direct-comparison.csv', direct_rows)

    requested_analysis = {'direct': 'direct', 'seeds': 'generation_seeds', 'curve': 'checkpoint_curve'}
    for analysis, panel, method, baseline, step, seeds, metrics in comparisons(has_lossavg):
        if section != 'full' and analysis != requested_analysis[section]:
            continue
        keys = [prompt_key(r) for r in panels[panel]]
        for metric in metrics:
            difference = np.column_stack([values(method, step, seed, metric, keys)
                                          - values(baseline, step, seed, metric, keys) for seed in seeds])
            row = {'analysis': analysis, 'panel': panel, 'metric': metric, 'method': method,
                   'baseline': baseline, 'step': step, 'generation_seeds': ','.join(map(str, seeds)),
                   'prompts': len(keys), 'training_runs_per_method': 1}
            paired_rows.append({**row, 'bootstrap_unit': 'prompt', 'confidence_level': 0.95,
                                'bootstrap_replicates': BOOTSTRAP_REPLICATES, **bootstrap(np, difference)})
            prompt_rows.extend({**row, 'scene_id': key[0], 'variant_id': key[1],
                                'mean_prompt_delta': float(delta.mean())} for key, delta in zip(keys, difference))
    csv_write(out / 'paired-differences.csv', paired_rows)
    csv_write(out / 'prompt-differences.csv', prompt_rows)
    if section != 'direct':
        make_plots(out, paired_rows, section, len(panels['subset']))
    atomic_json(out / 'summary.json', {'plan_id': plan['plan_id'], 'bundle_id': bundle['bundle_id'],
               'section': section, 'checkpoint_comparison': direct_rows, 'paired_comparisons': paired_rows,
               'noise_runtime': next(iter(bundle['sources'].values()))['noise_runtime'],
               'noise_pairing': plan['noise_pairing'],
               'lossavg': plan['lossavg'], 'bootstrap': {
                   'unit': 'prompt (all included generation seeds kept together)',
                   'conditioned_on': 'fixed trained checkpoints and listed generation seeds',
                   'interval': 'pointwise paired percentile 95%', 'replicates': BOOTSTRAP_REPLICATES,
                   'rng': 'NumPy PCG64', 'seed': BOOTSTRAP_SEED,
                   'win_rate': 'Per-image wins plus half ties, averaged within prompt then across prompts.'}})
    print(f'Wrote verified SD3.5-M generation tables and draft plots to {out}')
