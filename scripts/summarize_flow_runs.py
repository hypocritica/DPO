"""Compare completed flow evaluations on one fixed panel per backbone."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import digest, read_json, read_jsonl
from self_correct_dpo.evaluation.runner import verify_complete
from self_correct_dpo.runtime.pipeline import paired_delta


def collect(runs, reference_run=None):
    reference_run = str(Path(reference_run).resolve()) if reference_run else None
    runs = list(dict.fromkeys(str(Path(run).resolve()) for run in runs))
    if reference_run and reference_run not in runs:
        runs.append(reference_run)
    rows, groups, panels = [], {}, {}
    for run in runs:
        run = Path(run).resolve()
        info = read_json(run / 'training.json')
        config = info['config']; t = config['training']; family = config['model']['family']
        if family not in ('sd35_medium', 'zimage'):
            raise ValueError(f'Not a flow run: {run}')
        baseline = run / 'evaluations/step-0000000'
        if not (baseline / 'complete.json').exists():
            raise ValueError(f'Base evaluation is incomplete: {run}')
        verify_complete(baseline)
        base_meta = read_json(baseline / 'evaluation.json')['identity']
        signature = digest({'base': base_meta['base_identity'], 'inputs': base_meta['inputs'],
                            'sampling': base_meta['config']['sampling'],
                            'pairs_id': info['pairs_id'], 'seed': config['seed']})
        if family in groups and groups[family] != signature:
            raise ValueError(f'{family}: model/data/panel/scorer/sampling/seed differ; compare separately')
        groups[family] = signature
        base_scores = read_jsonl(baseline / 'scores/pickscore/scores.jsonl')
        for path in sorted((run / 'evaluations').glob('step-*')):
            if not (path / 'complete.json').exists():
                continue
            summary = verify_complete(path)
            meta = read_json(path / 'evaluation.json')['identity']
            if (meta['inputs'] != base_meta['inputs'] or meta['base_identity'] != base_meta['base_identity']
                    or meta['config']['sampling'] != base_meta['config']['sampling']):
                raise ValueError(f'Checkpoint evaluation differs from its base panel: {path}')
            pairs = t['batch_size'] * t['world_size'] * t['gradient_accumulation_steps']
            scores = read_jsonl(path / 'scores/pickscore/scores.jsonl')
            algorithm = ('sft' if t['objective'] == 'sft' else
                         'linear_dpo' if t['objective'] == 'linear_dpo' else
                         'mvpo' if t['noise_views'] > 1 else 'dpo')
            sft = algorithm == 'sft'
            panels[str(run), summary['step']] = scores
            rows.append({'family': family, 'algorithm': algorithm, 'method': config['method'],
                         'step': summary['step'], 'k': t['noise_views'],
                         'sampled_pair_rows_per_update': pairs,
                         'pairs_per_update': 0 if sft else pairs,
                         'pair_views_per_update': 0 if sft else pairs*t['noise_views'],
                         'winner_images_per_update': pairs,
                         'policy_image_views_per_update': pairs*t['noise_views']*(1 if sft else 2),
                         'beta': None if sft else t['beta'], 'learning_rate': t['learning_rate'],
                         'pickscore': summary['metrics']['pickscore']['mean'],
                         **paired_delta(scores, base_scores), 'run': str(run)})
    if reference_run:
        if len(groups) != 1:
            raise ValueError('Use --reference-run with one backbone at a time')
        for row in rows:
            reference = panels.get((reference_run, row['step']))
            row.update(reference_run=reference_run, delta_reference=None,
                       delta_reference_ci_low=None, delta_reference_ci_high=None,
                       reference_status='not_evaluated_at_this_step')
            if reference is not None:
                delta = paired_delta(panels[row['run'], row['step']], reference)
                row.update(delta_reference=delta['delta_base'],
                           delta_reference_ci_low=delta['delta_ci_low'],
                           delta_reference_ci_high=delta['delta_ci_high'], reference_status='complete')
    return sorted(rows, key=lambda r: (r['family'], r['step'], r['algorithm'], r['run']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--reference-run', type=Path, help='Also compare matched steps against this run (e.g. DPO)')
    p.add_argument('--out', type=Path, help='Optional new CSV; existing files are never overwritten')
    a = p.parse_args()
    rows = collect(a.runs, a.reference_run)
    if a.out and rows:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'results': rows,
        'interpretation': 'Validation screening only. CI is paired over prompts, not training seeds. '
        'Compare matching steps; SFT uses only winners, MVPO has extra views. '
        'Policy image-views exclude reference forwards and are not measured FLOPs. '
        'Checkpoint selection on this panel requires a separate final test; positive scores are not human win rates.'}, indent=2))


if __name__ == '__main__':
    main()
