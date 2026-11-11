"""Paper tables with explicit protocols, missingness, and score provenance.

Reads committed generated-image manifests and scores only. Export never needs
the training directory or model assets and never changes the statistical sample.
"""
import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path

from self_correct_dpo.common import digest, output_lock, read_json, sha256
from self_correct_dpo.evaluation.experiment_catalog import model_spec
from self_correct_dpo.evaluation.suite import (COMPOSITIONAL, FAMILIES, PANELS, filter_tasks,
    freeze_json, metrics_for, read_plan, safe_target, score_summary, validate_task_render)


def tex(value):
    escaped = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(escaped.get(c, c) for c in str(value))


def number(value):
    return '—' if value is None else f'{value:.6f}'


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def finite(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f'Invalid finite score field: {name}')
    return value


def collect(plan, out, args):
    tasks = filter_tasks(plan, args)
    if not tasks:
        raise ValueError('No planned tasks match the paper selection')
    rows, provenance = [], {}
    for task in tasks:
        target = safe_target(out, task['output'])
        selected_metrics = metrics_for(task, plan, args)
        base = {'task_id': task['task_id'], 'experiment_id': task.get('experiment_id', task['run']),
                'run': task['run'], 'family': task['family'], 'method': task.get('method', 'unregistered'),
                'method_label': task.get('method_label', task['run']), 'step': task['step'],
                'checkpoint_id': task['checkpoint']['checkpoint_id'], 'prompts': task['prompts'],
                'images': task['images'], 'benchmark': task['benchmark'],
                'base_model_id': digest(model_spec(task['model'])),
                'panel_mode': plan['panels'][task['benchmark']].get('mode', 'custom-unverified'),
                'seeds': ','.join(map(str, task['sampling']['seeds'])),
                'prompt_sha256': plan['panels'][task['benchmark']]['sha256'],
                'sampling_id': digest(task['sampling'])}
        try:
            run, complete = validate_task_render(task, plan, out)
        except (OSError, ValueError, KeyError, TypeError) as error:
            for metric in selected_metrics:
                rows.append({**base, 'metric': metric, 'status': 'missing_render' if isinstance(error, FileNotFoundError) else 'invalid',
                             'mean': None, 'prompt_standard_error': None, 'error': str(error)})
            continue
        for metric in selected_metrics:
            row = {**base, 'metric': metric, 'status': 'complete', 'mean': None,
                   'prompt_standard_error': None, 'error': ''}
            directory = target / 'scores' / metric
            try:
                summary = score_summary(target, metric, run, complete['manifest_sha256'])
                mean = finite(summary['mean'], 'mean')
                if summary.get('prompts') != task['prompts']:
                    raise ValueError('Score prompt count differs from the frozen task')
                stderr = summary.get('prompt_standard_error')
                if stderr is not None and finite(stderr, 'prompt_standard_error') < 0:
                    raise ValueError('Negative standard error')
                identity_file = directory / ('evaluation.json' if metric in COMPOSITIONAL else 'identity.json')
                identity = read_json(identity_file)['identity']
                # Image/run bindings are checked by score_summary; the remaining
                # identity distinguishes scorer assets, implementation and env.
                scoring_protocol = {k: v for k, v in identity.items() if k not in ('run_id', 'manifest_sha256', 'python')}
                protocol_id = digest(scoring_protocol)
                row.update(mean=mean, prompt_mean=summary.get('prompt_mean'), prompt_standard_error=stderr,
                           metric_protocol=summary.get('protocol', summary.get('convention', 'unspecified')),
                           metric_version=summary.get('version'), scorer_protocol_id=protocol_id,
                           evaluation_id=summary.get('evaluation_id'),
                           scope=('smoke-only' if 'smoke-only' in (base['panel_mode'], summary.get('mode'))
                                  else base['panel_mode']),
                           summary_path=str(directory / 'summary.json'))
                for key, value in sorted(summary.get('components', {}).items()):
                    row['component_' + key] = finite(value, key)
                provenance[task['task_id'] + '/' + metric] = {
                    'task': task, 'scoring_identity': identity, 'summary': summary,
                    'summary_sha256': sha256(directory / 'summary.json'),
                    'score_complete_sha256': sha256(directory / 'complete.json'),
                    'render_manifest_sha256': complete['manifest_sha256']}
                for failure in (target / 'failures/generate.json', target / 'failures' / f'score-{metric}.json'):
                    if failure.exists():
                        raise ValueError(f'Unresolved failure record: {failure}: {read_json(failure).get("error")}')
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status='missing_score' if isinstance(error, FileNotFoundError) else 'invalid',
                           mean=None, prompt_standard_error=None, error=str(error))
                # Do not leave structured numbers or provenance attached to an
                # invalid result and accidentally publish a partial valid row.
                for key in list(row):
                    if key.startswith('component_') or key == 'prompt_mean':
                        row.pop(key)
                provenance.pop(task['task_id'] + '/' + metric, None)
            rows.append(row)
    return rows, provenance


def make_tables(rows):
    task_rows = defaultdict(list)
    for row in rows:
        task_rows[row['task_id']].append(row)
    groups = defaultdict(list)
    for task_id, values in task_rows.items():
        first = values[0]
        protocol = {'family': first['family'], 'benchmark': first['benchmark'], 'prompt_sha256': first['prompt_sha256'],
                    'sampling_id': first['sampling_id'], 'base_model_id': first['base_model_id'],
                    'panel_mode': first['panel_mode'],
                    'scorers': {r['metric']: r.get('scorer_protocol_id') for r in values}}
        groups[(first['family'], first['benchmark'], digest(protocol))].append((first, values, protocol))
    files, index = {}, []
    for (family, benchmark, protocol_id), entries in sorted(groups.items()):
        entries.sort(key=lambda e: (e[0]['method'] != 'base', e[0]['method_label'], e[0]['run'], e[0]['step']))
        columns = sorted({r['metric'] for _, values, _ in entries for r in values})
        headers = ['Method', 'Step', 'Prompts', 'Images', *columns, 'Status']
        table_rows = []
        for first, values, _ in entries:
            by_metric = {r['metric']: r for r in values}
            status = 'complete' if all(r['status'] == 'complete' for r in values) else 'INCOMPLETE'
            table_rows.append([first['method_label'], str(first['step']), str(first['prompts']), str(first['images']),
                               *[number(by_metric.get(metric, {}).get('mean')) for metric in columns], status])
        stem = family + '-' + benchmark + '-' + protocol_id[:12]
        markdown = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        markdown += ['| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in row) + ' |' for row in table_rows]
        latex = [r'\begin{tabular}{' + 'l' + 'r' * (len(headers) - 2) + 'l}', r'\hline',
                 ' & '.join(tex(h) for h in headers) + r' \\', r'\hline']
        latex += [' & '.join('--' if value == '—' else tex(value) for value in row) + r' \\' for row in table_rows]
        latex += [r'\hline', r'\end{tabular}']
        scopes = sorted({r.get('scope', 'incomplete') for _, values, _ in entries for r in values})
        note = ('Scores keep their native scales; no automatic rescaling, best-checkpoint selection or significance claim. '
                'Missing values are not zero. Scope: ' + ', '.join(scopes) + '. Protocol: ' + protocol_id + '.')
        files[stem + '.md'] = '\n'.join(markdown) + '\n\n' + note + '\n'
        files[stem + '.tex'] = '% ' + note + '\n' + '\n'.join(latex) + '\n'
        index.append({'table': stem, 'family': family, 'benchmark': benchmark, 'protocol': entries[0][2],
                      'tasks': [e[0]['task_id'] for e in entries], 'metrics': columns, 'scopes': scopes})
    return files, index


def export_paper(plan, out, args):
    if not args.paper_name or Path(args.paper_name).name != args.paper_name or args.paper_name in ('.', '..'):
        raise ValueError('--paper-name must be a simple directory name')
    rows, provenance = collect(plan, out, args)
    complete = all(r['status'] == 'complete' for r in rows)
    if not complete and not args.allow_incomplete:
        counts = {state: sum(r['status'] == state for r in rows) for state in sorted({r['status'] for r in rows})}
        raise ValueError(f'Paper results incomplete: {counts}. Finish scoring or use --allow-incomplete for a labelled draft.')
    tables, index = make_tables(rows)
    result = {'schema_version': 1, 'plan_id': plan['plan_id'], 'complete': complete,
              'status': 'complete' if complete else 'INCOMPLETE DRAFT',
              'notes': ['Means keep their native scorer scales.', 'Standard errors in results.csv are across prompt means, not confidence intervals.',
                        'Tables split when base model, sampling, prompts or scorer identities differ.',
                        'Smoke-only and custom-unverified panels are not certified full benchmark results.',
                        'No training tensors or original model assets are needed for export.'],
              'panels': {name: plan['panels'][name] for name in sorted({r['benchmark'] for r in rows})},
              'rows': rows, 'tables': index, 'provenance': provenance}
    result['export_id'] = digest(result)
    destination = safe_target(out, 'paper/' + args.paper_name)
    payloads = {'results.csv': csv_text(rows), **tables}
    payloads['README.md'] = ('# Evaluation paper export\n\nStatus: ' + result['status'] + '\n\n' +
        '\n'.join('- ' + note for note in result['notes']) + '\n\n' +
        '\n'.join('- [' + table['table'] + '](' + table['table'] + '.md)' for table in index) + '\n')
    with output_lock(destination):
        freeze_json(destination / 'results.json', result)
        for relative, content in payloads.items():
            path = safe_target(destination, relative)
            if path.exists() and path.read_text() != content:
                raise ValueError(f'Paper export changed; use a new --paper-name: {path}')
            if not path.exists():
                temporary = path.with_name('.' + path.name + '.tmp')
                temporary.write_text(content)
                temporary.replace(path)
        receipt = {'export_id': result['export_id'], 'complete': complete,
                   'files_sha256': {name: sha256(destination / name) for name in sorted([*payloads, 'results.json'])}}
        freeze_json(destination / 'export.json', receipt)
    return {'output': str(destination), 'complete': complete, 'rows': len(rows), 'tables': len(index), 'export_id': result['export_id']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='Evaluation output containing frozen plan.json')
    parser.add_argument('--paper-name', default='paper-v1', help='Use a new name after adding/changing selected scores')
    parser.add_argument('--families', nargs='+', choices=FAMILIES)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--run-glob', nargs='+')
    parser.add_argument('--steps', nargs='+', type=int)
    parser.add_argument('--benchmarks', nargs='+', choices=PANELS)
    parser.add_argument('--metrics', nargs='+')
    parser.add_argument('--metric-group', choices=('legacy', 'modern'))
    parser.add_argument('--allow-incomplete', action='store_true', help='Export labelled draft with missing/invalid scores left empty')
    args = parser.parse_args(argv)
    try:
        plan, out = read_plan(args.out)
        result = export_paper(plan, out, args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['complete'] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'Paper export failed: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
