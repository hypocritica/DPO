"""Export the three selected Z-Image methods across Parti preference scores and GenEval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from self_correct_dpo.common import digest, output_lock, read_json, sha256
from self_correct_dpo.evaluation import paper, suite
from self_correct_dpo.evaluation.selected import identity
from self_correct_dpo.evaluation.experiment_catalog import model_spec

METHODS = ('base', 'dpo', 'stagewise_k4')
COLUMNS = (('partiprompts', 'hpsv3', 'Parti HPSv3'),
           ('partiprompts', 'unifiedreward', 'Parti UnifiedReward ACS (project mean)'),
           ('partiprompts', 'pickscore', 'Parti PickScore'),
           ('partiprompts', 'image_reward', 'Parti ImageReward'),
           ('geneval', 'geneval', 'GenEval macro accuracy [0,1]'))


def load_panel(directory, panel):
    plan, root = suite.read_plan(directory)
    receipt = read_json(root / 'selected-checkpoints.json')
    selection = receipt['selection']
    payload = (json.dumps(selection, indent=2) + '\n').encode()
    if (receipt['plan_id'] != plan['plan_id'] or panel not in receipt['benchmarks']
            or receipt['selection_sha256'] != hashlib.sha256(payload).hexdigest()
            or selection['identity_sha256'] != identity(selection)
            or selection['family'] != 'zimage' or selection['methods'] != list(METHODS)
            or plan.get('report_selection', {}).get('selection') != selection
            or plan.get('report_selection', {}).get('selection_id') != digest(selection)):
        raise ValueError('Invalid Z-Image selected checkpoint receipt')
    choices = {c['method']: c for c in selection['choices']}
    if set(choices) != set(METHODS[1:]) or len(selection['choices']) != 2:
        raise ValueError('Require exactly DPO and Stagewise K4 choices')
    metrics = [metric for name, metric, _ in COLUMNS if name == panel]
    args = argparse.Namespace(families=['zimage'], methods=list(METHODS), run_glob=None,
                              steps=None, benchmarks=[panel], metrics=metrics, metric_group=None)
    tasks = suite.filter_tasks(plan, args)
    if len(tasks) != 3 or {t['method'] for t in tasks} != set(METHODS):
        raise ValueError('Require exactly three Z-Image tasks per panel')
    for task in tasks:
        if task['prompts'] != (1632 if panel == 'partiprompts' else 553):
            raise ValueError('Full official panel required')
        if task['sampling']['seeds'] != ([17] if panel == 'partiprompts' else [17, 29, 43, 71]):
            raise ValueError('Frozen Z-Image seed protocol differs')
        if task['method'] == 'base':
            if task['step'] != 0 or task['checkpoint']['checkpoint_id'] != 'base':
                raise ValueError('Invalid Base checkpoint')
        else:
            choice = choices[task['method']]
            if type(choice['selected_step']) is not int or task['step'] != choice['selected_step']:
                raise ValueError('Task differs from chosen checkpoint step')
            saved = next(c for c in choice['checkpoints'] if c['step'] == choice['selected_step'])
            if (task['checkpoint']['checkpoint_id'] != saved['checkpoint_id']
                    or task['checkpoint']['training_sha256'] != choice['training_sha256']
                    or task['run_path'] != choice['run_path']):
                raise ValueError('Task differs from chosen training/checkpoint identity')
    rows, provenance = paper.collect(plan, root, args)
    if len(rows) != 3 * len(metrics) or any(r['status'] != 'complete' for r in rows):
        raise ValueError('All quantitative results must be complete and valid')
    for metric in metrics:
        group = [r for r in rows if r['metric'] == metric]
        for key in ('sampling_id', 'prompt_sha256', 'base_model_id', 'panel_mode', 'scorer_protocol_id'):
            if len({r[key] for r in group}) != 1:
                raise ValueError(f'Incompatible {panel}/{metric} {key}')
        if group[0]['panel_mode'] != 'full-benchmark':
            raise ValueError('Only verified full benchmark panels can be combined')
    return selection, tasks, rows, provenance, {'plan_id': plan['plan_id'],
        'plan_sha256': sha256(root / 'plan.json'), 'receipt_sha256': sha256(root / 'selected-checkpoints.json')}


def export(parti_out, geneval_out, destination):
    panels = [load_panel(parti_out, 'partiprompts'), load_panel(geneval_out, 'geneval')]
    if panels[0][0] != panels[1][0]:
        raise ValueError('Parti and GenEval must use the same chosen selection and provenance')
    def checkpoint_map(tasks):
        return {t['method']: (t['step'], t['checkpoint'], model_spec(t['model'])) for t in tasks}
    if checkpoint_map(panels[0][1]) != checkpoint_map(panels[1][1]):
        raise ValueError('Parti and GenEval model/checkpoint identities differ')
    if len({digest(model_spec(t['model'])) for panel in panels for t in panel[1]}) != 1:
        raise ValueError('All methods must share one frozen model identity')
    quantitative = [row for panel in panels for row in panel[2]]
    rows = []
    for method in METHODS:
        source = next(r for r in quantitative if r['method'] == method)
        row = {'method': method, 'step': source['step'], 'checkpoint_id': source['checkpoint_id']}
        for panel, metric, label in COLUMNS:
            row[label] = next(r['mean'] for r in quantitative
                              if r['method'] == method and r['benchmark'] == panel and r['metric'] == metric)
        rows.append(row)
    notes = ('Each column keeps its native scale and its labelled prompt panel. GenEval is official category-macro '
             'accuracy, not percent. UnifiedReward ACS is the project arithmetic mean of alignment, coherence and style. '
             'Parti uses 1632 prompts × seed 17; GenEval uses 553 prompts × four frozen seeds. '
             'Steps are manually chosen; no checkpoint ranking or significance claim.')
    headers = list(rows[0])
    cells = [[paper.number(v) if isinstance(v, float) else str(v) for v in row.values()] for row in rows]
    markdown = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    markdown += ['| ' + ' | '.join(row) + ' |' for row in cells]
    latex = [r'\begin{tabular}{lrllllll}', r'\hline', ' & '.join(map(paper.tex, headers)) + r' \\']
    latex += [' & '.join(map(paper.tex, row)) + r' \\' for row in cells]
    latex += [r'\hline', r'\end{tabular}']
    result = {'version': 1, 'complete': True, 'selection': panels[0][0], 'rows': rows,
              'quantitative_rows': quantitative, 'notes': notes,
              'sources': {'partiprompts': panels[0][4], 'geneval': panels[1][4]},
              'provenance': {key: value for panel in panels for key, value in panel[3].items()}}
    result['export_id'] = digest(result)
    destination = Path(destination).expanduser().resolve()
    with output_lock(destination):
        if destination.exists():
            raise ValueError('Report output already exists; choose a new --out')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.' + destination.name + '-', dir=destination.parent) as temporary:
            stage = Path(temporary) / 'report'
            stage.mkdir()
            suite.freeze_json(stage / 'results.json', result)
            (stage / 'results.csv').write_text(paper.csv_text(rows))
            (stage / 'table.md').write_text('\n'.join(markdown) + '\n\n' + notes + '\n')
            (stage / 'table.tex').write_text('% ' + notes + '\n' + '\n'.join(latex) + '\n')
            suite.freeze_json(stage / 'export.json', {'export_id': result['export_id'],
                'files_sha256': {p.name: sha256(p) for p in sorted(stage.iterdir())}})
            stage.rename(destination)
    return {'output': str(destination), 'export_id': result['export_id']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parti-out', required=True)
    parser.add_argument('--geneval-out', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(export(args.parti_out, args.geneval_out, args.out), indent=2))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(2, f'Z-Image export failed: {error}\n')


if __name__ == '__main__':
    main()
