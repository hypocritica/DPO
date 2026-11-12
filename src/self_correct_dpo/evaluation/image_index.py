"""Reusable image lookup table; never alters generation or score journals."""
import csv
import os
from pathlib import Path
import tempfile

from self_correct_dpo.common import digest, output_lock, read_jsonl, sha256
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.evaluation.suite import filter_tasks, safe_target, validate_task_render


def export_image_index(plan, out, args):
    tasks = filter_tasks(plan, args)
    if not tasks:
        raise ValueError('No planned tasks match image index selection')
    fields = ['family', 'method', 'method_label', 'experiment_id', 'step', 'checkpoint_id',
              'benchmark', 'scene_id', 'variant_id', 'seed', 'prompt', 'image_path', 'relative_image',
              'image_sha256', 'image_id', 'task_id', 'sampling_id', 'generation_batch_size']
    statuses, count = [], 0
    with output_lock(out / 'image-index'):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=out,
                                         prefix='.image-index-', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for task in tasks:
                    try:
                        _, complete = validate_task_render(task, plan, out)
                        render = safe_target(out, task['output']) / 'render'
                        records = read_jsonl(render / 'manifest.jsonl')
                        rows = []
                        for record in records:
                            image = safe_target(render, record['image'])
                            if not image.is_file():
                                raise FileNotFoundError(f'Missing indexed image: {image}')
                            row = {key: task.get(key, '') for key in ('family', 'method', 'method_label', 'experiment_id', 'step', 'benchmark', 'task_id')}
                            row.update({key: record[key] for key in ('scene_id', 'variant_id', 'seed', 'prompt', 'image_sha256', 'image_id')})
                            row.update(checkpoint_id=task['checkpoint']['checkpoint_id'], image_path=str(image),
                                       relative_image=str(image.relative_to(out)), sampling_id=digest(task['sampling']),
                                       generation_batch_size=task['sampling'].get('generation_batch_size', 1))
                            rows.append(row)
                        writer.writerows(rows)
                        count += len(rows)
                        statuses.append({'task_id': task['task_id'], 'output': task['output'], 'status': 'complete',
                                         'images': len(rows), 'manifest_sha256': complete['manifest_sha256']})
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        statuses.append({'task_id': task['task_id'], 'output': task['output'],
                                         'status': 'missing' if isinstance(error, FileNotFoundError) else 'invalid', 'error': str(error)})
                stream.flush()
                os.fsync(stream.fileno())
                result = {'plan_id': plan['plan_id'], 'images': count, 'tasks': statuses,
                          'complete': all(t['status'] == 'complete' for t in statuses),
                          'verification': 'Frozen render bindings, manifest checksums and file existence; image hashes copied from manifests, not recomputed',
                          'csv_sha256': sha256(temporary)}
                (out / 'image-index.json').unlink(missing_ok=True)
                temporary.replace(out / 'image-index.csv')
                atomic_json(out / 'image-index.json', result)
            finally:
                temporary.unlink(missing_ok=True)
    return result
