"""A local HTML contact sheet; selections never alter evaluation manifests."""
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from self_correct_dpo.common import digest, read_json, read_jsonl, sha256, verify_images
from self_correct_dpo.evaluation.suite import filter_tasks, safe_target, freeze_json, validate_task_render, score_summary


def make_gallery(plan, out, args):
    name = args.gallery_name
    if not name or Path(name).name != name or name in ('.', '..') or args.limit < 1:
        raise ValueError('Use a simple gallery name and a positive image limit')
    gallery = safe_target(out, f'galleries/{name}')
    candidates = []
    for task in filter_tasks(plan, args):
        target = safe_target(out, task['output'])
        if not (target / 'render/complete.json').exists():
            continue
        run, render_complete = validate_task_render(task, plan, out)
        _, rows = verify_images(target / 'render')
        scores = {}
        for journal in sorted((target / 'scores').glob('*/scores.jsonl')):
            if not (journal.parent / 'complete.json').exists():
                continue
            score_summary(target, journal.parent.name, run, render_complete['manifest_sha256'])
            complete = read_json(journal.parent / 'complete.json')
            if complete.get('scores_sha256') != sha256(journal):
                raise ValueError(f'Score journal checksum mismatch: {journal}')
            for row in read_jsonl(journal):
                key = (row['scene_id'], row['variant_id'], row['seed'])
                scores.setdefault(key, {})[row['metric']] = row['score']
        for row in rows:
            if args.scene_id and row['scene_id'] not in args.scene_id:
                continue
            if args.seed and row['seed'] not in args.seed:
                continue
            key = (row['scene_id'], row['variant_id'], row['seed'])
            candidates.append({'run': task['run'], 'family': task['family'], 'step': task['step'],
                'benchmark': task['benchmark'], **{k: row[k] for k in ('scene_id', 'variant_id', 'seed', 'prompt', 'image_sha256')},
                'image': str((target / 'render' / row['image']).resolve()), 'scores': scores.get(key, {})})
    # Put the same prompt and seed next to each other across checkpoints.
    candidates.sort(key=lambda r: (r['benchmark'], r['scene_id'], r['seed'], r['family'], r['run'], r['step']))
    selected = candidates[:args.limit]
    if not selected:
        raise ValueError('No completed generated images match the gallery filters')
    selection = {'plan_id': plan['plan_id'], 'purpose': 'qualitative display only; quantitative scores use all planned images',
                 'selection_rule': 'explicit prompt/seed filters then manifest order; no best-of selection',
                 'available': len(candidates), 'limit': args.limit, 'images': selected}
    selection['selection_id'] = digest(selection)
    freeze_json(gallery / 'selection.json', selection)
    escape = html.escape
    cards = []
    for row in selected:
        url = quote(os.path.relpath(row['image'], gallery), safe='/')
        label = f"{row['run']} · step {row['step']} · seed {row['seed']}"
        metrics = ' · '.join(f'{key}: {value:.4f}' for key, value in row['scores'].items())
        cards.append(f'<article><a href="{url}"><img loading="lazy" src="{url}" alt="{escape(row["prompt"], quote=True)}"></a>'
                     f'<b>{escape(label)}</b><p>{escape(row["prompt"])}</p><small>{escape(row["scene_id"])}<br>'
                     f'{escape(metrics)}</small><details><summary>Image path / SHA256</summary>'
                     f'<code>{escape(row["image"])}<br>{row["image_sha256"]}</code></details></article>')
    document = '<!doctype html><meta charset="utf-8"><title>Self-correct evaluation gallery</title>' \
        '<style>body{font:15px system-ui;margin:24px;background:#f5f5f5;color:#222}' \
        'main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}' \
        'article{background:white;padding:14px;border-radius:8px}img{width:100%;height:320px;object-fit:contain}' \
        'p,code{overflow-wrap:anywhere}small{color:#555}</style>' \
        '<h1>Self-correct evaluation gallery</h1><p>展示候选图；统计结果包含全部预定样本。' \
        '点击图片查看原图。selection.json 保存 prompt、seed、checkpoint 与图片路径。</p><main>' \
        + ''.join(cards) + '</main>'
    (gallery / 'index.html').write_text(document)
    return str(gallery / 'index.html')
