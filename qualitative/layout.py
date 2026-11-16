#!/usr/bin/env python3
"""Contact sheets, editable candidate selections and 174 mm paper layouts.

Selections are proposals until a human inspects the actual contact sheets.
No quality ranking or claim of MVPO superiority is inferred from prompt metadata.
"""
import argparse
import json
from pathlib import Path
import textwrap
from run import atomic_json, audit, image_path, read_json, valid_image

FAMILIES = ['sd15', 'sdxl', 'sd35_medium', 'zimage']
LABELS = dict(zip(FAMILIES, ['SD1.5', 'SDXL', 'SD3.5-M', 'Z-Image']))
METHODS = ['Base', 'DPO', 'MVPO']


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42})
    return plt


def save(fig, path, pdf=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Catch labels falling beyond the physical figure boundary, without tight bbox resizing.
    box = fig.bbox
    for item in fig.texts:
        b = item.get_window_extent(fig.canvas.get_renderer())
        if b.x0 < -1 or b.y0 < -1 or b.x1 > box.width + 1 or b.y1 > box.height + 1:
            raise ValueError(f'Text exceeds figure bounds: {item.get_text()}')
    fig.savefig(path.with_suffix('.png'), dpi=300, facecolor='white')
    if pdf:
        fig.savefig(path.with_suffix('.pdf'), facecolor='white')
    setup().close(fig)


def panel(fig, rect, path):
    from PIL import Image
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    if path and Path(path).is_file():
        with Image.open(path) as im:
            ax.imshow(im.convert('RGB'))
    else:
        ax.text(.5, .5, 'MISSING', ha='center', va='center', color='gray', transform=ax.transAxes)
    return ax


def contacts(root, families=None, smoke=False):
    # Pillow-only path: runs in the existing generation environments without matplotlib.
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    for family in families or FAMILIES:
        folder = root / family
        if not (folder / 'run_config.json').exists():
            continue
        audit(folder)
        plan = read_json(folder / 'run_config.json')
        for p in plan['prompts']:
            if smoke and p['id'] not in ('P02', 'P26'):
                continue
            sheet = Image.new('RGB', (1600, 2240), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((800, 15), f'{LABELS[family]} / {p["id"]}', font=font(28), fill='black', anchor='mt')
            draw.multiline_text((75, 65), textwrap.fill(p['prompt'], 115), font=font(20), fill='black', spacing=5)
            for col, method in enumerate(METHODS):
                draw.text((320 + col * 500, 148), method, font=font(24), fill='black', anchor='mt')
            for row, seed in enumerate(plan['spec']['sampling']['seeds']):
                y = 190 + row * 510
                draw.text((5, y + 220), f'seed\n{seed}', font=font(18), fill='black')
                for col, method in enumerate(METHODS):
                    path = image_path(folder, method, p['id'], seed)
                    x = 80 + col * 500
                    if valid_image(path, plan['plan_id']):
                        with Image.open(path) as original:
                            im = original.convert('RGB')
                            im.thumbnail((480, 480), Image.Resampling.LANCZOS)
                            sheet.paste(im, (x + (480-im.width)//2, y + (480-im.height)//2))
                    else:
                        draw.text((x+240, y+240), 'MISSING', font=font(24), fill='gray', anchor='mm')
            path = root / 'contact_sheets' / family / f'{p["id"]}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path, dpi=(300, 300))
        print(f'Contact sheets: {root / "contact_sheets" / family}', flush=True)


def case(root, family, p, seed, plan):
    paths = {m: image_path(root / family, m, p['id'], seed) for m in METHODS}
    if not all(valid_image(path, plan['plan_id']) for path in paths.values()):
        return None
    rows = [read_json(path.with_suffix('.json')) for path in paths.values()]
    if len({r['initial_noise_sha256'] for r in rows}) != 1:
        raise ValueError('Selected triplet noise differs')
    return {'backbone': family, 'prompt_id': p['id'], 'seed': seed,
            'prompt': p['prompt'], 'excerpt': p['prompt'][:125].rsplit(' ', 1)[0] + '…' if len(p['prompt']) > 125 else p['prompt'],
            'images': {m: str(path) for m, path in paths.items()},
            'reason': f'Unreviewed candidate: inspect {p["comparison_focus"]}; compare the whole matched triplet.',
            'reviewed': False}


def propose(root):
    # Diverse, prompt-based proposals; never rank methods using independently selected seeds.
    preferred = {'sd15': ['P02', 'P17', 'P07'], 'sdxl': ['P01', 'P09', 'P26'],
                 'sd35_medium': ['P22', 'P25', 'P30'], 'zimage': ['P26', 'P23', 'P29']}
    gallery_ids = {'sd15': ['P01', 'P02', 'P07', 'P17', 'P29'],
                   'sdxl': ['P09', 'P26', 'P05', 'P23', 'P11'],
                   'sd35_medium': ['P22', 'P25', 'P27', 'P15', 'P28'],
                   'zimage': ['P30', 'P04', 'P21', 'P12', 'P29']}
    comparison, gallery = [], []
    for family in FAMILIES:
        path = root / family / 'run_config.json'
        if not path.exists():
            continue
        plan = read_json(path)
        rows = {p['id']: p for p in plan['prompts']}
        for pid in preferred[family] + [p for p in rows if p not in preferred[family]]:
            candidates = [case(root, family, rows[pid], s, plan) for s in plan['spec']['sampling']['seeds']]
            available = next((c for c in candidates if c), None)
            if available:
                available['position'] = sum(c['backbone'] == family for c in comparison)
                comparison.append(available)
            if sum(c['backbone'] == family for c in comparison) == 3:
                break
        for pid in gallery_ids[family] + [p for p in rows if p not in gallery_ids[family]]:
            for seed in plan['spec']['sampling']['seeds']:
                im = image_path(root / family, 'MVPO', pid, seed)
                if valid_image(im, plan['plan_id']):
                    gallery.append({'backbone': family, 'prompt_id': pid, 'seed': seed,
                                    'position': sum(c['backbone'] == family for c in gallery),
                                    'image_path': str(im), 'reviewed': False})
                    break
            if sum(c['backbone'] == family for c in gallery) == 5:
                break
    for name, data in [('comparison_selection', comparison), ('gallery_selection', gallery)]:
        path = root / f'{name}.json'
        if path.exists():
            print(f'Preserving editable selection: {path}')
        else:
            atomic_json(path, {'status': 'UNREVIEWED prompt-diversity proposal; inspect contact sheets and edit', 'items': data})


def check_selection(root, selected, gallery=False):
    seen = set()
    for item in selected:
        family, pid, seed = item['backbone'], item['prompt_id'], item['seed']
        position = (family, item['position'])
        if position in seen:
            raise ValueError('Duplicate selection position')
        seen.add(position)
        plan = read_json(root / family / 'run_config.json')
        if seed not in plan['spec']['sampling']['seeds']:
            raise ValueError('Seed outside the fixed four')
        prompt = next(p for p in plan['prompts'] if p['id'] == pid)
        if gallery:
            expected = image_path(root / family, 'MVPO', pid, seed)
            if Path(item['image_path']).resolve() != expected.resolve() or not valid_image(expected, plan['plan_id']):
                raise ValueError('Gallery image must be an existing MVPO render')
        else:
            verified = case(root, family, prompt, seed, plan)
            if verified is None or item['images'] != verified['images'] or item['prompt'] != prompt['prompt']:
                raise ValueError('Selection is not a complete matched triplet')


def comparison(root, items, families, count, name, two_by_two=False):
    plt = setup()
    cols, rows = (2, 2) if two_by_two else (1, 2)
    width = 174 / 25.4
    block_w = .965 / cols
    image_in = width * (block_w - .03) / 3
    case_in = image_in + .47
    height = rows * (count * case_in + .30) + .12
    fig = plt.figure(figsize=(width, height))
    for index, family in enumerate(families):
        bx = .018 + (index % cols) * block_w
        top = 1 - .03 / height - (index // cols) / rows
        fig.text(bx + block_w / 2, top, LABELS[family], ha='center', va='top', fontsize=8, weight='bold')
        selected = sorted([c for c in items if c['backbone'] == family], key=lambda c: c['position'])[:count]
        for col, method in enumerate(METHODS):
            fig.text(bx + (col + .5) * block_w / 3, top - .17 / height,
                     'MVPO (Ours)' if method == 'MVPO' else method, ha='center', va='top', fontsize=7)
        for row in range(count):
            item = selected[row] if row < len(selected) else None
            y = top - (.32 + row * case_in + image_in) / height
            for col, method in enumerate(METHODS):
                panel(fig, [bx + col * block_w / 3, y, (block_w - .022) / 3, image_in / height], item['images'][method] if item else None)
            text = f'{item["prompt_id"]}, seed {item["seed"]}: {item["excerpt"]}' if item else 'No complete triplet available'
            fig.text(bx, y - .035 / height, textwrap.fill(text, 62 if cols == 2 else 120), va='top', fontsize=6.5)
    save(fig, root / 'figures' / name)


def gallery(root, items):
    plt = setup()
    width, height = 174 / 25.4, 6.65
    fig = plt.figure(figsize=(width, height))
    for row, family in enumerate(FAMILIES):
        top = .988 - row * .247
        fig.text(.015, top, LABELS[family], fontsize=8, weight='bold', va='top')
        selected = sorted([x for x in items if x['backbone'] == family], key=lambda x: x['position'])
        for col in range(5):
            item = selected[col] if col < len(selected) else None
            panel(fig, [.015 + col * .195, top - .212, .19, .19 * width / height], item['image_path'] if item else None)
            fig.text(.11 + col * .195, top - .219,
                     f'{item["prompt_id"]} / s{item["seed"]}' if item else 'MISSING', ha='center', va='top', fontsize=6.5)
    save(fig, root / 'figures/mvpo_gallery')


def export(root):
    cmp = read_json(root / 'comparison_selection.json')['items']
    gal = read_json(root / 'gallery_selection.json')['items']
    check_selection(root, cmp); check_selection(root, gal, gallery=True)
    comparison(root, cmp, FAMILIES, 2, 'qualitative_comparison', two_by_two=True)
    comparison(root, cmp, FAMILIES[2:], 3, 'qualitative_comparison_compact')
    comparison(root, cmp, FAMILIES[:2], 3, 'qualitative_comparison_supplement')
    gallery(root, gal)
    reviewed = all(x.get('reviewed') for x in cmp + gal) and len(cmp) == 12 and len(gal) == 20
    atomic_json(root / 'figures/layout_report.json', {'width_mm': 174, 'dpi': 300,
                'selection_visually_reviewed': reviewed, 'status': 'reviewed selection' if reviewed else 'PROVISIONAL: visual review still required',
                'comparison_cases': len(cmp), 'gallery_images': len(gal)})
    (root / 'figures/captions.txt').write_text(
        'Comparison. Qualitative samples from SD1.5, SDXL, SD3.5-M, and Z-Image using Base, DPO, '
        'and MVPO (Ours). Each triplet shares the unchanged prompt, seed, cached initial noise, '
        'and backbone-specific inference settings. The four-backbone layout shows two cases per '
        'backbone; the compact and supplementary layouts show three cases per backbone. '
        'Selection positions and provenance are recorded in comparison_selection.json. '
        'Cases are illustrative and do not establish aggregate performance.\n\n'
        'Gallery. MVPO samples from four backbones (one row per backbone), using a shared pool '
        'of 30 prompts and four fixed seeds. All displayed images reuse comparison-run outputs. '
        'Prompt IDs, seeds, and source PNGs are recorded in gallery_selection.json. '
        'Images are uniformly resized without retouching or cropping.\n\n'
        + ('Selections have been marked as visually reviewed.\n' if reviewed else
           'DRAFT STATUS: selections are unreviewed prompt-diversity proposals. Inspect actual contact '
           'sheets and edit selections before claiming visual curation or using these captions in a paper.\n'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['contacts', 'propose', 'export', 'all'])
    p.add_argument('--out', required=True)
    p.add_argument('--backbone', choices=FAMILIES)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); root = Path(a.out).resolve()
    if a.action in ('contacts', 'all'): contacts(root, [a.backbone] if a.backbone else None, smoke=a.smoke)
    if a.action in ('propose', 'all'): propose(root)
    if a.action in ('export', 'all'): export(root)

if __name__ == '__main__': main()
