"""Freeze the three published Diffusion-SDPO prompt panels, preserving every row."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.common import digest, load_suite, output_lock, read_json, sha256, write_json, write_jsonl
from self_correct_dpo.resources import load_resources

REPOSITORY = 'ATH-MaaS/Diffusion-SDPO'
REVISION = '84fb241c1b89705a247da8b0d6047798ca49830d'
SOURCES = {
    'pickv2': ('papv2.json', 487, '531f3ca923035b5f17dbcbc60a7d7dfa476219e59b0e99617bb3f0afabf5663b'),
    'hpsv2': ('hpsv2.json', 3200, '417a2ce00b3307780402c1d27528a0e41ed26651384687bbf3aa06c4c7e559fd'),
    'partiprompts': ('partiprompts.json', 1632, 'b11d46a91c4202461d6d1df69da0b9af5070851403574de6efd6736ccad22291'),
}


def panel_rows(data, name):
    if not isinstance(data, list) or not data:
        raise ValueError('Expected a nonempty JSON array of prompt records')
    rows = []
    for index, original in enumerate(data):
        if not isinstance(original, dict):
            raise ValueError(f'{name}[{index}] is not a prompt record')
        prompt = original.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'{name}[{index}] has no prompt')
        if original.get('neg_prompt', ''):
            raise ValueError('This preset requires empty per-prompt negative prompts')
        # Duplicate texts are deliberate benchmark entries. Never deduplicate,
        # strip, normalize, exclude monitor prompts, or alter their weighting.
        rows.append({'scene_id': f'{name}-{index:05d}-{digest(prompt)[:12]}',
            'variant_id': 'canonical', 'prompt': prompt, 'tag': name,
            'metadata': {'prompt': prompt, 'tag': name},
            'suite_mode': 'canonical-only-v1', 'template_family': 'canonical',
            'transform_type': 'canonical', 'source_index': index})
    return rows


def prepare(out, *, source_dir=None, download=False, smoke_prompts=None, sources=SOURCES):
    if (source_dir is None) == (not download):
        raise ValueError('Choose exactly one of source_dir or download')
    if smoke_prompts is not None and (type(smoke_prompts) is not int or smoke_prompts < 1):
        raise ValueError('smoke_prompts must be a positive integer')
    out = Path(out).expanduser().resolve()
    with output_lock(str(out) + '.prepare'):
        # Work in the destination filesystem, then publish the entire panel set.
        with tempfile.TemporaryDirectory(prefix='.' + out.name + '-', dir=out.parent) as temporary:
            stage = Path(temporary) / 'panels'
            stage.mkdir()
            (stage / 'sources').mkdir()
            panels = {}
            for name, (filename, count, checksum) in sources.items():
                url = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/prompts/{filename}'
                if download:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        payload = response.read(8 * 1024 * 1024 + 1)
                    if len(payload) > 8 * 1024 * 1024:
                        raise ValueError(f'Unexpectedly large prompt file: {filename}')
                else:
                    payload = (Path(source_dir) / filename).read_bytes()
                if hashlib.sha256(payload).hexdigest() != checksum:
                    raise ValueError(f'Published source checksum mismatch: {filename}; use revision {REVISION}')
                rows = panel_rows(json.loads(payload), name)
                if len(rows) != count:
                    raise ValueError(f'{name}: expected {count} rows, found {len(rows)}')
                if smoke_prompts is not None:
                    rows = rows[:smoke_prompts]
                (stage / 'sources' / filename).write_bytes(payload)
                target = stage / f'{name}.jsonl'
                write_jsonl(target, rows)
                load_suite(target)
                panels[name] = {'prompts': len(rows), 'source_prompts': count,
                    'unique_exact_texts': len({r['prompt'] for r in rows}),
                    'sha256': sha256(target), 'source_file': filename,
                    'source_sha256': checksum, 'source_url': url}
            manifest = {'repository': REPOSITORY, 'revision': REVISION,
                'mode': 'smoke-only' if smoke_prompts is not None else 'full-benchmark',
                'smoke_prompts': smoke_prompts,
                'policy': ('Keep the first N source rows for smoke only; preserve order and exact text.'
                           if smoke_prompts is not None else
                           'Preserve all source rows, order and exact text; no deduplication or overlap filtering.'),
                'note': 'hpsv2 is the published 3200-row panel, not the legacy HPDv2 test400 panel. '
                        'PickV2 is the full published panel and can overlap the training-monitor panel.',
                'panels': panels}
            write_json(stage / 'selection.json', manifest)
            if out.exists():
                if read_json(out / 'selection.json') != manifest:
                    raise ValueError('Existing prompt set has different provenance; choose another output directory')
                for name, item in panels.items():
                    if sha256(out / f'{name}.jsonl') != item['sha256'] or sha256(out / 'sources' / item['source_file']) != item['source_sha256']:
                        raise ValueError(f'Existing prompt set was modified: {name}')
            else:
                stage.rename(out)
            return {'out': str(out), **manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--download-sdpo', action='store_true', help='Download about 0.9 MB of pinned prompt JSONs; never downloads model weights')
    source.add_argument('--source-dir', help=f'Local prompts/ directory from Diffusion-SDPO revision {REVISION}')
    parser.add_argument('--resources', help='Resource registry used for the default output path')
    parser.add_argument('--out', help='Default: BENCHMARKS_ROOT/sd15/paper-eval')
    parser.add_argument('--smoke-prompts', type=int, help='Keep only the first N rows per panel for an explicitly labelled smoke test; use a separate --out')
    args = parser.parse_args(argv)
    if args.smoke_prompts is not None and not args.out:
        parser.error('--smoke-prompts requires a separate explicit --out')
    out = args.out or str(Path(load_resources(args.resources)['BENCHMARKS_ROOT']) / 'sd15/paper-eval')
    result = prepare(out, source_dir=args.source_dir, download=args.download_sdpo, smoke_prompts=args.smoke_prompts)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
