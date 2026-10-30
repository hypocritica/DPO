"""Derive SDXL pairs/panels from ready SD1.5 metadata; never open images or Parquet."""
import argparse
from collections import Counter
from hashlib import sha256 as new_hash
from itertools import islice
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import output_lock, read_json, sha256, write_json, write_jsonl
from self_correct_dpo.data.benchmarks import normalize_panel, select_monitor
from self_correct_dpo.data.pairs import validate_pair_rows
from self_correct_dpo.resources import load_resources


def json_rows(path, hasher=None):
    with Path(path).open('rb') as handle:
        for line in handle:
            if hasher is not None:
                hasher.update(line)
            if line.strip():
                yield json.loads(line)


def tokenizer_identity(model):
    files = {}
    for name in ('tokenizer', 'tokenizer_2'):
        directory = model / name
        inventory = sorted(p for p in directory.rglob('*') if p.is_file())
        if not inventory:
            raise ValueError(f'Missing local SDXL tokenizer files: {directory}')
        files.update({str(p.relative_to(model)): sha256(p) for p in inventory})
    return files


def allowed_texts(texts, tokenizers):
    allowed = [True] * len(texts)
    for tokenizer in tokenizers:
        limit = tokenizer.model_max_length
        if not isinstance(limit, int) or not 0 < limit < 1000000:
            raise ValueError('SDXL tokenizer must declare a finite model_max_length')
        ids = tokenizer(texts, padding=False, truncation=False,
                        add_special_tokens=True)['input_ids']
        if len(ids) != len(texts):
            raise ValueError('Tokenizer returned an unexpected batch size')
        allowed = [a and len(tokens) <= limit for a, tokens in zip(allowed, ids)]
    return allowed


def checked_output(path, identity, marker):
    """Reuse only a complete conversion with matching inputs and metadata hashes."""
    if not path.exists():
        return False
    if not (path / marker).is_file():
        raise ValueError(f'Existing output has no conversion marker: {path}; use a new output')
    record = read_json(path / marker)
    if record.get('conversion_identity') != identity:
        raise ValueError(f'Existing output differs from this conversion: {path}; use a new output')
    for name, expected in record['files_sha256'].items():
        if sha256(path / name) != expected:
            raise ValueError(f'Converted metadata changed: {path / name}')
    return True


def convert(source, out, model, panels_out, *, count=128, seed=17, batch_size=256,
            tokenizers=None):
    source, out, model, panels_out = [Path(p).expanduser().resolve()
                                     for p in (source, out, model, panels_out)]
    if source.name != 'pairs.jsonl':
        raise ValueError('Source must be a prepared pairs.jsonl, never raw Parquet')
    if batch_size < 1 or count < 1:
        raise ValueError('batch_size and monitor count must be positive')
    print(json.dumps({'phase': 'source_identity', 'source_pairs': str(source),
                      'out': str(out), 'panels_out': str(panels_out)}), flush=True)
    directories = (source.parent, out, panels_out)
    for i, left in enumerate(directories):
        for right in directories[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError('Source, converted data and panels must be separate directories')
    ready_path = source.parent / 'ready.json'
    ready = read_json(ready_path)
    source_sha = sha256(source)
    if ready.get('pairs_sha256') != source_sha or not ready.get('source_revision'):
        raise ValueError('SD1.5 ready.json does not match the source manifest/revision')
    audit_path = source.parent / 'audit.json'
    if audit_path.is_file() and read_json(audit_path).get('family') != 'sd15':
        raise ValueError('Source audit must identify prepared SD1.5 data')
    official = source.parent / 'official-test-unique.jsonl'
    identity = {'format': 1, 'source_pairs': str(source), 'source_sha256': source_sha,
                'source_ready_sha256': sha256(ready_path),
                'source_audit_sha256': sha256(audit_path) if audit_path.is_file() else None,
                'official_test_sha256': sha256(official), 'model': str(model),
                'tokenizers': tokenizer_identity(model), 'monitor_count': count, 'seed': seed,
                'out': str(out), 'panels_out': str(panels_out), 'code_sha256': sha256(__file__)}
    # Both locks are also used by the standard data/panel preparation helpers.
    with output_lock(out), output_lock(panels_out):
        have_data = checked_output(out, identity, 'ready.json')
        have_panels = checked_output(panels_out, identity, 'selection.json')
        if have_data and have_panels:
            print(json.dumps({'phase': 'reuse', 'pairs': str(out / 'pairs.jsonl'),
                              'monitor': str(panels_out / 'pickapic-monitor.jsonl')}), flush=True)
            return read_json(out / 'audit.json')
        if tokenizers is None:
            from transformers import AutoTokenizer
            tokenizers = [AutoTokenizer.from_pretrained(str(model / name), local_files_only=True)
                          for name in ('tokenizer', 'tokenizer_2')]
        if len(tokenizers) != 2:
            raise ValueError('Exactly two SDXL tokenizers are required')
        # Staging stays beside each destination, so publication is a directory rename.
        with tempfile.TemporaryDirectory(prefix='.' + out.name + '-convert-', dir=out.parent) as data_tmp, \
                tempfile.TemporaryDirectory(prefix='.' + panels_out.name + '-convert-', dir=panels_out.parent) as panel_tmp:
            data_stage, panel_stage = Path(data_tmp) / 'data', Path(panel_tmp) / 'panels'
            data_stage.mkdir()
            panel_stage.mkdir()
            before, after, ids, train_prompts = Counter(), Counter(), set(), set()
            source_digest = new_hash()
            rows = json_rows(source, source_digest)
            print(json.dumps({'phase': 'token_filter', 'source_pairs': 0,
                              'source_total': sum(ready['splits'].values())}), flush=True)
            with (data_stage / 'pairs.jsonl').open('w') as pairs, \
                    (data_stage / 'excluded.jsonl').open('w') as excluded:
                while True:
                    batch = list(islice(rows, batch_size))
                    if not batch:
                        break
                    # Metadata only: lexical path normalization, no image stat/hash/decode.
                    validate_pair_rows(batch, source.parent, verify_files=False,
                                       resolve_paths=False, in_place=True)
                    texts = [p['prompt'] for row in batch for p in row['prompts']]
                    flags = iter(allowed_texts(texts, tokenizers))
                    for row in batch:
                        pair_id, split = row['pair_id'], row['split']
                        if pair_id in ids:
                            raise ValueError(f'Duplicate source pair_id: {pair_id}')
                        ids.add(pair_id)
                        if row.get('storage', {}).get('kind') == 'arrow':
                            raise ValueError('Expected exported images, not Arrow-backed source pairs')
                        for side in ('winner', 'loser'):
                            checksum = row.get(side + '_sha256', '')
                            if not isinstance(checksum, str) or len(checksum) != 64:
                                raise ValueError(f'Missing prepared image SHA256: {pair_id}/{side}')
                        evidence = row.get('human_preference', {})
                        if evidence and evidence.get('source_revision') != ready['source_revision']:
                            raise ValueError(f'Source revision differs from ready.json: {pair_id}')
                        before[split] += 1
                        if split == 'train':
                            train_prompts.update(p['prompt'] for p in row['prompts'])
                        rejected = [p['variant_id'] for p in row['prompts'] if not next(flags)]
                        if rejected:
                            excluded.write(json.dumps({'pair_id': pair_id, 'split': split,
                                'reason': 'sdxl_token_length', 'variants': rejected}) + '\n')
                            continue
                        # Absolute references keep pointing to the same image bytes from any out.
                        pairs.write(json.dumps(row, ensure_ascii=False) + '\n')
                        after[split] += 1
                    if sum(before.values()) % (batch_size * 40) == 0:
                        print(json.dumps({'phase': 'token_filter', 'source_pairs': sum(before.values()),
                                          'kept': dict(after)}), flush=True)
            if (source_digest.hexdigest() != source_sha or dict(before) != ready.get('splits')
                    or sha256(ready_path) != identity['source_ready_sha256']):
                raise ValueError('Source manifest/ready changed or split counts do not match')
            if any(after[split] == 0 for split in ('train', 'validation', 'test')):
                raise ValueError('Tokenizer filtering left an empty split; inspect source/exclusions')
            # This caption export was already produced by SD1.5 preparation. No raw-data access.
            candidates = []
            print(json.dumps({'phase': 'monitor_panels', 'splits': dict(after)}), flush=True)
            official_digest = new_hash()
            official_rows = json_rows(official, official_digest)
            with (data_stage / 'official-test-unique.jsonl').open('w') as handle:
                while True:
                    batch = list(islice(official_rows, batch_size))
                    if not batch:
                        break
                    texts = [row.get('caption', row.get('prompt')) for row in batch]
                    if any(not isinstance(t, str) or not t.strip() for t in texts):
                        raise ValueError('Official caption export contains an empty/non-text caption')
                    for text, allowed in zip(texts, allowed_texts(texts, tokenizers)):
                        if allowed:
                            candidates.append({'caption': text})
                            handle.write(json.dumps({'caption': text}, ensure_ascii=False) + '\n')
            if official_digest.hexdigest() != identity['official_test_sha256']:
                raise ValueError('Official caption export changed during conversion')
            suite = normalize_panel(candidates, 'pickapic-test', exclude=train_prompts)
            monitor = select_monitor(suite, count, seed)
            monitor_ids = {row['scene_id'] for row in monitor}
            panels = {'pickapic-test': suite, 'pickapic-monitor': monitor,
                      'pickapic-test-remaining': [r for r in suite if r['scene_id'] not in monitor_ids]}
            for name, panel in panels.items():
                write_jsonl(panel_stage / (name + '.jsonl'), panel)
            selection = {'seed': seed, 'monitor_count': count, 'conversion_identity': identity,
                         'source_paths': {'pickapic': str(official), 'train_pairs': str(source)},
                         'panels': {name: {'prompts': len(panel),
                             'sha256': sha256(panel_stage / (name + '.jsonl'))}
                                    for name, panel in panels.items()},
                         'files_sha256': {name + '.jsonl': sha256(panel_stage / (name + '.jsonl'))
                                          for name in panels},
                         'policy': 'Both SDXL tokenizers; exclude every source train caption, including filtered rows.'}
            write_json(panel_stage / 'selection.json', selection)
            audit = {'family': 'sdxl', 'conversion_identity': identity, 'source_splits': dict(before),
                     'splits': dict(after), 'excluded_splits': dict(before - after),
                     'images_written_bytes': 0, 'image_policy': 'Reuse absolute references and original SHA256; no image reads.',
                     'split_policy': 'Retained pairs keep their source split; no regrouping or resplitting.',
                     'limitation': 'Derived subset of SD1.5 data; cannot recover rows excluded during SD1.5 preparation.'}
            write_json(data_stage / 'audit.json', audit)
            files = {name: sha256(data_stage / name) for name in
                     ('pairs.jsonl', 'excluded.jsonl', 'official-test-unique.jsonl', 'audit.json')}
            write_json(data_stage / 'ready.json', {'pairs_sha256': files['pairs.jsonl'],
                'splits': dict(after), 'source_revision': ready['source_revision'],
                'conversion_identity': identity, 'files_sha256': files})
            if tokenizer_identity(model) != identity['tokenizers']:
                raise ValueError('SDXL tokenizer files changed during conversion')
            # A retry can finish after panels were published but data publication was interrupted.
            for target, stage, exists, marker in ((panels_out, panel_stage, have_panels, 'selection.json'),
                                                  (out, data_stage, have_data, 'ready.json')):
                if exists:
                    if read_json(target / marker) != read_json(stage / marker):
                        raise ValueError(f'Existing conversion is not reproducible: {target}')
                else:
                    os.rename(stage, target)
            print(json.dumps({'phase': 'complete', 'pairs': str(out / 'pairs.jsonl'),
                              'monitor': str(panels_out / 'pickapic-monitor.jsonl'),
                              'source_splits': dict(before), 'splits': dict(after),
                              'images_written_bytes': 0}), flush=True)
            return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources')
    parser.add_argument('--source-pairs', help='Ready SD1.5 pairs.jsonl')
    parser.add_argument('--out', help='SDXL prepared-data directory')
    parser.add_argument('--model', help='Local SDXL model directory (only tokenizers are loaded)')
    parser.add_argument('--panels-out', help='SDXL benchmark directory')
    parser.add_argument('--monitor-count', type=int, default=128)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    resources = load_resources(args.resources)
    if not args.panels_out and Path(resources['MONITOR_PROMPTS_SDXL']).name != 'pickapic-monitor.jsonl':
        parser.error('MONITOR_PROMPTS_SDXL must end in pickapic-monitor.jsonl; or select --panels-out explicitly')
    convert(args.source_pairs or Path(resources['PICKAPIC_SD15_PREPARED']) / 'pairs.jsonl',
            args.out or resources['PICKAPIC_SDXL_PREPARED'], args.model or resources['SDXL_MODEL'],
            args.panels_out or Path(resources['MONITOR_PROMPTS_SDXL']).parent,
            count=args.monitor_count, seed=args.seed, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
