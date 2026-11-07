"""Add canonical metadata to an old Flow prompt panel in a NEW output file."""
import argparse
import copy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from self_correct_dpo.common import load_suite, read_jsonl, sha256, write_json, write_jsonl


def repair(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists() or out.with_suffix('.audit.json').exists():
        raise FileExistsError(out)
    original = read_jsonl(source)
    rows = copy.deepcopy(original)
    for row in rows:
        if row.get('variant_id') != 'canonical':
            raise ValueError('This repair only accepts canonical-only Flow panels')
        metadata = row.setdefault('metadata', {})
        for key in ('prompt', 'tag'):
            if key in metadata and metadata[key] != row[key]:
                raise ValueError(f'Conflicting existing metadata: {key}')
            metadata[key] = row[key]
        row.update(suite_mode='canonical-only-v1', template_family='canonical', transform_type='canonical')
    # Reuse the real loader to validate before publishing the new panel.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / 'panel.jsonl'
        write_jsonl(temporary, rows)
        load_suite(temporary)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    write_json(out.with_suffix('.audit.json'), {
        'kind': 'flow_monitor_metadata_repair', 'source': str(source),
        'source_sha256': sha256(source), 'prompts_sha256': sha256(out), 'count': len(rows),
        'prompt_text_and_order_preserved': True, 'training_data_reprocessed': False})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(repair(args.source, args.out))


if __name__ == '__main__':
    main()
