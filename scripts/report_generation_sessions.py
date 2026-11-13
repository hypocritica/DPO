"""Report completed renderer session timings and memory without loading models."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.out).expanduser().resolve()
    plan = json.loads((root / 'plan.json').read_text())
    for task in plan['tasks']:
        folder = root / task['output'] / 'render'
        records = [json.loads(p.read_text()) for p in (folder / 'sessions').glob('*.json')]
        if not records:
            print(f"{task['method']} step={task['step']}: no completed session")
            continue
        record = max(records, key=lambda r: r['started_unix'])
        images = record.get('generation_images_computed', 0)
        forward = record.get('generation_forward_seconds', 0)
        print(json.dumps({'method': task['method'], 'step': task['step'],
            'generation_batch_size': task['sampling'].get('generation_batch_size', 1),
            'images_computed': images, 'batches': record.get('generation_batches'),
            'forward_seconds': forward, 'session_seconds': record['elapsed_seconds'],
            'forward_images_per_second': images / forward if forward else None,
            'peak_allocated_GiB': record['peak_allocated_bytes'] / 1024**3,
            'peak_reserved_GiB': record['peak_reserved_bytes'] / 1024**3,
            'gpu': record['gpu']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
