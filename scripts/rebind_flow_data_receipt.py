"""Check certified data bytes; optionally write a new stat-bound receipt.

CPU-only, no image/model reads. The original report and prepared files are never
modified. --out is allowed only when pairs/ready/audit SHA256 all match the old
passed report; this is metadata rebinding, not a new data audit.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def checked_binding(path):
    from self_correct_dpo.common import sha256
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    checksum = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed while computing SHA256: {path}')
    return dict(path=str(path), size_bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=checksum)


def verify_and_rebind(pairs, report_path=None, out=None):
    from self_correct_dpo.common import output_lock, read_json
    from self_correct_dpo.flow_receipts import _passed, data_receipt
    from self_correct_dpo.runtime.checkpoints import atomic_json
    pairs = Path(pairs).expanduser().resolve(strict=True)
    report_path = Path(report_path or pairs.parent / 'validation-report.json').expanduser().resolve(strict=True)
    source = checked_binding(report_path)
    report = _passed(report_path, 'flow_data_validation')
    scope = report.get('scope', {})
    if scope.get('manifest_checks') is not True or scope.get('image_integrity') != 'verified_during_preparation':
        raise ValueError('Original report does not certify prepared data')
    if out is not None:
        out = Path(out).expanduser().resolve()
        if out == report_path:
            raise ValueError('The original report must be preserved; choose another --out')
    rebound = copy.deepcopy(report)
    evidence = {}
    for name, path in (('pairs', pairs), ('ready', pairs.parent/'ready.json'), ('audit', pairs.parent/'audit.json')):
        path = path.resolve(strict=True)
        expected = report['dataset_identity'][name]
        if Path(expected['path']).expanduser().resolve(strict=True) != path:
            raise ValueError(f'Original receipt belongs to another {name} path')
        print(f'[receipt] SHA256 {name}: {path} ({path.stat().st_size} bytes)', flush=True)
        observed = checked_binding(path)
        if observed['sha256'] != expected.get('sha256'):
            raise ValueError(f'SHA256 differs for {name}: {path}; expected={expected.get("sha256")}, '
                             f'observed={observed["sha256"]}. Data content changed; metadata rebinding refused.')
        changed = [key for key in ('size_bytes', 'mtime_ns') if expected[key] != observed[key]]
        evidence[name] = dict(expected=expected, observed=observed, changed_fields=changed)
        rebound['dataset_identity'][name] = observed
        print(f'[receipt] {name}: SHA256 MATCH; metadata changes={changed or "none"}; '
              f'expected_mtime_ns={expected["mtime_ns"]}; observed_mtime_ns={observed["mtime_ns"]}', flush=True)
    ready = read_json(pairs.parent/'ready.json')
    if (ready.get('pairs_sha256') != report['dataset_identity']['pairs']['sha256']
            or ready.get('source_revision') != report.get('source_revision')):
        raise ValueError('Original report and preparation identity disagree')
    if checked_binding(report_path) != source:
        raise ValueError('Original report changed during verification')
    for name, item in evidence.items():
        stat = Path(item['observed']['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['observed']['size_bytes'], item['observed']['mtime_ns']):
            raise ValueError(f'{name} changed after SHA256 verification')
    result = dict(status='verified', content_identical=True, source_report=str(report_path),
                  metadata_changed=any(item['changed_fields'] for item in evidence.values()), files=evidence)
    if out is not None:
        rebound['stat_rebinding'] = dict(method='sha256_identical_metadata_rebinding',
            created_at=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
            source_report=source, files=evidence, image_bytes_rechecked=False)
        with output_lock(out):
            if out.exists():
                previous = data_receipt(out, pairs)
                origin = previous.get('stat_rebinding', {}).get('source_report', {})
                without_provenance = lambda value: {k:v for k,v in value.items() if k != 'stat_rebinding'}
                if (without_provenance(previous) != without_provenance(rebound)
                        or origin.get('sha256') != source['sha256'] or origin.get('path') != source['path']):
                    raise FileExistsError(f'Refusing to overwrite an unrelated existing file: {out}')
                result['status'] = 'reused'
            else:
                atomic_json(out, rebound)
                data_receipt(out, pairs)
                result['status'] = 'rebound'
        result['report'] = str(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='Original passed report; default pairs.parent/validation-report.json')
    parser.add_argument('--out', type=Path, help='New receipt path; omit for read-only diagnosis. Reuses an identical valid output without overwriting.')
    args = parser.parse_args(argv)
    try:
        result = verify_and_rebind(args.pairs, args.report, args.out)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps(dict(status='failed', error=str(error)), ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
