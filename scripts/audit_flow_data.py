"""Certify prepared HPDv3 flow data once; report progress without reading images.

Run the complete audit in a CPU allocation after preparation. A passed report is
bound to immutable pairs/ready/audit files; launchers can reuse it using stat and
small-receipt checks. Image SHA256 checks remain enabled when training consumes
images. This command never rescans image bytes or changes a running preparation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def fingerprint(path, *, checksum=None):
    from self_correct_dpo.common import sha256
    path = Path(path).resolve()
    before = path.stat()
    checksum = checksum or sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Artifact changed during audit: {path}')
    return {'path': str(path), 'size_bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': checksum}


def log_evidence(path):
    """Read bounded tail plus initial selection metadata, not images/directories."""
    if path is None or not Path(path).is_file():
        return {}
    path = Path(path)
    with path.open('rb') as handle:
        beginning = handle.read(32768)
        handle.seek(max(0, path.stat().st_size - 65536))
        ending = handle.read(65536)
    result = {'path': str(path.resolve())}
    for block in (beginning, ending):
        for line in block.decode('utf-8', errors='replace').splitlines():
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(item, dict):
                continue
            if 'required_images_including_official_test' in item:
                result['selection'] = item
            if 'extracted_images' in item:
                result['latest_extraction_progress'] = item
            if item.get('phase') == 'verify_images':
                result['latest_verification_progress'] = item
    return result


def audit(pairs, *, extraction, raw=None, preparation_log=None, job_id=None):
    from self_correct_dpo.common import sha256
    from self_correct_dpo.data.pickapic import prompt_key, validate_human_pair
    from self_correct_dpo.data.validation import load_training_pairs

    pairs, extraction = Path(pairs).resolve(), Path(extraction).resolve()
    ready_path, audit_path = pairs.parent / 'ready.json', pairs.parent / 'audit.json'
    report = {
        'schema_version': 1, 'kind': 'flow_data_validation',
        'status': 'in_progress', 'passed': False,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset_identity': {}, 'checks': {},
        'scope': {'manifest_checks': False, 'image_integrity': 'not_yet_verified',
                  'image_bytes_rehashed': False, 'image_directory_scanned': False,
                  'raw_train_records_rechecked': False},
        'preparation': {'job_id': str(job_id) if job_id is not None else None,
                        'log': log_evidence(preparation_log)},
        'intended_policy': {'allowed_models': ['flux', 'hunyuan', 'kolors', 'real_images', 'sd3'],
                            'winner': 'official path1', 'loser': 'official path2',
                            'seed': 17, 'split_fractions': [.9, .05, .05],
                            'resolution_filter': None,
                            'heldout': 'Official HPDv3 test normalized captions, image UIDs and byte hashes',
                            'split_grouping': 'Connected components of normalized prompts, image UIDs and byte hashes'},
        'limitations': [
            'Unknown semantic paraphrases are not certified disjoint.',
            'Local validation/test splits are not the official HPDv3 test set.',
            'Image verification is reused from preparation; image bytes are checked again when consumed.',
            'Original train records/labels are checked for consistency inside manifest evidence, not rescanned against full train.json.',
            'Preparation audit/ready/extraction receipts are trusted local preparation evidence, not independently authenticated certificates.',
            'Certificate reuse assumes immutable prepared files; file stat bindings detect ordinary changes.',
        ],
    }
    required = {'pairs': pairs, 'ready': ready_path, 'audit': audit_path, 'extraction': extraction}
    report['artifacts'] = {name: {'path': str(path), 'exists': path.is_file()}
                           for name, path in required.items()}
    progress = report['preparation']['log']
    selection = progress.get('selection', {})
    current = progress.get('latest_extraction_progress', {})
    report['counts'] = {
        'selected_pairs_before_image_exclusions': selection.get('selected_pairs_before_image_exclusions'),
        'required_images_including_official_test': selection.get('required_images_including_official_test'),
        'metadata_exclusions': selection.get('exclusions'),
        'extracted_images_so_far': current.get('extracted_images'),
        'usable_pairs': None, 'final_split_counts': None,
    }
    if raw is not None:
        raw = Path(raw).resolve()
        report['source'] = {'path': str(raw), 'annotations': {}}
        for name in ('train.json', 'test.json'):
            path = raw / name
            if path.is_file():
                stat = path.stat()
                report['source']['annotations'][name] = {
                    'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                    'bytes_rehashed_in_progress': False}
        for name in ('.download-manifest.json', '.scd-download.json'):
            path = raw / name
            if path.is_file():
                receipt = json.loads(path.read_text())
                report['source']['download'] = {
                    'repo_id': receipt.get('repo_id'), 'revision': receipt.get('revision'),
                    'status': receipt.get('status', 'complete' if receipt.get('complete') else 'unknown'),
                    'download_source': receipt.get('download_source'), 'receipt': fingerprint(path)}
                break
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        report['checks']['completed_preparation'] = {'status': 'pending', 'missing': missing}
        if ready_path.is_file():
            report.update(status='failed')
            report['error'] = 'ready.json exists but required preparation artifacts are missing'
        return report
    try:
        baseline = {name: (path.stat().st_size, path.stat().st_mtime_ns)
                    for name, path in required.items()}
        ready = json.loads(ready_path.read_text())
        prepared = json.loads(audit_path.read_text())
        extracted = json.loads(extraction.read_text())
        raw = Path(raw or extracted['source']).resolve()
        receipt_path = next((raw / name for name in ('.download-manifest.json', '.scd-download.json')
                             if (raw / name).is_file()), None)
        if receipt_path is None:
            raise ValueError('Missing raw download receipt')
        receipt = json.loads(receipt_path.read_text())
        if not (receipt.get('status') == 'complete' or receipt.get('complete') is True):
            raise ValueError('Raw download receipt is not complete')
        revision = receipt['repo_id'] + '@' + receipt['revision']
        if ready.get('source_revision') != revision or prepared.get('source_revision') != revision:
            raise ValueError('Ready/audit/download source revisions disagree')
        if extracted.get('revision') != receipt['revision']:
            raise ValueError('Extraction source revision differs from download receipt')
        if prepared.get('family') != 'unfiltered':
            raise ValueError('Flow HPDv3-sub expects unfiltered preparation')
        expected_models = ['flux', 'hunyuan', 'kolors', 'real_images', 'sd3']
        if prepared.get('allowed_models') != expected_models:
            raise ValueError('Unexpected HPDv3-sub model subset')
        if prepared.get('seed') != 17 or list(prepared.get('fractions', [])) != [.9, .05, .05]:
            raise ValueError('Unexpected split seed/fractions')
        for key in ('train_json_sha256', 'test_json_sha256'):
            if not prepared.get(key) or prepared[key] != extracted.get(key):
                raise ValueError(f'Preparation/extraction disagree on {key}')
        downloaded_train = next((r for r in receipt.get('completed', [])
                                 if r.get('file') == 'train.json'), {})
        if len(downloaded_train.get('hash', '')) == 64 and downloaded_train['hash'] != prepared['train_json_sha256']:
            raise ValueError('Train annotation digest differs from download receipt')
        tests_path = raw / 'test.json'
        if sha256(tests_path) != prepared['test_json_sha256']:
            raise ValueError('Official test metadata changed')
        tests = json.loads(tests_path.read_text())
        heldout_prompts = {prompt_key(row['prompt']) for row in tests}
        heldout_uids = {row[side] for row in tests for side in ('path1', 'path2')}
        # This is the sole complete manifest audit. It does not open image files.
        rows = load_training_pairs(pairs, mode='none')
        counts = dict(Counter(row['split'] for row in rows))
        if counts != ready['splits'] or counts != prepared.get('splits'):
            raise ValueError('Manifest/ready/audit split counts disagree')
        if any(counts.get(split, 0) == 0 for split in ('train', 'validation', 'test')):
            raise ValueError('Prepared data has an empty split')
        if prepared.get('usable_pairs') != len(rows):
            raise ValueError('Audit usable-pair count differs from manifest')
        if prepared.get('raw_train_rows') != len(rows) + sum(prepared.get('exclusions', {}).values()):
            raise ValueError('Raw/usable/excluded counts do not balance')
        for row in rows:
            validate_human_pair(row)
            evidence = row['human_preference']
            if evidence['source'] != 'hpdv3' or evidence['source_revision'] != revision:
                raise ValueError('Manifest contains a different human-preference source')
            original = evidence['original_record']
            if original.get('model1') not in expected_models or original.get('model2') not in expected_models:
                raise ValueError('Manifest contains an excluded generator')
            if prompt_key(original['prompt']) in heldout_prompts or heldout_uids.intersection(evidence['image_uids']):
                raise ValueError('Official HPDv3 test prompt/image UID leaks into local splits')
            for side, original_side in (('winner', 'path1'), ('loser', 'path2')):
                if Path(row[side]).absolute() != (extraction.parent / original[original_side]).absolute():
                    raise ValueError('Winner/loser path differs from original path1/path2 ordering')
        for name, path in required.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != baseline[name]:
                raise ValueError(f'{name} changed during audit')
        identities = {name: fingerprint(path, checksum=ready['pairs_sha256'] if name == 'pairs' else None)
                      for name, path in required.items()}
        identities['download_receipt'] = fingerprint(receipt_path)
        report.update(status='passed', passed=True, source_revision=revision,
                      dataset_identity=identities, splits=counts,
                      counts={'usable_pairs': len(rows), 'raw_train_rows': prepared['raw_train_rows'],
                              'exclusions': prepared['exclusions'], 'extracted_images': extracted.get('images'),
                              'official_test_rows': len(tests)},
                      split_policy={'seed': 17, 'fractions': [.9, .05, .05],
                                    'connected_components': prepared.get('connected_components'),
                                    'official_hpdv3_heldout_checked': True,
                                    'pickapic_heldout_checked': bool(prepared.get('pickapic_heldout_inventory')),
                                    'semantic_paraphrases_certified': False})
        report['scope'].update(manifest_checks=True, image_integrity='verified_during_preparation')
        report['checks'] = {name: {'status': 'passed'} for name in (
            'completed_preparation', 'manifest_schema_and_sha256', 'split_disjointness',
            'source_revision_and_receipts', 'official_hpdv3_test_caption_and_uid_isolation',
            'winner_path1_and_original_label_order', 'model_subset', 'counts_balance')}
        report['checks']['official_test_image_hash_isolation'] = {
            'status': 'preparation_evidence', 'evidence': str(audit_path),
            'note': 'Preparation verified/hash-compared official test images; this audit does not reread them.'}
        report['checks']['original_train_record_provenance'] = {
            'status': 'preparation_evidence', 'evidence': str(audit_path),
            'note': 'Source annotation digests match preparation/extraction/download receipts; full train.json records are not rescanned.'}
    except (ValueError, KeyError, TypeError, OSError) as error:
        report.update(status='failed', passed=False, error=str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--extraction', required=True, type=Path, help='Completed extraction.json path')
    parser.add_argument('--raw', type=Path, help='Original downloaded HPDv3 root')
    parser.add_argument('--preparation-log', type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--out', type=Path, help='Defaults to pairs.parent/validation-report.json')
    args = parser.parse_args()
    report = audit(args.pairs, extraction=args.extraction, raw=args.raw,
                   preparation_log=args.preparation_log, job_id=args.job_id)
    from self_correct_dpo.data.preparation import atomic_json
    out = args.out or args.pairs.parent / 'validation-report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, report)
    print(json.dumps({'report': str(out.resolve()), 'status': report['status'], 'passed': report['passed']}))
    return 0 if report['passed'] else 2 if report['status'] == 'in_progress' else 1


if __name__ == '__main__':
    raise SystemExit(main())
