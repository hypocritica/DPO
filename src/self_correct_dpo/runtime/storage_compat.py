"""Exact source-version compatibility for the AOSS image-publication fix only."""
from pathlib import Path

from self_correct_dpo.common import digest, read_json, sha256
from self_correct_dpo.runtime.storage_compat_hashes import SOURCE_HASHES


def compatible_sources(recorded, current):
    if not isinstance(recorded, dict) or not isinstance(current, dict) or recorded.keys() != current.keys():
        return False
    for name, previous in recorded.items():
        observed = current[name]
        if previous == observed:
            continue
        approved = SOURCE_HASHES.get('evaluation.py' if name == 'runner.py' else name)
        if approved is None or (previous, observed) != (approved['original'], approved['fixed']):
            return False
    return True


def compatible_identity(recorded, current, *, code_key='code_sha256'):
    """All non-code identity fields must still match exactly."""
    return (isinstance(recorded, dict) and isinstance(current, dict)
            and code_key in recorded and code_key in current
            and {k: v for k, v in recorded.items() if k != code_key}
            == {k: v for k, v in current.items() if k != code_key}
            and compatible_sources(recorded[code_key], current[code_key]))


def record_compatibility(root, recorded, current):
    """Keep the original identity intact, and audit the exact files used now."""
    if recorded == current:
        return
    if not compatible_sources(recorded, current):
        raise ValueError('Source changes exceed the approved AOSS storage fix')
    from self_correct_dpo.runtime.checkpoints import atomic_json
    from self_correct_dpo.runtime import storage_compat_hashes
    report = {'fix': 'aoss-image-publication-v1', 'original': recorded, 'current': current,
              'policy_sha256': sha256(__file__),
              'approved_hashes_sha256': sha256(storage_compat_hashes.__file__)}
    path = Path(root) / ('storage-compatibility-' + digest(report)[:16] + '.json')
    if path.exists():
        if read_json(path) != report:
            raise ValueError('Storage compatibility audit record changed')
    else:
        atomic_json(path, report)
        print(f'Accepted reviewed storage-only source fix; original identity retained; audit: {path}', flush=True)
