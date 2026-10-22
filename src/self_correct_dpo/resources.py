"""One editable resource registry; environment and local overrides take priority."""
import os
from pathlib import Path
from string import Template

from self_correct_dpo.common import read_json

CONFIG_DIR = Path(__file__).resolve().parents[2] / 'configs'


def load_resources(path=None):
    registry = Path(path or os.environ.get('SCD_RESOURCES') or CONFIG_DIR / 'resources.json').expanduser().resolve()
    values = read_json(registry)
    local = registry.with_name(registry.stem + '.local.json')
    if local.is_file():
        values.update(read_json(local))
    resolved = {}

    def resolve(key, active=()):
        if key in resolved:
            return resolved[key]
        if key in active:
            raise ValueError(f'Resource reference cycle: {active + (key,)}')
        value = os.environ.get(key, values.get(key))
        if not isinstance(value, str) or not value:
            raise ValueError(f'Resource {key} must be a nonempty string')
        names = [m.group('named') or m.group('braced') for m in Template.pattern.finditer(value)
                 if m.group('named') or m.group('braced')]
        value = Template(value).substitute({name: resolve(name, active + (key,)) for name in names})
        target = Path(os.path.expanduser(value))
        resolved[key] = str((target if target.is_absolute() else registry.parent / target).resolve())
        return resolved[key]

    for key in values:
        resolve(key)
    return resolved
