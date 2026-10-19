"""Resolve explicitly supplied paths without machine-specific defaults."""
import os
import re
from pathlib import Path
from self_correct_dpo.common import read_json


def load_config(path, *, scorers=None, benchmarks=None):
    path = Path(path).expanduser().resolve()
    config = read_json(path)
    def resolve(value, label):
        value = os.path.expanduser(os.path.expandvars(value))
        if re.search(r"\$\{|\$[A-Za-z_]", value):
            raise ValueError(f"Set the environment variable used by {label}: {value}")
        target=Path(value)
        return str((target if target.is_absolute() else path.parent/target).resolve())
    fields = (("model", "path"), ("data", "pairs"), ("data", "diagnostic_suite"),
              ("aligned", "path"))
    for section, key in fields:
        value = config.get(section, {}).get(key)
        if value is None:
            continue
        config[section][key] = resolve(value, f'{section}.{key}')
    if config.get('model',{}).get('vae'):
        config['model']['vae']['path']=resolve(config['model']['vae']['path'],'model.vae.path')
    if config.get('evaluation',{}).get('enabled'):
        for key in ('prompts','scorers_config'):
            config['evaluation'][key]=resolve(config['evaluation'][key],f'evaluation.{key}')
    if config.get('scorers_config'):
        config['scorers_config']=resolve(config['scorers_config'],'scorers_config')
    if 'benchmarks' in config:
        selected=benchmarks if benchmarks is not None else config['benchmarks'].keys()
        config['benchmarks']={key:resolve(config['benchmarks'][key],f'benchmarks.{key}') for key in selected}
    if 'scorers' in config:
        selected = scorers if scorers is not None else config['scorers'].keys()
        config['scorers']={name:{key:resolve(value,f'scorers.{name}.{key}')
            if key in ('model','processor','checkpoint','open_clip_source','source','config','tokenizer','python') else value
            for key,value in config['scorers'][name].items()} for name in selected}
    return config
