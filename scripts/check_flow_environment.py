"""CPU-only flow API/version preflight; CUDA smoke remains separate."""
import argparse
import importlib.metadata as md
import json
import sys
from packaging.version import Version

LOCK = {'diffusers': '0.37.1', 'transformers': '4.57.1', 'peft': '0.17.1', 'accelerate': '1.14.0', 'huggingface-hub': '0.36.0', 'safetensors': '0.7.0', 'sentencepiece': '0.2.1', 'tokenizers': '0.22.2'}


def check_environment(family):
    versions = {'python':sys.version.split()[0]}
    errors=[]
    for name, expected in LOCK.items():
        try: versions[name]=md.version(name)
        except md.PackageNotFoundError: versions[name]='missing'
        if versions[name] != expected: errors.append(f'{name}: expected {expected}, found {versions[name]}')
    try:
        versions['torch']=md.version('torch')
        if Version(versions['torch'].split('+')[0]) < Version('2.2'):
            errors.append('Transformers4.57 requires torch>=2.2; preserve/clone the verified GPU runtime')
    except md.PackageNotFoundError: errors.append('torch missing')
    if not errors:
        import diffusers
        required = ('ZImagePipeline','ZImageTransformer2DModel') if family=='zimage' else ('StableDiffusion3Pipeline','SD3Transformer2DModel')
        for name in required:
            if not hasattr(diffusers,name): errors.append(f'diffusers.{name} unavailable')
    print(json.dumps(versions,indent=2))
    if errors: raise RuntimeError('Flow environment preflight failed:\n'+'\n'.join(errors))
    return versions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['sd35_medium','zimage'],required=True)
    a=p.parse_args();check_environment(a.family)


if __name__=='__main__':main()
