"""Export this standalone module for the private DPO repository; no Git/network mutations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'.gitignore','README.md','pyproject.toml','paths.sh','train.py','render.py','score.py','evaluate.py'}
DIRECTORIES={'src','scripts','launchers','configs','docs','tests'}
SUFFIXES={'.py','.sh','.md','.json','.txt','.toml'}


def package(out,root=ROOT):
    out,root=Path(out).resolve(),Path(root).resolve()
    if out.exists():raise FileExistsError(out)
    files=[]
    candidates=[p for p in root.iterdir() if p.is_file()]
    for name in sorted(DIRECTORIES):
        if (root/name).is_dir():candidates.extend((root/name).rglob('*'))
    for path in sorted(candidates):
        relative=path.relative_to(root)
        if path.is_symlink() or not path.is_file():continue
        if any(part.startswith('.') or part=='__pycache__' or part.endswith('.egg-info') for part in relative.parts):
            if str(relative)!='.gitignore':continue
        if len(relative.parts)==1:
            keep=relative.name in ROOT_FILES or (relative.name.startswith('requirements') and relative.suffix=='.txt')
        else:
            keep=relative.parts[0] in DIRECTORIES and path.suffix in SUFFIXES
            if relative.parts[0]=='configs' and relative.name.startswith('local'):keep=False
        if not keep:continue
        if path.stat().st_size>5*1024**2:raise ValueError(f'Unexpected large code file: {relative}')
        files.append((path,relative))
    required={'train.py','pyproject.toml','paths.sh','src/self_correct_dpo/training.py'}
    if not required.issubset({str(r) for _,r in files}):raise ValueError('Standalone module is incomplete')
    manifest={str(relative):sha256(path.read_bytes()).hexdigest() for path,relative in files}
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path,relative in files:archive.write(path,'DPO/'+relative.as_posix())
        archive.writestr('DPO/CODE_MANIFEST.json',json.dumps({'sha256':manifest,'note':'Code only; data/models/envs/outputs excluded.'},indent=2)+'\n')
    return {'archive':str(out),'files':len(files),'bytes':out.stat().st_size,'sha256':sha256(out.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(package(a.out),indent=2))


if __name__=='__main__':main()
