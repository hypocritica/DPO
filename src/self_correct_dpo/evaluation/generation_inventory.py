"""Read-only native evaluation metadata export; never packages model weights/images."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from datetime import datetime, timezone

NAMES = {'plan.json','batch.json','selection.json','training.json',
         'checkpoint-identity.json','render-config.json','run.json','manifest.jsonl',
         'identity.json','scores.jsonl','summary.json','complete.json','scorers.resolved.json',
         'partiprompts.jsonl','full.jsonl','subset.jsonl','curve.jsonl'}
PRUNE = {'images','samples','checkpoints','.git','logs','sessions','__pycache__','wandb','tensorboard'}


def discover(roots, exclude=None):
    """Find evaluation render headers even when no evaluate_suite plan exists."""
    files, issues, seen = [], [], set()
    for raw in roots:
        root=Path(raw).expanduser().resolve()
        if not root.is_dir():
            issues.append({'path':str(root),'status':'missing_root'});continue
        def onerror(error):
            issues.append({'path':error.filename,'status':'unreadable','error':str(error)})
        for directory, dirs, names in os.walk(root,followlinks=False,onerror=onerror):
            parent=Path(directory)
            dirs[:]=sorted(d for d in dirs if d not in PRUNE and not d.startswith('.')
                           and (exclude is None or not (parent/d).resolve().is_relative_to(exclude)))
            for name in sorted(names):
                path=parent/name
                if name not in NAMES or path.is_symlink() or path in seen:continue
                if name=='complete.json' and not any((parent/f).exists() for f in ('run.json','identity.json')):continue
                seen.add(path); files.append(path)
    return files,issues


def export_inventory(roots, out, max_file_mb=128, check_images=False):
    out=Path(out).expanduser().resolve()
    if out.exists():raise ValueError('Use a new inventory output directory (snapshot is immutable)')
    roots=[Path(r).expanduser().resolve() for r in roots]
    if any(r==out or r.is_relative_to(out) for r in roots):raise ValueError('Output cannot contain a source root')
    files,issues=discover(roots,exclude=out)
    out.mkdir(parents=True)
    index={'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
           'roots':[str(r) for r in roots],'files':[],'renders':[],'issues':issues,
           'note':'Read-only metadata snapshot; no model weights or images. Reuse must be verified against live files.'}
    with tarfile.open(out/'evaluation-metadata.tar.gz','w:gz') as archive:
        for path in files:
            try:
                if path.stat().st_size>max_file_mb*1024**2:
                    issues.append({'path':str(path),'status':'too_large','limit_mb':max_file_mb});continue
                before=path.stat();data=path.read_bytes();after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    issues.append({'path':str(path),'status':'changed_during_snapshot'});continue
                i=next(i for i,r in enumerate(roots) if path.is_relative_to(r))
                name=f'root-{i}/{path.relative_to(roots[i]).as_posix()}'
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
                archive.addfile(info,io.BytesIO(data))
                index['files'].append(dict(path=str(path),archive=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                if path.name=='run.json':
                    run=json.loads(data)
                    if not isinstance(run,dict) or 'sampling' not in run or 'suite' not in run:continue
                    record=dict(path=str(path),family=run.get('model',{}).get('family'),
                                adapter=run.get('config',{}).get('aligned'),sampling=run['sampling'],
                                prompts=len(run['suite']),expected_images=run.get('expected_images'),gpu=run.get('gpu'),
                                versions=run.get('versions'),complete=(path.parent/'complete.json').is_file(),
                                score_metrics=sorted(p.parent.name for p in (path.parent.parent/'scores').glob('*/scores.jsonl')))
                    if check_images and (path.parent/'manifest.jsonl').is_file():
                        rows=[json.loads(line) for line in (path.parent/'manifest.jsonl').read_text().splitlines() if line.strip()]
                        present=sum((path.parent/r['image']).is_file() for r in rows)
                        record.update(indexed_images=len(rows),present_images=present,missing_images=len(rows)-present)
                    index['renders'].append(record)
            except (OSError,ValueError,KeyError,TypeError) as error:
                issues.append({'path':str(path),'status':'unreadable_or_invalid','error':str(error)})
        payload=json.dumps(index,ensure_ascii=False,indent=2).encode()
        info=tarfile.TarInfo('inventory.json');info.size=len(payload);info.mode=0o600
        archive.addfile(info,io.BytesIO(payload))
    (out/'inventory.json').write_bytes(payload)
    (out/'README.txt').write_text('Send evaluation-metadata.tar.gz (contains inventory.json). No images or weights included.\n'
        f"Captured {len(index['renders'])} render headers; {len(issues)} issues.\n")
    return {'archive':str(out/'evaluation-metadata.tar.gz'),'renders':len(index['renders']),'files':len(index['files']),'issues':len(issues)}
