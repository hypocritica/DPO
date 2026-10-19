"""Select a tiny train/validation fixture from REAL prepared data, without relabeling."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_correct_dpo.common import sha256, write_json, write_jsonl
from self_correct_dpo.data import validate_pair_rows, validate_split_disjointness


def make_smoke_data(manifest,out,count=4):
    if count<1:raise ValueError('count must be positive')
    manifest,out=Path(manifest).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    chosen={'train':[],'validation':[]}
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line);split=row.get('split')
            if split in chosen and len(chosen[split])<count:chosen[split].append(row)
            if all(len(rows)==count for rows in chosen.values()):break
    if any(len(rows)<count for rows in chosen.values()):raise ValueError('Not enough train/validation rows')
    rows=validate_pair_rows(chosen['train']+chosen['validation'],manifest.parent)
    validate_split_disjointness(rows)
    out.mkdir(parents=True)
    for row in rows:
        if row.get('storage',{}).get('kind')!='arrow':
            for side in ('winner','loser'):row[side]=os.path.relpath(row[side],out)
    write_jsonl(out/'pairs.jsonl',rows)
    write_json(out/'selection.json',{'source':str(manifest),'source_sha256':sha256(manifest),
        'pair_ids':[r['pair_id'] for r in rows], 'note':'Technical smoke only; never use its loss/score as research evidence.'})
    return out/'pairs.jsonl'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=4,help='Rows per split')
    a=p.parse_args();print(make_smoke_data(a.pairs,a.out,a.count))


if __name__=='__main__':main()
