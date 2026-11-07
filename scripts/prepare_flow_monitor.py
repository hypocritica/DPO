"""Freeze one shared heldout prompt panel for both flow backbones."""
import argparse
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from self_correct_dpo.common import sha256,write_json,write_jsonl
from self_correct_dpo.data.pickapic import prompt_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--count',type=int,default=128)
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    train=set();validation={}
    with a.pairs.open() as handle:
        for line in handle:
            row=json.loads(line)
            prompt=next(v['prompt'] for v in row['prompts'] if v['variant_id']=='canonical')
            key=prompt_key(prompt)
            if row['split']=='train':train.add(key)
            elif row['split']=='validation':validation.setdefault(key,prompt)
    if train.intersection(validation):raise ValueError('Validation/training caption leakage')
    keys=sorted(validation)
    if a.count<1 or len(keys)<a.count:raise ValueError('Insufficient distinct heldout captions')
    random.Random(a.seed).shuffle(keys)
    rows=[{'scene_id':f'hpdv3-flow-{i:04d}','variant_id':'canonical','tag':'hpdv3-validation',
           'prompt':validation[key], 'metadata':{'prompt':validation[key], 'tag':'hpdv3-validation'},
           'suite_mode':'canonical-only-v1', 'template_family':'canonical',
           'transform_type':'canonical'} for i,key in enumerate(keys[:a.count])]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_jsonl(a.out,rows)
    write_json(a.out.with_suffix('.audit.json'),{'pairs_sha256':sha256(a.pairs),'prompts_sha256':sha256(a.out),
        'count':a.count,'seed':a.seed,'split':'validation','normalized_prompt_train_overlap':0})


if __name__=='__main__':main()
