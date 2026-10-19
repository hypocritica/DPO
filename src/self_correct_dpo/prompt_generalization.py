"""Frozen semantic prompt suites and scene-clustered, base-relative gains."""
from collections import defaultdict
import math
import random
import statistics
from self_correct_dpo.common import digest
from self_correct_dpo.pickapic import prompt_key


def freeze_suite(scenes, pairs):
    train = [r for r in pairs if r['split'] == 'train']
    training_texts = {prompt_key(p['prompt']) for r in train for p in r['prompts']}
    training_groups = {r['scene_id'] for r in train}
    by_id = {r['pair_id']: r for r in train}
    prompts, seen, cohorts = [], set(), set()
    for scene in scenes:
        sid = scene['scene_id']
        if sid in seen: raise ValueError('Duplicate diagnostic scene')
        seen.add(sid)
        cohort = scene['cohort']; cohorts.add(cohort)
        if cohort not in {'train', 'unseen'}: raise ValueError('Unknown diagnostic cohort')
        rewrites = scene['rewrites']
        if scene.get('semantics_reviewed') is not True or not scene.get('semantic_contract'):
            raise ValueError('Record visual semantics and review each rewrite before freezing')
        if len(rewrites) not in (2, 3): raise ValueError('Use two or three semantic rewrites')
        texts = [scene['original'], *rewrites]
        if any(not isinstance(x, str) or not x.strip() for x in texts) or len({prompt_key(x) for x in texts}) != len(texts):
            raise ValueError('Empty or duplicate diagnostic wording')
        if any(prompt_key(x) in training_texts for x in rewrites):
            raise ValueError('Evaluation rewrite appeared in training')
        if cohort == 'train':
            source = by_id.get(scene.get('source_pair_id'))
            if source is None or scene['original'] != source['prompts'][0]['prompt'] or sid != source['scene_id']:
                raise ValueError('Training scene must use the exact source original and semantic family ID')
        else:
            if scene.get('independence_reviewed') is not True:
                raise ValueError('Audit unseen visual scenes against training semantic groups before freezing')
            if sid in training_groups or prompt_key(scene['original']) in training_texts:
                raise ValueError('Unseen scene overlaps training')
        metadata = {'prompt': scene['original'], 'tag': cohort, 'semantic_contract': scene['semantic_contract']}
        for index, text in enumerate(texts):
            prompts.append({'scene_id': sid, 'variant_id': 'canonical' if index == 0 else f'rewrite-{index}',
                'prompt': text, 'tag': cohort, 'metadata': metadata, 'template_family': 'canonical' if index == 0 else 'rewrite',
                'transform_type': 'canonical' if index == 0 else 'semantic_rewrite'})
    if cohorts != {'train', 'unseen'}: raise ValueError('Diagnostic suite needs train and unseen scenes')
    return prompts


def gain_summary(suite, base, aligned, *, seed=17, bootstrap=2000):
    """Average seeds per expression, then rewrites per scene; bootstrap scenes jointly.

    score rows keyed by (scene_id, variant_id, seed). Aligned-base uses the same
    seed and evaluator. A single scene has no population uncertainty estimate.
    """
    def index(rows):
        result = {}
        for row in rows:
            key = (row['scene_id'], row['variant_id'], row['seed'])
            if key in result or not math.isfinite(row['score']): raise ValueError('Duplicate/nonfinite score')
            result[key] = float(row['score'])
        return result
    before, after = index(base), index(aligned)
    if set(before) != set(after): raise ValueError('Incomplete paired score coverage')
    expected = {(r['scene_id'], r['variant_id']) for r in suite}
    if {(s,v) for s,v,_ in before} != expected: raise ValueError('Score/suite coverage mismatch')
    seed_sets = [{seed for s,v,seed in before if (s,v) == key} for key in expected]
    if any(seeds != seed_sets[0] for seeds in seed_sets) or len(seed_sets[0]) < 2:
        raise ValueError('Each expression needs the same two or more fixed seeds')
    deltas = defaultdict(list)
    for (s,v,seed_), value in before.items(): deltas[s,v].append(after[s,v,seed_] - value)
    expressions = {key: statistics.mean(values) for key,values in deltas.items()}
    scenes = defaultdict(dict)
    tags = {r['scene_id']: r['tag'] for r in suite}
    for (sid, vid), value in expressions.items(): scenes[sid][vid] = value
    per_scene = []
    for sid, values in sorted(scenes.items()):
        original = values['canonical']
        rewritten = statistics.mean(v for k,v in values.items() if k != 'canonical')
        per_scene.append({'scene_id': sid, 'cohort': tags[sid], 'original_gain': original,
                          'rewrite_gain': rewritten, 'original_minus_rewrite_gain': original-rewritten,
                          'expression_gains': values})
    summary = {}
    for cohort in ('train', 'unseen'):
        rows = [r for r in per_scene if r['cohort'] == cohort]
        if not rows: raise ValueError('Missing cohort')
        fields = ('original_gain','rewrite_gain','original_minus_rewrite_gain')
        estimates = {field: statistics.mean(r[field] for r in rows) for field in fields}
        draws = {field: [] for field in fields}
        rng = random.Random(digest([seed, cohort]))
        if len(rows) > 1:
            for _ in range(bootstrap):
                sample = rng.choices(rows, k=len(rows))
                for field in fields: draws[field].append(statistics.mean(r[field] for r in sample))
        summary[cohort] = {'scenes': len(rows), 'estimates': estimates,
            'ci95': {field: [sorted(values)[int(.025*len(values))], sorted(values)[min(len(values)-1,int(.975*len(values)))]]
                      if values else None for field,values in draws.items()}}
    return {'per_scene': per_scene, 'summary': summary,
        'uncertainty': 'Paired scene bootstrap conditional on fixed seeds and audited rewrites; small diagnostic sets have weak population precision.',
        'interpretation': 'All gains are checkpoint minus base for the SAME expression; automatic metric alone is not a final generation-quality conclusion.'}
