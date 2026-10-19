"""Offline human preference adapter. Original Pick-a-Pic/HPDv3 labels are preserved.

The boundary accepts JSONL rows containing paths (image_0/image_1) rather than
embedding bytes. Parquet/HF-save-to-disk exports can be mapped at this boundary.
"""
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import unicodedata

from self_correct_dpo.common import digest, sha256
from self_correct_dpo.data import validate_pair_rows


def prompt_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def adapt_rows(raw, root, groups, *, source_revision, seed=17, fractions=(.9, .05, .05), image_resolver=None, pinned_groups=None, grouping_audited=True, preference_source='pickapic'):
    """Split connected components of prompt families, image UIDs and image hashes.

    groups maps original captions to audited semantic family IDs, including known
    paraphrases. Coverage is mandatory: string normalization alone cannot prove
    semantic disjointness. Repeated unordered pairs are removed; conflicting votes
    are quarantined, never majority-voted. Ties contribute leakage edges too.
    """
    if not source_revision.strip():
        raise ValueError('Record a dataset source revision')
    if preference_source not in ('pickapic','hpdv3'):
        raise ValueError('Unsupported human preference source')
    if len(fractions) != 3 or any(not math.isfinite(x) or x <= 0 for x in fractions) or abs(sum(fractions)-1) > 1e-8:
        raise ValueError('Need three positive split fractions summing to one')
    normalized = {}
    for caption, family in groups.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError('Semantic family IDs must be nonempty strings')
        key = prompt_key(caption)
        if key in normalized and normalized[key] != family:
            raise ValueError('Equivalent captions assigned to different groups')
        normalized[key] = family
    rows, exclusions, hashes, ids = [], [], {}, set()
    root = Path(root).resolve()
    for item in raw:
        pair_id = str(item.get('pair_id', item.get('ranking_id', '')))
        if not pair_id or pair_id in ids:
            raise ValueError('Missing or duplicate sample identifier')
        ids.add(pair_id)
        prompt = item.get('prompt', item.get('caption'))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'Missing original caption: {pair_id}')
        key = prompt_key(prompt)
        if key not in normalized:
            raise ValueError(f'Caption lacks audited semantic group: {pair_id}')
        labels = [item.get('label_0'), item.get('label_1')]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in labels):
            raise ValueError(f'Missing/invalid human labels: {pair_id}')
        if tuple(labels) not in {(1,0), (0,1), (.5,.5)}:
            raise ValueError(f'Expected hard human preference or explicit tie: {pair_id}')
        paths, checksums, uids = [], [], []
        if image_resolver:
            paths, checksums, uids = image_resolver(item)
        for side in (() if image_resolver else (0, 1)):
            path = Path(item[f'image_{side}']).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if str(path) not in hashes:
                hashes[str(path)] = sha256(path)
            paths.append(str(path)); checksums.append(hashes[str(path)])
            uids.append(str(item.get(f'image_{side}_uid', checksums[-1])))
        rows.append({'pair_id': pair_id, 'prompt': prompt, 'group': normalized[key],
                     'paths': paths, 'hashes': checksums, 'uids': uids, 'labels': labels,
                     'source_split': item.get('split'), 'has_label': item.get('has_label', True), 'storage': item.get('storage')})
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def union(a, b):
        parents[find(a)] = find(b)
    owners = {}
    duplicates = defaultdict(list)
    for i, row in enumerate(rows):
        keys = [('prompt', prompt_key(row['prompt'])), ('group', row['group'])]
        keys += [('image_hash', x) for x in row['hashes']]
        keys += [('image_uid', x) for x in row['uids']]
        for key in keys:
            if key in owners: union(i, owners[key])
            else: owners[key] = i
        duplicates[tuple(sorted(row['hashes']))].append(i)
    reject = {}
    for indices in duplicates.values():
        decisions = {rows[i]['hashes'][0 if rows[i]['labels'][0] == 1 else 1]
                     for i in indices if tuple(rows[i]['labels']) != (.5,.5) and rows[i]['has_label']}
        if len(decisions) > 1:
            reject.update({i: 'conflicting_human_votes' for i in indices})
        else:
            eligible = sorted((i for i in indices if tuple(rows[i]['labels']) != (.5,.5) and rows[i]['has_label']), key=lambda i: rows[i]['pair_id'])
            reject.update({i: 'duplicate_image_pair' for i in eligible[1:]})
    components = defaultdict(list)
    for i, row in enumerate(rows): components[find(i)].append(i)
    # Hash assignment is independent of input order; ratios are approximate at group level.
    assignments = {}
    for indices in components.values():
        component_id = digest(sorted(rows[i]['pair_id'] for i in indices))
        value = random.Random(digest([seed, component_id])).random()
        split = 'train' if value < fractions[0] else 'validation' if value < sum(fractions[:2]) else 'test'
        forced = {pinned_groups[rows[i]['group']] for i in indices if pinned_groups and rows[i]['group'] in pinned_groups}
        if len(forced) > 1: raise ValueError('Connected component has conflicting pinned assignments')
        if forced: split = next(iter(forced))
        for i in indices: assignments[i] = (split, component_id)
    pairs = []
    for i, row in enumerate(rows):
        reason = reject.get(i)
        if not row['has_label']: reason = 'unlabelled'
        elif tuple(row['labels']) == (.5,.5): reason = 'tie'
        elif row['hashes'][0] == row['hashes'][1]: reason = 'identical_images'
        if reason:
            exclusions.append({'pair_id': row['pair_id'], 'reason': reason, 'labels': row['labels'],
                               'prompt': row['prompt'], 'image_0': row['paths'][0], 'image_1': row['paths'][1]})
            continue
        winner = 0 if row['labels'][0] == 1 else 1
        split, component_id = assignments[i]
        pairs.append({'pair_id': row['pair_id'], 'scene_id': row['group'], 'split': split,
                      'winner': row['paths'][winner], 'loser': row['paths'][1-winner],
                      'winner_sha256': row['hashes'][winner], 'loser_sha256': row['hashes'][1-winner],
                      'prompt_group_id': row['group'], 'split_component_id': component_id,
                      'prompts': [{'variant_id': 'canonical', 'prompt': row['prompt'],
                                   'template_family': 'canonical', 'transform_type': 'canonical'}],
                      'human_preference': {'source': preference_source, 'source_revision': source_revision,
                          'labels': row['labels'], 'winner_index': winner, 'semantic_groups_audited': grouping_audited,
                          'image_uids': row['uids'], 'image_sha256': row['hashes'], 'original_prompt': row['prompt'], 'source_split': row['source_split']}})
        if row.get('storage'):
            storage = dict(row['storage'])
            storage['winner_column'] = f'jpg_{winner}'
            storage['loser_column'] = f'jpg_{1-winner}'
            pairs[-1]['storage'] = storage
    pairs = validate_pair_rows(pairs, root, verify_files=False)
    report = {'raw_rows': len(raw), 'usable_pairs': len(pairs), 'exclusions': dict(Counter(x['reason'] for x in exclusions)),
              'splits': dict(Counter(x['split'] for x in pairs)), 'connected_components': len(components),
              'largest_component_rows': max(map(len, components.values()), default=0),
              'seed': seed, 'fractions': fractions, 'source_revision': source_revision,
              'groups_id': digest(groups), 'limitation': 'Known paraphrase isolation relies on the supplied audited family mapping; unknown semantic aliases require review.'}
    return pairs, exclusions, report


def validate_human_pair(row):
    evidence = row.get('human_preference', {})
    labels = evidence.get('labels')
    if (evidence.get('source') not in ('pickapic','hpdv3') or not evidence.get('source_revision') or
        labels not in ([1,0], [0,1]) or evidence.get('winner_index') != (0 if labels[0] == 1 else 1)):
        raise ValueError(f"Pair {row['pair_id']} lacks original, decisive Pick-a-Pic human preference evidence")
    if evidence['source']=='hpdv3':
        original=evidence.get('original_record',{})
        if (evidence.get('preferred_field')!='path1' or labels!=[1,0] or
                not isinstance(original.get('path1'),str) or not isinstance(original.get('path2'),str) or
                original.get('prompt')!=evidence.get('original_prompt') or
                evidence.get('image_uids')!=[original['path1'],original['path2']]):
            raise ValueError('HPDv3 evidence must preserve the official preferred path1 ordering and original record')
    hashes = evidence.get('image_sha256', [])
    winner = evidence['winner_index']
    if len(hashes) != 2 or row.get('winner_sha256') != hashes[winner] or row.get('loser_sha256') != hashes[1-winner]:
        raise ValueError('Human label/image ordering does not match the original pair')
    if len(row['prompts']) != 1 or row['prompts'][0]['variant_id'] != 'canonical':
        raise ValueError('Human baseline trains only on original captions')
    if row['prompts'][0]['prompt'] != evidence.get('original_prompt'):
        raise ValueError('Training caption differs from the original human preference prompt')
