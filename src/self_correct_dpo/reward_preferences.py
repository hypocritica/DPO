"""Rank complete scored candidate pools; validate scalar AI preference evidence."""
from collections import defaultdict
from functools import lru_cache
import math
from pathlib import Path

from self_correct_dpo.common import digest, read_json, read_jsonl, sha256, verify_images
from self_correct_dpo.data import semantic_scene_key

SCHEMA = "canonical-scalar-preference-v1"


def select_extremes(candidates, expected_count=8, min_gap=0.0):
    if expected_count < 2 or not math.isfinite(min_gap) or min_gap < 0:
        raise ValueError("Invalid candidate count or score gap")
    if len(candidates) != expected_count:
        raise ValueError("Candidate count differs from the frozen protocol")
    if len({r['seed'] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate seed")
    if len({r['image_sha256'] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate image")
    if any(type(r['score']) not in (int, float) or not math.isfinite(r['score']) for r in candidates):
        raise ValueError("Non-finite or non-numeric reward")
    ordered = sorted(candidates, key=lambda r: (r['score'], r['seed'], r['image_sha256']))
    loser, winner = ordered[0], ordered[-1]
    if winner['score'] - loser['score'] <= min_gap:
        return None
    return {'winner': winner, 'loser': loser}


def collect_scored_candidates(runs, score_dirs, kind):
    if kind not in {'pickscore', 'geneval'} or len(runs) != len(score_dirs) or not runs:
        raise ValueError("Specify matching run/score paths and a supported reward")
    candidates, sources, scorers, bases, assignments = [], [], set(), set(), {}
    for run_path, score_path in zip(runs, score_dirs):
        root, scored = Path(run_path).resolve(), Path(score_path).resolve()
        run, manifest = verify_images(root)
        if run['arm'] != 'base' or run['model']['family'] != 'sdxl':
            raise ValueError("Controlled candidates require frozen SDXL base renders")
        bases.add(digest(run['base_identity']))
        report_path = scored / ('quality.json' if kind == 'pickscore' else 'evaluation.json')
        report = read_json(report_path)
        score_file = scored / 'scores.jsonl'
        if report['run_id'] != run['run_id'] or report['manifest_sha256'] != sha256(root / 'manifest.jsonl'):
            raise ValueError("Scores belong to another render")
        if report['scores_sha256'] != sha256(score_file) or report['evaluator_id'] != digest(report['evaluator']):
            raise ValueError("Score artifact or evaluator identity changed")
        expected_kind = 'pickscore' if kind == 'pickscore' else 'geneval-canonical-metadata-v1'
        if report['evaluator']['kind'] != expected_kind:
            raise ValueError("Wrong evaluator kind")
        if kind == 'pickscore' and report.get('interpretation') != 'raw canonical-caption cosine or scaled PickScore; no batch-dependent softmax':
            raise ValueError("PickScore must use a common canonical caption")
        scorers.add(report['evaluator_id'])
        values = read_jsonl(score_file)
        by_image = {v['image']: v for v in values}
        if len(values) != len(manifest) or len(by_image) != len(values) or set(by_image) != {v['image'] for v in manifest}:
            raise ValueError("Incomplete or duplicate scores")
        suite = {(v['scene_id'], v['variant_id']): v for v in run['suite']}
        sources.append({'run': str(root), 'scores': str(scored), 'run_id': run['run_id'],
                        'manifest_sha256': report['manifest_sha256'], 'scores_sha256': sha256(score_file),
                        'report_sha256': sha256(report_path), 'evaluator_id': report['evaluator_id']})
        for image in manifest:
            scored_image = by_image[image['image']]
            if any(scored_image.get(k) != v for k, v in image.items()):
                raise ValueError("Score image metadata differs from render manifest")
            row = suite[(image['scene_id'], image['variant_id'])]
            if row['variant_id'] != 'canonical' or row.get('split') not in {'train', 'validation'}:
                raise ValueError("Use independent canonical train/validation candidates")
            value = scored_image['score'] if kind == 'pickscore' else scored_image['correct']
            if kind == 'geneval':
                if type(value) is not bool:
                    raise ValueError("GenEval reward must be verified strict correctness")
                value = float(value)
            path = root / image['image']
            candidate = {'scene_id': row['scene_id'], 'metadata': row['metadata'], 'split': row['split'],
                         'seed': image['seed'], 'image': str(path), 'image_sha256': sha256(path),
                         'score': value, 'scoring_caption': row['metadata']['prompt']}
            for key in [('scene', candidate['scene_id']), ('semantics', semantic_scene_key(candidate)),
                        ('image', candidate['image_sha256'])]:
                if assignments.setdefault(key, row['split']) != row['split']:
                    raise ValueError("Cross-split candidate leakage")
            candidates.append(candidate)
    if len(bases) != 1 or len(scorers) != 1:
        raise ValueError("Candidate sources must share one base and one evaluator")
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row['scene_id']].append(row)
    for rows in grouped.values():
        if len({digest(r['metadata']) for r in rows}) != 1 or len({r['split'] for r in rows}) != 1:
            raise ValueError("Scene metadata or split mismatch")
    return grouped, sources



@lru_cache(maxsize=4)
def _verified_pools(fingerprints, kind):
    # One complete image audit per immutable source set in this process. Each
    # caller hashes the small source manifests/reports again before cache lookup.
    return collect_scored_candidates([f[0] for f in fingerprints], [f[1] for f in fingerprints], kind)


def source_fingerprints(sources, kind):
    if kind not in {'pickscore', 'geneval'}:
        raise ValueError("Unsupported reward kind")
    fingerprints = []
    for source in sources:
        root, scored = Path(source['run']), Path(source['scores'])
        report = scored / ('quality.json' if kind == 'pickscore' else 'evaluation.json')
        fingerprints.append((str(root), str(scored), sha256(report), sha256(scored / 'scores.jsonl'),
                             sha256(root / 'manifest.jsonl'), sha256(root / 'run.json'), sha256(root / 'complete.json')))
    return tuple(fingerprints)


def validate_reward_pair(row):
    ann = row.get('annotation', {})
    if row.get('reviewed') is not False or ann.get('kind') != SCHEMA or ann.get('passed') is not True:
        raise ValueError("Scalar preferences require explicit AI provenance")
    path = Path(ann.get('evidence_path', ''))
    if not path.is_file() or sha256(path) != ann.get('evidence_sha256'):
        raise ValueError("Scalar preference evidence missing or changed")
    ev = read_json(path)
    if ev.get('schema') != SCHEMA or ev.get('scene_id') != row['scene_id'] or ev.get('metadata_id') != digest(row['metadata']):
        raise ValueError("Scalar evidence belongs to another scene")
    if ev.get('split') != row['split'] or ev.get('scoring_caption') != row['metadata']['prompt']:
        raise ValueError("Split or scoring caption changed")
    sources = ev['sources']
    groups, actual_sources = _verified_pools(source_fingerprints(sources, ev['reward_kind']), ev['reward_kind'])
    if actual_sources != sources or ann.get('evaluator_id') != actual_sources[0]['evaluator_id']:
        raise ValueError("Scalar reward source artifacts changed")
    candidates = groups[row['scene_id']]
    if ev['candidates'] != candidates:
        raise ValueError("Candidate reward pool changed")
    selection = ev['selection']
    if selection.get('rule') != 'best-worst-fixed-k':
        raise ValueError("Unsupported preference selection rule")
    chosen = select_extremes(candidates, selection['candidate_count'], selection['min_gap'])
    if chosen is None:
        raise ValueError("Tied rewards cannot establish a preference")
    for side in ('winner', 'loser'):
        if chosen[side] != ev[side] or chosen[side]['image_sha256'] != row.get(side + '_sha256'):
            raise ValueError("Reward ordering does not establish the claimed label")
        if sha256(row[side]) != chosen[side]['image_sha256']:
            raise ValueError('Selected reward image changed')
        if str(Path(row[side]).resolve()) != chosen[side]['image']:
            raise ValueError("Reward image path mismatch")
    return True
