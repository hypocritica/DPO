"""Strict AI-label parsing and verifiable preference evidence; no GPU imports."""
import json
from pathlib import Path

from self_correct_dpo.common import digest, read_json, sha256


FIELDS = ("objects_present", "counts_correct", "relation_correct")
SCHEMA = "qwen-position-two-questions-v1"


def parse_answer(raw):
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Malformed JSON fence")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError("Judge must return exactly the three required fields")
    if any(value[k] is not None and type(value[k]) is not bool for k in FIELDS):
        raise ValueError("Judge values must be true, false, or null")
    return value


def consensus(raw_answers):
    if len(raw_answers) != 2:
        return {"status": "invalid_count", "eligible": False}
    try:
        first, second = map(parse_answer, raw_answers)
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"status": "parse_error", "eligible": False}
    if first != second:
        return {"status": "disagreement", "eligible": False, "answers": [first, second]}
    if any(first[k] is None for k in FIELDS):
        return {"status": "uncertain", "eligible": False, "answer": first}
    if not first["objects_present"] or not first["counts_correct"]:
        return {"status": "object_or_count_failure", "eligible": False, "answer": first}
    return {"status": "winner" if first["relation_correct"] else "loser",
            "eligible": True, "answer": first}


def judge_questions(metadata):
    objects = metadata["include"]
    if metadata["tag"] != "position" or len(objects) != 2 or objects[1]["position"][1] != 0:
        raise ValueError("Only two-object position scenes are supported")
    first, second = objects[1]["class"], objects[0]["class"]
    relation = objects[1]["position"][0]
    inverse = {"left of": "right of", "right of": "left of", "above": "below", "below": "above"}[relation]
    rules = ('Judge only visible image content. Do not assume the requested scene is present. '
             'Return JSON only with exactly keys objects_present, counts_correct, relation_correct. '
             'Each value must be true, false, or null if uncertain. objects_present means both named '
             'object classes are visible. counts_correct means exactly one of each named class. '
             'relation_correct means the requested relation is visibly satisfied, based on object centers; '
             'use null for ambiguous overlap or occlusion. Ignore beauty and photographic style. ')
    return [rules + f'Request: exactly one {first}, exactly one {second}; the {first} is {relation} the {second}.',
            rules + f'Check exactly one {second} and exactly one {first}. The required spatial statement is: '
                    f'the {second} is {inverse} the {first}.']


def validate_auto_pair(row):
    ann = row.get("annotation", {})
    if ann.get("kind") == "canonical-scalar-preference-v1":
        from self_correct_dpo.data.reward_preferences import validate_reward_pair
        return validate_reward_pair(row)
    if row.get("reviewed") is not False or ann.get("kind") != SCHEMA or ann.get("passed") is not True:
        raise ValueError("AI preferences require explicit passed provenance and reviewed=false")
    evidence_path = Path(ann.get("evidence_path", ""))
    if not evidence_path.is_file() or sha256(evidence_path) != ann.get("evidence_sha256"):
        raise ValueError("Automatic preference evidence missing or modified")
    evidence = read_json(evidence_path)
    if evidence.get("scene_id") != row["scene_id"] or evidence.get("metadata_id") != digest(row["metadata"]):
        raise ValueError("Preference evidence is for another scene")
    if not evidence.get("judge_id") or evidence.get("schema") != SCHEMA:
        raise ValueError("Missing judge identity")
    for side in ("winner", "loser"):
        record = evidence[side]
        if record["image_sha256"] != row.get(side + "_sha256"):
            raise ValueError("Preference evidence image hash mismatch")
        if consensus(record["raw_answers"])["status"] != side:
            raise ValueError("Preference evidence does not establish the claimed label")
        if record.get("questions") != judge_questions(row["metadata"]):
            raise ValueError("Preference questions differ from the fixed scene specification")
    if evidence["winner"]["image_sha256"] == evidence["loser"]["image_sha256"]:
        raise ValueError("Identical winner and loser")
    return True
