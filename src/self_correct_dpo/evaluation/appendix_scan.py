"""Small, dependency-free identities for the checkpoint/time diagnostic scan."""
from self_correct_dpo.common import digest


def validate_conditions(conditions):
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("a nonempty frozen condition list is required")
    names = set()
    for row in conditions:
        name, start, stop = row["condition"], row["index_start"], row["index_stop"]
        if (not isinstance(name, str) or not name or not all(c.isalnum() or c in "_-" for c in name)
                or name in names or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= 1000):
            raise ValueError("invalid/duplicate condition or scheduler-index interval")
        mode = row["mode"]
        if mode not in ("fixed_index", "stage_interval") or (mode == "fixed_index" and stop != start + 1):
            raise ValueError("invalid condition mode")
        expected_probability = 1.0 if mode == "fixed_index" else (stop - start) / 1000
        if row.get("probability") != expected_probability:
            raise ValueError("stage interval must record its schedule mass")
        names.add(name)
    return conditions


def fixed_conditions(family, indices):
    return validate_conditions([dict(condition=f"{family}_index{i}", mode="fixed_index",
                                    index_start=i, index_stop=i + 1, probability=1.0) for i in indices])


def scan_seed(scope, role, *parts):
    return int(digest(["mvpo-scan-v1", scope, role, *parts])[:16], 16) % (2 ** 63)


def validate_job(job, family):
    if (job.get("schema_version") != 1 or job.get("family") != family
            or job.get("research_phase") not in ("exploration", "confirmation")
            or job.get("job_id") != digest({k: v for k, v in job.items() if k != "job_id"})):
        raise ValueError("invalid frozen scan job identity")
    validate_conditions(job["conditions"])
    if type(job.get("skip_numerical_checks", False)) is not bool:
        raise ValueError("skip_numerical_checks must be boolean")
    if job.get("skip_numerical_checks") and job["research_phase"] != "exploration":
        raise ValueError("unchecked measurements are exploration only")
    if job["view_batch_size"] not in (1, 2, 4):
        raise ValueError("scan view batch must be 1, 2 or 4")
    if job["research_phase"] == "confirmation" and not job.get("excluded_pair_ids"):
        raise ValueError("confirmation requires an explicit exploration pair exclusion panel")
    return job
