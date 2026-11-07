"""Schema and helpers for reproducible research reports.

Every naturalness, emotion, and accent run writes a ``voix.research_report.v1``
document.  The schema is intentionally strict about provenance and about
missing metrics: a metric that was not measured must say ``unavailable`` with a
reason instead of being omitted or replaced by a placeholder.

This module does not run experiments and does not load models.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from voix.utils.paths import project_path

RESEARCH_REPORT_SCHEMA = "voix.research_report.v1"
TRACKS = ("naturalness", "emotion", "accent", "shared")
LEVELS = ("exploratory", "confirmatory", "deployment")
STATUSES = ("completed", "partial", "failed", "blocked")
METRIC_STATUSES = ("computed", "partial", "unavailable")
MOS_STATUSES = ("computed", "partial", "unavailable")

_SHA256_CHARS = set("0123456789abcdef")


def sha256_file(path: Union[str, Path]) -> str:
    """Return the lowercase SHA256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unavailable_metric(reason: str) -> Dict[str, Any]:
    """Build an explicit unavailable-metric entry."""
    if not str(reason).strip():
        raise ValueError("an unavailable metric requires a non-empty reason")
    return {"status": "unavailable", "reason": str(reason).strip()}


def computed_metric(value: Any, **extra: Any) -> Dict[str, Any]:
    """Build a computed-metric entry while preserving extra measured fields."""
    entry: Dict[str, Any] = {"status": "computed", "value": value}
    entry.update(extra)
    return entry


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return set(value.lower()) <= _SHA256_CHARS


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_metric_entries(entries: Any, *, field: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(entries, dict):
        return [f"{field} must be an object"]
    for name, entry in entries.items():
        prefix = f"{field}.{name}"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        status = entry.get("status")
        if status not in METRIC_STATUSES:
            errors.append(
                f"{prefix}.status must be one of {METRIC_STATUSES}, got {status!r}"
            )
            continue
        if status == "unavailable" and not _non_empty_string(entry.get("reason")):
            errors.append(f"{prefix}.reason is required when status is unavailable")
    return errors


def validate_report(report: Mapping[str, Any]) -> List[str]:
    """Return a list of schema errors; an empty list means valid."""
    if not isinstance(report, Mapping):
        return ["report must be an object"]
    errors: List[str] = []

    if report.get("schema_version") != RESEARCH_REPORT_SCHEMA:
        errors.append(
            f"schema_version must be {RESEARCH_REPORT_SCHEMA!r}, "
            f"got {report.get('schema_version')!r}"
        )
    track = report.get("track")
    if track not in TRACKS:
        errors.append(f"track must be one of {TRACKS}, got {track!r}")
    level = report.get("level")
    if level not in LEVELS:
        errors.append(f"level must be one of {LEVELS}, got {level!r}")
    status = report.get("status")
    if status not in STATUSES:
        errors.append(f"status must be one of {STATUSES}, got {status!r}")
    if not _non_empty_string(report.get("run_id")):
        errors.append("run_id must be a non-empty string")
    if not _non_empty_string(report.get("git_commit")):
        errors.append("git_commit must be a non-empty string (use 'unknown' if unavailable)")

    for key in ("config_sha256", "manifest_sha256"):
        value = report.get(key)
        if value == "not_applicable":
            continue
        if not _is_sha256(value):
            errors.append(
                f"{key} must be a 64-character lowercase SHA256 or 'not_applicable'"
            )

    seeds = report.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("seeds must be a non-empty list")
    elif not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        errors.append("seeds must contain only integers")

    errors.extend(_validate_metric_entries(report.get("metrics"), field="metrics"))

    if not isinstance(report.get("gates"), dict):
        errors.append("gates must be an object")

    limitations = report.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        errors.append("limitations must be a list of strings")

    mos = report.get("mos")
    if not isinstance(mos, dict):
        errors.append("mos must be an object")
    else:
        mos_status = mos.get("status")
        if mos_status not in MOS_STATUSES:
            errors.append(f"mos.status must be one of {MOS_STATUSES}, got {mos_status!r}")
        elif mos_status == "unavailable":
            if not _non_empty_string(mos.get("reason")):
                errors.append("mos.reason is required when mos.status is unavailable")
        else:
            n_ratings = mos.get("n_ratings")
            mean = mos.get("mean")
            if not isinstance(n_ratings, int) or isinstance(n_ratings, bool) or n_ratings <= 0:
                errors.append("mos.n_ratings must be a positive integer when MOS is computed")
            try:
                numeric_mean = float(mean)
            except (TypeError, ValueError):
                errors.append("mos.mean must be numeric when MOS is computed")
            else:
                if not math.isfinite(numeric_mean):
                    errors.append("mos.mean must be finite when MOS is computed")

    if "per_class" in report and not isinstance(report["per_class"], dict):
        errors.append("per_class must be an object when present")
    if "artifacts" in report and not isinstance(report["artifacts"], dict):
        errors.append("artifacts must be an object when present")
    return errors


def _git_commit(root: Optional[Path] = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or project_path()),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def _hash_or_not_applicable(
    path: Optional[Union[str, Path]],
    provided: Optional[str],
    *,
    name: str,
) -> str:
    if provided is not None:
        if provided != "not_applicable" and not _is_sha256(provided):
            raise ValueError(f"{name} must be a SHA256 or 'not_applicable'")
        return provided
    if path is None:
        return "not_applicable"
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{name} source does not exist: {resolved}")
    return sha256_file(resolved)


def build_report(
    *,
    track: str,
    level: str,
    run_id: str,
    metrics: Mapping[str, Any],
    gates: Optional[Mapping[str, Any]] = None,
    status: str = "completed",
    limitations: Iterable[str] = (),
    mos: Optional[Mapping[str, Any]] = None,
    seeds: Sequence[int] = (0,),
    config_path: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    config_sha256: Optional[str] = None,
    manifest_sha256: Optional[str] = None,
    git_commit: Optional[str] = None,
    per_class: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and validate a research report.

    ``config_sha256``/``manifest_sha256`` take precedence over the paths.  If
    neither is supplied, the field is ``not_applicable`` rather than a fake
    hash.  The default MOS entry is explicitly unavailable.
    """
    report: Dict[str, Any] = {
        "schema_version": RESEARCH_REPORT_SCHEMA,
        "track": track,
        "level": level,
        "run_id": run_id,
        "git_commit": git_commit if git_commit is not None else _git_commit(),
        "config_sha256": _hash_or_not_applicable(
            config_path, config_sha256, name="config_sha256"
        ),
        "manifest_sha256": _hash_or_not_applicable(
            manifest_path, manifest_sha256, name="manifest_sha256"
        ),
        "seeds": [int(seed) for seed in seeds],
        "metrics": dict(metrics),
        "gates": dict(gates or {}),
        "status": status,
        "limitations": [str(item) for item in limitations],
        "mos": dict(
            mos
            if mos is not None
            else {
                "status": "unavailable",
                "reason": "no human ratings were collected for this run",
            }
        ),
    }
    if per_class is not None:
        report["per_class"] = dict(per_class)
    if artifacts is not None:
        report["artifacts"] = dict(artifacts)
    if extra is not None:
        overlap = set(extra) & set(report)
        if overlap:
            raise ValueError(f"extra fields would overwrite report keys: {sorted(overlap)}")
        report.update(dict(extra))

    errors = validate_report(report)
    if errors:
        raise ValueError("invalid research report: " + "; ".join(errors))
    return report


def write_report(
    path: Union[str, Path],
    report: Mapping[str, Any],
    *,
    validate: bool = True,
) -> Path:
    """Write a report as UTF-8 JSON, validating it first by default."""
    if validate:
        errors = validate_report(report)
        if errors:
            raise ValueError("invalid research report: " + "; ".join(errors))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def load_report(
    path: Union[str, Path],
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """Read a research report and validate it by default."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if validate:
        errors = validate_report(payload)
        if errors:
            raise ValueError("invalid research report: " + "; ".join(errors))
    return payload


def report_path(track: str, run_id: str) -> Path:
    """Canonical location for a run report: ``results/<track>/<run_id>/report.json``."""
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    return project_path("results", track, run_id, "report.json")
