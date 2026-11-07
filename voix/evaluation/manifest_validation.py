"""Generic manifest validation for research corpora.

The validator is intentionally dataset-aware but model-free.  It checks the
properties that make a research split trustworthy: required columns, non-empty
speaker and split labels, speaker disjointness, duplicate clips, class balance,
duration, and optionally audio readability.

It never repairs a manifest.  A violation is reported and the manifest is
rejected.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from voix.utils.paths import project_path

MANIFEST_VALIDATION_SCHEMA = "voix.research_manifest_validation.v1"

# kind -> required metadata.  ``audio`` is a column name; GLOBE reconstructs the
# path from speaker_split/accent_class/clip_id instead.
KINDS: Dict[str, Dict[str, Any]] = {
    "crema_d": {
        "speaker": "actor_id",
        "split": "split",
        "label": "intended_emotion",
        "audio": "path",
        "duration": "duration_sec",
        "clip_id": "clip_id",
    },
    "globe": {
        "speaker": "client_id",
        "split": "speaker_split",
        "label": "accent_class",
        "audio": None,
        "duration": "duration_sec",
        "clip_id": "clip_id",
    },
    "ami": {
        "speaker": "speaker_id",
        "split": "split",
        "label": None,
        "audio": "wav_path",
        "duration": "duration_sec",
        "clip_id": "clip_id",
        "text": "text",
        "required_extra": ("text", "ref_clip_id", "ref_wav_path"),
    },
    "ami_naturalness": {
        "speaker": "speaker_id",
        "split": "split",
        "label": None,
        "audio": "wav_path",
        "duration": "duration_sec",
        "clip_id": "clip_id",
        "text": "text",
        "required_extra": ("text", "meeting_id"),
    },
    "generic": {
        "speaker": "speaker_id",
        "split": "split",
        "label": None,
        "audio": None,
        "duration": None,
        "clip_id": None,
    },
}

EXPECTED_SPLITS = {"train", "val", "test"}
_EMPTY_TOKENS = {"", "nan", "none", "null"}


def _empty_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(_EMPTY_TOKENS)


def load_manifest(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV, JSONL, or JSON-list manifest into a DataFrame."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".jsonl":
        return pd.read_json(source, lines=True)
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            payload = payload["samples"]
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list or contain a 'samples' list")
        return pd.DataFrame(payload)
    raise ValueError(f"unsupported manifest extension: {source.suffix!r}")


def _resolve_audio(
    row: Mapping[str, Any],
    *,
    kind: str,
    root: Path,
    audio_root: Optional[Path],
) -> Optional[Path]:
    audio_column = KINDS[kind].get("audio")
    if audio_column and row.get(audio_column):
        candidate = Path(str(row[audio_column]))
        return candidate if candidate.is_absolute() else root / candidate
    if kind == "globe" and audio_root is not None:
        return (
            audio_root
            / str(row["speaker_split"])
            / str(row["accent_class"])
            / f"{row['clip_id']}.wav"
        )
    return None


def validate_manifest(
    path: Union[str, Path],
    *,
    kind: str,
    root: Union[str, Path] = project_path(),
    audio_root: Optional[Union[str, Path]] = None,
    check_audio: bool = False,
    min_speakers_per_class: int = 0,
    min_hours_per_class: Optional[float] = None,
    trainable_labels: Optional[Sequence[str]] = None,
    max_audio_errors: int = 20,
) -> Dict[str, Any]:
    """Validate a manifest and return a machine-readable report."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    config = KINDS[kind]
    source = Path(path)
    root_path = Path(root)
    audio_root_path = Path(audio_root) if audio_root is not None else None
    trainable = (
        {str(label).strip().lower() for label in trainable_labels}
        if trainable_labels is not None
        else None
    )

    errors: List[str] = []
    warnings: List[str] = []
    if not source.exists():
        return {
            "schema_version": MANIFEST_VALIDATION_SCHEMA,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "manifest": str(source),
            "kind": kind,
            "passed": False,
            "errors": [f"manifest does not exist: {source}"],
            "warnings": [],
            "summary": {},
        }

    try:
        frame = load_manifest(source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema_version": MANIFEST_VALIDATION_SCHEMA,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "manifest": str(source),
            "kind": kind,
            "passed": False,
            "errors": [f"could not load manifest: {exc}"],
            "warnings": [],
            "summary": {},
        }

    if frame.empty:
        errors.append("manifest is empty")

    speaker_column = config["speaker"]
    split_column = config["split"]
    label_column = config.get("label")
    duration_column = config.get("duration")
    clip_column = config.get("clip_id")

    for column in (speaker_column, split_column):
        if column not in frame.columns:
            errors.append(f"required column missing: {column!r}")
    if label_column and label_column not in frame.columns:
        errors.append(f"required column missing: {label_column!r}")
    for column in config.get("required_extra", ()):
        if column not in frame.columns:
            errors.append(f"required column missing: {column!r}")
    if duration_column and duration_column not in frame.columns:
        warnings.append(f"duration column missing: {duration_column!r}")
    if clip_column and clip_column not in frame.columns:
        warnings.append(f"clip id column missing: {clip_column!r}")

    if errors:
        return {
            "schema_version": MANIFEST_VALIDATION_SCHEMA,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "manifest": str(source),
            "kind": kind,
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "summary": {"rows": int(len(frame))},
        }

    speakers = frame[speaker_column].astype(str).str.strip()
    splits = frame[split_column].astype(str).str.strip().str.lower()
    if _empty_mask(frame[speaker_column]).any():
        errors.append(f"{speaker_column!r} contains empty speaker ids")
    if _empty_mask(frame[split_column]).any():
        errors.append(f"{split_column!r} contains empty split labels")
    unknown_splits = sorted(set(splits) - EXPECTED_SPLITS)
    if unknown_splits:
        errors.append(
            f"{split_column!r} contains unsupported splits {unknown_splits}; "
            f"expected a subset of {sorted(EXPECTED_SPLITS)}"
        )

    text_column = config.get("text")
    if text_column and text_column in frame.columns:
        if _empty_mask(frame[text_column]).any():
            errors.append(f"{text_column!r} contains empty transcripts")

    frame = frame.assign(
        __speaker=speakers,
        __split=splits,
    )
    conflicts = frame.groupby("__speaker")["__split"].nunique()
    leaking = sorted(conflicts[conflicts > 1].index.tolist())
    if leaking:
        errors.append(
            f"{len(leaking)} speaker(s) appear in more than one split: "
            + ", ".join(leaking[:20])
        )

    if clip_column and clip_column in frame.columns:
        duplicates = frame[clip_column].astype(str).duplicated(keep=False)
        if duplicates.any():
            duplicate_ids = sorted(frame.loc[duplicates, clip_column].astype(str).unique())
            errors.append(
                f"{len(duplicate_ids)} duplicate clip id(s): " + ", ".join(duplicate_ids[:20])
            )

    split_summary: Dict[str, Any] = {}
    for split, subset in frame.groupby("__split"):
        split_summary[str(split)] = {
            "rows": int(len(subset)),
            "speakers": int(subset["__speaker"].nunique()),
            "hours": (
                float(subset[duration_column].sum() / 3600.0)
                if duration_column and duration_column in subset.columns
                else None
            ),
        }

    label_summary: Dict[str, Any] = {}
    if label_column and label_column in frame.columns:
        labels = frame[label_column].astype(str).str.strip().str.lower()
        frame = frame.assign(__label=labels)
        for label, subset in frame.groupby("__label"):
            train = subset[subset["__split"] == "train"]
            label_summary[str(label)] = {
                "rows": int(len(subset)),
                "speakers": int(subset["__speaker"].nunique()),
                "hours": (
                    float(subset[duration_column].sum() / 3600.0)
                    if duration_column and duration_column in subset.columns
                    else None
                ),
                "train_rows": int(len(train)),
                "train_speakers": int(train["__speaker"].nunique()),
                "train_hours": (
                    float(train[duration_column].sum() / 3600.0)
                    if duration_column and duration_column in train.columns
                    else None
                ),
                "trainable": trainable is None or str(label) in trainable,
            }
        for label, summary in sorted(label_summary.items()):
            if not summary["trainable"]:
                continue
            if summary["train_speakers"] < min_speakers_per_class:
                errors.append(
                    f"label {label!r} has {summary['train_speakers']} train speakers, "
                    f"below the required {min_speakers_per_class}"
                )
            if (
                min_hours_per_class is not None
                and summary["train_hours"] is not None
                and summary["train_hours"] < min_hours_per_class
            ):
                errors.append(
                    f"label {label!r} has {summary['train_hours']:.3f} train hours, "
                    f"below the required {min_hours_per_class:.3f}"
                )

    audio_errors: List[str] = []
    audio_checked = 0
    capped_warned = False
    if check_audio:
        for index, row in frame.iterrows():
            candidate = _resolve_audio(
                row,
                kind=kind,
                root=root_path,
                audio_root=audio_root_path,
            )
            if candidate is None:
                continue
            audio_checked += 1
            if len(audio_errors) < max_audio_errors:
                if not candidate.exists():
                    audio_errors.append(f"row {index}: missing audio {candidate}")
                else:
                    try:
                        import soundfile as sf

                        sf.info(str(candidate))
                    except Exception as exc:  # soundfile raises several backend errors
                        audio_errors.append(
                            f"row {index}: unreadable audio {candidate}: {exc}"
                        )
            elif len(audio_errors) == max_audio_errors and not capped_warned:
                warnings.append(
                    f"audio error list capped at {max_audio_errors}; remaining rows were "
                    "checked but not listed"
                )
                capped_warned = True
        errors.extend(audio_errors)

    return {
        "schema_version": MANIFEST_VALIDATION_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": str(source),
        "kind": kind,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "rows": int(len(frame)),
            "speakers": int(frame["__speaker"].nunique()),
            "splits": split_summary,
            "labels": label_summary,
            "trainable_labels": (
                sorted(trainable) if trainable is not None else sorted(label_summary)
            ),
            "evaluation_only_labels": (
                sorted(set(label_summary) - trainable) if trainable is not None else []
            ),
            "audio_checked": audio_checked,
            "audio_errors": len(audio_errors),
        },
    }


def write_validation_report(
    report: Mapping[str, Any],
    path: Union[str, Path],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
