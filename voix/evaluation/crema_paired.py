"""Build and validate the paired CREMA-D evaluation grid.

The grid is an evaluation artifact, not a training manifest.  It pairs, for one
actor and one standard CREMA-D sentence, the real neutral clip with every real
emotion clip that exists, and records the fixed identity reference used for
generation.

Design contract (see ``docs/MASTER.md`` section 5):

* one fixed neutral identity clip per actor;
* one real neutral clip per actor and sentence;
* one row per ``(actor, sentence, emotion)``;
* missing real emotion clips are kept with ``has_real_emotion_clip=False`` so
  generation can still run, but target-residual training must skip them;
* no face, CFVN, or CVAE involvement.  The identity reference is a real ECAPA
  embedding from the actor's neutral speech.

The module is pure pandas/NumPy and does not load models or audio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from voix.utils.paths import project_path

PAIRED_EVAL_SCHEMA = "voix.crema_paired_eval.v1"

# Verified against data/raw/crema_d/README.md lines 161-174.  Trailing list
# punctuation is omitted so the same string can be used as a WER reference.
SENTENCES: Dict[str, str] = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

CREMA_EMOTIONS: Tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
)
NON_NEUTRAL_EMOTIONS: Tuple[str, ...] = tuple(
    emotion for emotion in CREMA_EMOTIONS if emotion != "neutral"
)
INTENSITY_ORDER: Dict[str, int] = {"low": 0, "medium": 1, "high": 2, "unspecified": 3}

PAIRED_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "actor_id",
    "speaker_id",
    "sentence",
    "text",
    "emotion",
    "intensity",
    "split",
    "identity_clip_id",
    "identity_path",
    "neutral_clip_id",
    "neutral_path",
    "emotion_clip_id",
    "emotion_path",
    "has_real_emotion_clip",
    "real_intensity_label",
    "neutral_duration_sec",
    "emotion_duration_sec",
    "age",
    "sex",
)

_REQUIRED_MANIFEST_COLUMNS = (
    "clip_id",
    "path",
    "actor_id",
    "sentence",
    "intended_emotion",
    "intensity",
    "split",
    "duration_sec",
)


def _string(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _resolve(path: str, root: Union[str, Path]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def _check_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CREMA-D manifest is missing columns: {missing}")


def _pick_identity_clip(neutral: pd.DataFrame) -> pd.Series:
    """Pick one neutral clip per actor deterministically.

    Longest duration first gives ECAPA more speech to embed; clip id breaks
    ties.  Clipped or non-finite clips are rejected before selection.
    """
    usable = neutral[
        (neutral["finite"].astype(str).str.lower() == "true")
        & (pd.to_numeric(neutral["clipping_fraction"], errors="coerce").fillna(1.0) == 0.0)
    ]
    if usable.empty:
        usable = neutral
    ranked = usable.assign(
        _duration=pd.to_numeric(usable["duration_sec"], errors="coerce").fillna(0.0)
    ).sort_values(["_duration", "clip_id"], ascending=[False, True])
    return ranked.iloc[0]


def _pick_real_clip(rows: pd.DataFrame) -> Optional[pd.Series]:
    """Prefer the unspecified/full-expression clip, then high > medium > low."""
    if rows.empty:
        return None
    ranked = rows.assign(
        _intensity_rank=rows["intensity"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(INTENSITY_ORDER)
        .fillna(-1)
    ).sort_values(["_intensity_rank", "clip_id"], ascending=[False, True])
    return ranked.iloc[0]


def build_paired_grid(
    manifest: pd.DataFrame,
    *,
    split: str = "val",
    emotions: Sequence[str] = CREMA_EMOTIONS,
    max_actors: Optional[int] = None,
    root: Union[str, Path] = project_path(),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return the paired evaluation grid and a machine-readable summary.

    Missing neutral clips are skipped and reported.  Missing real emotion clips
    keep the row but set ``has_real_emotion_clip=False``; this lets E0 generate
    the condition without pretending a target residual exists.
    """
    _check_columns(manifest)
    split_key = str(split).strip().lower()
    frame = manifest.copy()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame = frame[frame["split"] == split_key].copy()
    if frame.empty:
        raise ValueError(f"no rows for split {split!r}")

    requested_emotions = [str(emotion).strip().lower() for emotion in emotions]
    unknown = sorted(set(requested_emotions) - set(CREMA_EMOTIONS))
    if unknown:
        raise ValueError(f"unsupported emotions: {unknown}")
    if "neutral" not in requested_emotions:
        raise ValueError("the paired grid must include the neutral condition")

    actors = sorted(frame["actor_id"].astype(str).unique())
    if max_actors is not None:
        actors = actors[: int(max_actors)]

    rows: List[Dict[str, Any]] = []
    skipped_missing_neutral: List[Dict[str, str]] = []
    duplicate_neutral: List[Dict[str, Any]] = []
    duplicate_emotion: List[Dict[str, Any]] = []

    for actor in actors:
        actor_rows = frame[frame["actor_id"].astype(str) == actor]
        neutral_rows = actor_rows[
            actor_rows["intended_emotion"].astype(str).str.strip().str.lower() == "neutral"
        ]
        if neutral_rows.empty:
            skipped_missing_neutral.append(
                {"actor_id": actor, "sentence": "*", "reason": "no neutral clip"}
            )
            continue
        identity = _pick_identity_clip(neutral_rows)

        for sentence, text in SENTENCES.items():
            sentence_neutral = neutral_rows[
                neutral_rows["sentence"].astype(str).str.strip().str.upper() == sentence
            ]
            if sentence_neutral.empty:
                skipped_missing_neutral.append(
                    {"actor_id": actor, "sentence": sentence, "reason": "no neutral clip"}
                )
                continue
            if len(sentence_neutral) > 1:
                duplicate_neutral.append(
                    {
                        "actor_id": actor,
                        "sentence": sentence,
                        "clip_ids": sorted(sentence_neutral["clip_id"].astype(str)),
                    }
                )
            neutral = sentence_neutral.sort_values("clip_id").iloc[0]

            for emotion in requested_emotions:
                if emotion == "neutral":
                    real = neutral
                    real_intensity = "unspecified"
                    has_real = True
                else:
                    candidates = actor_rows[
                        (actor_rows["sentence"].astype(str).str.strip().str.upper() == sentence)
                        & (
                            actor_rows["intended_emotion"].astype(str).str.strip().str.lower()
                            == emotion
                        )
                    ]
                    if len(candidates) > 1:
                        duplicate_emotion.append(
                            {
                                "actor_id": actor,
                                "sentence": sentence,
                                "emotion": emotion,
                                "clip_ids": sorted(candidates["clip_id"].astype(str)),
                            }
                        )
                    real = _pick_real_clip(candidates)
                    if real is None:
                        real_intensity = ""
                        has_real = False
                    else:
                        real_intensity = _string(real.get("intensity", ""))
                        has_real = True

                pair_id = f"{actor}_{sentence}"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "actor_id": actor,
                        "speaker_id": actor,
                        "sentence": sentence,
                        "text": text,
                        "emotion": emotion,
                        "intensity": 0.0 if emotion == "neutral" else 1.0,
                        "split": split_key,
                        "identity_clip_id": _string(identity["clip_id"]),
                        "identity_path": _string(identity["path"]),
                        "neutral_clip_id": _string(neutral["clip_id"]),
                        "neutral_path": _string(neutral["path"]),
                        "emotion_clip_id": _string(real["clip_id"]) if real is not None else "",
                        "emotion_path": _string(real["path"]) if real is not None else "",
                        "has_real_emotion_clip": bool(has_real),
                        "real_intensity_label": real_intensity,
                        "neutral_duration_sec": float(neutral["duration_sec"]),
                        "emotion_duration_sec": (
                            float(real["duration_sec"]) if real is not None else None
                        ),
                        "age": float(neutral.get("age", 30.0)),
                        "sex": _string(neutral.get("sex", "")),
                    }
                )

    grid = pd.DataFrame(rows, columns=list(PAIRED_COLUMNS))
    if grid.empty:
        raise ValueError("the paired grid is empty")

    per_emotion: Dict[str, Any] = {}
    for emotion in requested_emotions:
        subset = grid[grid["emotion"] == emotion]
        per_emotion[emotion] = {
            "rows": int(len(subset)),
            "actors": int(subset["actor_id"].nunique()),
            "sentences": int(subset["sentence"].nunique()),
            "with_real_clip": int(subset["has_real_emotion_clip"].sum()),
            "real_seconds": float(
                subset.loc[subset["has_real_emotion_clip"], "emotion_duration_sec"]
                .fillna(0.0)
                .sum()
            ),
        }

    summary = {
        "schema_version": PAIRED_EVAL_SCHEMA,
        "split": split_key,
        "actors": int(grid["actor_id"].nunique()),
        "actor_ids": sorted(grid["actor_id"].unique()),
        "sentences": int(grid["sentence"].nunique()),
        "emotions": requested_emotions,
        "rows": int(len(grid)),
        "pairs": int(grid["pair_id"].nunique()),
        "per_emotion": per_emotion,
        "identity_rule": (
            "per actor: longest neutral clip with finite audio and no clipping; "
            "tie-break by clip_id"
        ),
        "skipped_missing_neutral": skipped_missing_neutral,
        "duplicate_neutral": duplicate_neutral,
        "duplicate_emotion": duplicate_emotion,
    }
    return grid, summary


def validate_paired_grid(
    grid: pd.DataFrame,
    *,
    root: Union[str, Path] = project_path(),
    check_audio: bool = False,
) -> List[str]:
    """Return validation errors; an empty list means the grid is usable."""
    errors: List[str] = []
    missing = [column for column in PAIRED_COLUMNS if column not in grid.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if grid.empty:
        return ["grid is empty"]

    for column in ("actor_id", "speaker_id", "sentence", "text", "emotion", "split"):
        empty = grid[column].astype(str).str.strip().isin({"", "nan", "none", "null"})
        if empty.any():
            errors.append(f"{column!r} contains empty values")

    unknown_emotions = sorted(
        set(grid["emotion"].astype(str).str.lower()) - set(CREMA_EMOTIONS)
    )
    if unknown_emotions:
        errors.append(f"unsupported emotions: {unknown_emotions}")

    duplicates = grid.duplicated(subset=["actor_id", "sentence", "emotion"], keep=False)
    if duplicates.any():
        keys = grid.loc[duplicates, ["actor_id", "sentence", "emotion"]].astype(str)
        errors.append(
            "duplicate (actor, sentence, emotion) rows: "
            + "; ".join("|".join(row) for row in keys.drop_duplicates().to_numpy())
        )

    for sentence, expected in SENTENCES.items():
        subset = grid[grid["sentence"].astype(str).str.upper() == sentence]
        if subset.empty:
            continue
        wrong = subset[subset["text"].astype(str) != expected]
        if not wrong.empty:
            errors.append(
                f"sentence {sentence!r} has {len(wrong)} row(s) with the wrong text"
            )

    neutral = grid[grid["emotion"].astype(str).str.lower() == "neutral"]
    if neutral.empty:
        errors.append("grid has no neutral rows")
    else:
        mismatched = neutral[
            neutral["emotion_path"].astype(str) != neutral["neutral_path"].astype(str)
        ]
        if not mismatched.empty:
            errors.append("neutral rows must use the same path as their neutral reference")
        wrong_intensity = neutral[
            pd.to_numeric(neutral["intensity"], errors="coerce").fillna(-1.0) != 0.0
        ]
        if not wrong_intensity.empty:
            errors.append("neutral rows must have intensity 0.0")

    non_neutral = grid[grid["emotion"].astype(str).str.lower() != "neutral"]
    if not non_neutral.empty:
        wrong_intensity = non_neutral[
            pd.to_numeric(non_neutral["intensity"], errors="coerce").fillna(-1.0) <= 0.0
        ]
        if not wrong_intensity.empty:
            errors.append("non-neutral rows must have a positive intensity")

    for column in ("identity_path", "neutral_path"):
        empty = grid[column].astype(str).str.strip().isin({"", "nan", "none", "null"})
        if empty.any():
            errors.append(f"{column!r} contains empty paths")
            continue
        if check_audio:
            missing_paths = [
                path for path in grid[column].astype(str).unique() if not _resolve(path, root).exists()
            ]
            if missing_paths:
                errors.append(
                    f"{column!r} has {len(missing_paths)} missing file(s): "
                    + ", ".join(missing_paths[:10])
                )

    real = grid[grid["has_real_emotion_clip"].astype(bool)]
    if check_audio and not real.empty:
        missing_paths = [
            path
            for path in real["emotion_path"].astype(str).unique()
            if not _resolve(path, root).exists()
        ]
        if missing_paths:
            errors.append(
                f"'emotion_path' has {len(missing_paths)} missing real clip(s): "
                + ", ".join(missing_paths[:10])
            )

    # Every (actor, sentence) pair must have exactly one neutral row.
    counts = (
        grid.assign(_emotion=grid["emotion"].astype(str).str.lower())
        .query("_emotion == 'neutral'")
        .groupby(["actor_id", "sentence"])
        .size()
    )
    if not counts.empty and (counts != 1).any():
        bad = counts[counts != 1]
        errors.append(
            "every (actor, sentence) must have exactly one neutral row: "
            + ", ".join(f"{actor}|{sentence}={count}" for (actor, sentence), count in bad.items())
        )
    return errors


def write_paired_grid(
    grid: pd.DataFrame,
    path: Union[str, Path],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(destination, index=False)
    return destination
