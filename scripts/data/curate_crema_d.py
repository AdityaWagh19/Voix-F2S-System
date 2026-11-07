"""Curate CREMA-D into a speaker-disjoint emotion manifest.

CREMA-D has 7,442 clips from 91 actors with six intended emotions.  This script
parses the filename convention, joins actor demographics and the crowd-sourced
majority vote, checks each WAV, and writes a deterministic speaker-disjoint
manifest plus a gate report.

The 2.0 h / 30-speaker gate is computed, never lowered.  CREMA-D is about
5.26 h total, so it is expected to fail the 2 h per-emotion gate; the manifest
is therefore a pilot/evaluation artifact unless a larger licensed corpus is
added.  Pass ``--require-gate`` to turn that failure into a non-zero exit.

Usage:
    python scripts/data/curate_crema_d.py
    python scripts/data/curate_crema_d.py --require-gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = ROOT / "data" / "raw" / "crema_d"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_manifest.csv"
DEFAULT_REPORT = ROOT / "data" / "processed" / "crema_d_curation_report.json"

SCHEMA = "voix.crema_d_curation.v1"
ACCESSED = "2026-10-28"

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
EMOTION_CODES = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "disgust",
}
EMOTION_LETTERS = {
    "neutral": "N",
    "happy": "H",
    "sad": "S",
    "angry": "A",
    "fear": "F",
    "disgust": "D",
}
LEVELS = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}
CLIP_RE = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>[A-Z]{3})_(?P<level>[A-Z]{2})$"
)

TRAINABLE_GATE = {
    "min_speakers_per_class": 30,
    "min_hours_per_class": 2.0,
    "max_min_class_ratio": 1.5,
    "min_val_speakers": 5,
    "min_test_speakers": 5,
    "split_policy": "speaker-disjoint train/validation/test",
}


def parse_clip(stem: str) -> Optional[Dict[str, str]]:
    """Parse ``1001_DFA_ANG_XX`` into actor, sentence, emotion, and level."""
    match = CLIP_RE.match(stem)
    if not match:
        return None
    emotion = EMOTION_CODES.get(match.group("emotion"))
    level = LEVELS.get(match.group("level"))
    if emotion is None or level is None:
        return None
    return {
        "clip_id": stem,
        "actor_id": match.group("actor"),
        "sentence": match.group("sentence"),
        "intended_emotion": emotion,
        "intensity": level,
    }


def assign_splits(
    speakers: List[str],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    salt: str = "crema-d-v1",
) -> Dict[str, str]:
    """Deterministically assign speakers to train/val/test.

    Ordering is by SHA-256 of ``salt:speaker`` so the split is stable across
    machines and independent of filesystem order.  Every speaker appears in
    exactly one split.
    """
    unique = sorted(set(str(speaker) for speaker in speakers))
    if not unique:
        return {}
    ordered = sorted(
        unique,
        key=lambda speaker: hashlib.sha256(
            f"{salt}:{speaker}".encode("utf-8")
        ).hexdigest(),
    )
    total = len(ordered)
    n_test = max(1, round(total * test_fraction)) if total >= 3 else 0
    n_val = max(1, round(total * val_fraction)) if total >= 3 else 0
    if n_test + n_val >= total:
        n_test = max(1, total // 5) if total >= 5 else 0
        n_val = max(1, total // 5) if total >= 5 else 0
        while n_test + n_val >= total:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
    splits: Dict[str, str] = {}
    for index, speaker in enumerate(ordered):
        if index < n_test:
            splits[speaker] = "test"
        elif index < n_test + n_val:
            splits[speaker] = "val"
        else:
            splits[speaker] = "train"
    return splits


def load_demographics(root: Path) -> pd.DataFrame:
    path = root / "VideoDemographics.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["actor_id", "age", "sex", "race", "ethnicity"])
    frame = pd.read_csv(path, dtype={"ActorID": str})
    if "ActorID" not in frame.columns:
        return pd.DataFrame(columns=["actor_id", "age", "sex", "race", "ethnicity"])
    frame = frame.copy()
    frame["actor_id"] = frame["ActorID"].astype(str).str.zfill(4)
    frame = frame.rename(
        columns={
            "Age": "age",
            "Sex": "sex",
            "Race": "race",
            "Ethnicity": "ethnicity",
        }
    )
    keep = ["actor_id", "age", "sex", "race", "ethnicity"]
    for column in keep:
        if column not in frame.columns:
            frame[column] = None
    return frame[keep].drop_duplicates("actor_id")


def load_votes(root: Path) -> pd.DataFrame:
    """Load the voice-only perceptual votes.

    ``tabulatedVotes.csv`` is three concatenated 7,442-row blocks in the order
    voice-only, face-only, audiovisual (see ``tabulateVotesV2.r``).  The audio
    corpus must use the voice-only block; concatenating all three multiplies
    every clip by three and mixes modalities.  If the file is already a single
    block, it is used as-is.
    """
    empty = pd.DataFrame(
        columns=[
            "clip_id",
            "perceptual_vote",
            "agreement",
            "num_responses",
        ]
    )
    path = root / "processedResults" / "tabulatedVotes.csv"
    if not path.is_file():
        return empty
    frame = pd.read_csv(path)
    if "fileName" not in frame.columns:
        return empty
    frame = frame.copy()
    frame["clip_id"] = (
        frame["fileName"].astype(str).str.replace(r"\.wav$", "", regex=True)
    )

    unique_clips = frame["clip_id"].nunique()
    if len(frame) == 3 * unique_clips:
        block = len(frame) // 3
        blocks = [frame.iloc[index * block : (index + 1) * block] for index in range(3)]
        if all(
            block_frame["clip_id"].tolist() == blocks[0]["clip_id"].tolist()
            for block_frame in blocks[1:]
        ):
            frame = blocks[0].copy()
        else:
            raise ValueError(
                "tabulatedVotes.csv has 3 rows per clip but the blocks do not "
                "share one clip order; cannot identify the voice-only block"
            )
    elif len(frame) != unique_clips:
        raise ValueError(
            "tabulatedVotes.csv is neither one row per clip nor three ordered "
            f"blocks ({len(frame)} rows, {unique_clips} clips)"
        )

    frame["perceptual_vote"] = frame.get("emoVote")
    frame["agreement"] = pd.to_numeric(frame.get("agreement"), errors="coerce")
    frame["num_responses"] = pd.to_numeric(
        frame.get("numResponses"), errors="coerce"
    )
    return frame[["clip_id", "perceptual_vote", "agreement", "num_responses"]]


def check_audio(path: Path, min_sec: float, max_sec: float) -> Dict[str, Any]:
    """Read one WAV and return duration plus health facts."""
    info = sf.info(str(path))
    duration = float(info.frames) / float(info.samplerate)
    result: Dict[str, Any] = {
        "duration_sec": round(duration, 4),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "status": "ok",
        "error": None,
        "peak": None,
        "rms": None,
        "clipping_fraction": None,
        "finite": None,
    }
    errors: List[str] = []
    if duration < min_sec:
        errors.append(f"duration {duration:.3f}s < {min_sec:.3f}s")
    if duration > max_sec:
        errors.append(f"duration {duration:.3f}s > {max_sec:.3f}s")
    if info.samplerate <= 0:
        errors.append("invalid sample rate")
    if info.channels <= 0:
        errors.append("invalid channel count")
    if errors:
        result["status"] = "error"
        result["error"] = "; ".join(errors)
        return result
    try:
        samples, _ = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # pragma: no cover - corrupt-file guard
        result["status"] = "error"
        result["error"] = f"read failed: {exc}"
        return result
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    finite = bool(np.isfinite(samples).all())
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    clipping = float(np.mean(np.abs(samples) >= 0.999969)) if samples.size else 0.0
    result.update(
        {
            "peak": round(peak, 6),
            "rms": round(rms, 6),
            "clipping_fraction": round(clipping, 8),
            "finite": finite,
        }
    )
    if not finite:
        errors.append("non-finite samples")
    if clipping > 0.0:
        errors.append(f"clipping_fraction={clipping:.8f}")
    if errors:
        result["status"] = "error"
        result["error"] = "; ".join(errors)
    return result


def summarize(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for emotion in EMOTIONS:
        subset = frame[frame["intended_emotion"] == emotion]
        if subset.empty:
            summary[emotion] = {
                "clips": 0,
                "speakers": 0,
                "hours": 0.0,
                "train_speakers": 0,
                "val_speakers": 0,
                "test_speakers": 0,
                "mean_agreement": None,
                "perceptual_label_agreement": None,
            }
            continue
        agreement = pd.to_numeric(subset["agreement"], errors="coerce").dropna()
        votes = subset["perceptual_vote"].astype(str)
        expected = EMOTION_LETTERS[emotion]
        # The majority vote can be a tie such as ``N:H``; the intended label
        # counts as perceptually consistent when it is one of the tied labels.
        label_agreement = (
            float(votes.str.split(":").map(lambda parts: expected in parts).mean())
            if len(votes)
            else None
        )
        tie_fraction = (
            float(votes.str.contains(":").mean()) if len(votes) else None
        )
        summary[emotion] = {
            "clips": int(len(subset)),
            "speakers": int(subset["actor_id"].nunique()),
            "hours": round(float(subset["duration_sec"].sum()) / 3600.0, 3),
            "train_speakers": int(
                subset.loc[subset["split"] == "train", "actor_id"].nunique()
            ),
            "val_speakers": int(
                subset.loc[subset["split"] == "val", "actor_id"].nunique()
            ),
            "test_speakers": int(
                subset.loc[subset["split"] == "test", "actor_id"].nunique()
            ),
            "mean_agreement": round(float(agreement.mean()), 4)
            if len(agreement)
            else None,
            "perceptual_label_agreement": round(label_agreement, 4)
            if label_agreement is not None
            else None,
            "tie_fraction": round(tie_fraction, 4)
            if tie_fraction is not None
            else None,
        }
    return summary


def gate_failures(summary: Dict[str, Dict[str, Any]]) -> List[str]:
    failures: List[str] = []
    hours = [
        float(stats["hours"])
        for stats in summary.values()
        if float(stats["hours"]) > 0.0
    ]
    if hours:
        ratio = max(hours) / min(hours)
        if ratio > TRAINABLE_GATE["max_min_class_ratio"]:
            failures.append(
                f"max/min hours ratio {ratio:.3f} > "
                f"{TRAINABLE_GATE['max_min_class_ratio']:.3f}"
            )
    for emotion, stats in summary.items():
        if int(stats["clips"]) == 0:
            failures.append(f"{emotion}: no clips")
            continue
        if float(stats["hours"]) < TRAINABLE_GATE["min_hours_per_class"]:
            failures.append(
                f"{emotion}: {stats['hours']:.3f}h < "
                f"{TRAINABLE_GATE['min_hours_per_class']:.3f}h"
            )
        if int(stats["speakers"]) < TRAINABLE_GATE["min_speakers_per_class"]:
            failures.append(
                f"{emotion}: {stats['speakers']} speakers < "
                f"{TRAINABLE_GATE['min_speakers_per_class']}"
            )
        if int(stats["val_speakers"]) < TRAINABLE_GATE["min_val_speakers"]:
            failures.append(
                f"{emotion}: {stats['val_speakers']} val speakers < "
                f"{TRAINABLE_GATE['min_val_speakers']}"
            )
        if int(stats["test_speakers"]) < TRAINABLE_GATE["min_test_speakers"]:
            failures.append(
                f"{emotion}: {stats['test_speakers']} test speakers < "
                f"{TRAINABLE_GATE['min_test_speakers']}"
            )
    return failures


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def curate(
    root: Path,
    *,
    min_clip_sec: float = 0.5,
    max_clip_sec: float = 15.0,
    min_agreement: float = 0.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    workers: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    audio_dir = root / "AudioWAV"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"CREMA-D AudioWAV directory not found: {audio_dir}")
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no WAV files in {audio_dir}")

    rows: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    for path in wavs:
        parsed = parse_clip(path.stem)
        if parsed is None:
            excluded.append(
                {"clip_id": path.stem, "reason": "unparseable filename"}
            )
            continue
        rows.append(parsed)
    base = pd.DataFrame(rows)
    if base.empty:
        raise ValueError("no parseable CREMA-D clips")

    def _health(path: Path) -> Dict[str, Any]:
        return check_audio(path, min_clip_sec, max_clip_sec)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        health = list(pool.map(_health, wavs))
    health_by_id = {path.stem: item for path, item in zip(wavs, health)}
    base["health"] = base["clip_id"].map(health_by_id)
    failed_health = base["health"].map(
        lambda item: not isinstance(item, dict) or item.get("status") != "ok"
    )
    for _, row in base[failed_health].iterrows():
        item = row["health"] or {}
        excluded.append(
            {
                "clip_id": str(row["clip_id"]),
                "reason": str(item.get("error") or "missing health result"),
            }
        )
    base = base[~failed_health].copy()
    for key in (
        "duration_sec",
        "sample_rate",
        "channels",
        "peak",
        "rms",
        "clipping_fraction",
        "finite",
    ):
        base[key] = base["health"].map(lambda item, k=key: item.get(k))
    base = base.drop(columns=["health"])

    agreement_filter_removed = 0
    if min_agreement > 0.0:
        votes = load_votes(root)
        base = base.merge(votes, on="clip_id", how="left")
        before = len(base)
        base = base[
            pd.to_numeric(base["agreement"], errors="coerce") >= min_agreement
        ].copy()
        agreement_filter_removed = before - len(base)
    else:
        base = base.merge(load_votes(root), on="clip_id", how="left")

    demographics = load_demographics(root)
    base = base.merge(demographics, on="actor_id", how="left")

    splits = assign_splits(
        base["actor_id"].tolist(),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    base["split"] = base["actor_id"].map(splits)
    base["path"] = base["clip_id"].map(
        lambda clip_id: _display_path(audio_dir / f"{clip_id}.wav")
    )
    columns = [
        "clip_id",
        "path",
        "actor_id",
        "sentence",
        "intended_emotion",
        "intensity",
        "split",
        "duration_sec",
        "sample_rate",
        "channels",
        "peak",
        "rms",
        "clipping_fraction",
        "finite",
        "perceptual_vote",
        "agreement",
        "num_responses",
        "age",
        "sex",
        "race",
        "ethnicity",
    ]
    for column in columns:
        if column not in base.columns:
            base[column] = None
    base = base[columns].sort_values(
        ["split", "intended_emotion", "actor_id", "clip_id"]
    )
    base = base.reset_index(drop=True)

    summary = summarize(base)
    failures = gate_failures(summary)
    report = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": "https://github.com/CheyneyComputerScience/CREMA-D",
        "license": "ODbL 1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "accessed": ACCESSED,
        "root": str(root),
        "clips_total": int(len(wavs)),
        "clips_kept": int(len(base)),
        "clips_excluded": int(len(wavs) - len(base) - agreement_filter_removed),
        "agreement_filter_removed": int(agreement_filter_removed),
        "excluded": excluded[:1000],
        "excluded_count": int(len(excluded)),
        "speakers": int(base["actor_id"].nunique()),
        "hours_total": round(float(base["duration_sec"].sum()) / 3600.0, 3),
        "by_emotion": summary,
        "gate": {
            "thresholds": TRAINABLE_GATE,
            "pass": not failures,
            "training_eligible": not failures,
            "failures": failures,
        },
        "label_source": "intended emotion from the CREMA-D filename",
        "label_note": (
            "The voice-only crowd majority vote (the first of the three "
            "tabulated blocks) is joined as a perceptual audit label, not used "
            "as the training target."
        ),
        "split_salt": "crema-d-v1",
        "notes": [
            "CREMA-D is about 5.26 h across six emotions (~0.88 h/class), so it is expected to fail the 2 h/class gate.",
            "A failed gate makes this a pilot/evaluation manifest, not a learned-control training set.",
            "No threshold is lowered; add a larger licensed emotion corpus before training a production control.",
        ],
    }
    return base, report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-clip-sec", type=float, default=0.5)
    parser.add_argument("--max-clip-sec", type=float, default=15.0)
    parser.add_argument("--min-agreement", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="Exit non-zero when the 2 h/30-speaker gate fails.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    frame, report = curate(
        args.root,
        min_clip_sec=args.min_clip_sec,
        max_clip_sec=args.max_clip_sec,
        min_agreement=args.min_agreement,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        workers=args.workers,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.manifest, index=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Wrote {args.manifest} ({len(frame)} clips)")
    print(f"Wrote {args.report}")
    if args.require_gate and not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
