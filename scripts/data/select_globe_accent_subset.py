"""Select a balanced, speaker-disjoint GLOBE accent subset.

The selection is metadata-only.  It resolves the self-reported Common Voice
accent into one of six target classes, applies quality filters, caps each
speaker, fills each class to the requested duration, and assigns speakers to
train/val/test by a deterministic hash.  The output manifest carries the
Parquet coordinates (``source_file``, ``row_group``, ``row_in_group``) so the
audio fetch can read only the selected row groups.

The gate is intentionally strict: 2 hours and 30 speakers per class, a
max/min duration ratio of at most 1.5, and no speaker in more than one split.
A short class is reported as a failure; thresholds are never lowered.

Usage:
    python scripts/data/select_globe_accent_subset.py
    python scripts/data/select_globe_accent_subset.py --target-hours 2.0 --speaker-cap-sec 120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "data" / "processed" / "globe_with_speakers.parquet"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "globe_accent_selection_summary.json"

# One canonical self-reported label per target class.  Multi-accent strings are
# accepted only when exactly one token belongs to exactly one target class.
ACCENT_CLASSES: Dict[str, str] = {
    "us": "United States English",
    "england": "England English",
    "australian": "Australian English",
    "canadian": "Canadian English",
    "india_south_asia": "India and South Asia (India, Pakistan, Sri Lanka)",
    "irish": "Irish English",
}



def resolve_accent(label: object) -> Tuple[Optional[str], str]:
    """Return ``(class, reason)`` for a self-reported accent string."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None, "missing"
    text = str(label).strip()
    if not text or text.lower() == "nan":
        return None, "missing"
    # Common Voice accent strings can contain commas inside a label
    # ("India and South Asia (India, Pakistan, Sri Lanka)"), so tokenizing on
    # commas is unsafe.  Match the six known labels as substrings instead.
    matches = {
        key for key, label in ACCENT_CLASSES.items() if label in text
    }
    if len(matches) == 1:
        return next(iter(matches)), "ok"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "not_target"


def stable_split(client_id: str, seed: int, train_pct: int) -> str:
    digest = hashlib.sha256(f"{seed}:{client_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    remaining = 100 - train_pct
    if remaining <= 0:
        return "train"
    if bucket < train_pct + remaining / 2:
        return "val"
    return "test"


def cap_speaker_clips(
    group: pd.DataFrame,
    *,
    cap_sec: float,
    max_clips: int,
) -> pd.DataFrame:
    sort_columns = ["utmos", "snr", "wer", "duration_sec"]
    ascending = [False, False, True, False]
    if "_group_size" in group.columns:
        # Prefer clips from row groups that already yield many candidates, so
        # the audio pass does not fetch a 22 MB chunk for one clip.
        sort_columns = ["_group_size"] + sort_columns
        ascending = [False] + ascending
    ordered = group.sort_values(sort_columns, ascending=ascending)
    selected: List[int] = []
    total = 0.0
    for index, row in ordered.iterrows():
        duration = float(row["duration_sec"])
        if duration <= 0:
            continue
        if len(selected) >= max_clips:
            break
        if total + duration > cap_sec:
            continue
        selected.append(index)
        total += duration
        if total >= cap_sec:
            break
    return group.loc[selected]


def _shuffled_speakers(
    speakers: List[str],
    *,
    seed: int,
    accent: str,
) -> List[str]:
    return sorted(
        speakers,
        key=lambda client_id: hashlib.sha256(
            f"{seed}:{accent}:{client_id}".encode("utf-8")
        ).hexdigest(),
    )


def select_class(
    frame: pd.DataFrame,
    *,
    accent: str,
    target_sec: float,
    cap_sec: float,
    max_clips: int,
    balance_gender: bool,
    seed: int,
) -> pd.DataFrame:
    class_frame = frame[frame["accent_class"] == accent]
    if class_frame.empty:
        return class_frame

    # Per-speaker cap first, then choose speakers until the class target.
    capped = (
        class_frame.groupby("client_id", group_keys=False)
        .apply(
            lambda group: cap_speaker_clips(
                group, cap_sec=cap_sec, max_clips=max_clips
            )
        )
        .reset_index(drop=True)
    )
    if capped.empty:
        return capped

    available = (
        capped.groupby("client_id")["duration_sec"].sum().sort_values(ascending=False)
    )
    speakers = list(available.index)

    if balance_gender:
        gender = (
            capped.groupby("client_id")["predicted_gender"]
            .agg(lambda values: values.mode().iloc[0] if len(values.mode()) else "unknown")
            .to_dict()
        )
        groups: Dict[str, List[str]] = {}
        for client_id in speakers:
            groups.setdefault(str(gender.get(client_id, "unknown")), []).append(client_id)
        ordered_groups = [
            _shuffled_speakers(groups[key], seed=seed, accent=accent)
            for key in sorted(groups, key=lambda key: (-len(groups[key]), key))
        ]
        ordered: List[str] = []
        cursor = 0
        while any(cursor < len(group) for group in ordered_groups):
            for group in ordered_groups:
                if cursor < len(group):
                    ordered.append(group[cursor])
            cursor += 1
    else:
        ordered = _shuffled_speakers(speakers, seed=seed, accent=accent)

    chosen: List[pd.DataFrame] = []
    total = 0.0
    for client_id in ordered:
        if total >= target_sec:
            break
        speaker_frame = capped[capped["client_id"] == client_id]
        if speaker_frame.empty:
            continue
        chosen.append(speaker_frame)
        total += float(speaker_frame["duration_sec"].sum())
    if not chosen:
        return capped.iloc[0:0]
    return pd.concat(chosen, ignore_index=True)


def select_class_row_group(
    frame: pd.DataFrame,
    *,
    accent: str,
    target_sec: float,
    cap_sec: float,
    max_clips: int,
    balance_gender: bool,
    seed: int,
) -> pd.DataFrame:
    """Select clips row-group first to minimize the audio fetch.

    The speaker cap is still strict.  Row groups are ranked by the number of
    distinct speakers they contain, then by clip count, so the pass prefers
    chunks that buy both speaker diversity and low transfer overhead.
    """
    class_frame = frame[frame["accent_class"] == accent].copy()
    if class_frame.empty:
        return class_frame
    group_sizes = class_frame.groupby(["source_file", "row_group"]).size()
    class_frame["_group_size"] = [
        int(group_sizes.loc[(source, row_group)])
        for source, row_group in zip(
            class_frame["source_file"], class_frame["row_group"]
        )
    ]
    capped = (
        class_frame.groupby("client_id", group_keys=False)
        .apply(
            lambda group: cap_speaker_clips(
                group, cap_sec=cap_sec, max_clips=max_clips
            )
        )
        .reset_index(drop=True)
    )
    if capped.empty:
        return capped

    gender_by_speaker = (
        capped.groupby("client_id")["predicted_gender"]
        .agg(lambda values: values.mode().iloc[0] if len(values.mode()) else "unknown")
        .to_dict()
    )
    stats = (
        capped.groupby(["source_file", "row_group"])
        .agg(
            clips=("common_voice_path", "size"),
            speakers=("client_id", "nunique"),
            seconds=("duration_sec", "sum"),
        )
        .reset_index()
        .sort_values(
            ["speakers", "clips", "seconds"],
            ascending=[False, False, False],
        )
    )

    selected: List[int] = []
    speaker_seconds: Dict[str, float] = {}
    gender_seconds: Dict[str, float] = {}
    total = 0.0
    gender_cap = target_sec * 0.55 if balance_gender else float("inf")

    def try_add(position: int, row: pd.Series, *, enforce_gender: bool) -> bool:
        nonlocal total
        if total >= target_sec:
            return False
        client_id = str(row["client_id"])
        duration = float(row["duration_sec"])
        used = speaker_seconds.get(client_id, 0.0)
        if used + duration > cap_sec:
            return False
        gender = str(gender_by_speaker.get(client_id, "unknown"))
        if enforce_gender and balance_gender:
            other = "female" if gender == "male" else "male"
            current = gender_seconds.get(gender, 0.0)
            other_current = gender_seconds.get(other, 0.0)
            if current + duration > gender_cap and other_current < target_sec - gender_cap:
                return False
        selected.append(position)
        speaker_seconds[client_id] = used + duration
        gender_seconds[gender] = gender_seconds.get(gender, 0.0) + duration
        total += duration
        return True

    positions_by_group: Dict[tuple, List[int]] = {}
    for position, row in capped.iterrows():
        key = (row["source_file"], row["row_group"])
        positions_by_group.setdefault(key, []).append(int(position))

    for _, group_row in stats.iterrows():
        key = (group_row["source_file"], group_row["row_group"])
        for position in positions_by_group.get(key, []):
            row = capped.iloc[position]
            try_add(position, row, enforce_gender=True)
        if total >= target_sec:
            break

    if total < target_sec:
        # Fill any remaining gap without the soft gender cap.  The speaker cap
        # remains strict.
        selected_set = set(selected)
        for _, group_row in stats.iterrows():
            key = (group_row["source_file"], group_row["row_group"])
            for position in positions_by_group.get(key, []):
                if position in selected_set:
                    continue
                if try_add(position, capped.iloc[position], enforce_gender=False):
                    selected_set.add(position)
            if total >= target_sec:
                break

    return capped.iloc[selected].reset_index(drop=True)


def assign_splits(
    frame: pd.DataFrame,
    *,
    seed: int,
    min_eval_speakers: int,
) -> pd.DataFrame:
    speakers = sorted(frame["client_id"].dropna().unique().tolist())
    # Keep 5% each for val/test once a class has enough speakers; this leaves
    # roughly 90% of the class in train without weakening the speaker-disjoint
    # invariant.  Small classes still get a 90/5/5 split.
    train_pct = 90 if len(speakers) >= 30 else 80
    split_map = {
        client_id: stable_split(client_id, seed, train_pct) for client_id in speakers
    }
    frame = frame.copy()
    frame["speaker_split"] = frame["client_id"].map(split_map)

    # Guarantee a usable val/test speaker count per accent.  Moving the
    # lowest-duration train speakers keeps the train split as large as the
    # 30-speaker gate allows.
    for accent in sorted(frame["accent_class"].unique()):
        accent_speakers = sorted(
            frame.loc[frame["accent_class"] == accent, "client_id"].unique().tolist()
        )
        if len(accent_speakers) < 3:
            continue
        target_eval = min(
            max(1, min_eval_speakers),
            max(1, len(accent_speakers) // 10),
        )
        for required in ("val", "test"):
            present = sum(
                1 for client_id in accent_speakers if split_map[client_id] == required
            )
            while present < target_eval:
                current = frame[frame["accent_class"] == accent]
                durations = (
                    current[current["speaker_split"] == "train"]
                    .groupby("client_id")["duration_sec"]
                    .sum()
                    .sort_values()
                )
                candidates = [
                    client_id
                    for client_id in durations.index
                    if split_map.get(client_id) == "train"
                ]
                train_speakers = sum(
                    1
                    for client_id in accent_speakers
                    if split_map[client_id] == "train"
                )
                if not candidates or train_speakers <= 30:
                    break
                moved = candidates[0]
                split_map[moved] = required
                frame.loc[frame["client_id"] == moved, "speaker_split"] = required
                present += 1

    # A speaker must not appear in two splits anywhere in the selection.
    conflicts = (
        frame.groupby("client_id")["speaker_split"].nunique().loc[lambda s: s > 1]
    )
    if len(conflicts):
        raise RuntimeError(f"speaker split conflict: {list(conflicts.index)[:5]}")
    return frame


def summarize_class(frame: pd.DataFrame, *, target_sec: float) -> Dict[str, object]:
    if frame.empty:
        return {
            "clips": 0,
            "speakers": 0,
            "hours": 0.0,
            "train": {"clips": 0, "speakers": 0, "hours": 0.0},
            "val": {"clips": 0, "speakers": 0, "hours": 0.0},
            "test": {"clips": 0, "speakers": 0, "hours": 0.0},
            "gender_clips": {},
        }
    result: Dict[str, object] = {
        "clips": int(len(frame)),
        "speakers": int(frame["client_id"].nunique()),
        "hours": round(float(frame["duration_sec"].sum()) / 3600.0, 3),
        "target_hours": round(target_sec / 3600.0, 3),
        "gender_clips": {
            str(key): int(value)
            for key, value in frame["predicted_gender"].value_counts().items()
        },
    }
    for split in ("train", "val", "test"):
        subset = frame[frame["speaker_split"] == split]
        result[split] = {
            "clips": int(len(subset)),
            "speakers": int(subset["client_id"].nunique()),
            "hours": round(float(subset["duration_sec"].sum()) / 3600.0, 3),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--target-hours", type=float, default=2.0)
    parser.add_argument("--min-speakers", type=int, default=30)
    parser.add_argument("--max-ratio", type=float, default=1.5)
    parser.add_argument("--speaker-cap-sec", type=float, default=120.0)
    parser.add_argument("--max-clips-per-speaker", type=int, default=40)
    parser.add_argument("--min-clip-sec", type=float, default=1.0)
    parser.add_argument("--max-clip-sec", type=float, default=30.0)
    parser.add_argument("--min-utmos", type=float, default=3.0)
    parser.add_argument("--max-wer", type=float, default=0.1)
    parser.add_argument("--min-snr", type=float, default=15.0)
    parser.add_argument("--globe-splits", nargs="+", default=["train"])
    parser.add_argument(
        "--evaluation-only",
        nargs="*",
        default=[],
        choices=list(ACCENT_CLASSES),
        help="Thin classes to keep for evaluation but exclude from the training gate.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["row_group", "speaker"],
        default="row_group",
        help="row_group minimizes audio-chunk overhead; speaker is the legacy mode.",
    )
    parser.add_argument(
        "--min-eval-speakers",
        type=int,
        default=5,
        help="Preferred val/test speaker count per class (capped at 10% of speakers).",
    )
    parser.add_argument("--no-gender-balance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest.is_file() and not args.overwrite:
        print(f"[skip] {args.manifest} exists; use --overwrite to rebuild")
        return
    if not args.input.is_file():
        raise SystemExit(f"missing joined metadata: {args.input}")

    print(f"[read] {args.input}")
    frame = pd.read_parquet(args.input)
    source_rows = len(frame)

    frame = frame[frame["split"].isin(args.globe_splits)].copy()
    rows_after_globe_split = len(frame)
    frame = frame[frame["client_id"].notna() & frame["duration_sec"].notna()].copy()
    frame["duration_sec"] = pd.to_numeric(frame["duration_sec"], errors="coerce")
    frame = frame[
        (frame["duration_sec"] >= args.min_clip_sec)
        & (frame["duration_sec"] <= args.max_clip_sec)
    ].copy()

    resolved = frame["accent_self_reported"].map(resolve_accent)
    frame["accent_class"] = [value[0] for value in resolved]
    frame["accent_reason"] = [value[1] for value in resolved]
    label_reasons = frame["accent_reason"].value_counts().to_dict()
    rows_with_target_accent = int(frame["accent_class"].notna().sum())

    quality_before = len(frame)
    frame = frame[
        (pd.to_numeric(frame["utmos"], errors="coerce") >= args.min_utmos)
        & (pd.to_numeric(frame["wer"], errors="coerce") <= args.max_wer)
        & (pd.to_numeric(frame["snr"], errors="coerce") >= args.min_snr)
    ].copy()

    candidates = frame[frame["accent_class"].notna()].copy()
    target_sec = float(args.target_hours) * 3600.0
    selected: List[pd.DataFrame] = []
    selector = (
        select_class_row_group
        if args.selection_mode == "row_group"
        else select_class
    )
    for accent in ACCENT_CLASSES:
        chosen = selector(
            candidates,
            accent=accent,
            target_sec=target_sec,
            cap_sec=args.speaker_cap_sec,
            max_clips=args.max_clips_per_speaker,
            balance_gender=not args.no_gender_balance,
            seed=args.seed,
        )
        selected.append(chosen)
    manifest = pd.concat(selected, ignore_index=True)
    if manifest.empty:
        raise SystemExit("no clips selected")
    manifest = assign_splits(
        manifest,
        seed=args.seed,
        min_eval_speakers=args.min_eval_speakers,
    )

    manifest["clip_id"] = manifest["common_voice_path"].map(
        lambda path: Path(str(path)).stem
    )
    columns = [
        "clip_id",
        "common_voice_path",
        "client_id",
        "accent_class",
        "accent_self_reported",
        "speaker_split",
        "split",
        "duration_sec",
        "source_file",
        "row_group",
        "row_in_group",
        "snr",
        "utmos",
        "wer",
        "predicted_accent",
        "predicted_gender",
        "predicted_age",
        "common_voice_sentence",
    ]
    manifest = manifest[columns].sort_values(
        ["accent_class", "speaker_split", "client_id", "clip_id"]
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)

    evaluation_only = set(args.evaluation_only)
    classes: Dict[str, Dict[str, object]] = {}
    gate_failures: List[str] = []
    durations: List[float] = []
    for accent in ACCENT_CLASSES:
        subset = manifest[manifest["accent_class"] == accent]
        summary = summarize_class(subset, target_sec=target_sec)
        summary["evaluation_only"] = accent in evaluation_only
        classes[accent] = summary
        if accent in evaluation_only:
            if summary["clips"] == 0:
                gate_failures.append(f"{accent}: evaluation-only class has no clips")
            continue
        if summary["hours"] > 0:
            durations.append(float(summary["hours"]))
        if summary["hours"] < args.target_hours:
            gate_failures.append(
                f"{accent}: {summary['hours']:.3f}h < {args.target_hours:.3f}h"
            )
        if summary["speakers"] < args.min_speakers:
            gate_failures.append(
                f"{accent}: {summary['speakers']} speakers < {args.min_speakers}"
            )
        if summary["train"]["speakers"] < args.min_speakers:
            gate_failures.append(
                f"{accent}: {summary['train']['speakers']} train speakers < {args.min_speakers}"
            )
    trainable = [accent for accent in ACCENT_CLASSES if accent not in evaluation_only]
    ratio = None
    if len(durations) == len(trainable) and durations:
        ratio = max(durations) / min(durations)
        if ratio > args.max_ratio:
            gate_failures.append(
                f"max/min duration ratio {ratio:.3f} > {args.max_ratio:.3f}"
            )
    elif trainable:
        gate_failures.append("one or more trainable accent classes have no selected clips")

    row_group_yield = (
        manifest.groupby(["source_file", "row_group"])
        .agg(clips=("clip_id", "size"), seconds=("duration_sec", "sum"))
        .reset_index()
        .sort_values(["clips", "seconds"], ascending=False)
    )
    summary = {
        "input": str(args.input),
        "manifest": str(args.manifest),
        "source_rows": source_rows,
        "rows_after_globe_split": int(rows_after_globe_split),
        "rows_with_target_accent": int(rows_with_target_accent),
        "accent_resolution": {
            str(key): int(value) for key, value in label_reasons.items()
        },
        "rows_before_quality": int(quality_before),
        "candidate_rows_with_target_accent": int(len(candidates)),
        "filters": {
            "target_hours": args.target_hours,
            "min_speakers": args.min_speakers,
            "max_ratio": args.max_ratio,
            "speaker_cap_sec": args.speaker_cap_sec,
            "max_clips_per_speaker": args.max_clips_per_speaker,
            "min_eval_speakers": args.min_eval_speakers,
            "min_clip_sec": args.min_clip_sec,
            "max_clip_sec": args.max_clip_sec,
            "min_utmos": args.min_utmos,
            "max_wer": args.max_wer,
            "min_snr": args.min_snr,
            "globe_splits": args.globe_splits,
            "selection_mode": args.selection_mode,
            "gender_balance": not args.no_gender_balance,
        },
        "classes": classes,
        "evaluation_only_classes": sorted(evaluation_only),
        "selected_clips": int(len(manifest)),
        "selected_hours": round(float(manifest["duration_sec"].sum()) / 3600.0, 3),
        "selected_speakers": int(manifest["client_id"].nunique()),
        "duration_ratio": round(ratio, 4) if ratio is not None else None,
        "row_groups": int(len(row_group_yield)),
        "row_group_yield": {
            "min_clips": int(row_group_yield["clips"].min()),
            "median_clips": float(row_group_yield["clips"].median()),
            "max_clips": int(row_group_yield["clips"].max()),
            "clips_under_5_per_group": int((row_group_yield["clips"] < 5).sum()),
        },
        "gate_pass": not gate_failures,
        "gate_failures": gate_failures,
        "notes": [
            "Labels are self-reported Common Voice accents, not classifier predictions.",
            "Thresholds are fixed by the plan; a short class fails rather than being lowered.",
            "The manifest is metadata-only; audio is fetched later by source_file/row_group.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
