"""Build a deterministic, speaker-disjoint, gender-balanced VoxCeleb2 manifest.

The manifest is the contract between download, feature extraction, training,
and evaluation.  It is intentionally independent of the old fused-feature
corpus: every clip is selected from the raw VoxCeleb2 ZIP directory listing,
and every split is speaker-disjoint by construction.

Design
------
* **All dev speakers are kept.**  Speaker diversity is the most important
  property for a face-to-voice model, so the subset is formed by taking a fixed
  number of clips per speaker rather than dropping speakers.
* **Clips are sampled across videos, not just the first N utterances.**  A
  round-robin over a speaker's videos, with a stable hash ordering inside each
  video, gives deterministic visual diversity.
* **Gender balance is at the clip level.**  If one gender has more speakers,
  those speakers contribute fewer clips so the two genders contribute roughly
  the same number of training clips.
* **Splits are speaker-disjoint.**  Dev speakers are split into train/val by a
  deterministic stride within each gender.  Official VoxCeleb2 test speakers
  form the held-out test split and never appear in dev.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEV_SET = "dev"
TEST_SET = "test"
MANIFEST_COLUMNS = [
    "clip_id",
    "speaker_id",
    "video_id",
    "utterance_id",
    "split",
    "gender",
    "shard",
    "video_path",
    "set",
    "source_zip",
]


@dataclass(frozen=True)
class ClipRecord:
    """One VoxCeleb2 clip before split/sample assignment."""

    set: str
    speaker_id: str
    video_id: str
    utterance_id: str
    shard: int
    gender: str
    source_zip: str

    @property
    def clip_id(self) -> str:
        return f"{self.set}/{self.speaker_id}/{self.video_id}/{self.utterance_id}"

    @property
    def video_path(self) -> str:
        return f"{self.speaker_id}/{self.video_id}/{self.utterance_id}.mp4"

    def as_row(self, split: str) -> dict:
        return {
            "clip_id": self.clip_id,
            "speaker_id": self.speaker_id,
            "video_id": self.video_id,
            "utterance_id": self.utterance_id,
            "split": split,
            "gender": self.gender,
            "shard": int(self.shard),
            "video_path": self.video_path,
            "set": self.set,
            "source_zip": self.source_zip,
        }


def stable_int(text: str) -> int:
    """Stable 64-bit integer hash (independent of Python's hash seed)."""
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def parse_zip_member(
    name: str,
    *,
    set_name: str,
    shard: int,
    source_zip: str,
    gender: str = "u",
) -> Optional[ClipRecord]:
    """Parse a VoxCeleb2 ZIP member into a ``ClipRecord``.

    Accepted layouts:
      * ``dev/mp4/id00019/videoId/00001.mp4``
      * ``test/mp4/id00019/videoId/00001.mp4``
      * ``mp4/id00019/videoId/00001.mp4`` (official test ZIP)
    """
    normalized = str(name).replace("\\", "/").strip("/")
    if not normalized.lower().endswith(".mp4"):
        return None
    parts = normalized.split("/")
    try:
        mp4_index = parts.index("mp4")
    except ValueError:
        return None
    tail = parts[mp4_index + 1 :]
    if len(tail) != 3:
        return None
    speaker_id, video_id, filename = tail
    if not (speaker_id.startswith("id") and speaker_id[2:].isdigit()):
        return None
    if not video_id:
        return None
    utterance_id = filename[:-4]
    if not utterance_id.isdigit():
        return None
    return ClipRecord(
        set=set_name,
        speaker_id=speaker_id,
        video_id=video_id,
        utterance_id=utterance_id,
        shard=int(shard),
        gender=str(gender or "u").lower(),
        source_zip=source_zip,
    )


def load_meta_genders(meta_path: str | Path) -> dict[str, str]:
    """Read ``vox2_meta.csv`` and return speaker -> m/f/u."""
    path = Path(meta_path)
    if not path.exists():
        raise FileNotFoundError(f"VoxCeleb2 metadata not found: {path}")
    meta = pd.read_csv(path)
    meta.columns = [str(c).strip() for c in meta.columns]
    id_col = next((c for c in meta.columns if c.lower() == "voxceleb2 id"), None)
    gender_col = next((c for c in meta.columns if c.lower() == "gender"), None)
    if id_col is None or gender_col is None:
        raise ValueError(
            f"Expected 'VoxCeleb2 ID' and 'Gender' columns in {path}; "
            f"got {meta.columns.tolist()}"
        )
    genders: dict[str, str] = {}
    for speaker, gender in zip(meta[id_col], meta[gender_col]):
        key = str(speaker).strip()
        value = str(gender).strip().lower()
        if key:
            genders[key] = value if value in {"m", "f"} else "u"
    return genders


def assign_speaker_splits(
    speaker_genders: Mapping[str, str],
    val_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign each dev speaker to train or val, stratified by gender.

    The split is a deterministic stride over sorted speaker IDs within each
    gender.  No random seed is involved, so rerunning the builder produces the
    same split on any machine.
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    step = max(2, int(round(1.0 / float(val_fraction))))
    assignments: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for speaker, gender in speaker_genders.items():
        groups.setdefault(str(gender), []).append(str(speaker))

    for gender, speakers in sorted(groups.items()):
        ordered = sorted(speakers)
        if len(ordered) < 2:
            for speaker in ordered:
                assignments[speaker] = "train"
            continue
        val_speakers = set(ordered[::step])
        # Always leave at least one train speaker in the group.
        if len(val_speakers) >= len(ordered):
            val_speakers = set(ordered[1::step])
        for speaker in ordered:
            assignments[speaker] = "val" if speaker in val_speakers else "train"

    train = {s for s, split in assignments.items() if split == "train"}
    val = {s for s, split in assignments.items() if split == "val"}
    if train & val:
        raise AssertionError("train/val speaker overlap")
    return assignments


def sample_speaker_clips(
    clips: Sequence[ClipRecord],
    max_clips: int,
    *,
    seed: str = "voix-v2",
) -> list[ClipRecord]:
    """Deterministically sample clips across a speaker's videos.

    Clips are grouped by video and ordered by a stable hash inside each video.
    A round-robin over videos then picks at most ``max_clips`` clips while
    covering as many distinct videos as possible.  This avoids selecting only
    the first video or only the lowest-numbered utterances.
    """
    if max_clips <= 0 or max_clips >= len(clips):
        return sorted(clips, key=lambda c: c.clip_id)

    by_video: dict[str, list[ClipRecord]] = {}
    for clip in clips:
        by_video.setdefault(clip.video_id, []).append(clip)
    for video_id, items in by_video.items():
        by_video[video_id] = sorted(
            items, key=lambda c: stable_int(f"{seed}:{c.clip_id}")
        )

    videos = sorted(by_video)
    selected: list[ClipRecord] = []
    round_index = 0
    while len(selected) < max_clips:
        added = False
        for video_id in videos:
            items = by_video[video_id]
            if round_index < len(items):
                selected.append(items[round_index])
                added = True
                if len(selected) >= max_clips:
                    break
        if not added:
            break
        round_index += 1
    return sorted(selected, key=lambda c: c.clip_id)


def clips_per_speaker_for_gender(
    speaker_counts: Mapping[str, int],
    target: int,
    *,
    min_clips: int = 4,
    balance: bool = True,
) -> dict[str, int]:
    """Return a per-gender clip cap that balances total clip counts.

    The gender with fewer speakers keeps ``target`` clips per speaker; the
    larger gender is scaled down so the two genders contribute approximately
    the same number of clips.  ``min_clips`` prevents the larger gender from
    being reduced to an unusably small sample.
    """
    target = max(1, int(target))
    known = {g: int(n) for g, n in speaker_counts.items() if int(n) > 0}
    if not balance or len(known) < 2:
        return {g: target for g in speaker_counts}

    smallest_gender = min(known, key=lambda g: (known[g], g))
    smallest_count = known[smallest_gender]
    caps: dict[str, int] = {}
    for gender, count in speaker_counts.items():
        if count <= 0:
            caps[gender] = target
            continue
        if gender == smallest_gender:
            caps[gender] = target
        else:
            caps[gender] = max(
                min_clips,
                min(target, int(round(target * smallest_count / count))),
            )
    return caps


def build_manifest(
    records: Iterable[ClipRecord],
    *,
    val_fraction: float = 0.1,
    max_clips_per_speaker: int = 8,
    balance_gender: bool = True,
    min_clips_per_speaker: int = 4,
    seed: str = "voix-v2",
) -> pd.DataFrame:
    """Build the final manifest with splits and balanced clip sampling."""
    records = list(records)
    if not records:
        raise ValueError("No clip records were provided")

    dev_records = [r for r in records if r.set == DEV_SET]
    test_records = [r for r in records if r.set == TEST_SET]
    if not dev_records:
        raise ValueError("No dev records were provided")
    if not test_records:
        raise ValueError("No official test records were provided")

    dev_speakers = {r.speaker_id for r in dev_records}
    test_speakers = {r.speaker_id for r in test_records}
    overlap = dev_speakers & test_speakers
    if overlap:
        raise AssertionError(f"dev/test speaker overlap: {sorted(overlap)[:5]}")

    dev_genders = {}
    for r in dev_records:
        dev_genders.setdefault(r.speaker_id, r.gender)
    assignments = assign_speaker_splits(dev_genders, val_fraction=val_fraction)

    # Group records by split and speaker.
    grouped: dict[tuple[str, str], list[ClipRecord]] = {}
    for record in dev_records:
        split = assignments[record.speaker_id]
        grouped.setdefault((split, record.speaker_id), []).append(record)
    for record in test_records:
        grouped.setdefault(("test", record.speaker_id), []).append(record)

    # Compute gender-balanced clip caps per split.
    caps: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        speaker_genders = {
            speaker: records_[0].gender
            for (split_, speaker), records_ in grouped.items()
            if split_ == split
        }
        counts: dict[str, int] = {}
        for gender in speaker_genders.values():
            counts[gender] = counts.get(gender, 0) + 1
        caps[split] = clips_per_speaker_for_gender(
            counts,
            target=max_clips_per_speaker,
            min_clips=min_clips_per_speaker,
            balance=balance_gender,
        )

    rows: list[dict] = []
    for (split, speaker), speaker_clips in sorted(grouped.items()):
        gender = speaker_clips[0].gender
        cap = caps[split].get(gender, max_clips_per_speaker)
        selected = sample_speaker_clips(speaker_clips, cap, seed=seed)
        rows.extend(record.as_row(split) for record in selected)

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame = frame.sort_values(
        ["set", "shard", "speaker_id", "video_id", "utterance_id"]
    ).reset_index(drop=True)

    # Hard invariants: no duplicate clips, no speaker across splits.
    if frame["clip_id"].duplicated().any():
        duplicates = frame.loc[frame["clip_id"].duplicated(), "clip_id"].head(5)
        raise AssertionError(f"duplicate clips in manifest: {duplicates.tolist()}")
    speakers_per_split = frame.groupby("split")["speaker_id"].nunique()
    if frame.groupby("speaker_id")["split"].nunique().max() > 1:
        raise AssertionError("a speaker appears in more than one split")
    if speakers_per_split.get("train", 0) <= 0 or speakers_per_split.get("val", 0) <= 0:
        raise AssertionError("train and val must both be non-empty")
    if speakers_per_split.get("test", 0) <= 0:
        raise AssertionError("test must be non-empty")

    return frame


def manifest_summary(
    frame: pd.DataFrame,
    *,
    val_fraction: float,
    max_clips_per_speaker: int,
    balance_gender: bool,
    manifest_path: Optional[str | Path] = None,
) -> dict:
    """Compute a reproducible summary of a manifest."""
    summary: dict = {
        "clips": int(len(frame)),
        "speakers": int(frame["speaker_id"].nunique()),
        "val_fraction_requested": float(val_fraction),
        "max_clips_per_speaker": int(max_clips_per_speaker),
        "balance_gender": bool(balance_gender),
        "splits": {},
        "shards": {},
        "gender_by_split": {},
        "clips_per_speaker": {},
    }
    for split, group in frame.groupby("split"):
        summary["splits"][split] = {
            "clips": int(len(group)),
            "speakers": int(group["speaker_id"].nunique()),
            "videos": int(group["video_id"].nunique()),
            "gender": {str(k): int(v) for k, v in group["gender"].value_counts().items()},
        }
        summary["gender_by_split"][split] = {
            str(g): int(n) for g, n in group.groupby("gender")["clip_id"].count().items()
        }
        summary["clips_per_speaker"][split] = {
            "min": int(group.groupby("speaker_id").size().min()),
            "max": int(group.groupby("speaker_id").size().max()),
            "mean": float(group.groupby("speaker_id").size().mean()),
        }
    for shard, group in frame.groupby("shard"):
        summary["shards"][str(int(shard))] = {
            "clips": int(len(group)),
            "speakers": int(group["speaker_id"].nunique()),
        }
    if manifest_path is not None:
        path = Path(manifest_path)
        if path.exists():
            summary["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return summary


def save_manifest(
    frame: pd.DataFrame,
    output_path: str | Path,
    summary_path: Optional[str | Path] = None,
    **summary_kwargs,
) -> dict:
    """Write the manifest CSV and an optional JSON summary."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = manifest_summary(frame, manifest_path=output, **summary_kwargs)
    if summary_path is not None:
        summary_file = Path(summary_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a manifest CSV."""
    frame = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    return frame
