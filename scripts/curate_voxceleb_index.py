"""VoxCeleb2 speaker and utterance index curation script.

Generates a deterministic curated_index.csv selecting 2,000 speakers
(1,000 female + 1,000 male) and 50 cross-session utterances per speaker
from the VoxCeleb2 metadata files.

Usage:
    python scripts/curate_voxceleb_index.py \
        --meta   /path/to/vox2_meta.csv \
        --txt    /path/to/vox2_dev_txt/ \
        --output data/processed/curated_index.csv \
        --seed   42 \
        --speakers 2000 \
        --utterances 50

Output CSV columns:
    speaker_id   : VoxCeleb2 speaker ID  (e.g. id00001)
    video_id     : YouTube video ID      (e.g. 0Xr3JLQKWV8)
    utterance_id : Utterance segment ID  (e.g. 00001)
    gender       : m / f
    nationality  : Country string from VoxCeleb2 meta

Selection strategy (see Phase 2 plan, Section 3.5):
  1. Filter speakers with >= min_utterances_per_speaker clips available.
  2. Stratified sample: exactly target_speakers/2 male + female.
  3. Per speaker: sample utterances ACROSS different video sessions
     (cross-session sampling to maximise intra-speaker acoustic variation).
  4. Deterministic: fixed numpy seed produces identical index every run.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Minimum utterances a speaker must have to qualify for selection
# ---------------------------------------------------------------------------
MIN_UTTERANCES_PER_SPEAKER: int = 50


def load_meta(meta_path: str) -> dict[str, dict]:
    """Load vox2_meta.csv and return dict keyed by speaker_id.

    VoxCeleb2 meta format (tab-separated):
        VoxCeleb2 ID  | VGGFace2 ID | Gender | Nationality | Set
    """
    speakers: dict[str, dict] = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("VoxCeleb"):
                continue  # skip header
            parts = [p.strip() for p in row]
            if len(parts) < 4:
                continue
            spk_id   = parts[0]
            gender   = parts[2].lower()          # 'm' or 'f'
            nation   = parts[3] if len(parts) > 3 else "unknown"
            split    = parts[4] if len(parts) > 4 else "dev"
            if split.lower() != "dev":
                continue  # use dev set only for training
            speakers[spk_id] = {"gender": gender, "nationality": nation}
    return speakers


def scan_utterances(txt_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Scan VoxCeleb2 txt directory to build utterance inventory.

    Returns dict: speaker_id -> list of (video_id, utterance_id) tuples.

    txt_dir structure expected:
        txt_dir/
            id00001/
                AbcXyz123/
                    00001.txt
                    00002.txt
                    ...
    """
    inventory: dict[str, list[tuple[str, str]]] = defaultdict(list)
    txt_root = Path(txt_dir)
    for spk_dir in txt_root.iterdir():
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        for vid_dir in spk_dir.iterdir():
            if not vid_dir.is_dir():
                continue
            vid_id = vid_dir.name
            for utt_file in vid_dir.glob("*.txt"):
                utt_id = utt_file.stem
                inventory[spk_id].append((vid_id, utt_id))
    return dict(inventory)


def cross_session_sample(
    utterances: list[tuple[str, str]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Sample n utterances spread across different video sessions.

    Groups utterances by video_id, then samples proportionally from
    each group to maximise cross-session diversity.

    Args:
        utterances: List of (video_id, utterance_id) for a single speaker.
        n:          Number of utterances to sample.
        rng:        NumPy random generator for determinism.

    Returns:
        List of n (video_id, utterance_id) tuples.
    """
    by_video: dict[str, list[str]] = defaultdict(list)
    for vid_id, utt_id in utterances:
        by_video[vid_id].append(utt_id)

    video_ids = sorted(by_video.keys())
    n_videos = len(video_ids)
    selected: list[tuple[str, str]] = []

    # Distribute quota proportionally across videos, at least 1 per video
    quotas = [max(1, n // n_videos)] * n_videos
    remainder = n - sum(quotas)
    for i in range(remainder):
        quotas[i % n_videos] += 1

    for vid_id, quota in zip(video_ids, quotas):
        pool = by_video[vid_id]
        take = min(quota, len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        selected.extend((vid_id, pool[idx]) for idx in chosen)

    # If we couldn't fill quota (some videos had <1 clip), top up from any video
    if len(selected) < n:
        flat = [(v, u) for v in video_ids for u in by_video[v]
                if (v, u) not in set(selected)]
        if flat:
            extra = rng.choice(len(flat), size=min(n - len(selected), len(flat)),
                               replace=False)
            selected.extend(flat[i] for i in extra)

    return selected[:n]


def curate_index(
    meta_path: str,
    txt_dir: str,
    output_path: str,
    seed: int = 42,
    target_speakers: int = 2000,
    utterances_per_speaker: int = 50,
) -> int:
    """Generate the curated training index CSV.

    Args:
        meta_path:              Path to vox2_meta.csv.
        txt_dir:                Path to VoxCeleb2 txt/ directory.
        output_path:            Output CSV file path.
        seed:                   Random seed for reproducibility.
        target_speakers:        Total speakers to select (half male, half female).
        utterances_per_speaker: Utterances per speaker (cross-session sampled).

    Returns:
        Total number of rows written to the output CSV.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    print(f"Loading speaker metadata from {meta_path}...")
    speakers = load_meta(meta_path)
    print(f"  Found {len(speakers)} dev speakers in metadata.")

    print(f"Scanning utterance inventory from {txt_dir}...")
    inventory = scan_utterances(txt_dir)
    print(f"  Found utterances for {len(inventory)} speakers.")

    # Filter speakers with enough utterances
    qualified: dict[str, dict] = {}
    for spk_id, meta in speakers.items():
        if spk_id in inventory and len(inventory[spk_id]) >= MIN_UTTERANCES_PER_SPEAKER:
            qualified[spk_id] = meta

    print(f"  {len(qualified)} speakers have >= {MIN_UTTERANCES_PER_SPEAKER} utterances.")

    # Stratified split by gender
    males   = [s for s, m in qualified.items() if m["gender"] == "m"]
    females = [s for s, m in qualified.items() if m["gender"] == "f"]
    n_per_gender = target_speakers // 2

    if len(males) < n_per_gender:
        raise ValueError(f"Not enough male speakers: {len(males)} < {n_per_gender}")
    if len(females) < n_per_gender:
        raise ValueError(f"Not enough female speakers: {len(females)} < {n_per_gender}")

    sel_males   = rng.choice(sorted(males),   size=n_per_gender, replace=False).tolist()
    sel_females = rng.choice(sorted(females), size=n_per_gender, replace=False).tolist()
    selected_speakers = sel_males + sel_females
    print(f"Selected {len(sel_males)} male + {len(sel_females)} female speakers.")

    # Build output rows
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spk_id in selected_speakers:
        utterances = inventory[spk_id]
        sampled = cross_session_sample(utterances, utterances_per_speaker, rng)
        meta = qualified[spk_id]
        for vid_id, utt_id in sampled:
            rows.append({
                "speaker_id":   spk_id,
                "video_id":     vid_id,
                "utterance_id": utt_id,
                "gender":       meta["gender"],
                "nationality":  meta["nationality"],
            })

    # Write CSV
    fieldnames = ["speaker_id", "video_id", "utterance_id", "gender", "nationality"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate VoxCeleb2 training index (cross-session, stratified)."
    )
    parser.add_argument("--meta",        required=True, help="Path to vox2_meta.csv")
    parser.add_argument("--txt",         required=True, help="Path to vox2 txt/ directory")
    parser.add_argument("--output",      default="data/processed/curated_index.csv")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--speakers",    type=int, default=2000)
    parser.add_argument("--utterances",  type=int, default=50)
    args = parser.parse_args()

    total = curate_index(
        meta_path=args.meta,
        txt_dir=args.txt,
        output_path=args.output,
        seed=args.seed,
        target_speakers=args.speakers,
        utterances_per_speaker=args.utterances,
    )
    print(f"Curation complete: {total} training samples indexed.")


if __name__ == "__main__":
    main()