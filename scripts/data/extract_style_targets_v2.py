"""Extract per-speaker StyleTTS2 style targets from VoxCeleb2 audio.

The legacy ``data/processed/style_pairs.pt`` artifact has only 497 rows and no
speaker labels.  Matching those rows to the 6,111 V2 speakers by ECAPA cosine
is noisy: even an oracle linear map from the raw voice embedding to the matched
target collapses to ~0.92 across-speaker cosine.  The adapter therefore learns
the target mean instead of a voice-dependent style.

This script computes the 256-D ``[timbre | prosody]`` style vector directly
from each V2 speaker's own audio using the frozen StyleTTS2 style and predictor
encoders.  A single clip is a noisy speaker estimate (within-speaker style
cosine 0.711 vs 0.627 across speakers), so the target is the mean over several
clips per speaker; split-half reliability of that mean is 0.85.

Usage:
    python scripts/data/extract_style_targets_v2.py \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --videos data/raw/voxceleb2/clips \
        --clips-per-speaker 5 \
        --output data/processed/style_targets_voxceleb2_5clip.pt
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import h5py
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.video_io import extract_audio  # noqa: E402


def _decode(values) -> set[str]:
    return {
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    }


def _save(
    output: Path,
    styles: List[torch.Tensor],
    speaker_ids: List[str],
    clip_ids: List[str],
    source_clip_ids: List[List[str]],
    n_clips: List[int],
    splits: List[str],
    genders: List[str],
    clips_per_speaker: int,
) -> None:
    payload = {
        "style": torch.stack(styles).float(),
        "speaker_ids": speaker_ids,
        "clip_ids": clip_ids,
        "source_clip_ids": source_clip_ids,
        "n_clips": n_clips,
        "split": splits,
        "gender": genders,
        "clips_per_speaker": int(clips_per_speaker),
        "source": (
            "StyleTTS2 style_encoder + predictor_encoder; "
            f"mean over up to {clips_per_speaker} clips per speaker"
        ),
        "schema_version": "StyleTargetV2",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-speaker StyleTTS2 style targets from VoxCeleb2"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "features_v2.h5",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=ROOT / "data" / "raw" / "voxceleb2" / "clips",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "style_targets_voxceleb2_5clip.pt",
    )
    parser.add_argument(
        "--clips-per-speaker",
        type=int,
        default=5,
        help="Maximum clips averaged into one speaker target.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional limit for a smoke run; writes the same artifact format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output instead of resuming by speaker id.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clips_per_speaker < 1:
        raise ValueError("--clips-per-speaker must be at least 1")

    index = pd.read_csv(args.index, dtype=str)
    with h5py.File(args.features, "r") as store:
        done = _decode(store["clip_ids"][:])
    rows = index[index.clip_id.isin(done)].copy()
    if rows.empty:
        raise ValueError("No extracted clips found; run extraction first")

    # Deterministic clip selection: sort by clip id, keep the first N clips.
    rows = rows.sort_values(["speaker_id", "clip_id"])
    rows = rows.groupby("speaker_id", sort=True).head(
        int(args.clips_per_speaker)
    )
    grouped: Dict[str, list] = {
        str(speaker): group.to_dict("records")
        for speaker, group in rows.groupby("speaker_id", sort=True)
    }

    existing: Dict[str, int] = {}
    styles: List[torch.Tensor] = []
    speaker_ids: List[str] = []
    clip_ids: List[str] = []
    source_clip_ids: List[List[str]] = []
    n_clips: List[int] = []
    splits: List[str] = []
    genders: List[str] = []
    if args.output.exists() and not args.overwrite:
        previous = torch.load(args.output, map_location="cpu", weights_only=False)
        if str(previous.get("schema_version", "")) != "StyleTargetV2":
            raise ValueError(
                f"{args.output} is not a StyleTargetV2 artifact; pass --overwrite"
            )
        previous_clips = int(previous.get("clips_per_speaker", 1))
        if previous_clips != int(args.clips_per_speaker):
            raise ValueError(
                f"{args.output} was built with clips_per_speaker={previous_clips}, "
                f"not {args.clips_per_speaker}; pass --overwrite"
            )
        styles = list(previous["style"].float())
        speaker_ids = list(previous["speaker_ids"])
        clip_ids = list(previous["clip_ids"])
        source_clip_ids = [list(v) for v in previous.get("source_clip_ids", [])]
        n_clips = [int(v) for v in previous.get("n_clips", [1] * len(styles))]
        splits = list(previous["split"])
        genders = list(previous["gender"])
        existing = {speaker: i for i, speaker in enumerate(speaker_ids)}
        print(f"Resuming {len(existing)} speakers from {args.output}")

    pending = [speaker for speaker in grouped if speaker not in existing]
    if args.max_speakers is not None:
        pending = pending[: int(args.max_speakers)]
    if not pending:
        print("Nothing to do; all requested speakers are already present.")
        return

    total_clips = sum(len(grouped[speaker]) for speaker in pending)
    print("=" * 72)
    print("StyleTTS2 target extraction - VoxCeleb2")
    print("=" * 72)
    print(f"Index: {args.index} ({len(grouped):,} speakers with extracted clips)")
    print(f"Videos: {args.videos}")
    print(f"Output: {args.output}")
    print(
        f"Pending: {len(pending):,} speakers / {total_clips:,} clips "
        f"(up to {args.clips_per_speaker} clips per speaker)"
    )

    from styletts2 import tts

    load_start = time.time()
    model = tts.StyleTTS2()
    print(f"StyleTTS2 loaded in {time.time() - load_start:.1f}s on {model.device}")

    failures: List[tuple[str, str, str]] = []
    processed_clips = 0
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="voix_style_targets_") as temporary:
        temporary_path = Path(temporary)
        for number, speaker in enumerate(pending, start=1):
            speaker_styles: List[torch.Tensor] = []
            used_clips: List[str] = []
            first_row = grouped[speaker][0]
            for clip_number, row in enumerate(grouped[speaker]):
                video = Path(str(row["video_path"]))
                if not video.is_absolute():
                    video = args.videos / video
                wav = temporary_path / f"{number:06d}_{clip_number:02d}.wav"
                processed_clips += 1
                if not video.exists():
                    failures.append(
                        (speaker, str(row["clip_id"]), f"missing_video:{video}")
                    )
                    continue
                if not extract_audio(
                    video,
                    wav,
                    sample_rate=args.sample_rate,
                    max_seconds=args.max_seconds,
                ):
                    failures.append(
                        (speaker, str(row["clip_id"]), "audio_extract_failed")
                    )
                    continue
                try:
                    with torch.no_grad():
                        style = model.compute_style(str(wav))
                    style = style.detach().cpu().float().reshape(-1)
                    if style.numel() != 256:
                        raise ValueError(
                            f"expected 256-D style, got {style.numel()}"
                        )
                except Exception as exc:  # noqa: BLE001 - report the exact clip
                    failures.append(
                        (
                            speaker,
                            str(row["clip_id"]),
                            f"style_failed:{type(exc).__name__}:{exc}",
                        )
                    )
                    continue
                speaker_styles.append(style)
                used_clips.append(str(row["clip_id"]))

            if not speaker_styles:
                continue
            styles.append(torch.stack(speaker_styles).mean(0))
            speaker_ids.append(speaker)
            clip_ids.append(str(first_row["clip_id"]))
            source_clip_ids.append(used_clips)
            n_clips.append(len(used_clips))
            splits.append(str(first_row["split"]))
            genders.append(str(first_row["gender"]))

            if number % 200 == 0:
                elapsed = time.time() - started
                speaker_rate = number / max(elapsed, 1e-9)
                clip_rate = processed_clips / max(elapsed, 1e-9)
                eta = (len(pending) - number) / max(speaker_rate, 1e-9) / 60.0
                _save(
                    args.output,
                    styles,
                    speaker_ids,
                    clip_ids,
                    source_clip_ids,
                    n_clips,
                    splits,
                    genders,
                    args.clips_per_speaker,
                )
                print(
                    f"  {number}/{len(pending)} speakers | "
                    f"{processed_clips}/{total_clips} clips | "
                    f"{clip_rate:.1f} clips/s | ETA {eta:.1f} min | "
                    f"failures={len(failures)}"
                )

    _save(
        args.output,
        styles,
        speaker_ids,
        clip_ids,
        source_clip_ids,
        n_clips,
        splits,
        genders,
        args.clips_per_speaker,
    )
    elapsed = time.time() - started
    print("-" * 72)
    print(
        f"Done: {len(styles)} speakers total | {len(failures)} clip failures | "
        f"{elapsed / 60:.1f} min"
    )
    if failures:
        print("Failure sample:")
        for speaker, clip, reason in failures[:10]:
            print(f"  {speaker}/{clip}: {reason}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
