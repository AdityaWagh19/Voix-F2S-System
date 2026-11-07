"""Build a speaker-disjoint VoxCeleb2 corpus for StyleTTS2 LoRA fine-tuning.

The StyleTTS2 LibriTTS checkpoint is an audiobook model.  This script prepares
real conversational speech and real ASR transcripts so the prosody-diffusion
branch can be adapted with LoRA.  It does not touch the identity path: the CFVN,
CVAE, and style adapter are not loaded or modified.

What it does:

1. Selects gender-balanced speakers from the existing speaker-disjoint V2
   split (train or val), deterministically by SHA-256.
2. Picks N target clips and one different-video reference clip per speaker.
3. Extracts 24 kHz mono WAV from the VoxCeleb2 MP4 with the bundled ffmpeg.
4. Transcribes with Whisper and records the real ASR confidence fields.
5. Filters on duration, text length, average log-probability, compression
   ratio, and no-speech probability.
6. Writes a manifest, a transcript cache, and an honest corpus report.

Everything is resumable: existing WAVs and cached transcripts are reused.

Usage:
    python scripts/data/build_styletts2_lora_corpus.py --train-speakers 300 \
        --val-speakers 50 --clips-per-speaker 2 --whisper-model base
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv"
DEFAULT_CLIPS = ROOT / "data" / "raw" / "voxceleb2" / "clips"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "styletts2_lora"

MIN_SPEAKER_CLIPS = 3  # two targets plus one distinct reference

MANIFEST_COLUMNS = [
    "clip_id",
    "speaker_id",
    "split",
    "gender",
    "wav_path",
    "ref_clip_id",
    "ref_wav_path",
    "duration_sec",
    "text",
    "whisper_model",
    "avg_logprob",
    "no_speech_prob",
    "compression_ratio",
]


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def select_speakers(
    index: pd.DataFrame,
    split: str,
    count: int,
    seed: str,
    min_clips: int = MIN_SPEAKER_CLIPS,
) -> List[str]:
    """Pick a deterministic, gender-balanced speaker list for one split.

    Speakers need at least ``min_clips`` clips in distinct videos so every
    target can have a different-video reference.  If one gender is short, the
    remaining slots are filled from the other pool; the report records the
    actual counts rather than pretending the target was met.
    """
    frame = index[index["split"] == split]
    if frame.empty:
        return []
    counts = frame.groupby("speaker_id")["clip_id"].size()
    videos = frame.groupby("speaker_id")["video_id"].nunique()
    eligible = sorted(
        speaker
        for speaker in counts.index
        if counts[speaker] >= min_clips and videos[speaker] >= min_clips
    )
    gender = frame.groupby("speaker_id")["gender"].first()
    female = [s for s in eligible if str(gender[s]).lower().startswith("f")]
    male = [s for s in eligible if str(gender[s]).lower().startswith("m")]
    female.sort(key=lambda speaker: stable_key(seed, f"{split}:f:{speaker}"))
    male.sort(key=lambda speaker: stable_key(seed, f"{split}:m:{speaker}"))

    target_female = count // 2
    target_male = count - target_female
    chosen = female[:target_female] + male[:target_male]
    used = set(chosen)
    for pool in (female, male):
        if len(chosen) >= count:
            break
        for speaker in pool:
            if speaker not in used:
                chosen.append(speaker)
                used.add(speaker)
                if len(chosen) >= count:
                    break
    return chosen


def select_clips(
    index: pd.DataFrame,
    speaker: str,
    targets: int,
    seed: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return target rows plus one reference from a different video."""
    frame = index[(index["split"].isin(["train", "val"])) & (index["speaker_id"] == speaker)]
    frame = frame.copy()
    frame["_key"] = [stable_key(seed, f"{speaker}:{clip}") for clip in frame["clip_id"]]
    frame = frame.sort_values(["_key", "clip_id"])

    chosen: List[Dict[str, Any]] = []
    used_videos: set[str] = set()
    for row in frame.to_dict("records"):
        video = str(row["video_id"])
        if video in used_videos:
            continue
        chosen.append(row)
        used_videos.add(video)
        if len(chosen) == targets + 1:
            break
    if len(chosen) < targets + 1:
        return [], None
    return chosen[:targets], chosen[targets]


def extract_wav(
    source: Path,
    destination: Path,
    ffmpeg: str,
    sample_rate: int = 24000,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


def safe_name(clip_id: str) -> str:
    return clip_id.replace("/", "__") + ".wav"


def display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_transcript_cache(path: Path, whisper_model: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("whisper_model") != whisper_model:
                continue
            clip_id = row.get("clip_id")
            if clip_id:
                cache[str(clip_id)] = row
    return cache


def append_transcript(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def transcribe_wav(wav_path: Path, model, device: str) -> Dict[str, Any]:
    import librosa

    wave, _ = librosa.load(str(wav_path), sr=16000)
    result = model.transcribe(
        wave,
        language="en",
        fp16=device == "cuda",
        condition_on_previous_text=False,
        verbose=False,
    )
    segments = result.get("segments") or []
    if segments:
        total = sum(max(0.01, float(s.get("end", 0)) - float(s.get("start", 0))) for s in segments)
        avg_logprob = sum(
            float(s.get("avg_logprob", -10.0))
            * max(0.01, float(s.get("end", 0)) - float(s.get("start", 0)))
            for s in segments
        ) / max(total, 1e-6)
        no_speech = max(float(s.get("no_speech_prob", 0.0)) for s in segments)
        compression = max(float(s.get("compression_ratio", 1.0)) for s in segments)
    else:
        avg_logprob = -10.0
        no_speech = 1.0
        compression = 1.0
    return {
        "text": str(result.get("text", "")).strip(),
        "language": result.get("language", "en"),
        "avg_logprob": round(avg_logprob, 4),
        "no_speech_prob": round(no_speech, 4),
        "compression_ratio": round(compression, 4),
        "segments": len(segments),
    }


def quality_reason(
    record: Dict[str, Any],
    *,
    min_duration: float,
    max_duration: float,
    min_logprob: float,
    max_compression: float,
    max_no_speech: float,
    max_chars: int,
) -> Optional[str]:
    """Return a drop reason, or None when the clip passes every filter."""
    duration = float(record["duration_sec"])
    if duration < min_duration:
        return "too_short"
    if duration > max_duration:
        return "too_long"
    text = str(record["text"]).strip()
    if len(text.split()) < 3:
        return "too_few_words"
    if len(text) > max_chars:
        return "text_too_long"
    if float(record["avg_logprob"]) < min_logprob:
        return "low_logprob"
    if float(record["compression_ratio"]) > max_compression:
        return "high_compression"
    if float(record["no_speech_prob"]) > max_no_speech:
        return "no_speech"
    return None


def build_manifest(
    index: pd.DataFrame,
    *,
    clips_root: Path,
    output_dir: Path,
    train_speakers: int,
    val_speakers: int,
    clips_per_speaker: int,
    whisper_model: str,
    device: str,
    seed: str,
    min_duration: float,
    max_duration: float,
    min_logprob: float,
    max_compression: float,
    max_no_speech: float,
    max_chars: int,
    ffmpeg: str,
    whisper_model_obj,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    import soundfile as sf

    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "transcripts.jsonl"
    cache = load_transcript_cache(cache_path, whisper_model)

    selections: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]] = []
    for split, count in (("train", train_speakers), ("val", val_speakers)):
        for speaker in select_speakers(index, split, count, seed):
            targets, reference = select_clips(index, speaker, clips_per_speaker, seed)
            if reference is None:
                continue
            selections.append((split, targets, reference))

    rows: List[Dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    started = time.time()
    total_targets = sum(len(targets) for _, targets, _ in selections)

    def ensure_wav(row: Dict[str, Any]) -> Path:
        clip_id = str(row["clip_id"])
        destination = wav_dir / safe_name(clip_id)
        if not destination.is_file() or destination.stat().st_size < 1000:
            source = clips_root / str(row["video_path"])
            if not source.is_file():
                raise FileNotFoundError(f"missing source video: {source}")
            extract_wav(source, destination, ffmpeg)
        return destination

    processed = 0
    for split, targets, reference in selections:
        try:
            reference_path = ensure_wav(reference)
        except Exception as exc:  # per-speaker guard
            drop_reasons[f"error:{type(exc).__name__}"] += len(targets)
            processed += len(targets)
            continue
        for target in targets:
            processed += 1
            clip_id = str(target["clip_id"])
            try:
                wav_path = ensure_wav(target)
                info = sf.info(str(wav_path))
                duration = float(info.frames) / float(info.samplerate)
                cached = cache.get(clip_id)
                if cached is None:
                    transcript = transcribe_wav(wav_path, whisper_model_obj, device)
                    transcript.update(
                        {
                            "clip_id": clip_id,
                            "whisper_model": whisper_model,
                            "duration_sec": round(duration, 4),
                        }
                    )
                    append_transcript(cache_path, transcript)
                    cache[clip_id] = transcript
                else:
                    transcript = cached
                record = {
                    "clip_id": clip_id,
                    "speaker_id": str(target["speaker_id"]),
                    "split": split,
                    "gender": str(target.get("gender", "")),
                    "wav_path": display_path(wav_path),
                    "ref_clip_id": str(reference["clip_id"]),
                    "ref_wav_path": display_path(reference_path),
                    "duration_sec": round(duration, 4),
                    "text": str(transcript.get("text", "")).strip(),
                    "whisper_model": whisper_model,
                    "avg_logprob": float(transcript.get("avg_logprob", -10.0)),
                    "no_speech_prob": float(transcript.get("no_speech_prob", 1.0)),
                    "compression_ratio": float(transcript.get("compression_ratio", 1.0)),
                }
                reason = quality_reason(
                    record,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_logprob=min_logprob,
                    max_compression=max_compression,
                    max_no_speech=max_no_speech,
                    max_chars=max_chars,
                )
                if reason is not None:
                    drop_reasons[reason] += 1
                    continue
                rows.append(record)
            except Exception as exc:  # per-clip guard, reported not hidden
                drop_reasons[f"error:{type(exc).__name__}"] += 1
            if processed % 25 == 0 or processed == total_targets:
                print(
                    f"[corpus] {processed}/{total_targets} targets, "
                    f"{len(rows)} kept, {sum(drop_reasons.values())} dropped",
                    flush=True,
                )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    report = {
        "schema_version": "voix.styletts2_lora_corpus.v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "source": "VoxCeleb2 V2 speaker-disjoint index",
        "whisper_model": whisper_model,
        "thresholds": {
            "min_duration_sec": min_duration,
            "max_duration_sec": max_duration,
            "min_avg_logprob": min_logprob,
            "max_compression_ratio": max_compression,
            "max_no_speech_prob": max_no_speech,
            "max_text_chars": max_chars,
            "min_words": 3,
        },
        "requested": {
            "train_speakers": train_speakers,
            "val_speakers": val_speakers,
            "clips_per_speaker": clips_per_speaker,
        },
        "selected": {
            "speakers_before_filter": len(selections),
            "speakers": len({row["speaker_id"] for row in rows}),
            "clips": int(len(manifest)),
            "hours": round(float(manifest["duration_sec"].sum()) / 3600.0, 3)
            if not manifest.empty
            else 0.0,
            "by_split": {
                split: {
                    "clips": int((manifest["split"] == split).sum()),
                    "speakers": int(
                        manifest.loc[manifest["split"] == split, "speaker_id"].nunique()
                    ),
                    "hours": round(
                        float(
                            manifest.loc[
                                manifest["split"] == split, "duration_sec"
                            ].sum()
                        )
                        / 3600.0,
                        3,
                    ),
                }
                for split in ("train", "val")
            },
            "by_gender": manifest.groupby("gender").size().to_dict()
            if not manifest.empty
            else {},
        },
        "dropped": {
            "clips": int(sum(drop_reasons.values())),
            "reasons": dict(drop_reasons),
        },
        "elapsed_sec": round(time.time() - started, 1),
        "notes": [
            "Transcripts are real Whisper output, not ground-truth text; avg_logprob and compression_ratio are recorded.",
            "Reference clips come from a different video of the same speaker.",
            "VoxCeleb2 is research/non-commercial; this corpus is not for commercial training.",
            "The identity path (CFVN/CVAE/adapter) is not touched by this corpus.",
        ],
    }
    return manifest, report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--clips-root", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-speakers", type=int, default=300)
    parser.add_argument("--val-speakers", type=int, default=50)
    parser.add_argument("--clips-per-speaker", type=int, default=2)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--seed", default="styletts2-lora-v1")
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--min-logprob", type=float, default=-0.8)
    parser.add_argument("--max-compression", type=float, default=2.4)
    parser.add_argument("--max-no-speech", type=float, default=0.85)
    parser.add_argument("--max-chars", type=int, default=300)
    return parser.parse_args(argv)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    index = pd.read_csv(args.index)
    required = {"clip_id", "speaker_id", "video_id", "split", "gender", "video_path"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"index missing columns: {sorted(missing)}")

    import imageio_ffmpeg
    import whisper

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[corpus] loading Whisper {args.whisper_model} on {args.device}", flush=True)
    model = whisper.load_model(args.whisper_model, device=args.device)
    manifest, report = build_manifest(
        index,
        clips_root=args.clips_root,
        output_dir=args.output_dir,
        train_speakers=args.train_speakers,
        val_speakers=args.val_speakers,
        clips_per_speaker=args.clips_per_speaker,
        whisper_model=args.whisper_model,
        device=args.device,
        seed=args.seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_logprob=args.min_logprob,
        max_compression=args.max_compression,
        max_no_speech=args.max_no_speech,
        max_chars=args.max_chars,
        ffmpeg=ffmpeg,
        whisper_model_obj=model,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    report_path = args.output_dir / "corpus_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selected"], indent=2), flush=True)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
