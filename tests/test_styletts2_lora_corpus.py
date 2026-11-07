"""Tests for the StyleTTS2 LoRA corpus builder.

The corpus is the input to a fine-tune, so these pin the parts that would
silently poison it: deterministic speaker selection, a reference clip that is
actually different from the target, quality filtering, and a manifest that
keeps the reference/target invariant after the real build path runs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "data" / "build_styletts2_lora_corpus.py"

spec = importlib.util.spec_from_file_location("build_styletts2_lora_corpus", SCRIPT)
assert spec and spec.loader
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


def _index(speakers: int = 12, clips: int = 5) -> pd.DataFrame:
    rows = []
    for speaker_index in range(speakers):
        split = "train" if speaker_index < speakers - 3 else "val"
        for clip_index in range(clips):
            rows.append(
                {
                    "clip_id": f"{split}/id{speaker_index:05d}/video{clip_index}/00001",
                    "speaker_id": f"id{speaker_index:05d}",
                    "video_id": f"video{clip_index}",
                    "split": split,
                    "gender": "f" if speaker_index % 2 == 0 else "m",
                    "video_path": f"id{speaker_index:05d}/video{clip_index}/00001.mp4",
                }
            )
    return pd.DataFrame(rows)


def test_speaker_selection_is_deterministic_and_balanced():
    index = _index()
    first = corpus.select_speakers(index, "train", 6, "seed-a")
    second = corpus.select_speakers(index, "train", 6, "seed-a")
    assert first == second
    assert len(first) == 6
    gender = index.groupby("speaker_id")["gender"].first()
    assert sum(gender[speaker] == "f" for speaker in first) == 3
    assert sum(gender[speaker] == "m" for speaker in first) == 3


def test_clip_selection_uses_a_different_video_for_the_reference():
    index = _index()
    targets, reference = corpus.select_clips(index, "id00000", 2, "seed-a")
    assert len(targets) == 2
    assert reference is not None
    target_videos = {row["video_id"] for row in targets}
    assert reference["video_id"] not in target_videos
    assert reference["clip_id"] not in {row["clip_id"] for row in targets}


def test_quality_reason_covers_each_filter():
    good = {
        "duration_sec": 4.0,
        "text": "this is a real sentence",
        "avg_logprob": -0.3,
        "compression_ratio": 1.2,
        "no_speech_prob": 0.1,
    }
    kwargs = dict(
        min_duration=1.5,
        max_duration=12.0,
        min_logprob=-0.8,
        max_compression=2.4,
        max_no_speech=0.85,
        max_chars=300,
    )
    assert corpus.quality_reason(good, **kwargs) is None
    assert corpus.quality_reason({**good, "duration_sec": 0.5}, **kwargs) == "too_short"
    assert corpus.quality_reason({**good, "duration_sec": 20.0}, **kwargs) == "too_long"
    assert corpus.quality_reason({**good, "text": "hi"}, **kwargs) == "too_few_words"
    assert corpus.quality_reason({**good, "avg_logprob": -1.5}, **kwargs) == "low_logprob"
    assert corpus.quality_reason({**good, "compression_ratio": 3.0}, **kwargs) == "high_compression"
    assert corpus.quality_reason({**good, "no_speech_prob": 0.99}, **kwargs) == "no_speech"


def test_build_manifest_keeps_target_and_reference_separate(tmp_path, monkeypatch):
    index = _index(speakers=5, clips=4)
    for row in index.itertuples():
        source = tmp_path / "clips" / row.video_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"fake video")

    def fake_extract(source, destination, ffmpeg, sample_rate=24000):
        destination.parent.mkdir(parents=True, exist_ok=True)
        wave = 0.01 * np.sin(np.linspace(0, 2 * np.pi * 220 * 2.0, 48000))
        sf.write(destination, wave.astype(np.float32), 24000)

    def fake_transcribe(wav_path, model, device):
        return {
            "text": "this is a deterministic transcript for the test",
            "language": "en",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.1,
            "segments": 1,
        }

    monkeypatch.setattr(corpus, "extract_wav", fake_extract)
    monkeypatch.setattr(corpus, "transcribe_wav", fake_transcribe)
    manifest, report = corpus.build_manifest(
        index,
        clips_root=tmp_path / "clips",
        output_dir=tmp_path / "out",
        train_speakers=2,
        val_speakers=1,
        clips_per_speaker=1,
        whisper_model="base",
        device="cpu",
        seed="test",
        min_duration=1.5,
        max_duration=12.0,
        min_logprob=-0.8,
        max_compression=2.4,
        max_no_speech=0.85,
        max_chars=300,
        ffmpeg="ffmpeg",
        whisper_model_obj=None,
    )

    assert len(manifest) == 3
    assert set(manifest["split"]) == {"train", "val"}
    assert (manifest["clip_id"] != manifest["ref_clip_id"]).all()
    assert report["selected"]["clips"] == 3
    assert report["dropped"]["clips"] == 0
    assert (tmp_path / "out" / "transcripts.jsonl").is_file()
    cached = [
        json.loads(line)
        for line in (tmp_path / "out" / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(cached) == 3
