"""Build a speaker-disjoint AMI pilot corpus for naturalness adaptation.

AMI is used only by the naturalness track.  This script never touches the
CFVN, CVAE, style adapter, or any emotion/accent artifact.  It downloads real
headset-mic meeting audio, cuts utterances using the human word annotations,
and writes a manifest with explicit split and firewall provenance.

Pilot split (10 meetings, ~5.5 h):
    train: IS1000a-d, ES2002a-d
    val:   ES2003a
    test:  ES2014a

The four participant groups are disjoint, so the pilot is speaker-disjoint
across train/val/test even though the full AMI corpus is not.

Usage:
    python scripts/data/build_ami_naturalness_corpus.py --download --workers 4
    python scripts/data/build_ami_naturalness_corpus.py --meetings ES2002a --download
"""
from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = ROOT / "data" / "raw" / "ami" / "annotations"
DEFAULT_AUDIO_ROOT = ROOT / "data" / "raw" / "ami" / "amicorpus"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "ami_naturalness"
BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

PILOT_SPLITS: Dict[str, Tuple[str, ...]] = {
    "train": (
        "IS1000a",
        "IS1000b",
        "IS1000c",
        "IS1000d",
        "ES2002a",
        "ES2002b",
        "ES2002c",
        "ES2002d",
    ),
    "val": ("ES2003a",),
    "test": ("ES2014a",),
}

MIN_SEGMENT_SEC = 1.0
MAX_SEGMENT_SEC = 15.0
MAX_GAP_SEC = 0.70
MAX_WORDS = 60

MANIFEST_COLUMNS = [
    "clip_id",
    "meeting_id",
    "speaker_id",
    "split",
    "wav_path",
    "text",
    "duration_sec",
    "start_sec",
    "end_sec",
    "channel",
    "headset",
    "source",
    "license",
    "track",
]


def parse_meetings(path: Path) -> Dict[str, Dict[str, object]]:
    root = ET.parse(path).getroot()
    meetings: Dict[str, Dict[str, object]] = {}
    for meeting in root.findall("meeting"):
        meeting_id = str(meeting.attrib.get("observation"))
        speakers = []
        for speaker in meeting.findall("speaker"):
            speakers.append(
                {
                    "global_name": str(speaker.attrib.get("global_name")),
                    "channel": int(speaker.attrib.get("channel")),
                    "agent": str(speaker.attrib.get("nxt_agent")),
                }
            )
        meetings[meeting_id] = {
            "type": meeting.attrib.get("type"),
            "duration": float(meeting.attrib.get("duration", 0.0)),
            "speakers": speakers,
        }
    return meetings


def parse_words(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    root = ET.parse(path).getroot()
    words: List[Dict[str, object]] = []
    for node in root.findall("w"):
        try:
            start = float(node.attrib.get("starttime"))
            end = float(node.attrib.get("endtime"))
        except (TypeError, ValueError):
            continue
        text = (node.text or "").strip()
        if not text:
            continue
        words.append(
            {
                "start": start,
                "end": max(start, end),
                "text": text,
                "punc": node.attrib.get("punc") == "true",
            }
        )
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return words


def _render_text(words: Sequence[Dict[str, object]]) -> str:
    parts: List[str] = []
    for word in words:
        token = str(word["text"])
        if word["punc"]:
            if parts:
                parts[-1] = parts[-1] + token
            continue
        parts.append(token)
    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_words(
    words: Sequence[Dict[str, object]],
    *,
    min_sec: float = MIN_SEGMENT_SEC,
    max_sec: float = MAX_SEGMENT_SEC,
    max_gap: float = MAX_GAP_SEC,
) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    current: List[Dict[str, object]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        text = _render_text(current)
        word_count = sum(1 for item in current if not item["punc"])
        if end - start >= min_sec and word_count >= 2 and len(text) >= 3:
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "n_words": word_count,
                }
            )
        current = []

    for word in words:
        if not current:
            current.append(word)
            continue
        gap = float(word["start"]) - float(current[-1]["end"])
        span = float(word["end"]) - float(current[0]["start"])
        sentence_end = bool(current[-1]["punc"]) and str(current[-1]["text"]) in {
            ".",
            "?",
            "!",
        }
        if gap > max_gap or span > max_sec or len(current) >= MAX_WORDS or sentence_end:
            flush()
            current.append(word)
        else:
            current.append(word)
    flush()
    return segments


def download_file(url: str, destination: Path, *, retries: int = 3) -> None:
    if destination.is_file() and destination.stat().st_size > 1_000_000:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            import requests

            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(destination)
            return
        except Exception as exc:  # pragma: no cover - network guard
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def audio_path(audio_root: Path, meeting_id: str, channel: int) -> Path:
    return (
        audio_root
        / meeting_id
        / "audio"
        / f"{meeting_id}.Headset-{channel}.wav"
    )


def load_audio(path: Path, start: float, end: float, target_sr: int = 24000) -> np.ndarray:
    info = sf.info(str(path))
    start_frame = max(0, int(round(start * info.samplerate)))
    end_frame = min(info.frames, int(round(end * info.samplerate)))
    audio, sample_rate = sf.read(
        str(path), start=start_frame, stop=end_frame, dtype="float32"
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        audio = resample_poly(audio, target_sr, sample_rate).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def write_manifest(rows: List[Dict[str, object]], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = frame[MANIFEST_COLUMNS]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def build(
    annotations: Path,
    audio_root: Path,
    output: Path,
    splits: Dict[str, Sequence[str]],
    *,
    download: bool,
    workers: int,
    max_meetings: Optional[int],
) -> Dict[str, object]:
    meetings = parse_meetings(annotations / "corpusResources" / "meetings.xml")
    selected: List[Tuple[str, str]] = []
    for split, meeting_ids in splits.items():
        for meeting_id in meeting_ids:
            if meeting_id not in meetings:
                raise KeyError(f"meeting {meeting_id} is not in meetings.xml")
            selected.append((split, meeting_id))
    if max_meetings is not None:
        selected = selected[:max_meetings]

    speaker_splits: Dict[str, str] = {}
    for split, meeting_id in selected:
        for speaker in meetings[meeting_id]["speakers"]:
            name = str(speaker["global_name"])
            previous = speaker_splits.get(name)
            if previous is not None and previous != split:
                raise RuntimeError(
                    f"speaker {name} appears in both {previous} and {split}; "
                    "the pilot must remain speaker-disjoint"
                )
            speaker_splits[name] = split

    tasks: List[Tuple[str, str, Dict[str, object]]] = []
    for split, meeting_id in selected:
        for speaker in meetings[meeting_id]["speakers"]:
            tasks.append((split, meeting_id, dict(speaker)))

    if download:
        print(f"Downloading {len(tasks)} headset file(s) with {workers} worker(s) ...")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    download_file,
                    f"{BASE_URL}/{meeting_id}/audio/{meeting_id}.Headset-{int(speaker['channel'])}.wav",
                    audio_path(audio_root, meeting_id, int(speaker["channel"])),
                ): (meeting_id, speaker)
                for _, meeting_id, speaker in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                meeting_id, speaker = futures[future]
                future.result()
                if index % 8 == 0 or index == len(futures):
                    print(f"  downloaded {index}/{len(futures)} ({meeting_id})", flush=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    wav_root = output / "wav"
    for split, meeting_id, speaker in tasks:
        channel = int(speaker["channel"])
        agent = str(speaker["agent"])
        speaker_id = str(speaker["global_name"])
        source_audio = audio_path(audio_root, meeting_id, channel)
        words_path = annotations / "words" / f"{meeting_id}.{agent}.words.xml"
        if not source_audio.is_file():
            failures.append(
                {"meeting": meeting_id, "speaker": speaker_id, "error": "missing audio"}
            )
            continue
        words = parse_words(words_path)
        if not words:
            failures.append(
                {"meeting": meeting_id, "speaker": speaker_id, "error": "missing words"}
            )
            continue
        segments = segment_words(words)
        for index, segment in enumerate(segments):
            clip_id = f"{meeting_id}_{speaker_id}_{index:04d}"
            destination = (
                wav_root / split / meeting_id / speaker_id / f"{clip_id}.wav"
            )
            if not destination.is_file():
                try:
                    audio = load_audio(
                        source_audio,
                        float(segment["start"]),
                        float(segment["end"]),
                    )
                except Exception as exc:  # pragma: no cover - runtime guard
                    failures.append(
                        {
                            "meeting": meeting_id,
                            "speaker": speaker_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                if audio.size == 0:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(destination), audio, 24000, subtype="PCM_16")
            info = sf.info(str(destination))
            rows.append(
                {
                    "clip_id": clip_id,
                    "meeting_id": meeting_id,
                    "speaker_id": speaker_id,
                    "split": split,
                    "wav_path": str(destination),
                    "text": str(segment["text"]),
                    "duration_sec": float(info.duration),
                    "start_sec": float(segment["start"]),
                    "end_sec": float(segment["end"]),
                    "channel": channel,
                    "headset": f"Headset-{channel}",
                    "source": "AMI corpus headset microphone",
                    "license": "CC BY 4.0",
                    "track": "naturalness",
                }
            )
        print(
            f"  {split}/{meeting_id}/{speaker_id}: {len(segments)} segment(s)",
            flush=True,
        )

    manifest_path = output / "manifest.csv"
    frame = write_manifest(rows, manifest_path)
    summary: Dict[str, object] = {
        "schema_version": "voix.ami_naturalness_corpus.v1",
        "track": "naturalness",
        "level": "exploratory",
        "annotations": str(annotations),
        "audio_root": str(audio_root),
        "output": str(output),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "splits": {split: list(ids) for split, ids in splits.items()},
        "meetings": [meeting_id for _, meeting_id in selected],
        "rows": int(len(frame)),
        "hours": float(frame["duration_sec"].sum() / 3600.0) if len(frame) else 0.0,
        "speakers": int(frame["speaker_id"].nunique()) if len(frame) else 0,
        "speakers_by_split": {
            split: sorted(
                set(frame.loc[frame["split"] == split, "speaker_id"].astype(str))
            )
            for split in splits
        },
        "rows_by_split": {
            split: int((frame["split"] == split).sum()) for split in splits
        },
        "hours_by_split": {
            split: float(
                frame.loc[frame["split"] == split, "duration_sec"].sum() / 3600.0
            )
            for split in splits
        },
        "failures": failures,
        "firewall": (
            "AMI is used only for StyleTTS2 naturalness adaptation and evaluation. "
            "It is not used to train the CFVN, CVAE, style adapter, emotion adapter, "
            "or accent adapter, and it is never used as face-derived identity data."
        ),
        "limitations": [
            "Pilot split is 10 meetings; it is not the full AMI corpus.",
            "AMI word timings are forced-alignment annotations; a small number are known incomplete.",
            "Headset audio is close-talking and cleaner than far-field meeting audio.",
            "No human MOS/CMOS exists for any generated audio.",
        ],
        "mos": {"status": "unavailable", "reason": "no human ratings collected"},
    }
    report_path = output / "corpus_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote {manifest_path} ({len(frame)} rows, {summary['hours']:.2f} h, "
        f"{summary['speakers']} speakers)"
    )
    print(f"Wrote {report_path}")
    return summary


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AMI naturalness pilot")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--meetings", default=None, help="Comma-separated override")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-meetings", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    splits = PILOT_SPLITS
    if args.meetings:
        splits = {"train": tuple(args.meetings.split(","))}
    build(
        args.annotations,
        args.audio_root,
        args.output,
        splits,
        download=args.download,
        workers=args.workers,
        max_meetings=args.max_meetings,
    )


if __name__ == "__main__":
    main()
