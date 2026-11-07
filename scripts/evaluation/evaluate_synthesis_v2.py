"""Evaluate synthesized VOIX audio with real objective metrics.

Unlike the old ``scripts/benchmark.py``, this harness never writes ``-1`` or a
made-up number.  Every metric reports ``status`` plus either a measured value or
an explicit reason it is unavailable.

Manifest format::

    {
      "reference_text": "The human voice reflects ...",
      "reference_wavs": ["data/reference/speaker.wav"],
      "samples": [
        {
          "audio": "demo/sample.wav",
          "face": "demo/face.png",              // optional
          "reference_wav": "data/reference/spk.wav",  // optional
          "group": "subject_01"                 // candidate set for diversity
        }
      ]
    }

Example:
    python scripts/evaluation/evaluate_synthesis_v2.py --manifest artifacts/synthesis_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.audio_metrics import (
    embedding_diversity,
    fad_proxy,
    transcribe_wer,
)
from voix.evaluation.metrics import face_voice_consistency, speaker_similarity


def _resolve(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate)


def evaluate_manifest(
    manifest: Dict[str, Any],
    *,
    device: str = "cpu",
    whisper_model: str = "base",
    cfvn_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> Dict[str, Any]:
    samples = list(manifest.get("samples", []))
    if not samples:
        raise ValueError("manifest has no samples")

    audio_paths = [_resolve(item.get("audio")) for item in samples]
    missing_audio = [path for path in audio_paths if not path or not Path(path).exists()]
    if missing_audio:
        raise FileNotFoundError(f"missing synthesized audio: {missing_audio}")

    report: Dict[str, Any] = {
        "n_samples": len(samples),
        "device": device,
        "metrics": {},
        "per_sample": [],
    }

    default_reference = str(manifest.get("reference_text", "")).strip()
    references = [
        str(item.get("reference_text") or default_reference).strip()
        for item in samples
    ]
    if all(references):
        report["metrics"]["wer"] = transcribe_wer(
            audio_paths, references, model_size=whisper_model, device=device
        )
    else:
        report["metrics"]["wer"] = {
            "status": "unavailable",
            "reason": "one or more samples have no reference_text",
            "wer": None,
        }

    secs: List[float] = []
    face_scores: List[float] = []
    for item, audio in zip(samples, audio_paths):
        entry: Dict[str, Any] = {"audio": audio}
        reference = _resolve(item.get("reference_wav"))
        if reference:
            similarity = speaker_similarity(audio, reference, device=device)
            entry["speaker_similarity"] = similarity
            if similarity is not None:
                secs.append(float(similarity))
        face = _resolve(item.get("face"))
        if face:
            consistency = face_voice_consistency(
                face,
                audio,
                device=device,
                **({"cfvn_path": cfvn_path} if cfvn_path else {}),
                **({"scaler_path": scaler_path} if scaler_path else {}),
            )
            entry["face_voice_consistency"] = consistency
            if consistency is not None:
                face_scores.append(float(consistency))
        report["per_sample"].append(entry)

    report["metrics"]["speaker_similarity"] = {
        "status": "computed" if secs else "unavailable",
        "n": len(secs),
        "mean": float(sum(secs) / len(secs)) if secs else None,
        "reason": None if secs else "no sample had a readable reference_wav",
    }
    report["metrics"]["face_voice_consistency"] = {
        "status": "computed" if face_scores else "unavailable",
        "n": len(face_scores),
        "mean": float(sum(face_scores) / len(face_scores)) if face_scores else None,
        "reason": None if face_scores else "no sample had a face or CFVN/scaler",
    }

    reference_wavs = [
        _resolve(path) for path in manifest.get("reference_wavs", []) if path
    ]
    reference_wavs = [path for path in reference_wavs if path and Path(path).exists()]
    report["metrics"]["fad_proxy"] = fad_proxy(reference_wavs, audio_paths)

    groups: Dict[str, List[str]] = {}
    for item, audio in zip(samples, audio_paths):
        groups.setdefault(str(item.get("group", "default")), []).append(audio)
    report["metrics"]["diversity"] = {
        group: embedding_diversity(paths, device=device)
        for group, paths in groups.items()
    }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthesized VOIX audio")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--cfvn", default=None)
    parser.add_argument("--scaler", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = evaluate_manifest(
        manifest,
        device=args.device,
        whisper_model=args.whisper_model,
        cfvn_path=args.cfvn,
        scaler_path=args.scaler,
    )
    text = json.dumps(report, indent=2)
    output = args.output or args.manifest.with_name("synthesis_evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
