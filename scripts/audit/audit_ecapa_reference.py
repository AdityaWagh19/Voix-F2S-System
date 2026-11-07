"""Measure an ECAPA speaker-similarity reference distribution on real AMI audio.

The face-voice audit reports cosine similarity between synthesized voices.  A
raw cosine is hard to interpret without knowing what real speakers look like in
the same embedding space.  This script embeds real AMI headset clips and
reports:

* same-speaker cosine (two clips from one speaker) -- the identity ceiling;
* between-speaker cosine within gender and across genders -- what "different
  person" looks like for this ECAPA model;
* where the calibrated face-voice pairs fall in that real distribution.

This is an objective speaker-embedding reference, not a human identity study.

Usage:
    python scripts/audit/audit_ecapa_reference.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "ami_naturalness" / "manifest.csv"
DEFAULT_FACE_AUDIT = (
    ROOT / "results" / "audit" / "style_calibration" / "style_calibration_audit.json"
)
DEFAULT_OUT = ROOT / "results" / "audit" / "ecapa_reference" / "ecapa_reference.json"
ECAPA_DIR = ROOT / "checkpoints" / "speechbrain" / "spkrec-ecapa"
FEMALE_SPEAKERS = {"FEE005", "FEE055", "FIE081"}


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _embed(model, path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sample_rate != 16000:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(tensor).squeeze().detach().cpu().numpy()
    return _unit(np.asarray(embedding, dtype=np.float64).reshape(-1))


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _select_clips(manifest: pd.DataFrame, per_speaker: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for speaker, group in manifest.groupby("speaker_id"):
        usable = group[
            (group["duration_sec"] >= 3.0) & (group["duration_sec"] <= 10.0)
        ].sort_values(["duration_sec", "clip_id"], ascending=[False, True])
        for _, row in usable.head(per_speaker).iterrows():
            path = Path(str(row["wav_path"]))
            if not path.is_file():
                continue
            selected.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "speaker_id": str(speaker),
                    "gender": "female" if speaker in FEMALE_SPEAKERS else "male",
                    "duration_sec": float(row["duration_sec"]),
                    "path": str(path),
                }
            )
    return selected


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--face-audit", type=Path, default=DEFAULT_FACE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-speaker", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest)
    selected = _select_clips(manifest, per_speaker=args.per_speaker)
    n_speakers = len({entry["speaker_id"] for entry in selected})
    if n_speakers < 6:
        raise RuntimeError(f"too few speakers: {n_speakers}")

    from speechbrain.inference.speaker import EncoderClassifier

    print(f"[setup] loading ECAPA; embedding {len(selected)} clips", flush=True)
    model = EncoderClassifier.from_hparams(
        source=str(ECAPA_DIR),
        savedir=str(ECAPA_DIR),
        run_opts={"device": args.device},
    )

    embeddings: Dict[str, List[np.ndarray]] = {}
    entries: List[Dict[str, Any]] = []
    for entry in selected:
        vector = _embed(model, Path(entry["path"]))
        embeddings.setdefault(entry["speaker_id"], []).append(vector)
        entries.append({**entry, "embedding_norm": float(np.linalg.norm(vector))})
        print(f"[embed] {entry['clip_id']} ({entry['gender']})", flush=True)

    same_speaker: List[float] = []
    same_gender_between: List[float] = []
    cross_gender: List[float] = []
    between_pairs: List[Dict[str, Any]] = []
    speakers = sorted(embeddings)
    for index, speaker_a in enumerate(speakers):
        vectors = embeddings[speaker_a]
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                same_speaker.append(float(np.dot(vectors[i], vectors[j])))
        for speaker_b in speakers[index + 1 :]:
            gender_a = "female" if speaker_a in FEMALE_SPEAKERS else "male"
            gender_b = "female" if speaker_b in FEMALE_SPEAKERS else "male"
            for vector_a in vectors:
                for vector_b in embeddings[speaker_b]:
                    cosine = float(np.dot(vector_a, vector_b))
                    between_pairs.append(
                        {
                            "a": speaker_a,
                            "b": speaker_b,
                            "a_gender": gender_a,
                            "b_gender": gender_b,
                            "cosine": cosine,
                        }
                    )
                    if gender_a == gender_b:
                        same_gender_between.append(cosine)
                    else:
                        cross_gender.append(cosine)

    face_pairs: List[Dict[str, Any]] = []
    if args.face_audit.is_file():
        audit = json.loads(args.face_audit.read_text(encoding="utf-8"))
        rows = {
            row["name"]: row
            for row in audit.get("conditions", [])
            if float(row.get("alpha", -1.0)) == 1.0
        }
        names = list(rows)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = rows[names[i]], rows[names[j]]
                cosine = float(
                    np.dot(_unit(np.asarray(a["ecapa_vector"])), _unit(np.asarray(b["ecapa_vector"])))
                )
                face_pairs.append(
                    {
                        "a": a["face_label"],
                        "b": b["face_label"],
                        "cosine": cosine,
                    }
                )

    def percentile_rank(value: float, reference: Sequence[float]) -> Optional[float]:
        if not reference:
            return None
        array = np.asarray(reference, dtype=np.float64)
        return float((array < value).mean() * 100.0)

    report = {
        "schema_version": "voix.ecapa_reference.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory_reference",
        "manifest": str(args.manifest),
        "n_speakers": n_speakers,
        "n_clips": len(entries),
        "clips": entries,
        "same_speaker": _summary(same_speaker),
        "between_speaker_same_gender": _summary(same_gender_between),
        "between_speaker_cross_gender": _summary(cross_gender),
        "same_speaker_pairs": same_speaker,
        "between_speaker_pairs": between_pairs,
        "calibrated_face_pairs": face_pairs,
        "calibrated_face_pair_summary": _summary(
            [pair["cosine"] for pair in face_pairs]
        ),
        "calibrated_face_pair_percentiles_vs_same_gender_between": {
            f"{pair['a']} | {pair['b']}": percentile_rank(
                pair["cosine"], same_gender_between
            )
            for pair in face_pairs
        },
        "notes": [
            "AMI headset clips are real human speech, resampled to 16 kHz for ECAPA.",
            "Same-speaker pairs use two different clips from one speaker and are an "
            "upper reference for identity preservation, not a perfect ceiling.",
            "Between-speaker values include all cross-speaker clip pairs; same-gender "
            "and cross-gender distributions are reported separately.",
            "These are speaker-embedding similarities, not human identity judgments.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "same_speaker": report["same_speaker"],
            "between_speaker_same_gender": report["between_speaker_same_gender"],
            "between_speaker_cross_gender": report["between_speaker_cross_gender"],
            "calibrated_face_pair_summary": report["calibrated_face_pair_summary"],
        },
        indent=2,
    ))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
