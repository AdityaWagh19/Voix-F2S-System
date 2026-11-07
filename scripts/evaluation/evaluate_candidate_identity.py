"""Measure candidate identity consistency and face-to-voice retrieval.

The showcase has two different questions that are easy to conflate:

* **Candidates** are intentionally different plausible identities for one face.
  They should each be face-consistent, but they should not collapse into the
  same speaker.
* **Expression variants** (accent/emotion) are controls on one identity.  They
  should stay close to the primary candidate while changing the requested
  expression.

This script reports both.  Every metric is either measured or explicitly
``unavailable``; missing audio, models, or ECAPA do not become zeros.

Usage:
    python scripts/evaluation/evaluate_candidate_identity.py
    python scripts/evaluation/evaluate_candidate_identity.py --device cuda --output results/candidate_identity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.metrics import face_voice_consistency, speaker_similarity

SCHEMA = "voix.candidate_identity.v1"


def _resolve(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate)


def _summarize(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    items = list(values)
    measured = [float(value) for value in items if value is not None]
    total = len(items)
    if not measured:
        return {
            "status": "unavailable",
            "n": 0,
            "total": total,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "status": "computed" if len(measured) == total else "partial",
        "n": len(measured),
        "total": total,
        "mean": round(sum(measured) / len(measured), 4),
        "min": round(min(measured), 4),
        "max": round(max(measured), 4),
    }


def _pairwise_matrix(
    paths: List[str],
    similarity_fn: Callable[[str, str], Optional[float]],
) -> Dict[str, Any]:
    matrix: List[List[Optional[float]]] = []
    pairs: List[Dict[str, Any]] = []
    for i, path_a in enumerate(paths):
        row: List[Optional[float]] = []
        for j, path_b in enumerate(paths):
            if i == j:
                row.append(1.0)
                continue
            if j < i:
                row.append(matrix[j][i])
                continue
            value = similarity_fn(path_a, path_b)
            row.append(value)
            if value is not None:
                pairs.append(
                    {
                        "a": Path(path_a).name,
                        "b": Path(path_b).name,
                        "similarity": value,
                    }
                )
        matrix.append(row)

    measured = [pair["similarity"] for pair in pairs]
    summary = _summarize(measured)
    summary.update({"n_pairs": len(pairs), "matrix": matrix, "pairs": pairs})
    return summary


def evaluate_candidate_identity(
    manifest: Dict[str, Any],
    *,
    primary_audio: str,
    face: str,
    device: str = "cpu",
    cfvn_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    pipeline: Any = None,
) -> Dict[str, Any]:
    samples = list(manifest.get("samples", []))
    if not samples:
        raise ValueError("manifest has no samples")

    primary = _resolve(primary_audio)
    face_path = _resolve(face)
    if not Path(primary).exists():
        raise FileNotFoundError(f"primary candidate audio not found: {primary}")
    if not Path(face_path).exists():
        raise FileNotFoundError(f"face image not found: {face_path}")

    candidates = [
        item for item in samples if str(item.get("group", "")) == "01_Candidates"
    ]
    expressions = [
        item
        for item in samples
        if str(item.get("group", "")) in {"02_Accents", "03_Emotions"}
    ]
    if not candidates:
        raise ValueError("manifest has no 01_Candidates samples")

    cfvn_kwargs: Dict[str, Any] = {"device": device}
    if cfvn_path:
        cfvn_kwargs["cfvn_path"] = cfvn_path
    if scaler_path:
        cfvn_kwargs["scaler_path"] = scaler_path
    if pipeline is not None:
        # Reuse the face extractors instead of rebuilding ArcFace/FaceMesh for
        # every asset.  The CFVN itself is cached inside the metrics module.
        cfvn_kwargs["pipeline"] = pipeline

    def measure(item: Dict[str, Any]) -> Dict[str, Any]:
        audio = _resolve(str(item.get("audio", "")))
        return {
            "audio": audio,
            "group": item.get("group"),
            "condition": item.get("condition"),
            "face_voice_cosine": face_voice_consistency(
                face_path, audio, **cfvn_kwargs
            ),
            "speaker_similarity_to_primary": speaker_similarity(
                audio, primary, device=device
            ),
        }

    candidate_results = [measure(item) for item in candidates]
    expression_results = [measure(item) for item in expressions]

    candidate_paths = [item["audio"] for item in candidate_results]
    candidate_pairwise = _pairwise_matrix(
        candidate_paths,
        lambda a, b: speaker_similarity(a, b, device=device),
    )

    report: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": "demo/synthesis_manifest.json",
        "face": face_path,
        "primary_audio": primary,
        "device": device,
        "candidates": candidate_results,
        "candidate_pairwise_similarity": candidate_pairwise,
        "candidate_face_voice": _summarize(
            [item["face_voice_cosine"] for item in candidate_results]
        ),
        "candidate_to_primary": _summarize(
            [
                item["speaker_similarity_to_primary"]
                for item in candidate_results
                if item["audio"] != primary
            ]
        ),
        "expressions": expression_results,
        "expression_face_voice": _summarize(
            [item["face_voice_cosine"] for item in expression_results]
        ),
        "expression_identity_preservation": _summarize(
            [item["speaker_similarity_to_primary"] for item in expression_results]
        ),
        "notes": [
            "Candidates are intended to be distinct plausible identities; lower pairwise similarity is more diverse, not worse.",
            "Expression variants are intended to preserve the primary identity; higher similarity to the primary is better.",
            "Face-to-voice cosine is a CFVN retrieval score, not a probability and not a MOS.",
            "Unavailable metrics are reported as unavailable, never as zero.",
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure candidate diversity and expression identity preservation"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "demo" / "synthesis_manifest.json",
    )
    parser.add_argument(
        "--primary",
        default="demo/01_candidates/candidate_1_primary.wav",
    )
    parser.add_argument("--face", default="demo/subject_face.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cfvn", default=None)
    parser.add_argument("--scaler", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "candidate_identity.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    # Build one pipeline so face extraction is reused across all assets.  The
    # CFVN is cached in voix.evaluation.metrics.  If pipeline construction
    # fails, the measurement still runs and reports unavailable metrics rather
    # than inventing values.
    pipeline = None
    try:
        from voix.inference.pipeline import VoixPipeline

        pipeline = VoixPipeline(device=args.device)
    except Exception as exc:
        print(f"[candidate-identity] pipeline reuse unavailable: {exc}")

    report = evaluate_candidate_identity(
        manifest,
        primary_audio=args.primary,
        face=args.face,
        device=args.device,
        cfvn_path=args.cfvn,
        scaler_path=args.scaler,
        pipeline=pipeline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
