"""Evaluate emotion and accent controls against a measured manifest.

This harness never fabricates a metric. Prosody is measured from audio,
speaker similarity uses ECAPA, and WER is optional because it needs Whisper
and ``ffmpeg``. Classifiers and MOS stay ``unavailable`` until real models and
real listener ratings exist.

Manifest format (``voix.expression.v1``)::

    {
      "schema_version": "voix.expression.v1",
      "reference_text": "The human voice reflects both physical anatomy and lived culture.",
      "samples": [
        {
          "audio": "demo/03_emotions/emotion_01_neutral.wav",
          "control": "emotion",
          "label": "neutral",
          "intensity": 0.0,
          "speaker_id": "subject_face"
        },
        {
          "audio": "demo/03_emotions/emotion_02_happy.wav",
          "control": "emotion",
          "label": "happy",
          "intensity": 0.85,
          "speaker_id": "subject_face",
          "neutral_reference": "demo/03_emotions/emotion_01_neutral.wav"
        },
        {
          "audio": "demo/02_accents/accent_02_british_rp.wav",
          "control": "accent",
          "label": "british_rp",
          "intensity": 0.85,
          "neutral_reference": "demo/02_accents/accent_01_general_american.wav"
        }
      ]
    }

Relative audio paths resolve against the project root unless ``--root`` is
given. ``neutral_reference`` is optional but required for the
speaker-preservation gate. Intensity sweeps need at least three samples with at
least two distinct intensity values.

Examples::

    # Prosody, speaker preservation, and explicit unavailable classifiers
    python scripts/evaluation/evaluate_expression_controls.py \
        --manifest results/expression_manifest.json

    # Include WER (requires openai-whisper and ffmpeg on PATH)
    python scripts/evaluation/evaluate_expression_controls.py \
        --manifest results/expression_manifest.json --compute-wer

    # Enforce the pre-registered gates in CI
    python scripts/evaluation/evaluate_expression_controls.py \
        --manifest results/expression_manifest.json --compute-wer \
        --max-wer 0.20 --require-pass
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.expression_metrics import (  # noqa: E402
    check_expression_gates,
    evaluate_expression_manifest,
    load_expression_manifest,
)


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def compute_wer_by_condition(
    manifest: Dict[str, Any],
    *,
    root: Path,
    device: str = "cpu",
    model_size: str = "base",
) -> Dict[str, Any]:
    """Transcribe each condition with Whisper and report WER per condition.

    The function is opt-in because Whisper model loading is expensive and
    requires an ``ffmpeg`` executable. Missing references or audio produce an
    explicit ``unavailable`` entry rather than a zero.
    """
    from voix.evaluation.audio_metrics import transcribe_wer

    default_reference = str(manifest.get("reference_text", "")).strip()
    groups: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for sample in manifest["samples"]:
        control = str(sample["control"]).strip().lower()
        label = str(sample["label"]).strip().lower()
        audio = _resolve(str(sample["audio"]), root)
        reference = str(sample.get("reference_text") or default_reference).strip()
        groups[f"{control}:{label}"].append((audio, reference))

    conditions: Dict[str, Any] = {}
    for key, entries in sorted(groups.items()):
        missing = [str(path) for path, _ in entries if not path.exists()]
        if missing:
            conditions[key] = {
                "status": "unavailable",
                "reason": f"missing audio: {missing}",
                "wer": None,
            }
            continue
        if not all(reference for _, reference in entries):
            conditions[key] = {
                "status": "unavailable",
                "reason": "missing reference_text",
                "wer": None,
            }
            continue
        conditions[key] = transcribe_wer(
            [path for path, _ in entries],
            [reference for _, reference in entries],
            model_size=model_size,
            device=device,
        )

    computed = {
        key: float(value["wer"])
        for key, value in conditions.items()
        if value.get("status") == "computed" and value.get("wer") is not None
    }
    if not computed:
        status = "unavailable"
    elif len(computed) == len(conditions):
        status = "computed"
    else:
        status = "partial"
    return {
        "status": status,
        "conditions": conditions,
        "mean": float(sum(computed.values()) / len(computed)) if computed else None,
        "worst": max(computed.values()) if computed else None,
        "reason": (
            None
            if computed
            else "Whisper, ffmpeg, or reference text unavailable for every condition"
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate emotion and accent controls without fabricating metrics"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "expression_controls.json",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Base directory for relative audio paths (default: project root)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--compute-wer",
        action="store_true",
        help="Transcribe with Whisper and compute per-condition WER",
    )
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument(
        "--wer",
        type=float,
        default=None,
        help="External measured WER to use for the gate (overrides computed worst)",
    )
    parser.add_argument(
        "--max-wer",
        type=float,
        default=None,
        help="Fail the WER gate above this value; requires --wer or --compute-wer",
    )
    parser.add_argument("--min-speaker-similarity", type=float, default=0.80)
    parser.add_argument("--min-classifier-accuracy", type=float, default=0.50)
    parser.add_argument("--min-monotonicity-rho", type=float, default=0.80)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit non-zero unless every measured gate passes",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = load_expression_manifest(args.manifest)
    root = args.root.resolve()

    report = evaluate_expression_manifest(
        manifest,
        root=root,
        device=args.device,
    )

    wer_report: Optional[Dict[str, Any]] = None
    if args.compute_wer:
        wer_report = compute_wer_by_condition(
            manifest,
            root=root,
            device=args.device,
            model_size=args.whisper_model,
        )
        report["wer"] = wer_report

    gate_wer = args.wer
    if gate_wer is None and wer_report is not None:
        gate_wer = wer_report.get("worst")
    gates = check_expression_gates(
        report,
        wer=gate_wer,
        max_wer=args.max_wer,
        min_speaker_similarity=args.min_speaker_similarity,
        min_classifier_accuracy=args.min_classifier_accuracy,
        min_monotonicity_rho=args.min_monotonicity_rho,
    )

    output = {
        "manifest": str(args.manifest),
        "generated_by": "scripts/evaluation/evaluate_expression_controls.py",
        "report": report,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Manifest: {args.manifest}")
    print(f"Prosody: {report['prosody']['status']}")
    print(f"Speaker preservation: {report['speaker_preservation']['status']}")
    print(f"Emotion classifier: {report['emotion_classifier']['status']}")
    print(f"Accent classifier: {report['accent_classifier']['status']}")
    print(f"MOS: {report['mos']['status']} ({report['mos']['reason']})")
    if wer_report is not None:
        print(f"WER: {wer_report['status']} (worst={wer_report['worst']})")
    print(f"Gates: {gates['status']}")
    for name, gate in gates["gates"].items():
        value = gate.get("value")
        detail = f"value={value}" if value is not None else gate.get("status")
        print(f"  - {name}: {gate.get('passed')} ({detail})")
    print(f"Wrote {args.output}")

    if args.require_pass and gates["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
