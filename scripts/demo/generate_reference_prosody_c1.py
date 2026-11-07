"""C1: reference-prosody transfer with the frozen StyleTTS2 style encoder.

The face-to-voice path produces a 256-D style vector whose first 128 values are
timbre and whose second 128 values are prosody.  C1 keeps the face-derived
timbre half and replaces the prosody half with the mean prosody of real
CREMA-D training clips for the requested emotion.  No residual is trained and
no recognizer is used as a controller.

The reference clips are selected from the CREMA-D ``train`` split only, with a
gender match and one clip per actor, so the evaluation splits stay untouched.
This is a research transform.  It is not wired into serving and no perceptual
emotion-recognition claim is attached to it.

Usage:
    python scripts/demo/generate_reference_prosody_c1.py --device cuda
"""
from __future__ import annotations

import os

# The repo's TensorFlow install is unusable (einops' TF backend crashes on
# `tensorflow.Tensor`).  Keep transformers and einops on the torch path.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_FACE_REPORT = (
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json"
)
DEFAULT_CALIBRATION = ROOT / "results" / "audit" / "style_calibration" / "calibration.json"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_manifest.csv"
DEFAULT_STYLE_CACHE = ROOT / "data" / "processed" / "crema_d_style_cache_train.npz"
DEFAULT_OUTPUT = ROOT / "results" / "emotion" / "c1"
DEFAULT_TEXT = (
    "The human voice carries identity, emotion, and the rhythm of lived "
    "experience."
)
EMOTIONS: Tuple[str, ...] = ("happy", "sad", "angry")
INTENSITY_ORDER = {"unspecified": 3, "high": 2, "medium": 1, "low": 0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_sha256(vector: torch.Tensor) -> str:
    return hashlib.sha256(
        np.asarray(vector.detach().cpu(), dtype=np.float32).round(8).tobytes()
    ).hexdigest()


def _demo_module():
    path = ROOT / "scripts" / "demo" / "generate_research_demo.py"
    spec = importlib.util.spec_from_file_location("generate_research_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the C1 reference-prosody set")
    parser.add_argument("--faces", nargs="+", default=None)
    parser.add_argument("--face-report", type=Path, default=DEFAULT_FACE_REPORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--calibration-alpha", type=float, default=1.0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--style-cache", type=Path, default=DEFAULT_STYLE_CACHE)
    parser.add_argument("--references-per-class", type=int, default=8)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--loudness-lufs", type=float, default=-23.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _reference_styles(
    *,
    manifest_path: Path,
    cache_path: Path,
    sex: str,
    emotions: Sequence[str],
    per_class: int,
    seed: int,
) -> Dict[str, Any]:
    """Select one clip per training actor and average its prosody half."""
    frame = pd.read_csv(manifest_path)
    required = {"clip_id", "actor_id", "intended_emotion", "intensity", "split", "sex"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CREMA-D manifest is missing columns: {missing}")

    cache = np.load(cache_path, allow_pickle=True)
    paths = cache["paths"].astype(str)
    styles = cache["styles"].astype(np.float32)
    index_by_stem = {Path(path).stem: index for index, path in enumerate(paths)}

    frame = frame[frame["split"].astype(str) == "train"].copy()
    frame = frame[frame["sex"].astype(str) == str(sex)].copy()
    frame = frame[frame["clip_id"].astype(str).isin(index_by_stem)].copy()
    if frame.empty:
        raise ValueError(f"no cached CREMA-D training references for {sex}")
    frame = frame[frame["intended_emotion"].astype(str).isin(set(emotions))].copy()

    frame["_intensity_rank"] = (
        frame["intensity"].astype(str).str.lower().map(INTENSITY_ORDER).fillna(-1)
    )
    frame = frame.sort_values(
        ["intended_emotion", "actor_id", "_intensity_rank", "clip_id"],
        ascending=[True, True, False, True],
    )
    frame = frame.drop_duplicates(subset=["intended_emotion", "actor_id"], keep="first")

    selected: Dict[str, Any] = {}
    for emotion in emotions:
        subset = frame[frame["intended_emotion"].astype(str) == str(emotion)]
        actors = sorted(subset["actor_id"].astype(str).unique())
        if not actors:
            raise ValueError(f"no {sex} {emotion} references in the training cache")
        rng = np.random.default_rng(int(seed) + sum(ord(char) for char in f"{sex}{emotion}"))
        take = min(int(per_class), len(actors))
        chosen = sorted(rng.choice(actors, size=take, replace=False).tolist())
        rows = subset[subset["actor_id"].astype(str).isin(set(chosen))]
        rows = rows.sort_values("actor_id")
        clip_ids = rows["clip_id"].astype(str).tolist()
        style_rows = np.stack(
            [styles[index_by_stem[clip_id], 128:] for clip_id in clip_ids]
        )
        mean_prosody = style_rows.mean(axis=0).astype(np.float32)
        selected[str(emotion)] = {
            "emotion": str(emotion),
            "sex": str(sex),
            "split": "train",
            "n_references": int(len(clip_ids)),
            "actor_ids": chosen,
            "clip_ids": clip_ids,
            "prosody_mean_norm": float(np.linalg.norm(mean_prosody)),
            "prosody_mean_sha256": hashlib.sha256(
                mean_prosody.astype(np.float32).round(8).tobytes()
            ).hexdigest(),
            "prosody_mean": mean_prosody,
        }
    return selected


def _face_styles(
    demo: Any,
    pipeline: Any,
    *,
    faces: Sequence[str],
    face_report: Mapping[str, Any],
    calibration: Mapping[str, Any],
    alpha: float,
) -> List[Dict[str, Any]]:
    entries = {entry["label"]: entry for entry in face_report["entries"]}
    missing = [label for label in faces if label not in entries]
    if missing:
        raise KeyError(f"faces missing from the diversity report: {missing}")

    voices: List[Dict[str, Any]] = []
    for label in faces:
        entry = entries[label]
        face_path = Path(str(entry["face"]))
        if not face_path.is_file():
            raise FileNotFoundError(f"face image not found: {face_path}")
        if _sha256(face_path) != str(entry["face_sha256"]):
            raise RuntimeError(f"face image hash mismatch for {label}")
        style, gender, age = demo._face_style(pipeline, face_path)
        reference_key = (
            "style_without_geo_vector"
            if "style_without_geo_vector" in entry
            else "style_vector"
        )
        expected = np.asarray(entry[reference_key], dtype=np.float64).reshape(-1)
        actual = np.asarray(style.detach().cpu(), dtype=np.float64).reshape(-1)
        if not np.allclose(actual, expected, atol=1e-5, rtol=0.0):
            raise RuntimeError(f"recomputed face style does not match for {label}")
        gender_key = "female" if int(gender) == 0 else "male"
        calibrated = demo.apply_translation(
            style, calibration, gender_key, alpha=float(alpha)
        )
        voices.append(
            {
                "id": "_".join(label.split("_")[:2]),
                "label": label,
                "face_path": str(face_path),
                "face_sha256": _sha256(face_path),
                "gender": int(gender),
                "sex": "Female" if int(gender) == 0 else "Male",
                "age": float(age),
                "derived_style_sha256": demo._style_sha256(style),
                "calibrated_style_sha256": demo._style_sha256(calibrated),
                "calibrated_style": calibrated,
            }
        )
    return voices


def _write_readme(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# C1: reference-prosody transfer",
        "",
        "C1 keeps the frozen face-derived timbre half of the StyleTTS2 style",
        "vector and replaces the prosody half with the mean prosody of real",
        "CREMA-D **training** clips for the requested emotion.",
        "",
        "> This is a research transform. It is not wired into serving, and no",
        "> MOS, CMOS, or human emotion-identification study was run.",
        "",
        "## References",
        "",
        f"- CREMA-D manifest: `{report['inputs']['manifest']}`",
        f"- Cached train styles: `{report['inputs']['style_cache']}`",
        f"- Split: `train` only",
        f"- Gender match: yes",
        f"- References per class: {report['config']['references_per_class']} (one clip per actor)",
        "",
        "| Sex | Emotion | References | Mean prosody norm |",
        "| --- | --- | --- | --- |",
    ]
    for sex, emotions in report["references"].items():
        for emotion, entry in emotions.items():
            lines.append(
                f"| {sex} | {emotion} | {entry['n_references']} | "
                f"{entry['prosody_mean_norm']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Clips",
            "",
            "| Voice | Condition | Audio |",
            "| --- | --- | --- |",
        ]
    )
    for row in report["rows"]:
        lines.append(f"| {row['voice_label']} | {row['condition']} | `{row['wav_path']}` |")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- Clips generated: {report['summary']['n_clips']}",
            f"- Timbre half preserved per voice: {report['summary']['timbre_preserved']}",
            f"- Styles distinct from neutral: {report['summary']['styles_distinct']}",
            f"- Minimum ECAPA-to-neutral (all rows): {report['summary']['min_ecapa_to_neutral']}",
            f"- Minimum ECAPA-to-neutral (emotion rows): {report['summary']['min_ecapa_to_neutral_emotional']}",
            "",
            "## Limitations",
            "",
            "- CREMA-D is acted speech; the references are not a naturalistic corpus.",
            "- Only three demo voices and three emotions were tested.",
            "- The timbre and prosody halves of StyleTTS2's style vector may not be",
            "  fully independent; replacing one half is an approximation.",
            "- Panel recognition results, if present, are model outputs, not human",
            "  perception.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    demo = _demo_module()
    faces = list(args.faces) if args.faces else list(demo.DEFAULT_VOICES)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    face_report = json.loads(args.face_report.read_text(encoding="utf-8"))
    calibration = demo.load_calibration(args.calibration)
    demo.validate_calibration(calibration)

    pipeline = demo._load_pipeline(args.device)
    voices = _face_styles(
        demo,
        pipeline,
        faces=faces,
        face_report=face_report,
        calibration=calibration,
        alpha=float(args.calibration_alpha),
    )
    # Release the visual extractors before the vocoder claims GPU memory.
    pipeline._arcface = None
    pipeline._facemesh = None
    pipeline._demog = None
    pipeline._feature_builder = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    references: Dict[str, Dict[str, Any]] = {}
    for sex in ("Female", "Male"):
        references[sex] = _reference_styles(
            manifest_path=args.manifest,
            cache_path=args.style_cache,
            sex=sex,
            emotions=EMOTIONS,
            per_class=int(args.references_per_class),
            seed=int(args.seed),
        )

    print("[setup] loading StyleTTS2", flush=True)
    tts = pipeline.tts
    tts.load()
    if tts._model is None or getattr(tts, "_use_fallback", True):
        raise RuntimeError("the neural StyleTTS2 backend failed to load")
    sample_rate = int(tts.sample_rate)

    rows: List[Dict[str, Any]] = []
    for voice in voices:
        conditions = [("neutral", voice["calibrated_style"], None)]
        for emotion in EMOTIONS:
            reference = references[voice["sex"]][emotion]
            prosody = torch.from_numpy(reference["prosody_mean"]).float().reshape(1, 128)
            style = torch.cat(
                [voice["calibrated_style"][:, :128], prosody.to(voice["calibrated_style"].device)],
                dim=-1,
            )
            conditions.append((emotion, style, reference))
        for condition, style, reference in conditions:
            name = f"{voice['id']}__{condition if condition == 'neutral' else 'emotion_' + condition}"
            destination = audio_dir / f"{name}.wav"
            synthesis_s: Optional[float] = None
            if args.skip_existing and destination.is_file():
                duration = float(sf.info(str(destination)).duration)
            else:
                print(f"[generate] {name}", flush=True)
                started = time.time()
                waveform = demo._synthesize(
                    tts,
                    text=args.text,
                    style=style,
                    gender=int(voice["gender"]),
                    age=float(voice["age"]),
                    seed=int(args.seed),
                    diffusion_steps=int(args.diffusion_steps),
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    f0_scale=1.0,
                    loudness_lufs=float(args.loudness_lufs),
                )
                duration = demo._save(destination, waveform, sample_rate)
                synthesis_s = float(time.time() - started)
                print(
                    f"  {name}: {duration:.2f}s in {synthesis_s:.1f}s",
                    flush=True,
                )
            rows.append(
                {
                    "name": name,
                    "voice_id": voice["id"],
                    "voice_label": voice["label"],
                    "track": "baseline" if condition == "neutral" else "emotion",
                    "condition": condition,
                    "label": f"{voice['label']}: {condition}",
                    "text": args.text,
                    "wav_path": str(destination),
                    "audio_sha256": _sha256(destination),
                    "sample_rate": sample_rate,
                    "duration_sec": float(duration),
                    "seed": int(args.seed),
                    "diffusion_steps": int(args.diffusion_steps),
                    "alpha": float(args.alpha),
                    "beta": float(args.beta),
                    "style_sha256": _vector_sha256(style),
                    "timbre_sha256": _vector_sha256(style[:, :128]),
                    "calibrated_style_sha256": voice["calibrated_style_sha256"],
                    "reference": (
                        None
                        if reference is None
                        else {
                            "sex": reference["sex"],
                            "n_references": reference["n_references"],
                            "clip_ids": reference["clip_ids"],
                            "prosody_mean_sha256": reference["prosody_mean_sha256"],
                        }
                    ),
                    "timing": {"synthesis_s": synthesis_s, "metrics_s": None},
                }
            )

    index = pd.DataFrame(rows)
    index_path = output_dir / "index.csv"
    index.to_csv(index_path, index=False)

    print("[metrics] measuring audio", flush=True)
    demo._attach_audio_metrics(rows, text=args.text)

    print("[ecapa] identity check", flush=True)
    from speechbrain.inference.speaker import EncoderClassifier

    ecapa_dir = ROOT / "checkpoints" / "speechbrain" / "spkrec-ecapa"
    ecapa = EncoderClassifier.from_hparams(
        source=str(ecapa_dir), savedir=str(ecapa_dir), run_opts={"device": args.device}
    )
    neutral_vectors: Dict[str, np.ndarray] = {}
    for row in rows:
        vector = demo._ecapa_embedding(ecapa, Path(row["wav_path"]), sample_rate)
        row["ecapa_vector"] = vector.tolist()
        if row["condition"] == "neutral":
            neutral_vectors[row["voice_id"]] = vector
    for row in rows:
        row["ecapa_to_neutral"] = float(
            np.dot(neutral_vectors[row["voice_id"]], demo._unit(row["ecapa_vector"]))
        )
    del ecapa
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timbre_preserved = all(
        len({str(row["timbre_sha256"]) for row in rows if row["voice_id"] == voice["id"]}) == 1
        for voice in voices
    )
    styles_distinct = all(
        len({str(row["style_sha256"]) for row in rows if row["voice_id"] == voice["id"]})
        == 1 + len(EMOTIONS)
        for voice in voices
    )
    min_ecapa = min(float(row["ecapa_to_neutral"]) for row in rows)
    emotional = [row for row in rows if row["condition"] != "neutral"]
    min_ecapa_emotional = min(
        float(row["ecapa_to_neutral"]) for row in emotional
    )

    report = {
        "schema_version": "voix.reference_prosody_c1.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "track": "emotion",
        "experiment": "C1",
        "level": "research_transform",
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "style_cache": str(args.style_cache),
            "style_cache_sha256": _sha256(args.style_cache),
            "face_report": str(args.face_report),
            "calibration": str(args.calibration),
            "index": str(index_path),
            "index_sha256": _sha256(index_path),
        },
        "config": {
            "faces": faces,
            "emotions": list(EMOTIONS),
            "references_per_class": int(args.references_per_class),
            "text": args.text,
            "seed": int(args.seed),
            "diffusion_steps": int(args.diffusion_steps),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "loudness_lufs": float(args.loudness_lufs),
            "calibration_alpha": float(args.calibration_alpha),
            "style_composition": "concat(face_calibrated_timbre[:128], reference_prosody_mean[128:])",
        },
        "voices": [
            {key: value for key, value in voice.items() if key != "calibrated_style"}
            for voice in voices
        ],
        "references": {
            sex: {
                emotion: {
                    key: value
                    for key, value in entry.items()
                    if key != "prosody_mean"
                }
                for emotion, entry in emotions.items()
            }
            for sex, emotions in references.items()
        },
        "rows": rows,
        "summary": {
            "n_clips": len(rows),
            "timbre_preserved": bool(timbre_preserved),
            "styles_distinct": bool(styles_distinct),
            "min_ecapa_to_neutral": float(min_ecapa),
            "min_ecapa_to_neutral_emotional": float(min_ecapa_emotional),
        },
        "limitations": [
            "Research transform; not wired into serving.",
            "CREMA-D is acted speech.",
            "Only three voices and three emotions were tested.",
            "No MOS, CMOS, or human emotion-identification study was run.",
            "StyleTTS2 timbre/prosody halves may not be fully independent.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_readme(output_dir / "README.md", report)
    print(f"Wrote {report_path} ({len(rows)} clips)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
