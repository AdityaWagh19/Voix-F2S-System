"""VOIX human listening (MOS) protocol harness.

This script does **not** synthesize MOS scores. It prepares blinded stimuli and
a rating sheet for human listeners, then computes real MOS statistics only when
an actual ratings CSV is supplied.

Protocol:
  1. Run this script to generate ``artifacts/mos_stimuli/``, ``manifest.json``,
     and ``ratings_template.csv``.
  2. Human raters listen to each blinded WAV and fill in 1-5 scores for
     naturalness, biometric fidelity, and speaker similarity.
  3. Re-run with ``--ratings path/to/ratings.csv`` to compute means, standard
     deviations, and 95% confidence intervals.

Usage:
  python scripts/evaluation/run_mos_evaluation.py --num-subjects 7
  python scripts/evaluation/run_mos_evaluation.py --ratings artifacts/mos_ratings.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SCORE_FIELDS = ("naturalness", "biometric_fidelity", "speaker_similarity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VOIX human MOS protocol harness")
    p.add_argument("--num-subjects", type=int, default=7, help="Number of test subjects")
    p.add_argument("--output", default=str(ROOT / "artifacts" / "mos_evaluation_report.json"))
    p.add_argument("--stimuli-dir", default=str(ROOT / "artifacts" / "mos_stimuli"))
    p.add_argument("--ratings", default=None, help="CSV produced from ratings_template.csv")
    p.add_argument("--text", default="The human voice reflects both physical anatomy and lived culture.")
    return p.parse_args()


def _subject_list(limit: int) -> List[Dict[str, Any]]:
    subjects = [
        ("C1_01_young_female_Zendaya.jpg", "Young female", 0, 26),
        ("C1_04_adult_male_George_Clooney.jpg", "Adult male", 1, 62),
        ("C2_08_south_asian_female_Priyanka_Chopra.jpg", "South Asian female", 0, 41),
        ("C2_11_black_male_Barack_Obama.jpg", "Black male", 1, 62),
        ("C2_15_latino_male_Pedro_Pascal.jpg", "Latino male", 1, 48),
        ("C3_17_prominent_jaw_Arnold_Schwarzenegger.jpg", "Prominent jaw male", 1, 76),
        ("SD_07_rhotacism_Barbara_Walters.jpg", "Rhoticism female", 0, 80),
    ]
    return [
        {"filename": f, "description": d, "gender": g, "age": a}
        for f, d, g, a in subjects[: max(1, limit)]
    ]


def _prepare_stimuli(args: argparse.Namespace) -> Dict[str, Any]:
    import soundfile as sf

    from voix.inference.pipeline import VoixPipeline

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    pipeline = VoixPipeline(device=device, enable_watermarking=False)
    stimuli_dir = Path(args.stimuli_dir)
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    counter = 1
    for subject in _subject_list(args.num_subjects):
        face_path = ROOT / "data" / "test_subjects" / subject["filename"]
        if not face_path.exists():
            print(f"[MOS] Skipping missing subject image: {face_path}")
            continue
        outputs = pipeline.generate(
            face_input=str(face_path),
            text=args.text,
            num_voices=3,
            temperature=1.1,
            gender=subject["gender"],
            age=subject["age"],
            watermark=False,
        )
        for output in outputs:
            stimulus_id = f"stim_{counter:03d}"
            counter += 1
            wav = output["waveform"].detach().cpu().squeeze().numpy()
            audio_path = stimuli_dir / f"{stimulus_id}.wav"
            sf.write(str(audio_path), wav, output["sample_rate"])
            manifest.append({
                "stimulus_id": stimulus_id,
                "audio_path": str(audio_path.relative_to(ROOT)),
                "subject_filename": subject["filename"],
                "subject_description": subject["description"],
                "candidate_index": output["voice_index"],
                "backend": output.get("backend", ""),
                "sample_rate": output["sample_rate"],
                "duration_seconds": round(len(wav) / output["sample_rate"], 3),
            })

    template_path = stimuli_dir.parent / "mos_ratings_template.csv"
    with open(template_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rater_id", "stimulus_id", *SCORE_FIELDS, "comments"],
        )
        writer.writeheader()
        for item in manifest:
            writer.writerow({
                "rater_id": "",
                "stimulus_id": item["stimulus_id"],
                "naturalness": "",
                "biometric_fidelity": "",
                "speaker_similarity": "",
                "comments": "",
            })

    manifest_path = stimuli_dir.parent / "mos_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"stimuli": manifest}, f, indent=2)

    return {
        "status": "awaiting_human_ratings",
        "human_mos": False,
        "message": (
            "Stimuli and a blank rating sheet were generated. No MOS scores are "
            "reported until real listener ratings are supplied via --ratings."
        ),
        "num_stimuli": len(manifest),
        "stimuli_dir": str(stimuli_dir),
        "manifest": str(manifest_path),
        "ratings_template": str(template_path),
        "scores": None,
        "stimuli": manifest,
    }


def _mean_ci(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
        # Normal-approximation 95% CI; adequate for the expected n > 20.
        half = 1.96 * std / math.sqrt(n)
    else:
        std = 0.0
        half = 0.0
    return {
        "n": n,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "ci95_low": round(max(1.0, mean - half), 3),
        "ci95_high": round(min(5.0, mean + half), 3),
    }


def _compute_ratings(args: argparse.Namespace) -> Dict[str, Any]:
    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings CSV not found: {ratings_path}")

    scores: Dict[str, List[float]] = {field: [] for field in SCORE_FIELDS}
    raters = set()
    per_stimulus: Dict[str, Dict[str, List[float]]] = {}
    with open(ratings_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rater = (row.get("rater_id") or "").strip()
            stimulus = (row.get("stimulus_id") or "").strip()
            if not rater or not stimulus:
                continue
            parsed = {}
            for field in SCORE_FIELDS:
                raw = (row.get(field) or "").strip()
                if not raw:
                    continue
                value = float(raw)
                if not 1.0 <= value <= 5.0:
                    raise ValueError(f"{field}={value} for {stimulus} is outside 1-5")
                parsed[field] = value
                scores[field].append(value)
                per_stimulus.setdefault(stimulus, {f: [] for f in SCORE_FIELDS})[field].append(value)
            if parsed:
                raters.add(rater)

    if not raters:
        raise ValueError(
            f"No complete ratings found in {ratings_path}. Fill in the template "
            "generated by this script before requesting MOS statistics."
        )

    return {
        "status": "complete",
        "human_mos": True,
        "num_raters": len(raters),
        "num_ratings": sum(len(v) for v in scores.values()),
        "ratings_csv": str(ratings_path),
        "scores": {field: _mean_ci(values) for field, values in scores.items()},
        "per_stimulus": {
            stimulus: {field: _mean_ci(values) for field, values in fields.items()}
            for stimulus, fields in sorted(per_stimulus.items())
        },
    }


def main() -> None:
    args = parse_args()
    print("=" * 68)
    print("VOIX Human MOS Protocol")
    print("=" * 68)

    if args.ratings:
        report = _compute_ratings(args)
        if report["scores"]:
            for field in SCORE_FIELDS:
                s = report["scores"][field]
                print(f"  {field:22s} {s['mean']:.2f} +/- {s['std']:.2f} "
                      f"(95% CI {s['ci95_low']:.2f}-{s['ci95_high']:.2f}, n={s['n']})")
    else:
        report = _prepare_stimuli(args)
        print(f"  Prepared {report['num_stimuli']} blinded stimuli.")
        print(f"  Manifest:  {report['manifest']}")
        print(f"  Ratings:   {report['ratings_template']}")
        print("  NO MOS SCORES COMPUTED. Human ratings are required.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report:    {out_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
