"""Summarize emotion reports for the controlled scalar-modifier comparison.

The direction gate is only meaningful when the post-synthesis scalar profile is
held constant.  This script reads already-written reports and prints a compact
table with the scalar mode, direction effects, and identity drift.  It does not
run models and does not change any report.

Example::

    python scripts/evaluation/summarize_emotion_controls.py \
        --report e1_prosody=results/emotion/e1/report_prosody.json \
        --report e1_prosody_controlled=results/emotion/e1c/report_prosody.json \
        --report e2_main=results/emotion/e2/report_main.json \
        --report e2_main_controlled=results/emotion/e2c/report_main.json \
        --output-json results/emotion/controlled_comparison.json \
        --output-md results/emotion/CONTROLLED_COMPARISON.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Named report to include; repeat for each condition",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def _load(label_path: str) -> Dict[str, Any]:
    if "=" not in label_path:
        raise ValueError(f"--report must be LABEL=PATH, got {label_path!r}")
    label, path = label_path.split("=", 1)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"label": label, "path": path, "report": payload}


def _wer_block(report: Dict[str, Any]) -> Dict[str, Any]:
    value = report.get("wer")
    if not isinstance(value, dict):
        value = report.get("metrics", {}).get("wer", {})
    return value if isinstance(value, dict) else {}


def _identity_block(report: Dict[str, Any]) -> Dict[str, float]:
    value = (
        report.get("metrics", {})
        .get("ecapa_generated_neutral", {})
        .get("value", {})
    )
    if not isinstance(value, dict):
        return {}
    neutral_key = next(
        (key for key in value if key.startswith("neutral:")), None
    )
    if neutral_key is None:
        return {}
    neutral = float(value[neutral_key])
    drops = {}
    for key, cosine in value.items():
        if key == neutral_key:
            continue
        emotion = key.split(":", 1)[-1]
        drops[emotion] = {
            "cosine_to_generated_neutral": float(cosine),
            "drop": float(neutral - float(cosine)),
        }
    return drops


def summarize(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    conditions: Dict[str, Any] = {}
    for entry in entries:
        report = entry["report"]
        generation = report.get("generation", {})
        if not isinstance(generation, dict):
            generation = {}
        directions = (
            report.get("gates", {})
            .get("e0_directions", {})
            .get("checks", {})
        )
        identity = _identity_block(report)
        wer = _wer_block(report)
        wer_conditions = (
            wer.get("conditions", {}) if isinstance(wer.get("conditions"), dict) else {}
        )
        rows: Dict[str, Any] = {}
        for emotion, checks in directions.items():
            emotion_wer = next(
                (
                    outcome
                    for key, outcome in wer_conditions.items()
                    if str(key).endswith(f":{emotion}")
                ),
                None,
            )
            row: Dict[str, Any] = {
                "identity": identity.get(emotion),
                "wer": emotion_wer,
            }
            for metric, check in checks.items():
                row[metric] = {
                    "mean_delta": check.get("mean_delta"),
                    "ci_low": check.get("ci_low"),
                    "ci_high": check.get("ci_high"),
                    "passed": check.get("passed"),
                }
            rows[str(emotion)] = row
        passed = sum(
            1
            for checks in directions.values()
            for check in checks.values()
            if check.get("passed")
        )
        total = sum(len(checks) for checks in directions.values())
        conditions[entry["label"]] = {
            "path": entry["path"],
            "run_id": report.get("run_id"),
            "scalar_modifiers": generation.get("scalar_modifiers"),
            "emotions_generated": generation.get("emotions_generated"),
            "direction_checks_passed": int(passed),
            "direction_checks_total": int(total),
            "neutral_zero": report.get("gates", {})
            .get("neutral_zero", {})
            .get("passed"),
            "identity_gate": report.get("gates", {})
            .get("identity", {})
            .get("passed"),
            "classifier_status": report.get("metrics", {})
            .get("classifier_accuracy", {})
            .get("status"),
            "wer_status": wer.get("status"),
            "wer_mean": wer.get("mean"),
            "wer_worst": wer.get("worst"),
            "rows": rows,
        }
    return {"conditions": conditions}


def markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Emotion condition comparison",
        "",
        "Direction effects use paired CREMA-D validation actors. A direction",
        "check is only interpretable when `scalar_modifiers` is `neutral`.",
        "`None` means the run predates the field and used each emotion's",
        "post-synthesis scalar profile; those runs are confounded.",
        "",
        "`e2_label_shuffled_neutral_scalars` is the corrected label-shuffled",
        "null (labels permuted, targets fixed). The `e2_style_shuffled_*` rows",
        "are historical diagnostics: they preserve the label-target association",
        "and are not valid nulls.",
        "",
        "WER is measured with Whisper on the generated audio against the CREMA-D",
        "transcript. It is paired but noisy on short clips; `unavailable` means it",
        "was not requested or could not be measured.",
        "",
        "| Condition | Scalars | Directions passed | Neutral-zero | Identity gate | WER mean |",
        "|---|---|---|---|---|---|",
    ]
    for label, block in summary["conditions"].items():
        scalars = block["scalar_modifiers"]
        if scalars is None:
            scalars = "pre-field (emotion profile)"
        wer_mean = block.get("wer_mean")
        wer_text = f"{float(wer_mean):.4f}" if wer_mean is not None else "unavailable"
        lines.append(
            f"| {label} | {scalars} | "
            f"{block['direction_checks_passed']}/{block['direction_checks_total']} | "
            f"{block['neutral_zero']} | {block['identity_gate']} | {wer_text} |"
        )
    lines.extend(
        [
            "",
            "| Condition | Emotion | F0 delta (Hz) [95% CI] | Duration delta (s) [95% CI] | ECAPA drop vs gen. neutral | WER |",
            "|---|---|---|---|---|---|",
        ]
    )
    for label, block in summary["conditions"].items():
        for emotion, row in block["rows"].items():
            f0 = row.get("f0_mean_hz") or {}
            duration = row.get("duration_seconds") or {}
            identity = row.get("identity") or {}
            f0_text = (
                f"{f0.get('mean_delta'):+.2f} "
                f"[{f0.get('ci_low'):+.2f}, {f0.get('ci_high'):+.2f}]"
                if f0.get("mean_delta") is not None
                else "unavailable"
            )
            duration_text = (
                f"{duration.get('mean_delta'):+.3f} "
                f"[{duration.get('ci_low'):+.3f}, {duration.get('ci_high'):+.3f}]"
                if duration.get("mean_delta") is not None
                else "unavailable"
            )
            drop = identity.get("drop")
            drop_text = f"{drop:.3f}" if drop is not None else "unavailable"
            wer_outcome = row.get("wer") or {}
            wer_value = wer_outcome.get("wer")
            wer_text = (
                f"{float(wer_value):.4f}"
                if wer_value is not None
                else "unavailable"
            )
            lines.append(
                f"| {label} | {emotion} | {f0_text} | {duration_text} | "
                f"{drop_text} | {wer_text} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.report:
        raise SystemExit("at least one --report LABEL=PATH is required")
    entries = [_load(item) for item in args.report]
    summary = summarize(entries)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown(summary), encoding="utf-8")
        print(f"Wrote {args.output_md}")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
