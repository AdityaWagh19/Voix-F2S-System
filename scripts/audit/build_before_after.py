"""Build the before/after audio comparison for the VOIX research demo.

The comparison has three parts:

1. controlled style-manifold pairs: same face/text/seed, alpha 0 (raw face
   style) vs alpha 1.0 (calibrated).  This is the causal before/after for the
   hiss fix.
2. demo-level before/after: the preserved v1 gallery vs the v2 3-voice demo.
   The identities differ, so this is a distributional comparison, not a paired
   identity test.
3. negative controls: real-AMI reference style, N1 LoRA, and diffusion steps.

Every number is copied from an existing audit report.  No metric is recomputed
and no perceptual claim is made.

Usage:
    python scripts/audit/build_before_after.py
"""
from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
V2_REPORT = ROOT / "results" / "demo" / "research_v2" / "report.json"
V1_REPORT = (
    ROOT / "results" / "audit" / "baseline_audio" / "snapshot_20261105" / "report.json"
)
V1_METRICS = ROOT / "results" / "audit" / "baseline_audio" / "audio_metrics.json"
CALIBRATION = (
    ROOT / "results" / "audit" / "style_calibration" / "style_calibration_audit.json"
)
MATRIX = ROOT / "results" / "audit" / "synthesis_matrix" / "synthesis_matrix.json"
OUTPUT = ROOT / "results" / "audit" / "before_after"
SELECTED = (
    "C2_12_black_female_Lupita_Nyongo",
    "C1_02_young_male_Timothee_Chalamet",
    "C2_09_east_asian_male_Jackie_Chan",
)


def _metric(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    return None if value is None else float(value)


def _row(
    *,
    experiment: str,
    group: str,
    label: str,
    voice: str,
    variant: str,
    source: Path,
    destination: Path,
    metrics: Mapping[str, Any],
    note: str,
) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copyfile(source, destination)
        copied = destination.relative_to(OUTPUT).as_posix()
    else:
        copied = None
    return {
        "experiment": experiment,
        "group": group,
        "label": label,
        "voice": voice,
        "variant": variant,
        "audio": copied,
        "noise_floor_dbfs": _metric(metrics, "noise_floor_dbfs"),
        "snr_proxy_db": _metric(metrics, "snr_proxy_db"),
        "band_4000_6000_share_db": _metric(metrics, "speech_band_4000_6000_share_db"),
        "band_6000_8000_share_db": _metric(metrics, "speech_band_6000_8000_share_db"),
        "f0_median_hz": _metric(metrics, "nat_f0_median_hz")
        or _metric(metrics, "f0_median_hz"),
        "duration_s": _metric(metrics, "duration_s"),
        "note": note,
    }


def main() -> int:
    if not V2_REPORT.is_file():
        raise FileNotFoundError(f"v2 report not found: {V2_REPORT}")
    v2 = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    v1 = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    v1_metrics = {
        clip["file"]: clip
        for clip in json.loads(V1_METRICS.read_text(encoding="utf-8"))["clips"]
    }
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    # 1. Controlled alpha pairs for the three selected voices.
    by_face_alpha: Dict[str, Dict[float, Mapping[str, Any]]] = {}
    for condition in calibration["conditions"]:
        by_face_alpha.setdefault(condition["face_label"], {})[
            float(condition["alpha"])
        ] = condition
    for face_label in SELECTED:
        pair = by_face_alpha.get(face_label, {})
        voice = "_".join(face_label.split("_")[:2])
        for alpha, variant in ((0.0, "raw_face_style"), (1.0, "calibrated")):
            condition = pair.get(alpha)
            if condition is None:
                continue
            rows.append(
                _row(
                    experiment="controlled_alpha",
                    group=face_label,
                    label=f"{voice} alpha {alpha:g}",
                    voice=voice,
                    variant=variant,
                    source=Path(condition["wav_path"]),
                    destination=OUTPUT / "audio" / "controlled" / f"{voice}__alpha{alpha:g}.wav",
                    metrics=condition["metrics"],
                    note=(
                        "Same face, prompt, seed, and synthesis settings; only the "
                        "style-manifold translation differs."
                    ),
                )
            )

    # 2. Demo-level v1 vs v2 neutrals.
    v1_audio = V1_REPORT.parent / "audio"
    v1_metric = v1_metrics.get("neutral.wav")
    if v1_metric is not None:
        rows.append(
            _row(
                experiment="demo_level",
                group="legacy_v1",
                label="v1 neutral (single face)",
                voice="legacy",
                variant="v1",
                source=v1_audio / "neutral.wav",
                destination=OUTPUT / "audio" / "demo" / "v1__neutral.wav",
                metrics=v1_metric,
                note=(
                    "Preserved v1 gallery: one different face, raw face style, "
                    "12 conditions."
                ),
            )
        )
    for row in v2["conditions"]:
        if row["track"] != "baseline":
            continue
        rows.append(
            _row(
                experiment="demo_level",
                group=row["voice_label"],
                label=f"v2 neutral ({row['voice_id']})",
                voice=row["voice_id"],
                variant="v2",
                source=Path(row["wav_path"]),
                destination=OUTPUT / "audio" / "demo" / f"v2__{row['name']}.wav",
                metrics=row["metrics"],
                note=(
                    "v2 gallery: calibrated style, one of three selected "
                    "face-derived voices."
                ),
            )
        )

    # 3. Negative controls from the controlled synthesis matrix.
    matrix_by_name = {condition["name"]: condition for condition in matrix["conditions"]}
    for name, variant, note in (
        ("base", "face_style", "Raw face style, current demo path."),
        ("lora", "n1_lora", "N1 AMI LoRA adapter applied to the same face style."),
        ("steps5", "diffusion_5", "Five diffusion steps; hiss persists."),
        ("steps30", "diffusion_30", "Thirty diffusion steps; hiss persists."),
        ("real_ref", "real_ami_style", "Real AMI reference style; same model, prompt, and seed."),
    ):
        condition = matrix_by_name.get(name)
        if condition is None:
            continue
        rows.append(
            _row(
                experiment="negative_controls",
                group="synthesis_matrix",
                label=name,
                voice="legacy_face",
                variant=variant,
                source=Path(condition["wav_path"]),
                destination=OUTPUT / "audio" / "controls" / f"{name}.wav",
                metrics=condition["metrics"],
                note=note,
            )
        )

    output_json = OUTPUT / "comparison.json"
    output_json.write_text(
        json.dumps(
            {
                "schema_version": "voix.before_after.v1",
                "level": "exploratory",
                "v2_report": str(V2_REPORT),
                "v1_report": str(V1_REPORT),
                "calibration_audit": str(CALIBRATION),
                "synthesis_matrix": str(MATRIX),
                "rows": rows,
                "notes": [
                    "Controlled alpha pairs are paired by face, prompt, seed, and synthesis settings.",
                    "Demo-level v1/v2 rows use different faces; compare distributions, not identities.",
                    "Negative controls come from the controlled synthesis matrix.",
                    "No MOS/CMOS or perceptual judgment is included.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = list(rows[0])
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    sections = []
    for group, group_rows in groups.items():
        cards = []
        for row in group_rows:
            audio = (
                f'<audio controls preload="none" src="{html.escape(row["audio"])}"></audio>'
                if row["audio"]
                else "<p class=\"missing\">audio unavailable</p>"
            )
            metrics = " · ".join(
                f"{name} {value:.2f}"
                for name, value in (
                    ("noise dBFS", row["noise_floor_dbfs"]),
                    ("SNR dB", row["snr_proxy_db"]),
                    ("6-8k dB", row["band_6000_8000_share_db"]),
                    ("F0 Hz", row["f0_median_hz"]),
                )
                if value is not None
            )
            cards.append(
                f"""
                <article class="card">
                  <h3>{html.escape(row['label'])}</h3>
                  <p class="tag">{html.escape(row['variant'])}</p>
                  {audio}
                  <p class="metrics">{html.escape(metrics)}</p>
                  <p class="note">{html.escape(row['note'])}</p>
                </article>
                """
            )
        sections.append(
            f"<section><h2>{html.escape(group)}</h2><div class=\"grid\">{''.join(cards)}</div></section>"
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VOIX before/after audio comparison</title>
<style>
  :root {{ color-scheme: dark; --bg:#0f1115; --panel:#181c23; --line:#2b313c; --ink:#e8ecf3; --muted:#9aa6b8; --accent:#7cc4ff; }}
  body {{ margin:0; font:15px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--ink); }}
  header, main {{ max-width:1180px; margin:0 auto; padding:24px; }}
  header {{ border-bottom:1px solid var(--line); }}
  h1 {{ font-size:28px; margin:0 0 8px; }}
  h2 {{ color:var(--accent); font-size:18px; }}
  h3 {{ font-size:15px; margin:0 0 4px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:12px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; }}
  audio {{ width:100%; margin:6px 0; }}
  .tag, .metrics, .note {{ color:var(--muted); font-size:12px; }}
  .note {{ font-size:13px; }}
  .missing {{ color:#ff9a9a; }}
  code {{ color:#d6e6ff; }}
</style></head><body>
<header>
  <h1>VOIX before/after audio comparison</h1>
  <p class="muted">Controlled alpha pairs, demo-level v1/v2 snapshots, and negative controls. No MOS/CMOS or perceptual rating is claimed.</p>
  <p class="muted">Every metric is copied from the linked audit reports; see <a href="comparison.csv">comparison.csv</a> and <a href="comparison.json">comparison.json</a>.</p>
</header>
<main>{''.join(sections)}</main>
</body></html>
"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'comparison.json'}")
    print(f"Wrote {OUTPUT / 'comparison.csv'}")
    print(f"Wrote {OUTPUT / 'index.html'}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
