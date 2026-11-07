"""Remove clips that failed the post-download GLOBE audio health check.

The audit is deliberately strict: any source clipping fails a clip.  The
selection gate itself (2.0 h and 30 speakers per trainable class) is checked
again after pruning.  The unpruned manifest is preserved for auditability.

Usage:
    python scripts/data/prune_globe_manifest.py
    python scripts/data/prune_globe_manifest.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_AUDIT = ROOT / "data" / "processed" / "globe_accent_audio_audit.json"
DEFAULT_BACKUP = ROOT / "data" / "processed" / "globe_accent_manifest_unpruned.csv"
DEFAULT_REPORT = ROOT / "data" / "processed" / "globe_accent_manifest_prune_report.json"
DEFAULT_AUDIO_DIR = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_EXCLUDED_DIR = ROOT / "data" / "processed" / "globe_audio_excluded"

TRAINABLE = ["us", "england", "australian", "canadian", "india_south_asia"]


def audio_path(audio_dir: Path, row: pd.Series) -> Path:
    return (
        audio_dir
        / str(row["speaker_split"])
        / str(row["accent_class"])
        / f"{row['clip_id']}.wav"
    )


def summarize(frame: pd.DataFrame, audio_dir: Path) -> Dict[str, object]:
    classes: Dict[str, Dict[str, object]] = {}
    for accent, subset in frame.groupby("accent_class"):
        actual_seconds = 0.0
        for _, row in subset.iterrows():
            path = audio_path(audio_dir, row)
            if path.is_file():
                actual_seconds += float(sf.info(path).duration)
        classes[str(accent)] = {
            "clips": int(len(subset)),
            "speakers": int(subset["client_id"].nunique()),
            "pre_trim_hours": round(
                float(subset["duration_sec"].sum()) / 3600.0, 3
            ),
            "post_trim_hours": round(actual_seconds / 3600.0, 3),
            "train_speakers": int(
                subset.loc[subset["speaker_split"] == "train", "client_id"].nunique()
            ),
            "val_speakers": int(
                subset.loc[subset["speaker_split"] == "val", "client_id"].nunique()
            ),
            "test_speakers": int(
                subset.loc[subset["speaker_split"] == "test", "client_id"].nunique()
            ),
        }
    return classes


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--excluded-dir", type=Path, default=DEFAULT_EXCLUDED_DIR)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-hours", type=float, default=2.0)
    parser.add_argument("--min-speakers", type=int, default=30)
    parser.add_argument("--min-eval-speakers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    failed = {item["clip_id"] for item in audit.get("failures", [])}
    if len(failed) != int(audit.get("failed", len(failed))):
        raise SystemExit(
            "audit failure list is truncated; rerun the audit with a larger limit"
        )
    removed = manifest[manifest["clip_id"].isin(failed)]
    kept = manifest[~manifest["clip_id"].isin(failed)].copy()

    # On a re-run the manifest is already pruned, so the excluded rows must come
    # from the preserved unpruned manifest.  This keeps the move idempotent and
    # stops a re-run from overwriting the original exclusion report.
    if args.backup.is_file():
        excluded_source = pd.read_csv(args.backup)
        pending = excluded_source[
            ~excluded_source["clip_id"].isin(set(manifest["clip_id"]))
        ]
    else:
        pending = removed

    before = summarize(manifest, args.audio_dir)
    after = summarize(kept, args.audio_dir)
    gate_failures: List[str] = []
    for accent in TRAINABLE:
        stats = after.get(accent)
        if not stats:
            gate_failures.append(f"{accent}: no clips after pruning")
            continue
        if float(stats["post_trim_hours"]) < args.min_hours:
            gate_failures.append(
                f"{accent}: {stats['post_trim_hours']:.3f}h < {args.min_hours:.3f}h"
            )
        if int(stats["speakers"]) < args.min_speakers:
            gate_failures.append(
                f"{accent}: {stats['speakers']} speakers < {args.min_speakers}"
            )
        if int(stats["train_speakers"]) < args.min_speakers:
            gate_failures.append(
                f"{accent}: {stats['train_speakers']} train speakers < {args.min_speakers}"
            )
        if int(stats["val_speakers"]) < args.min_eval_speakers:
            gate_failures.append(
                f"{accent}: {stats['val_speakers']} val speakers < {args.min_eval_speakers}"
            )
        if int(stats["test_speakers"]) < args.min_eval_speakers:
            gate_failures.append(
                f"{accent}: {stats['test_speakers']} test speakers < {args.min_eval_speakers}"
            )

    conflicts = kept.groupby("client_id")["speaker_split"].nunique()
    if int(conflicts.max()) > 1:
        gate_failures.append("speaker split conflict after pruning")

    moved_excluded = 0
    if not args.dry_run:
        for _, row in pending.iterrows():
            source = audio_path(args.audio_dir, row)
            if not source.is_file():
                continue
            destination = (
                args.excluded_dir
                / str(row["speaker_split"])
                / str(row["accent_class"])
                / source.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                source.unlink()
            else:
                shutil.move(str(source), str(destination))
            moved_excluded += 1

    already_pruned = (
        len(removed) == 0 and args.backup.is_file() and args.report.is_file()
    )
    if already_pruned:
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        previous["excluded_dir"] = str(args.excluded_dir)
        previous["moved_excluded"] = (
            int(previous.get("moved_excluded", 0)) + moved_excluded
        )
        previous["excluded_wavs_present"] = int(
            sum(
                audio_path(args.excluded_dir, row).is_file()
                for _, row in pending.iterrows()
            )
        )
        moved_note = (
            "Excluded WAVs are moved to globe_audio_excluded for auditability, "
            "so a glob over globe_audio cannot pick them up."
        )
        notes = [
            note
            for note in previous.get("notes", [])
            if not note.startswith("Excluded WAVs remain on disk")
        ]
        if moved_note not in notes:
            notes.append(moved_note)
        previous["notes"] = notes
        if args.dry_run:
            print(json.dumps(previous, indent=2), flush=True)
            print(
                f"[dry-run] {len(pending)} excluded WAVs would be moved; "
                "manifest and report not rewritten"
            )
        else:
            args.report.write_text(json.dumps(previous, indent=2), encoding="utf-8")
            print(json.dumps(previous, indent=2), flush=True)
            print(
                f"[already pruned] no failed clips in {args.manifest}; "
                "kept the original exclusion report"
            )
            print(f"Moved {moved_excluded} excluded WAVs to {args.excluded_dir}")
        if gate_failures:
            raise SystemExit(1)
        return

    report = {
        "manifest": str(args.manifest),
        "audit": str(args.audit),
        "excluded_dir": str(args.excluded_dir),
        "moved_excluded": moved_excluded,
        "pending_excluded": int(len(pending)),
        "before": before,
        "after": after,
        "removed_clips": int(len(removed)),
        "removed_hours_pre_trim": round(
            float(removed["duration_sec"].sum()) / 3600.0, 3
        ),
        "removed_by_class": {
            str(key): int(value)
            for key, value in removed["accent_class"].value_counts().items()
        },
        "removed_by_reason": {
            str(key): int(value)
            for key, value in (
                removed["clip_id"].map(
                    lambda clip_id: next(
                        (
                            item.get("error", "").split("=")[0]
                            for item in audit.get("failures", [])
                            if item["clip_id"] == clip_id
                        ),
                        "unknown",
                    )
                )
            )
            .value_counts()
            .items()
        },
        "gate_pass": not gate_failures,
        "gate_failures": gate_failures,
        "notes": [
            "The unpruned manifest is preserved as a separate CSV.",
            "Excluded WAVs are moved to globe_audio_excluded for auditability, so a glob over globe_audio cannot pick them up.",
            "The post-download audit, not the pre-trim selection summary, is authoritative for the 2.0 h gate.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        print("[dry-run] manifest not rewritten")
        return
    if args.backup.is_file():
        print(f"[backup] {args.backup} already exists; keeping it")
    else:
        shutil.copy2(args.manifest, args.backup)
    kept.to_csv(args.manifest, index=False)
    print(f"Wrote {args.manifest} ({len(kept)} clips)")
    print(f"Wrote {args.report}")
    if gate_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
