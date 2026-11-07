"""Tests for the research report schema and provenance helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voix.evaluation.research_report import (
    RESEARCH_REPORT_SCHEMA,
    build_report,
    load_report,
    report_path,
    sha256_file,
    unavailable_metric,
    validate_report,
    write_report,
)


def _minimal_report(**overrides):
    payload = {
        "schema_version": RESEARCH_REPORT_SCHEMA,
        "track": "emotion",
        "level": "exploratory",
        "run_id": "unit-test",
        "git_commit": "deadbeef",
        "config_sha256": "not_applicable",
        "manifest_sha256": "not_applicable",
        "seeds": [0],
        "metrics": {"accuracy": {"status": "computed", "value": 0.5}},
        "gates": {},
        "status": "completed",
        "limitations": ["unit test only"],
        "mos": {"status": "unavailable", "reason": "no ratings"},
    }
    payload.update(overrides)
    return payload


def test_valid_report_passes():
    assert validate_report(_minimal_report()) == []


def test_missing_required_field_fails():
    report = _minimal_report()
    del report["run_id"]
    assert any("run_id" in error for error in validate_report(report))


def test_invalid_track_level_status_fail():
    assert any("track" in error for error in validate_report(_minimal_report(track="other")))
    assert any("level" in error for error in validate_report(_minimal_report(level="pilot")))
    assert any("status" in error for error in validate_report(_minimal_report(status="done")))


def test_unavailable_metric_requires_reason():
    report = _minimal_report(metrics={"wer": {"status": "unavailable"}})
    errors = validate_report(report)
    assert any("wer.reason" in error for error in errors)


def test_unavailable_mos_requires_reason():
    report = _minimal_report(mos={"status": "unavailable"})
    assert any("mos.reason" in error for error in validate_report(report))


def test_computed_mos_requires_ratings_and_mean():
    report = _minimal_report(mos={"status": "computed", "n_ratings": 0, "mean": 4.0})
    assert any("n_ratings" in error for error in validate_report(report))
    report = _minimal_report(mos={"status": "computed", "n_ratings": 3, "mean": float("nan")})
    assert any("mos.mean" in error for error in validate_report(report))


def test_hash_must_be_sha256_or_not_applicable():
    report = _minimal_report(config_sha256="abc")
    assert any("config_sha256" in error for error in validate_report(report))


def test_build_report_hashes_files(tmp_path: Path):
    config = tmp_path / "config.yaml"
    manifest = tmp_path / "manifest.csv"
    config.write_text("steps: 10\n", encoding="utf-8")
    manifest.write_text("clip,path\n1,a.wav\n", encoding="utf-8")

    report = build_report(
        track="accent",
        level="exploratory",
        run_id="hash-test",
        metrics={"accuracy": {"status": "computed", "value": 0.6}},
        seeds=[0, 1],
        config_path=config,
        manifest_path=manifest,
        limitations=["unit test"],
    )
    assert report["config_sha256"] == sha256_file(config)
    assert report["manifest_sha256"] == sha256_file(manifest)
    assert report["seeds"] == [0, 1]
    assert report["mos"]["status"] == "unavailable"


def test_build_report_rejects_invalid_extra_overlap():
    with pytest.raises(ValueError):
        build_report(
            track="emotion",
            level="exploratory",
            run_id="overlap",
            metrics={},
            extra={"track": "accent"},
        )


def test_write_and_load_roundtrip(tmp_path: Path):
    report = build_report(
        track="naturalness",
        level="exploratory",
        run_id="roundtrip",
        metrics={"wer": {"status": "computed", "value": 0.1}},
        limitations=["no MOS"],
    )
    destination = write_report(tmp_path / "report.json", report)
    loaded = load_report(destination)
    assert loaded == report
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == (
        RESEARCH_REPORT_SCHEMA
    )


def test_unavailable_metric_helper_rejects_empty_reason():
    with pytest.raises(ValueError):
        unavailable_metric("   ")
    assert unavailable_metric("no classifier") == {
        "status": "unavailable",
        "reason": "no classifier",
    }


def test_report_path_layout():
    path = report_path("accent", "globe_a0")
    assert path.parts[-3:] == ("accent", "globe_a0", "report.json")
