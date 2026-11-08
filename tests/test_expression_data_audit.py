"""Tests for the emotion/accent corpus audit.

The audit is a data/license decision artifact, so these tests pin the
invariants that matter: every dataset declares a license and commercial-use
status, the summary partitions are complete and disjoint, sampled/paper
figures are not presented as full-corpus measurements, and the local scan
distinguishes "present" from "absent" without fabricating a match.
"""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit" / "audit_expression_data.py"

spec = importlib.util.spec_from_file_location("audit_expression_data", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

REQUIRED_DATASET_KEYS = {
    "id",
    "name",
    "role",
    "training_role",
    "speaker_id_available",
    "speaker_id_notes",
    "balance",
    "source",
    "license",
    "license_url",
    "commercial_use",
    "size",
    "local_expected_paths",
    "fit",
    "gaps",
    "verification",
}
COMMERCIAL_STATUSES = {
    "allowed",
    "allowed_with_obligations",
    "requires_agreement",
    "research_only_unless_licensed",
    "research_only",
}
TRAINING_ROLES = {"primary", "secondary", "evaluation", "pilot_evaluation", "research"}
SPEAKER_ID_STATUSES = {"full", "partial", "join_required", "none"}
BALANCE_STATUSES = {
    "balanced",
    "expected_balanced",
    "imbalanced",
    "single_class",
    "not_labeled",
    "unknown",
}


class TestDatasetInventory:
    def test_schema_and_unique_ids(self):
        result = audit.build_audit()
        assert result["schema_version"] == audit.AUDIT_SCHEMA
        assert result["scope_document"] == "docs/SCOPE.md"
        ids = [dataset["id"] for dataset in result["datasets"]]
        assert len(ids) == len(set(ids)) == 13
        assert result["generated_at"].endswith("UTC")

    def test_every_dataset_declares_license_and_status(self):
        result = audit.build_audit()
        for dataset in result["datasets"]:
            missing = REQUIRED_DATASET_KEYS - set(dataset)
            assert not missing, f"{dataset['id']} missing {missing}"
            assert dataset["license"]
            assert dataset["license_url"].startswith("http")
            assert dataset["commercial_use"] in COMMERCIAL_STATUSES
            assert dataset["verification"], dataset["id"]
            assert dataset["training_role"] in TRAINING_ROLES, dataset["id"]
            assert dataset["speaker_id_available"] in SPEAKER_ID_STATUSES, dataset["id"]
            assert dataset["speaker_id_notes"], dataset["id"]
            assert dataset["balance"]["status"] in BALANCE_STATUSES, dataset["id"]
            assert dataset["balance"]["notes"], dataset["id"]

    def test_training_roles_match_license_risk(self):
        result = audit.build_audit()
        for dataset in result["datasets"]:
            if dataset["training_role"] in {"primary", "secondary"}:
                assert dataset["commercial_use"] in {
                    "allowed",
                    "allowed_with_obligations",
                    "requires_agreement",
                }, dataset["id"]
            if dataset["commercial_use"] == "research_only":
                assert dataset["training_role"] == "research", dataset["id"]

    def test_globe_v3_requires_speaker_id_join(self):
        dataset = next(d for d in audit.DATASETS if d["id"] == "globe_v3")
        assert dataset["speaker_id_available"] == "join_required"
        assert "common_voice_path" in dataset["speaker_id_notes"]
        assert "client_id" in dataset["speaker_id_notes"]
        assert dataset["balance"]["status"] == "imbalanced"
        assert "partial" in dataset["balance"]["notes"].lower()

    def test_summary_partitions_are_complete_and_disjoint(self):
        result = audit.build_audit()
        summary = result["summary"]
        groups = [
            set(summary["commercial_usable_with_obligations"]),
            set(summary["commercial_only_after_agreement_or_purchase"]),
            set(summary["research_only"]),
        ]
        ids = {dataset["id"] for dataset in result["datasets"]}
        assert set().union(*groups) == ids
        assert sum(len(group) for group in groups) == len(ids)

    def test_recommendations_name_the_first_acquisitions(self):
        result = audit.build_audit()
        recommendation = result["recommendation"]
        emotion = " ".join(recommendation["emotion_first_acquisition"])
        accent = " ".join(recommendation["accent_first_acquisition"])
        excluded = " ".join(recommendation["do_not_train_commercially_on"])
        assert "CREMA-D" in emotion
        assert "MSP-Podcast" in emotion
        assert "GLOBE_V3" in accent
        assert "EdAcc" in accent
        assert "Svarah" in accent
        for name in ("ESD", "RAVDESS", "IEMOCAP", "L2-ARCTIC", "VoxCeleb2"):
            assert name in excluded

    def test_recommended_inventory_is_balanced_and_standard(self):
        result = audit.build_audit()
        inventory = result["recommended_control_inventory"]
        assert inventory["status"] == "proposed_not_yet_trained"

        emotion = inventory["emotion"]
        assert emotion["train"] == [
            "neutral",
            "happy",
            "sad",
            "angry",
            "fear",
            "disgust",
        ]
        assert "whispered" in emotion["remove_or_move"]
        assert "surprised" in emotion["add_when_commercially_usable_data_exists"]
        assert "whispered" not in emotion["train"]

        accent = inventory["accent"]
        assert accent["train"] == [
            "US English (General American)",
            "British English (England English)",
            "Australian English",
            "Canadian English",
            "Indian/South Asian English",
        ]
        for thin in (
            "Irish English",
            "Scottish English",
            "Welsh English",
            "New Zealand English",
            "South African English",
            "Northern Irish English",
        ):
            assert thin in accent["evaluation_only"]
            assert thin not in accent["train"]
        assert "British RP" in accent["rename"]

        thresholds = inventory["balance_thresholds"]
        assert thresholds["min_speakers_per_class"] == 30
        assert thresholds["min_hours_per_class"] == 2.0
        assert thresholds["max_min_class_ratio"] == 1.5
        assert thresholds["per_speaker_seconds_cap"] == 120
        assert thresholds["max_clips_per_speaker"] == 40
        assert thresholds["pre_trim_target_hours"] == 2.5
        assert "speaker-disjoint" in thresholds["split_policy"]

    def test_summary_training_roles_partition_the_inventory(self):
        result = audit.build_audit()
        roles = result["summary"]["training_roles"]
        ids = {dataset["id"] for dataset in result["datasets"]}
        grouped = [set(ids_for_role) for ids_for_role in roles.values()]
        assert set().union(*grouped) == ids
        assert sum(len(group) for group in grouped) == len(ids)
        assert result["summary"]["speaker_id_recovery_required"] == ["globe_v3"]


class TestMeasurementDiscipline:
    def test_globe_v3_paper_figures_are_not_top_level_measurements(self):
        dataset = next(d for d in audit.DATASETS if d["id"] == "globe_v3")
        size = dataset["size"]
        assert size["rows"] == 723532
        assert sum(size["splits"].values()) == size["rows"]
        assert size["sample_rate_hz"] == 24000
        assert "speakers" not in size
        assert "accents" not in size
        assert "hours" not in size
        paper = size["paper_reported"]
        assert paper["speakers"] == 23519
        assert paper["accents"] == 164
        assert paper["hours"] == 535
        assert "recompute" in paper["note"].lower()
        assert any("recompute" in gap.lower() for gap in dataset["gaps"])

    def test_openslr83_uses_verified_speaker_counts_only(self):
        dataset = next(d for d in audit.DATASETS if d["id"] == "openslr83")
        size = dataset["size"]
        assert size["speakers"] == 120
        assert size["transcriptions"] == 17877
        counts = size["dialect_speaker_counts"]
        assert counts["Irish"] == {"total": 5, "male": 3, "female": 2}
        assert counts["Scottish"] == {"total": 39, "male": 11, "female": 28}

    def test_research_only_corpora_are_marked(self):
        for dataset in audit.DATASETS:
            if dataset["id"] in {"esd", "iemocap", "l2_arctic", "voxceleb2"}:
                assert dataset["commercial_use"] == "research_only"
        ravdess = next(d for d in audit.DATASETS if d["id"] == "ravdess")
        assert ravdess["commercial_use"] == "research_only_unless_licensed"


class TestLocalScan:
    def test_absent_roots_do_not_fabricate_a_match(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit, "ROOT", tmp_path)
        monkeypatch.setattr(
            audit, "DATASETS", [audit.DATASETS[0]]
        )
        monkeypatch.setattr(audit, "_path", lambda *parts: tmp_path / "missing" / Path(*parts))

        scan = audit.scan_local()

        assert scan["dataset_matches"] == []
        assert scan["expression_corpora_present"] is False
        assert scan["voxceleb2_present"] is False
        assert scan["roots_present"] == []

    def test_present_dataset_is_detected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit, "ROOT", tmp_path)
        monkeypatch.setattr(
            audit,
            "DATASETS",
            [
                {
                    "id": "fake_accent",
                    "local_expected_paths": ["data/raw/fake_accent"],
                }
            ],
        )
        monkeypatch.setattr(audit, "_path", lambda *parts: tmp_path / "missing" / Path(*parts))
        (tmp_path / "data" / "raw" / "fake_accent").mkdir(parents=True)

        scan = audit.scan_local()

        assert scan["dataset_matches"] == [
            {"id": "fake_accent", "paths": [str(Path("data/raw/fake_accent"))]}
        ]
        assert scan["expression_corpora_present"] is True

    def test_main_writes_a_parseable_audit(self, tmp_path, monkeypatch):
        output = tmp_path / "nested" / "audit.json"
        monkeypatch.setattr(
            audit, "parse_args", lambda: types.SimpleNamespace(output=output)
        )

        audit.main()

        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["schema_version"] == audit.AUDIT_SCHEMA
        assert len(payload["datasets"]) == 13
