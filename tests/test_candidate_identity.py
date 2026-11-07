"""Tests for the candidate identity / expression preservation measurement."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_candidate_identity.py"

spec = importlib.util.spec_from_file_location("evaluate_candidate_identity", SCRIPT)
assert spec and spec.loader
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class TestSummaries:
    def test_partial_summary_does_not_treat_missing_as_zero(self):
        result = identity._summarize([0.5, None, 0.7])
        assert result["status"] == "partial"
        assert result["n"] == 2
        assert result["total"] == 3
        assert result["mean"] == 0.6
        assert result["min"] == 0.5
        assert result["max"] == 0.7

    def test_all_missing_is_unavailable(self):
        result = identity._summarize([None, None])
        assert result["status"] == "unavailable"
        assert result["mean"] is None

    def test_generator_input_is_counted_once(self):
        result = identity._summarize(value for value in [0.1, None, 0.3])
        assert result["total"] == 3
        assert result["n"] == 2


class TestPairwiseMatrix:
    def test_matrix_is_symmetric_and_uses_measured_pairs(self):
        paths = ["a.wav", "b.wav", "c.wav"]

        def similarity(a, b):
            return {"a.wav|b.wav": 0.5, "a.wav|c.wav": 0.2, "b.wav|c.wav": 0.1}[
                f"{Path(a).name}|{Path(b).name}"
            ]

        result = identity._pairwise_matrix(paths, similarity)

        assert result["status"] == "computed"
        assert result["n_pairs"] == 3
        assert result["matrix"] == [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
        assert result["mean"] == 0.2667


class TestEvaluateCandidateIdentity:
    def test_reports_candidates_and_expression_preservation(self, tmp_path, monkeypatch):
        for name in (
            "candidate_1_primary.wav",
            "candidate_2_alternative.wav",
            "candidate_3_contrasting.wav",
            "accent_01.wav",
            "emotion_01.wav",
            "face.png",
        ):
            (tmp_path / name).touch()

        manifest = {
            "samples": [
                {
                    "audio": str(tmp_path / "candidate_1_primary.wav"),
                    "group": "01_Candidates",
                    "condition": "Candidate 1",
                },
                {
                    "audio": str(tmp_path / "candidate_2_alternative.wav"),
                    "group": "01_Candidates",
                    "condition": "Candidate 2",
                },
                {
                    "audio": str(tmp_path / "candidate_3_contrasting.wav"),
                    "group": "01_Candidates",
                    "condition": "Candidate 3",
                },
                {
                    "audio": str(tmp_path / "accent_01.wav"),
                    "group": "02_Accents",
                    "condition": "General American",
                },
                {
                    "audio": str(tmp_path / "emotion_01.wav"),
                    "group": "03_Emotions",
                    "condition": "Neutral",
                },
            ]
        }

        def fake_face_voice(face, audio, **kwargs):
            return 0.11 if "candidate" in Path(audio).name else 0.22

        def fake_similarity(a, b, device="cpu"):
            names = {Path(a).name, Path(b).name}
            if len(names) == 1:
                return 1.0
            if "candidate" in Path(a).name and "candidate" in Path(b).name:
                return 0.50
            return 0.95

        monkeypatch.setattr(identity, "face_voice_consistency", fake_face_voice)
        monkeypatch.setattr(identity, "speaker_similarity", fake_similarity)

        report = identity.evaluate_candidate_identity(
            manifest,
            primary_audio=str(tmp_path / "candidate_1_primary.wav"),
            face=str(tmp_path / "face.png"),
        )

        assert report["schema_version"] == identity.SCHEMA
        assert len(report["candidates"]) == 3
        assert len(report["expressions"]) == 2
        assert report["candidate_face_voice"]["mean"] == 0.11
        assert report["expression_face_voice"]["mean"] == 0.22
        assert report["expression_identity_preservation"]["mean"] == 0.95
        assert report["candidate_pairwise_similarity"]["n_pairs"] == 3
        assert report["candidate_pairwise_similarity"]["mean"] == 0.5
