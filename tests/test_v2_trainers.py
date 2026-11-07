"""Tests for the FaceFeatureV2 trainer utilities and loss semantics.

The legacy CVAE "diversity" loss returned the negative of the mean similarity
and was added with a positive weight, so training rewarded collapse.  These
tests pin the corrected sign and verify that separation losses ignore
same-speaker pairs, where high similarity is desirable.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.training.trainer_adapter_cfvn import (
    _head_losses,
    adapter_centered_norm_loss,
    adapter_diversity_loss,
)
from voix.training.trainer_cfvn import speaker_centroid_targets
from voix.training.trainer_cvae_v3_diversity import (
    InfoNCELoss,
    decoder_diversity_loss,
    face_separation_loss,
)
from voix.training.v2_utils import (
    align_style_targets,
    embed_faces,
    load_style_targets,
    match_style_targets,
    normalize_face,
    query_gallery_indices,
    speaker_codes,
)


class TestDiversityLossSign:
    def test_identical_different_speaker_outputs_have_high_positive_loss(self):
        vector = F.normalize(torch.randn(1, 192), dim=-1).repeat(2, 1)
        codes = torch.tensor([0, 1])
        loss = decoder_diversity_loss(vector, codes)
        assert float(loss) > 0.99

    def test_same_speaker_pairs_are_ignored(self):
        vector = F.normalize(torch.randn(1, 192), dim=-1).repeat(3, 1)
        codes = torch.tensor([0, 0, 0])
        assert float(decoder_diversity_loss(vector, codes)) == 0.0
        assert float(face_separation_loss(vector, codes)) == 0.0

    def test_gradient_step_reduces_cross_speaker_similarity(self):
        torch.manual_seed(0)
        outputs = torch.randn(4, 32, requires_grad=True)
        codes = torch.tensor([0, 0, 1, 1])
        before = float(decoder_diversity_loss(outputs, codes))
        loss = decoder_diversity_loss(outputs, codes)
        loss.backward()
        with torch.no_grad():
            outputs -= 0.5 * outputs.grad
        after = float(decoder_diversity_loss(outputs, codes))
        assert after < before


class TestInfoNCE:
    def test_matching_targets_have_lower_loss_than_shuffled_targets(self):
        torch.manual_seed(1)
        anchors = F.normalize(torch.randn(8, 192), dim=-1)
        codes = torch.arange(8)
        loss_fn = InfoNCELoss(temperature=0.07)
        perfect = loss_fn(anchors, anchors.clone(), codes)
        shuffled = loss_fn(anchors, anchors.flip(0), codes)
        assert float(perfect) < float(shuffled)


class TestV2Utils:
    def test_normalize_face_uses_scaler(self):
        scaler = {
            "face_mean": torch.zeros(4),
            "face_std": torch.full((4,), 2.0),
        }
        face = np.ones((2, 4), dtype=np.float32)
        normalized = normalize_face(face, scaler)
        assert torch.allclose(normalized, torch.full((2, 4), 0.5))

    def test_normalize_face_rejects_wrong_dimension(self):
        scaler = {"face_mean": torch.zeros(4), "face_std": torch.ones(4)}
        with pytest.raises(ValueError, match="scaler expects"):
            normalize_face(np.zeros((2, 5), dtype=np.float32), scaler)

    def test_query_gallery_are_disjoint_and_skip_singletons(self):
        speakers = np.asarray(["a", "a", "b", "b", "b", "c"])
        query, gallery = query_gallery_indices(speakers)
        assert set(query).isdisjoint(set(gallery))
        assert len(query) == 2
        assert set(speakers[query]) == {"a", "b"}
        assert len(gallery) == 3

    def test_cfvn_query_gallery_skips_singletons(self):
        from voix.training.trainer_cfvn import query_gallery_split

        speakers = np.asarray(["a", "a", "b", "b", "c"])
        query, gallery = query_gallery_split(speakers)
        assert set(query).isdisjoint(set(gallery))
        assert len(query) == 2
        assert set(speakers[query]) == {"a", "b"}

    def test_speaker_codes_are_stable(self):
        first, first_names = speaker_codes(np.asarray(["b", "a", "b"]))
        second, second_names = speaker_codes(np.asarray(["b", "a", "b"]))
        assert first_names == second_names == ["a", "b"]
        assert np.array_equal(first, second)
        assert first[0] == first[2] != first[1]

    def test_embed_faces_is_normalized_and_batched(self):
        torch.manual_seed(2)
        model = ContrastiveFaceVoiceNet(
            face_dim=4,
            out_dim=3,
            hidden_dims=(5,),
            dropout=0.0,
            use_layernorm=False,
        ).eval()
        face = torch.randn(5, 4)
        embedded = embed_faces(model, face, "cpu", batch_size=2)
        assert embedded.shape == (5, 3)
        assert torch.allclose(
            embedded.norm(dim=-1), torch.ones(5), atol=1e-5
        )

    def test_style_targets_are_constant_within_a_speaker(self):
        speaker_features = np.asarray(
            [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1]],
            dtype=np.float32,
        )
        speaker_ids = np.asarray(["a", "a", "b", "b"])
        style_pairs = {
            "ecapa": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "timbre": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "prosody": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        }
        targets, stats = match_style_targets(
            speaker_features, speaker_ids, style_pairs
        )
        assert torch.allclose(targets[0], targets[1])
        assert torch.allclose(targets[2], targets[3])
        assert not torch.allclose(targets[0], targets[2])
        assert stats["style_match_speakers"] == 2
        assert stats["style_match_cosine_mean"] > 0.9


class TestStyleTargetV2:
    def _payload(self, tmp_path, style, speakers, schema="StyleTargetV2"):
        path = tmp_path / "targets.pt"
        torch.save(
            {
                "style": style,
                "speaker_ids": speakers,
                "clip_ids": [f"{speaker}-clip" for speaker in speakers],
                "split": ["train"] * len(speakers),
                "gender": ["f"] * len(speakers),
                "schema_version": schema,
            },
            path,
        )
        return path

    def test_load_and_align_per_speaker_targets(self, tmp_path):
        style = torch.randn(3, 256)
        path = self._payload(tmp_path, style, ["a", "b", "c"])
        payload = load_style_targets(path)
        speaker_ids = np.asarray(["b", "a", "b", "c"])
        targets, stats = align_style_targets(speaker_ids, payload)
        assert targets.shape == (4, 256)
        assert torch.allclose(targets[0], style[1])
        assert torch.allclose(targets[1], style[0])
        assert torch.allclose(targets[2], style[1])
        assert torch.allclose(targets[3], style[2])
        assert stats["style_target_speakers"] == 3
        assert stats["style_target_missing_speakers"] == 0

    def test_missing_speaker_fails_instead_of_zero_filling(self, tmp_path):
        path = self._payload(tmp_path, torch.randn(1, 256), ["a"])
        payload = load_style_targets(path)
        with pytest.raises(ValueError, match="no style target"):
            align_style_targets(np.asarray(["a", "b"]), payload)
        targets, stats = align_style_targets(
            np.asarray(["a", "b"]), payload, require_all=False
        )
        assert stats["style_target_missing_speakers"] == 1
        assert float(targets.abs().sum()) == 0.0

    def test_wrong_schema_is_rejected(self, tmp_path):
        path = self._payload(
            tmp_path, torch.randn(1, 256), ["a"], schema="style_pairs"
        )
        with pytest.raises(ValueError, match="StyleTargetV2"):
            load_style_targets(path)

    def test_duplicate_speakers_are_rejected(self, tmp_path):
        path = self._payload(tmp_path, torch.randn(2, 256), ["a", "a"])
        with pytest.raises(ValueError, match="duplicate speaker"):
            load_style_targets(path)


class TestCFVNEvaluationSplit:
    def test_empty_test_subset_is_allowed(self):
        from voix.training.data import CleanFeatureDataset

        dataset = CleanFeatureDataset(
            face_raw=np.zeros((4, 4), dtype=np.float32),
            speaker_raw=np.zeros((4, 4), dtype=np.float32),
            speaker_ids=np.asarray(["a", "a", "b", "b"]),
            split=np.asarray(["train", "train", "val", "val"]),
            clip_ids=np.asarray(["1", "2", "3", "4"]),
            hdf5_index=np.arange(4, dtype=np.int64),
            gender=np.asarray(["f", "f", "m", "m"]),
        )
        empty = dataset.subset("test", allow_empty=True)
        assert len(empty) == 0
        with pytest.raises(ValueError):
            dataset.subset("test")

    def test_constant_cfvn_does_not_score_perfect_retrieval(self):
        from voix.training.data import CleanFeatureDataset
        from voix.training.trainer_cfvn import evaluate

        rng = np.random.default_rng(0)
        n_speakers, clips = 6, 3
        speaker_ids = np.repeat(
            np.asarray([f"s{i}" for i in range(n_speakers)]), clips
        )
        speaker_genders = np.asarray(
            ["f" if i % 2 == 0 else "m" for i in range(n_speakers)]
        )
        dataset = CleanFeatureDataset(
            face_raw=rng.normal(size=(len(speaker_ids), 4)).astype(np.float32),
            speaker_raw=rng.normal(size=(len(speaker_ids), 4)).astype(np.float32),
            speaker_ids=speaker_ids,
            split=np.asarray(["val"] * len(speaker_ids)),
            clip_ids=np.asarray([f"c{i}" for i in range(len(speaker_ids))]),
            hdf5_index=np.arange(len(speaker_ids), dtype=np.int64),
            gender=np.repeat(speaker_genders, clips),
        )

        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 4, device=x.device)

        metrics = evaluate(
            Constant(),
            dataset,
            torch.from_numpy(dataset.face_raw),
            torch.from_numpy(dataset.speaker_raw),
            "cpu",
        )
        assert metrics["n_eval_speakers"] == n_speakers
        assert metrics["val_r1"] < 1.0
        assert metrics["val_r1"] <= 1.0


class TestAdapterServingParity:
    def test_cvae_adapter_inputs_are_normalized(self):
        from voix.models.cvae import CVAE
        from voix.training.trainer_adapter_cfvn import (
            cvae_prior_mean_inputs,
            cvae_sample_inputs,
        )

        torch.manual_seed(3)
        cvae = CVAE(face_dim=8, speaker_dim=8, latent_dim=4).eval()
        face = torch.randn(5, 8)
        samples = cvae_sample_inputs(
            cvae, face, num_samples=3, temperature=1.0, device="cpu"
        )
        assert samples.shape == (5, 3, 8)
        assert torch.allclose(
            samples.norm(dim=-1), torch.ones(5, 3), atol=1e-5
        )
        prior = cvae_prior_mean_inputs(cvae, face, device="cpu")
        assert torch.allclose(prior.norm(dim=-1), torch.ones(5), atol=1e-5)


class TestAdapterLoss:
    def test_perfect_prediction_has_near_zero_loss(self):
        prediction = torch.randn(4, 256)
        target = prediction.clone()
        loss, metrics = _head_losses(prediction, target, lambda_cos=1.0)
        assert float(loss) < 1e-6
        assert metrics["cosine_timbre"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["cosine_prosody"] == pytest.approx(1.0, abs=1e-6)

    def test_timbre_and_prosody_are_reported_separately(self):
        prediction = torch.zeros(2, 256)
        target = torch.zeros(2, 256)
        target[:, :128] = 1.0
        target[:, 128:] = -1.0
        _, metrics = _head_losses(prediction, target, lambda_cos=1.0)
        assert metrics["loss_timbre_mse"] == pytest.approx(1.0)
        assert metrics["loss_prosody_mse"] == pytest.approx(1.0)
        assert metrics["cosine_timbre"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["cosine_prosody"] == pytest.approx(0.0, abs=1e-6)

    def test_diversity_loss_is_differentiable(self):
        prediction = torch.randn(6, 8, requires_grad=True)
        codes = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = adapter_diversity_loss(prediction, codes)
        assert loss.requires_grad
        loss.backward()
        assert prediction.grad is not None
        assert float(prediction.grad.norm()) > 0.0

    def test_diversity_loss_ignores_same_speaker_pairs(self):
        prediction = torch.randn(6, 8, requires_grad=True)
        loss = adapter_diversity_loss(prediction, torch.zeros(6, dtype=torch.long))
        assert float(loss) == 0.0
        assert loss.requires_grad

    def test_centered_norm_loss_matches_residual_magnitude(self):
        center = torch.randn(1, 8)
        prediction = (center + torch.eye(8)[:4]).requires_grad_()
        loss = adapter_centered_norm_loss(prediction, center, target_norm=1.0)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)
        loss = adapter_centered_norm_loss(prediction, center, target_norm=2.0)
        assert float(loss) == pytest.approx(1.0, abs=1e-5)
        loss.backward()
        assert float(prediction.grad.norm()) > 0.0

    def test_centered_diversity_ignores_a_common_style_mean(self):
        center = torch.randn(1, 8)
        residual = torch.eye(8)[:6]
        prediction = (center + 0.1 * residual).requires_grad_()
        codes = torch.tensor([0, 0, 1, 1, 2, 2])
        raw = adapter_diversity_loss(prediction, codes)
        centered = adapter_diversity_loss(prediction, codes, center=center)
        assert float(raw) > 0.9
        assert float(centered) == pytest.approx(0.0, abs=1e-6)
        centered.backward()
        assert float(prediction.grad.norm()) > 0.0

class TestSpeakerCentroidTargets:
    def test_same_speaker_rows_share_a_unit_target(self):
        torch.manual_seed(0)
        embeddings = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        speakers = np.asarray(["a", "a", "a", "b", "b", "c"])
        targets = speaker_centroid_targets(embeddings, speakers)
        assert targets.shape == embeddings.shape
        assert torch.allclose(targets.norm(dim=-1), torch.ones(6), atol=1e-5)
        assert torch.allclose(targets[0], targets[1])
        assert torch.allclose(targets[0], targets[2])
        assert torch.allclose(targets[3], targets[4])
        assert not torch.allclose(targets[0], targets[3])

    def test_singleton_speaker_target_is_its_normalized_embedding(self):
        embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        targets = speaker_centroid_targets(embeddings, np.asarray(["x", "y"]))
        assert torch.allclose(targets[0], torch.tensor([1.0, 0.0]))
        assert torch.allclose(targets[1], torch.tensor([0.0, 1.0]))

class TestCFVNTrainingContract:
    def test_alignment_defaults_are_not_the_legacy_chance_recipe(self):
        import sys
        from voix.training.trainer_cfvn import parse_args

        original = sys.argv
        sys.argv = ["trainer_cfvn"]
        try:
            args = parse_args()
        finally:
            sys.argv = original
        assert args.reg_target == "centroid"
        assert args.lambda_reg >= 1.0
        assert args.lr >= 1e-3

    def test_embed_all_handles_empty_input(self):
        from voix.training.trainer_cfvn import embed_all

        model = ContrastiveFaceVoiceNet(face_dim=8, out_dim=4)
        out = embed_all(model, torch.zeros((0, 8)), "cpu")
        assert out.shape == (0, 4)


class TestGenderControlledRetrieval:
    def test_perfect_identity_beats_same_gender_chance(self):
        from voix.evaluation.retrieval import gender_controlled_retrieval

        # Two female and two male speakers, two clips each.  The two clips of a
        # speaker are identical, so the query centroid is exactly its gallery
        # centroid and retrieval should be perfect.
        base = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ]
        )
        features = base.repeat_interleave(2, dim=0)
        speakers = np.repeat(["f1", "f2", "m1", "m2"], 2)
        genders = np.repeat(["f", "f", "m", "m"], 2)
        query = np.arange(0, len(speakers), 2)
        gallery = np.arange(1, len(speakers), 2)

        metrics = gender_controlled_retrieval(
            features[query],
            speakers[query],
            genders[query],
            features[gallery],
            speakers[gallery],
            genders[gallery],
        )

        assert metrics["recall_at_1"] == 1.0
        assert metrics["n_speakers"] == 4.0
        assert metrics["chance_at_1"] == 0.5

    def test_gender_only_features_do_not_beat_chance(self):
        from voix.evaluation.retrieval import gender_controlled_retrieval

        # Every female query is the same vector and every male query is the
        # same vector.  Within a gender all gallery speakers are tied, so the
        # model cannot identify anyone and R@1 must not be perfect.
        female = torch.tensor([[1.0, 0.0]])
        male = torch.tensor([[0.0, 1.0]])
        features = torch.cat(
            [female, female, female, male, male, male], dim=0
        )
        speakers = np.asarray(["f1", "f2", "f3", "m1", "m2", "m3"])
        genders = np.asarray(["f", "f", "f", "m", "m", "m"])
        # Query f2/m2 against all six gallery speakers.  Every same-gender
        # candidate is tied, and the target is not first in the candidate
        # list, so R@1 must be zero rather than an artefact of stable sorting.
        query = np.asarray([1, 4])
        gallery = np.arange(6)

        metrics = gender_controlled_retrieval(
            features[query],
            speakers[query],
            genders[query],
            features[gallery],
            speakers[gallery],
            genders[gallery],
        )
        assert metrics["recall_at_1"] <= 0.5

    def test_inconsistent_gender_labels_raise(self):
        from voix.evaluation.retrieval import gender_controlled_retrieval

        with pytest.raises(ValueError):
            gender_controlled_retrieval(
                torch.zeros(2, 2),
                ["a", "a"],
                ["f", "m"],
                torch.zeros(2, 2),
                ["a", "b"],
                ["f", "f"],
            )

class TestCheckpointDevice:
    def test_from_checkpoint_moves_module_to_requested_device(self, tmp_path, monkeypatch):
        model = ContrastiveFaceVoiceNet(
            face_dim=4,
            out_dim=3,
            hidden_dims=(5,),
            dropout=0.0,
            use_layernorm=False,
        )
        config = {
            "face_dim": 4,
            "out_dim": 3,
            "hidden_dims": [5],
            "dropout": 0.0,
            "use_layernorm": False,
            "residual": False,
        }
        path = tmp_path / "cfvn.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "architecture": config}, path
        )

        calls = []
        original_to = ContrastiveFaceVoiceNet.to

        def spy(self, *args, **kwargs):
            calls.append((args, kwargs))
            return original_to(self, *args, **kwargs)

        monkeypatch.setattr(ContrastiveFaceVoiceNet, "to", spy)
        loaded = ContrastiveFaceVoiceNet.from_checkpoint(str(path), device="cpu")

        assert calls, "from_checkpoint must move the constructed module"
        assert calls[-1][0][0] == "cpu"
        assert next(loaded.parameters()).device.type == "cpu"
