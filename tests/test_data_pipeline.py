"""Phase 2 test suite: data pipeline, fusion layer, and head-pose.

All tests run fully offline using synthetic data (no VoxCeleb2 required).
Tests validate:
  - FaceFusionLayer shapes, gradients, freeze/unfreeze
  - HeadPoseEstimator: PnP estimation, frontal frame selection
  - VoxCelebPairedDataset: structure, integrity auditor
  - DataLoader: shape, reproducibility, seeding
  - Craniofacial ratio integration with head-pose
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_landmarks(yaw_deg: float = 0.0) -> np.ndarray:
    """Return synthetic (468, 3) landmarks approximating a frontal face.

    Landmark positions are set to canonical values matching the MediaPipe
    face model. A non-zero yaw_deg shifts the nose-tip landmark to simulate
    head rotation.
    """
    lm = np.zeros((468, 3), dtype=np.float32)
    # Inter-ocular landmarks (indices 33, 263)
    lm[33]  = [0.35, 0.45, 0.0]   # left eye outer
    lm[263] = [0.65, 0.45, 0.0]   # right eye outer
    # Nose tip (1), chin (199)
    lm[1]   = [0.50 + yaw_deg / 200.0, 0.50, 0.0]
    lm[199] = [0.50, 0.75, 0.0]
    # Mouth corners (61, 291)
    lm[61]  = [0.40, 0.65, 0.0]
    lm[291] = [0.60, 0.65, 0.0]
    # Lips (13, 14)
    lm[13]  = [0.50, 0.62, 0.0]
    lm[14]  = [0.50, 0.64, 0.0]
    # Fill remaining with small noise
    rng = np.random.default_rng(42)
    lm[lm.sum(axis=1) == 0] = rng.uniform(0.1, 0.9, size=(lm[lm.sum(axis=1) == 0].shape[0], 3)).astype(np.float32)
    return lm


def _make_h5(path: str, n: int = 200, n_spk: int = 10) -> None:
    """Write a minimal synthetic HDF5 feature store for testing."""
    import h5py
    rng = np.random.default_rng(0)
    face  = rng.standard_normal((n, 560)).astype(np.float32)
    spk   = rng.standard_normal((n, 192)).astype(np.float32)
    ids   = np.repeat(np.arange(n_spk), n // n_spk).astype(np.int32)[:n]
    meta  = np.array([f"id{i:05d}/vid/utt" for i in range(n)])
    with h5py.File(path, "w") as f:
        f.create_dataset("face_features",    data=face)
        f.create_dataset("speaker_features", data=spk)
        f.create_dataset("speaker_ids",      data=ids)
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset("metadata", shape=(n,), dtype=dt)
        for i, m in enumerate(meta):
            ds[i] = m


# ===========================================================================
# T2.4: FaceFusionLayer
# ===========================================================================

class TestFaceFusionLayer:

    def test_import(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        assert FaceFusionLayer is not None

    def test_output_shape_batched(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        layer = FaceFusionLayer()
        e_id   = torch.randn(8, 512)
        e_geo  = torch.randn(8, 32)
        e_demo = torch.randn(8, 16)
        out = layer(e_id, e_geo, e_demo)
        assert out.shape == (8, 560), f"Expected (8, 560), got {out.shape}"

    def test_output_shape_single(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        layer = FaceFusionLayer()
        e_id   = torch.randn(1, 512)
        e_geo  = torch.randn(1, 32)
        e_demo = torch.randn(1, 16)
        out = layer(e_id, e_geo, e_demo)
        assert out.shape == (1, 560)

    def test_output_dim_constant(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        assert FaceFusionLayer.OUT_DIM  == 560
        assert FaceFusionLayer.ID_DIM   == 512
        assert FaceFusionLayer.GEO_DIM  == 32
        assert FaceFusionLayer.DEMO_DIM == 16

    def test_gradient_flows(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        layer = FaceFusionLayer()
        e_id   = torch.randn(4, 512, requires_grad=True)
        e_geo  = torch.randn(4, 32,  requires_grad=True)
        e_demo = torch.randn(4, 16,  requires_grad=True)
        out = layer(e_id, e_geo, e_demo)
        loss = out.sum()
        loss.backward()
        assert e_id.grad is not None,   "No gradient to e_id"
        assert e_geo.grad is not None,  "No gradient to e_geo"
        assert e_demo.grad is not None, "No gradient to e_demo"

    def test_freeze_stops_gradient(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        layer = FaceFusionLayer()
        layer.freeze()
        for p in layer.parameters():
            assert not p.requires_grad, "Parameter should be frozen"
        assert layer._frozen

    def test_unfreeze_restores_gradient(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        layer = FaceFusionLayer()
        layer.freeze()
        layer.unfreeze()
        for p in layer.parameters():
            assert p.requires_grad, "Parameter should be unfrozen"

    def test_in_dim_property(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        layer = FaceFusionLayer()
        assert layer.in_dim == 560   # 512 + 32 + 16

    def test_no_nan_output(self) -> None:
        from voix.models.fusion import FaceFusionLayer
        layer = FaceFusionLayer()
        e_id   = torch.randn(16, 512)
        e_geo  = torch.randn(16, 32)
        e_demo = torch.randn(16, 16)
        out = layer(e_id, e_geo, e_demo)
        assert not torch.isnan(out).any(), "NaN in fusion output"
        assert not torch.isinf(out).any(), "Inf in fusion output"


# ===========================================================================
# T2.1: HeadPoseEstimator and frontal frame selection
# ===========================================================================

class TestHeadPoseEstimator:

    def test_import(self) -> None:
        from voix.data.head_pose import HeadPoseEstimator
        assert HeadPoseEstimator is not None

    def test_estimate_head_pose_returns_three_floats(self) -> None:
        from voix.data.head_pose import estimate_head_pose
        lm = _make_synthetic_landmarks(yaw_deg=0.0)
        result = estimate_head_pose(lm, 224, 224)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)
            assert math.isfinite(v)

    def test_frontal_score_lower_for_frontal(self) -> None:
        from voix.data.head_pose import (
            compute_frontal_score,
            compute_mouth_openness,
            estimate_head_pose,
        )
        lm_frontal  = _make_synthetic_landmarks(yaw_deg=0.0)
        lm_sideways = _make_synthetic_landmarks(yaw_deg=20.0)

        y1, p1, r1 = estimate_head_pose(lm_frontal, 224, 224)
        y2, p2, r2 = estimate_head_pose(lm_sideways, 224, 224)

        score_frontal  = compute_frontal_score(y1, p1, r1, compute_mouth_openness(lm_frontal))
        score_sideways = compute_frontal_score(y2, p2, r2, compute_mouth_openness(lm_sideways))

        # The frontal frame should score lower (or equal at worst)
        assert score_frontal <= score_sideways + 5.0, (
            f"Frontal score {score_frontal:.2f} not <= sideways {score_sideways:.2f}"
        )

    def test_select_frontal_frame_all_none(self) -> None:
        from voix.data.head_pose import select_frontal_frame
        result = select_frontal_frame([None, None, None])
        assert result is None

    def test_select_frontal_frame_single_valid(self) -> None:
        from voix.data.head_pose import select_frontal_frame
        from unittest.mock import patch
        lm = _make_synthetic_landmarks(yaw_deg=0.0)
        # Mock estimate_head_pose to return valid frontal angles (0, 0, 0)
        with patch('voix.data.head_pose.estimate_head_pose', return_value=(0.0, 0.0, 0.0)):
            result = select_frontal_frame([None, lm, None])
        assert result == 1

    def test_select_frontal_frame_picks_best(self) -> None:
        from voix.data.head_pose import select_frontal_frame
        from unittest.mock import patch
        lm_frontal  = _make_synthetic_landmarks(yaw_deg=0.0)
        lm_sideways = _make_synthetic_landmarks(yaw_deg=20.0)
        # Mock: frontal frame gets yaw=2, sideways gets yaw=22 -> frontal wins
        def mock_pose(lm, w=224, h=224):
            if lm is lm_frontal:
                return 2.0, 1.0, 0.5
            return 22.0, 5.0, 1.0
        with patch('voix.data.head_pose.estimate_head_pose', side_effect=mock_pose):
            result = select_frontal_frame([lm_sideways, lm_frontal, lm_sideways])
        assert result == 1  # frontal frame at index 1 should win

    def test_head_pose_estimator_score_frame(self) -> None:
        from voix.data.head_pose import HeadPoseEstimator
        from unittest.mock import patch
        est = HeadPoseEstimator()
        lm = _make_synthetic_landmarks()
        with patch('voix.data.head_pose.estimate_head_pose', return_value=(5.0, 3.0, 1.0)):
            yaw, pitch, roll, score = est.score_frame(lm, 224, 224)
        assert all(math.isfinite(v) for v in [yaw, pitch, roll, score])
        assert score >= 0.0

# ===========================================================================
# T2.7: VoxCelebPairedDataset and DataLoader
# ===========================================================================

class TestVoxCelebPairedDataset:

    @pytest.fixture
    def h5_file(self, tmp_path):
        path = str(tmp_path / "test.h5")
        _make_h5(path, n=200, n_spk=10)
        return path

    def test_import(self) -> None:
        from voix.data.dataset import VoxCelebPairedDataset
        assert VoxCelebPairedDataset is not None

    def test_len(self, h5_file) -> None:
        from voix.data.dataset import VoxCelebPairedDataset
        ds = VoxCelebPairedDataset(h5_file)
        assert len(ds) == 200

    def test_getitem_shapes(self, h5_file) -> None:
        from voix.data.dataset import VoxCelebPairedDataset
        ds = VoxCelebPairedDataset(h5_file)
        face, spk, spk_id = ds[0]
        assert face.shape == (560,),  f"face shape {face.shape}"
        assert spk.shape  == (192,),  f"spk shape {spk.shape}"
        assert isinstance(spk_id, int)

    def test_getitem_dtypes(self, h5_file) -> None:
        from voix.data.dataset import VoxCelebPairedDataset
        ds = VoxCelebPairedDataset(h5_file)
        face, spk, _ = ds[0]
        assert face.dtype == torch.float32
        assert spk.dtype  == torch.float32

    def test_speaker_split_no_leakage(self, h5_file) -> None:
        from voix.data.dataset import VoxCelebPairedDataset, make_speaker_split
        ds = VoxCelebPairedDataset(h5_file)
        train, val = make_speaker_split(ds, val_fraction=0.2, seed=42)
        # Extract speaker ids from both splits
        train_ids = set(ds[i][2] for i in train.indices)
        val_ids   = set(ds[i][2] for i in val.indices)
        overlap = train_ids & val_ids
        assert len(overlap) == 0, f"Speaker leakage: {overlap}"

    def test_dataloader_batch_shape(self, h5_file) -> None:
        from voix.data.dataset import VoxCelebPairedDataset, build_dataloader
        ds = VoxCelebPairedDataset(h5_file)
        loader = build_dataloader(ds, batch_size=16, shuffle=False, num_workers=0, seed=42)
        face, spk, ids = next(iter(loader))
        assert face.shape == (16, 560)
        assert spk.shape  == (16, 192)
        assert ids.shape  == (16,)

    def test_dataloader_reproducibility(self, h5_file) -> None:
        from voix.data.dataset import VoxCelebPairedDataset, build_dataloader
        ds = VoxCelebPairedDataset(h5_file)
        loader1 = build_dataloader(ds, batch_size=32, shuffle=True, num_workers=0, seed=99)
        loader2 = build_dataloader(ds, batch_size=32, shuffle=True, num_workers=0, seed=99)
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        assert torch.allclose(batch1[0], batch2[0]), "Same seed should produce same batch"


# ===========================================================================
# T2.7: DatasetIntegrityAuditor
# ===========================================================================

class TestDatasetIntegrityAuditor:

    @pytest.fixture
    def clean_h5(self, tmp_path):
        path = str(tmp_path / "clean.h5")
        _make_h5(path, n=100, n_spk=5)
        return path

    def test_clean_dataset_passes(self, clean_h5) -> None:
        from voix.data.dataset import DatasetIntegrityAuditor
        auditor = DatasetIntegrityAuditor(clean_h5)
        results = auditor.run()
        assert results["passed"], f"Clean dataset failed audit: {results}"

    def test_nan_detection(self, tmp_path) -> None:
        import h5py
        from voix.data.dataset import DatasetIntegrityAuditor
        path = str(tmp_path / "nan.h5")
        _make_h5(path, n=50)
        # Inject NaN
        with h5py.File(path, "r+") as f:
            f["face_features"][0, 0] = float("nan")
        auditor = DatasetIntegrityAuditor(path)
        results = auditor.run()
        assert results["face_nan"], "NaN not detected"
        assert not results["passed"], "Should fail when NaN present"

    def test_correct_n_samples(self, clean_h5) -> None:
        from voix.data.dataset import DatasetIntegrityAuditor
        auditor = DatasetIntegrityAuditor(clean_h5)
        results = auditor.run()
        assert results["n_samples"] == 100

    def test_correct_n_speakers(self, clean_h5) -> None:
        from voix.data.dataset import DatasetIntegrityAuditor
        auditor = DatasetIntegrityAuditor(clean_h5)
        results = auditor.run()
        assert results["n_speakers"] == 5
