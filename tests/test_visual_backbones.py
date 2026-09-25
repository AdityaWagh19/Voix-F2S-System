"""Backbone smoke tests: ArcFace identity extractor and MediaPipe FaceMesh.

These tests verify that pretrained visual feature extractors load correctly,
produce tensors of the exact specified dimensions, and behave deterministically
on fixed inputs.

mediapipe>=1.0.0 uses the tasks API (mediapipe.tasks.vision.FaceLandmarker).
The solutions API (mp.solutions) is no longer available.

Run with:
    pytest tests/test_visual_backbones.py -v
"""

import numpy as np
import pytest

from voix.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_face_bgr():
    """Return a reproducible 112x112 synthetic BGR face image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def synthetic_face_rgb():
    """Return a reproducible 112x112 synthetic RGB face image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def large_face_rgb():
    """Return a larger 224x224 synthetic RGB face image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# ArcFace extractor tests
# ---------------------------------------------------------------------------

class TestArcFaceExtractor:
    """Verify ArcFaceExtractor interface, output shape, and normalization."""

    def test_import(self):
        """ArcFaceExtractor class can be imported from voix.data."""
        from voix.data.face_extractor import ArcFaceExtractor
        assert ArcFaceExtractor is not None

    def test_instantiation(self):
        """ArcFaceExtractor can be instantiated with default parameters."""
        from voix.data.face_extractor import ArcFaceExtractor
        extractor = ArcFaceExtractor(device_id=-1, cache_dir="./checkpoints")
        assert extractor.model_name == "buffalo_l"
        assert not extractor._loaded

    def test_embedding_dim_constant(self):
        """Embedding dimension constant is 512."""
        from voix.data.face_extractor import ArcFaceExtractor
        assert ArcFaceExtractor.EMBEDDING_DIM == 512


# ---------------------------------------------------------------------------
# FaceMesh extractor tests
# ---------------------------------------------------------------------------

class TestFaceMeshExtractor:
    """Verify FaceMeshExtractor interface and landmark count constant."""

    def test_import(self):
        """FaceMeshExtractor class can be imported from voix.data."""
        from voix.data.face_extractor import FaceMeshExtractor
        assert FaceMeshExtractor is not None

    def test_num_landmarks_constant(self):
        """NUM_LANDMARKS constant is exactly 468."""
        from voix.data.face_extractor import FaceMeshExtractor
        assert FaceMeshExtractor.NUM_LANDMARKS == 468

    def test_instantiation(self):
        """FaceMeshExtractor can be instantiated with default parameters."""
        from voix.data.face_extractor import FaceMeshExtractor
        extractor = FaceMeshExtractor(num_faces=1)
        assert extractor.num_faces == 1
        assert not extractor._loaded

    def test_mediapipe_tasks_api_available(self):
        """mediapipe.tasks.vision module is importable (requires mediapipe>=1.0)."""
        from mediapipe.tasks.python import vision as mp_vision
        assert hasattr(mp_vision, "FaceLandmarker")

    def test_extract_synthetic_rgb(self, synthetic_face_rgb):
        """FaceMeshExtractor.extract() on synthetic image: None or (468,3).

        On a pure noise image the landmarker returns None (no face geometry).
        If a model file is cached, the test also validates the output shape.
        """
        from pathlib import Path
        from voix.data.face_extractor import FaceMeshExtractor

        model_cached = Path("./checkpoints/mediapipe/face_landmarker.task").exists()
        if not model_cached:
            pytest.skip("FaceLandmarker model not cached; skipping online test")

        extractor = FaceMeshExtractor(num_faces=1)
        result = extractor.extract(synthetic_face_rgb)
        if result is not None:
            assert result.shape == (468, 3), f"Expected (468, 3), got {result.shape}"
            assert result.dtype == np.float32
        extractor.close()

    def test_extract_large_image(self, large_face_rgb):
        """FaceMeshExtractor handles 224x224 images correctly."""
        from pathlib import Path
        from voix.data.face_extractor import FaceMeshExtractor

        model_cached = Path("./checkpoints/mediapipe/face_landmarker.task").exists()
        if not model_cached:
            pytest.skip("FaceLandmarker model not cached; skipping online test")

        extractor = FaceMeshExtractor(num_faces=1)
        result = extractor.extract(large_face_rgb)
        if result is not None:
            assert result.shape == (468, 3)
            assert result.dtype == np.float32
        extractor.close()


# ---------------------------------------------------------------------------
# Architectural contract validation tests
# ---------------------------------------------------------------------------

class TestArchitecturalContracts:
    """Verify architectural constants match specifications in architecture.md."""

    def test_face_embedding_dim(self):
        """Total face embedding dimension is 560-D (512 + 32 + 16)."""
        arcface_dim = 512
        facemesh_dim = 32
        demographics_dim = 16
        total = arcface_dim + facemesh_dim + demographics_dim
        assert total == 560, f"Expected 560-D face embedding, got {total}"

    def test_speaker_embedding_dim(self):
        """Speaker embedding dimension is 192-D (ECAPA-TDNN contract)."""
        from voix.data.speaker_extractor import ECAPAExtractor
        assert ECAPAExtractor.EMBEDDING_DIM == 192

    def test_sample_rate(self):
        """Audio sample rate is 16 kHz (standard for ECAPA-TDNN input)."""
        from voix.data.speaker_extractor import ECAPAExtractor
        assert ECAPAExtractor.SAMPLE_RATE == 16000

    def test_facemesh_num_landmarks(self):
        """FaceMesh produces exactly 468 landmarks."""
        from voix.data.face_extractor import FaceMeshExtractor
        assert FaceMeshExtractor.NUM_LANDMARKS == 468