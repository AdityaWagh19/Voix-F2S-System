"""ArcFace and MediaPipe FaceMesh feature extraction wrappers.

All extractors are immutable feature computers with parameters frozen at
instantiation. No gradient computation is performed through these modules.

MediaPipe note: mediapipe>=1.0.0 removed the legacy mp.solutions API.
This module targets mediapipe>=1.0.0 using the mediapipe.tasks API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


class ArcFaceExtractor:
    """Frozen ArcFace ResNet-50 identity embedding extractor.

    Produces L2-normalized 512-D identity embeddings invariant to
    illumination and minor pose variation. Uses InsightFace buffalo_l
    model pack which bundles RetinaFace detection + ArcFace recognition.

    Args:
        model_name: InsightFace model pack name. Default: "buffalo_l".
        device_id: GPU device index. 0 for primary GPU, -1 for CPU.
        cache_dir: Directory to cache downloaded model weights.
    """

    EMBEDDING_DIM: int = 512

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device_id: int = 0,
        cache_dir: str = "./checkpoints",
    ) -> None:
        self.model_name = model_name
        self.device_id = device_id
        self.cache_dir = str(Path(cache_dir).resolve())
        self._app = None
        self._loaded = False

    def _load(self) -> None:
        """Lazy-load model on first use to avoid VRAM usage at import time."""
        from insightface.app import FaceAnalysis

        os.makedirs(self.cache_dir, exist_ok=True)
        self._app = FaceAnalysis(
            name=self.model_name,
            root=self.cache_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=self.device_id, det_size=(112, 112))
        self._loaded = True

    def extract(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-D L2-normalized identity embedding from a BGR image.

        Args:
            image_bgr: HxWx3 uint8 BGR numpy array (OpenCV convention).

        Returns:
            Float32 array of shape (512,) with L2 norm == 1.0,
            or None if no face is detected in the image.
        """
        if not self._loaded:
            self._load()

        faces = self._app.get(image_bgr)
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.normed_embedding.astype(np.float32)
        return embedding

    def extract_batch(self, images_bgr: list[np.ndarray]) -> list[Optional[np.ndarray]]:
        """Extract embeddings from a list of BGR images."""
        return [self.extract(img) for img in images_bgr]


class FaceMeshExtractor:
    """MediaPipe FaceMesh 478-point 3D landmark extractor.

    Uses mediapipe.tasks.vision.FaceLandmarker (mediapipe>=1.0.0 API).
    Returns 468 canonical landmarks (first 468 of the 478 output set,
    excluding the 10 iris landmarks) for cross-version consistency.

    Args:
        num_faces: Maximum number of faces to detect. Default: 1.
        min_detection_confidence: Detection confidence threshold.
        model_path: Optional path to a local .task model file.
                    If None, downloads from MediaPipe CDN on first use.
    """

    NUM_LANDMARKS: int = 468
    _MODEL_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(
        self,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        self.num_faces = num_faces
        self.min_detection_confidence = min_detection_confidence
        self.model_path = model_path
        self._landmarker = None
        self._loaded = False

    def _download_model(self) -> str:
        """Download the FaceLandmarker model task file if not cached."""
        import urllib.request

        cache_dir = Path("./checkpoints/mediapipe")
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "face_landmarker.task"

        if not model_path.exists():
            print(f"Downloading MediaPipe FaceLandmarker model to {model_path}...")
            urllib.request.urlretrieve(self._MODEL_URL, model_path)

        return str(model_path)

    def _load(self) -> None:
        """Lazy-load MediaPipe FaceLandmarker on first use."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_file = self.model_path or self._download_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_file)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_score=self.min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._loaded = True

    def extract(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 facial landmarks from an RGB image.

        Args:
            image_rgb: HxWx3 uint8 RGB numpy array.

        Returns:
            Float32 array of shape (468, 3) containing normalized (x, y, z)
            coordinates in [0, 1] range, or None if no face is detected.
        """
        import mediapipe as mp

        if not self._loaded:
            self._load()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        # Take first 468 canonical landmarks (exclude 10 iris landmarks)
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks[:self.NUM_LANDMARKS]],
            dtype=np.float32,
        )

        assert coords.shape == (self.NUM_LANDMARKS, 3), (
            f"Expected ({self.NUM_LANDMARKS}, 3) landmarks, got {coords.shape}"
        )
        return coords

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()