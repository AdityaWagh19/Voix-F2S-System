"""Head-pose estimation and frontal frame selection (T2.1).

Estimates 3D head pose (yaw, pitch, roll) from MediaPipe FaceMesh landmarks
using OpenCV solvePnP with a canonical 3D facial anchor model.

Frontal frame selection metric (from Phase 2 plan):
    S_t = |yaw_t| + 1.5|pitch_t| + 0.5|roll_t| + mouth_openness_t * 100
    t* = argmin_t S_t
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Canonical 3D facial anchor model (6 points, millimetres)
# Nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
# ---------------------------------------------------------------------------
_ANCHOR_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ],
    dtype=np.float64,
)

_ANCHOR_IDX = [1, 199, 33, 263, 61, 291]
_UPPER_LIP: int = 13
_LOWER_LIP: int = 14


def estimate_head_pose(
    landmarks: np.ndarray,
    image_width: int = 224,
    image_height: int = 224,
) -> tuple:
    """Estimate (yaw, pitch, roll) in degrees from MediaPipe 468 landmarks."""
    import cv2

    pts_2d = np.array(
        [[landmarks[i][0] * image_width, landmarks[i][1] * image_height]
         for i in _ANCHOR_IDX],
        dtype=np.float64,
    )
    focal = float(image_width)
    cam_matrix = np.array(
        [[focal, 0, image_width / 2],
         [0, focal, image_height / 2],
         [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        _ANCHOR_3D, pts_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)

    # Euler angle extraction from rotation matrix using atan2.
    # More stable than RQDecomp3x3 which has gimbal lock / sign ambiguity.
    # Convention: yaw around Y, pitch around X, roll around Z.
    pitch_rad = float(np.arctan2(-rmat[2, 0], np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)))
    yaw_rad   = float(np.arctan2(rmat[1, 0] / np.cos(pitch_rad), rmat[0, 0] / np.cos(pitch_rad)))
    roll_rad  = float(np.arctan2(rmat[2, 1] / np.cos(pitch_rad), rmat[2, 2] / np.cos(pitch_rad)))

    yaw_deg   = float(np.degrees(yaw_rad))
    pitch_deg = float(np.degrees(pitch_rad))
    roll_deg  = float(np.degrees(roll_rad))

    return yaw_deg, pitch_deg, roll_deg


def compute_mouth_openness(landmarks: np.ndarray) -> float:
    """Normalized vertical lip gap (relative to inter-ocular distance)."""
    ioc = float(np.linalg.norm(landmarks[33] - landmarks[263]))
    if ioc < 1e-6:
        return 0.0
    return float(np.linalg.norm(landmarks[_UPPER_LIP] - landmarks[_LOWER_LIP])) / ioc


def compute_frontal_score(yaw: float, pitch: float, roll: float, mouth: float) -> float:
    """S_t = |yaw| + 1.5|pitch| + 0.5|roll| + mouth_openness * 100."""
    return abs(yaw) + 1.5 * abs(pitch) + 0.5 * abs(roll) + mouth * 100.0


def select_frontal_frame(
    landmarks_per_frame: list,
    image_width: int = 224,
    image_height: int = 224,
    max_yaw: float = 25.0,
    max_pitch: float = 25.0,
) -> Optional[int]:
    """Return index of most frontal/neutral frame, or None if none qualify."""
    best_idx = None
    best_score = float("inf")
    for idx, lm in enumerate(landmarks_per_frame):
        if lm is None:
            continue
        yaw, pitch, roll = estimate_head_pose(lm, image_width, image_height)
        if abs(yaw) > max_yaw or abs(pitch) > max_pitch:
            continue
        score = compute_frontal_score(yaw, pitch, roll, compute_mouth_openness(lm))
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


class HeadPoseEstimator:
    """Stateful wrapper for batch head-pose estimation and frame selection."""

    def __init__(self, max_yaw: float = 25.0, max_pitch: float = 25.0) -> None:
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch

    def select_best_frame(
        self,
        landmarks_per_frame: list,
        image_width: int = 224,
        image_height: int = 224,
    ) -> Optional[int]:
        """Return index of best frontal frame from a list of landmark arrays."""
        return select_frontal_frame(
            landmarks_per_frame, image_width, image_height,
            self.max_yaw, self.max_pitch,
        )

    def score_frame(
        self,
        landmarks: np.ndarray,
        image_width: int = 224,
        image_height: int = 224,
    ) -> tuple:
        """Return (yaw, pitch, roll, frontal_score) for a single frame."""
        yaw, pitch, roll = estimate_head_pose(landmarks, image_width, image_height)
        mouth = compute_mouth_openness(landmarks)
        score = compute_frontal_score(yaw, pitch, roll, mouth)
        return yaw, pitch, roll, score