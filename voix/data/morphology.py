"""Craniofacial morphological ratio extractor (32-D).

Computes 32 anatomical distance ratios and angular measurements from
MediaPipe FaceMesh 468-point 3D landmarks. All distances are normalized
by inter-ocular distance (IOC) to remove camera focal length and zoom effects.

Landmark indices reference MediaPipe canonical face model:
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Canonical landmark index constants
# ---------------------------------------------------------------------------
IOC_LEFT: int = 33      # Left eye outer canthus
IOC_RIGHT: int = 263    # Right eye outer canthus

# Vertical face geometry
FOREHEAD_TOP: int = 10
CHIN_BOTTOM: int = 152
LEFT_CHEEK: int = 234
RIGHT_CHEEK: int = 454

# Jaw landmarks
JAW_LEFT: int = 172
JAW_RIGHT: int = 397
JAW_CENTER: int = 152

# Nose landmarks
NOSE_TIP: int = 4
NOSE_BOTTOM: int = 2
NOSE_LEFT: int = 102
NOSE_RIGHT: int = 331
NOSE_BRIDGE: int = 6

# Lip landmarks
UPPER_LIP_TOP: int = 13
LOWER_LIP_BOTTOM: int = 14
LIP_LEFT_CORNER: int = 61
LIP_RIGHT_CORNER: int = 291
PHILTRUM_TOP: int = 164

# Eye landmarks
EYE_LEFT_INNER: int = 133
EYE_RIGHT_INNER: int = 362

# Orbital landmarks
BROW_LEFT: int = 70
BROW_RIGHT: int = 300


def _dist(landmarks: np.ndarray, a: int, b: int) -> float:
    """Compute Euclidean distance between two landmarks."""
    return float(np.linalg.norm(landmarks[a] - landmarks[b]))


def _ioc_distance(landmarks: np.ndarray) -> float:
    """Compute inter-ocular distance for scale normalization."""
    return _dist(landmarks, IOC_LEFT, IOC_RIGHT)


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle in degrees between two 3D vectors."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_craniofacial_ratios(landmarks: np.ndarray) -> np.ndarray:
    """Compute 32-D normalized craniofacial morphological ratio vector.

    All distance-based features are divided by inter-ocular distance (IOC)
    to achieve scale invariance. Angular features are in degrees.

    Args:
        landmarks: Float32 array of shape (468, 3) in normalized [0,1]
                   coordinate space from MediaPipe FaceMesh.

    Returns:
        Float32 array of shape (32,) containing morphological ratios.
        Returns zero vector if landmarks are invalid.

    Feature index mapping:
        0:  Face height / IOC
        1:  Face width (cheek) / IOC
        2:  Jaw width / IOC
        3:  Mandible angle (degrees)
        4:  Nose width / IOC
        5:  Nose height / IOC
        6:  Nose bridge length / IOC
        7:  Eye separation / IOC (== 1.0 by definition, sanity check)
        8:  Inner eye separation / IOC
        9:  Lower face ratio (chin to lip / face height)
        10: Upper face ratio (forehead to eye / face height)
        11: Lip width / IOC
        12: Upper lip height / IOC
        13: Lower lip height / IOC
        14: Philtrum length / IOC
        15: Left cheekbone offset / IOC
        16: Right cheekbone offset / IOC
        17: Nose tip to chin / IOC
        18: Nose tip to forehead / IOC
        19: Brow span / IOC
        20: Brow to eye distance (left) / IOC
        21: Brow to eye distance (right) / IOC
        22: Eye width (left) / IOC
        23: Eye width (right) / IOC
        24: Face aspect ratio (height / width)
        25: Mandible to nose ratio
        26: Lip to jaw ratio
        27: Nose to lip ratio
        28: Jaw angle left (degrees)
        29: Jaw angle right (degrees)
        30: Chin protrusion (z-depth) relative to cheeks
        31: Nose tip protrusion (z-depth) relative to bridge
    """
    if landmarks is None or landmarks.shape != (468, 3):
        return np.zeros(32, dtype=np.float32)

    ioc = _ioc_distance(landmarks)
    if ioc < 1e-9:
        return np.zeros(32, dtype=np.float32)

    features = np.zeros(32, dtype=np.float32)

    # 0: Face height / IOC
    features[0] = _dist(landmarks, FOREHEAD_TOP, CHIN_BOTTOM) / ioc

    # 1: Face width (cheek to cheek) / IOC
    features[1] = _dist(landmarks, LEFT_CHEEK, RIGHT_CHEEK) / ioc

    # 2: Jaw width / IOC
    features[2] = _dist(landmarks, JAW_LEFT, JAW_RIGHT) / ioc

    # 3: Mandible angle (degrees) - angle at chin vertex
    v_left = landmarks[JAW_LEFT] - landmarks[JAW_CENTER]
    v_right = landmarks[JAW_RIGHT] - landmarks[JAW_CENTER]
    features[3] = _angle_deg(v_left, v_right)

    # 4: Nose width / IOC
    features[4] = _dist(landmarks, NOSE_LEFT, NOSE_RIGHT) / ioc

    # 5: Nose height (bridge to tip) / IOC
    features[5] = _dist(landmarks, NOSE_BRIDGE, NOSE_TIP) / ioc

    # 6: Nose bridge length / IOC
    features[6] = _dist(landmarks, BROW_LEFT, NOSE_TIP) / ioc

    # 7: Inter-ocular distance / IOC (sanity check = 1.0)
    features[7] = ioc / ioc  # == 1.0 always

    # 8: Inner eye separation / IOC
    features[8] = _dist(landmarks, EYE_LEFT_INNER, EYE_RIGHT_INNER) / ioc

    # 9: Lower face ratio (nose-tip to chin / total height)
    face_height = _dist(landmarks, FOREHEAD_TOP, CHIN_BOTTOM)
    lower_face = _dist(landmarks, NOSE_TIP, CHIN_BOTTOM)
    features[9] = lower_face / (face_height + 1e-9)

    # 10: Upper face ratio (forehead to brow center / total height)
    brow_center = (landmarks[BROW_LEFT] + landmarks[BROW_RIGHT]) / 2
    forehead_to_brow = np.linalg.norm(landmarks[FOREHEAD_TOP] - brow_center)
    features[10] = float(forehead_to_brow) / (face_height + 1e-9)

    # 11: Lip width / IOC
    features[11] = _dist(landmarks, LIP_LEFT_CORNER, LIP_RIGHT_CORNER) / ioc

    # 12 & 13: Upper and lower lip height / IOC
    lip_center_x = (landmarks[LIP_LEFT_CORNER] + landmarks[LIP_RIGHT_CORNER]) / 2
    features[12] = float(np.linalg.norm(landmarks[UPPER_LIP_TOP] - lip_center_x)) / ioc
    features[13] = float(np.linalg.norm(landmarks[LOWER_LIP_BOTTOM] - lip_center_x)) / ioc

    # 14: Philtrum length / IOC
    features[14] = _dist(landmarks, PHILTRUM_TOP, UPPER_LIP_TOP) / ioc

    # 15 & 16: Cheekbone lateral offsets / IOC
    face_center_x = (landmarks[IOC_LEFT][0] + landmarks[IOC_RIGHT][0]) / 2
    features[15] = abs(float(landmarks[LEFT_CHEEK][0]) - face_center_x) / ioc
    features[16] = abs(float(landmarks[RIGHT_CHEEK][0]) - face_center_x) / ioc

    # 17: Nose tip to chin / IOC
    features[17] = _dist(landmarks, NOSE_TIP, CHIN_BOTTOM) / ioc

    # 18: Nose tip to forehead / IOC
    features[18] = _dist(landmarks, NOSE_TIP, FOREHEAD_TOP) / ioc

    # 19: Brow span / IOC
    features[19] = _dist(landmarks, BROW_LEFT, BROW_RIGHT) / ioc

    # 20 & 21: Brow to eye distance (left and right) / IOC
    features[20] = float(np.linalg.norm(landmarks[BROW_LEFT] - landmarks[IOC_LEFT])) / ioc
    features[21] = float(np.linalg.norm(landmarks[BROW_RIGHT] - landmarks[IOC_RIGHT])) / ioc

    # 22 & 23: Eye width / IOC (outer to inner canthus)
    features[22] = _dist(landmarks, IOC_LEFT, EYE_LEFT_INNER) / ioc
    features[23] = _dist(landmarks, IOC_RIGHT, EYE_RIGHT_INNER) / ioc

    # 24: Face aspect ratio (height / width)
    face_width = _dist(landmarks, LEFT_CHEEK, RIGHT_CHEEK)
    features[24] = face_height / (face_width + 1e-9)

    # 25: Mandible to nose ratio
    features[25] = _dist(landmarks, JAW_LEFT, JAW_RIGHT) / (_dist(landmarks, NOSE_LEFT, NOSE_RIGHT) + 1e-9)

    # 26: Lip to jaw ratio
    features[26] = _dist(landmarks, LIP_LEFT_CORNER, LIP_RIGHT_CORNER) / (_dist(landmarks, JAW_LEFT, JAW_RIGHT) + 1e-9)

    # 27: Nose to lip ratio
    features[27] = _dist(landmarks, NOSE_LEFT, NOSE_RIGHT) / (_dist(landmarks, LIP_LEFT_CORNER, LIP_RIGHT_CORNER) + 1e-9)

    # 28 & 29: Jaw angles left and right (degrees at jaw corners)
    v_jaw_l = landmarks[JAW_LEFT] - landmarks[CHIN_BOTTOM]
    v_chin_l = landmarks[FOREHEAD_TOP] - landmarks[CHIN_BOTTOM]
    features[28] = _angle_deg(v_jaw_l, v_chin_l)

    v_jaw_r = landmarks[JAW_RIGHT] - landmarks[CHIN_BOTTOM]
    features[29] = _angle_deg(v_jaw_r, v_chin_l)

    # 30: Chin z-protrusion relative to cheeks (depth cue)
    cheek_z = (landmarks[LEFT_CHEEK][2] + landmarks[RIGHT_CHEEK][2]) / 2
    features[30] = float(landmarks[CHIN_BOTTOM][2]) - cheek_z

    # 31: Nose tip z-protrusion relative to nose bridge
    features[31] = float(landmarks[NOSE_TIP][2]) - float(landmarks[NOSE_BRIDGE][2])

    return features.astype(np.float32)
