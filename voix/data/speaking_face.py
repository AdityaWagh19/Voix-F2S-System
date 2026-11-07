"""Speaking-face selection for paired face/voice corpus extraction.

VoxCeleb clips are cut from interview-style videos.  The old extractor used the
largest face and stopped at the first frontal frame, which frequently selected
the interviewer or a background face instead of the speaking subject.  This
module tracks every detected face across sampled frames and chooses the track
whose mouth motion best correlates with the clip's audio envelope.

The module is deliberately model-agnostic: callers pass an InsightFace
``FaceAnalysis`` instance (or any object exposing ``det_model.detect`` and the
``models`` mapping).  This keeps the selection logic unit-testable without
loading ONNX models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# iBUG 68-point landmark indices
# ---------------------------------------------------------------------------
LM68_LEFT_EYE_OUTER = 36
LM68_RIGHT_EYE_OUTER = 45
LM68_LEFT_EYE_INNER = 39
LM68_RIGHT_EYE_INNER = 42
LM68_MOUTH_LEFT = 48
LM68_MOUTH_RIGHT = 54
LM68_MOUTH_UPPER_INNER = 62
LM68_MOUTH_LOWER_INNER = 66


@dataclass
class FaceDetection:
    """One face detected in one sampled frame."""

    frame_index: int
    time_sec: float
    bbox: np.ndarray                 # (4,) x1, y1, x2, y2 in pixels
    det_score: float
    kps: Optional[np.ndarray] = None  # (5, 2) InsightFace keypoints
    landmarks_68: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    mouth_open: Optional[float] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class FaceTrack:
    """A face followed across sampled frames."""

    track_id: int
    detections: list[FaceDetection] = field(default_factory=list)

    def add(self, detection: FaceDetection) -> None:
        self.detections.append(detection)

    def sorted_detections(self) -> list[FaceDetection]:
        return sorted(self.detections, key=lambda d: d.time_sec)

    @property
    def last(self) -> Optional[FaceDetection]:
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.time_sec)

    def mouth_series(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, mouth_openness) for detections with valid landmarks."""
        points = [
            (d.time_sec, d.mouth_open)
            for d in self.sorted_detections()
            if d.mouth_open is not None and np.isfinite(d.mouth_open)
        ]
        if not points:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        times, mouths = zip(*points)
        return np.asarray(times, dtype=np.float32), np.asarray(mouths, dtype=np.float32)


@dataclass
class TrackScore:
    """Interpretable score components for one track."""

    track_id: int
    score: float
    av_sync: float
    mouth_var: float
    mouth_mean: float
    face_area: float
    frontal: float
    det_score: float
    centrality: float
    n_frames: int
    av_pairs: int = 0


@dataclass
class SelectedFace:
    """The chosen speaking face for one clip."""

    embedding: np.ndarray             # (512,) L2-normalized ArcFace embedding
    bbox: np.ndarray                  # (4,) best-frame bbox
    frame_index: int
    time_sec: float
    landmarks_68: Optional[np.ndarray]
    pose: Optional[np.ndarray]
    gender: Optional[int]
    age: Optional[float]
    det_score: float
    track_id: int
    track_len: int
    n_faces: int
    n_tracks: int
    av_sync: float
    av_pairs: int
    mouth_var: float
    mouth_mean: float
    face_area: float
    frontal: float
    centrality: float
    score: float
    frames_used: int


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def mouth_openness(landmarks_68: Optional[np.ndarray]) -> Optional[float]:
    """Normalized inner-lip gap from iBUG 68 landmarks.

    The gap is divided by inter-ocular distance so it is invariant to face
    scale.  Returns None when the landmarks are missing or degenerate.
    """
    if landmarks_68 is None:
        return None
    lm = np.asarray(landmarks_68)
    if lm.ndim != 2 or lm.shape[0] < 68 or lm.shape[1] < 2:
        return None
    ioc = float(np.linalg.norm(lm[LM68_LEFT_EYE_OUTER, :2] - lm[LM68_RIGHT_EYE_OUTER, :2]))
    if not np.isfinite(ioc) or ioc < 1e-6:
        return None
    gap = float(
        np.linalg.norm(
            lm[LM68_MOUTH_UPPER_INNER, :2] - lm[LM68_MOUTH_LOWER_INNER, :2]
        )
    )
    if not np.isfinite(gap):
        return None
    return gap / ioc


def frontality_from_pose(pose: Optional[np.ndarray]) -> float:
    """Map InsightFace pose (pitch, yaw, roll in degrees) to [0, 1]."""
    if pose is None:
        return 0.5
    values = np.asarray(pose, dtype=np.float32).reshape(-1)
    if values.size < 3 or not np.isfinite(values[:3]).all():
        return 0.5
    mean_abs = float(np.mean(np.abs(values[:3])))
    return float(1.0 / (1.0 + mean_abs / 25.0))


def centrality_score(bbox: np.ndarray, frame_width: int, frame_height: int) -> float:
    """1.0 at the frame centre, 0.0 at the farthest corner."""
    if frame_width <= 0 or frame_height <= 0:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    max_dist = float(np.hypot(frame_width / 2.0, frame_height / 2.0))
    dist = float(np.hypot(cx - frame_width / 2.0, cy - frame_height / 2.0))
    return float(max(0.0, 1.0 - dist / max(max_dist, 1e-6)))


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over finite pairs; 0.0 if undefined."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    a, b = a[mask], b[mask]
    if float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else 0.0


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def assign_detections_to_tracks(
    tracks: list[FaceTrack],
    detections: list[FaceDetection],
    iou_threshold: float = 0.25,
    max_gap_frames: int = 2,
) -> tuple[list[FaceTrack], list[FaceDetection]]:
    """Greedily match detections to existing tracks by IoU.

    A track is only extended when the new detection is within
    ``max_gap_frames`` of its last detection.  This prevents a face from being
    stitched across a camera cut or a long occlusion.

    Returns the updated track list and the unmatched detections.  New tracks
    are *not* created here; the caller decides how to seed them.
    """
    unmatched = list(detections)
    if not tracks or not unmatched:
        return tracks, unmatched

    candidates: list[tuple[float, int, int]] = []
    for ti, track in enumerate(tracks):
        last = track.last
        if last is None:
            continue
        for di, det in enumerate(unmatched):
            if int(det.frame_index) - int(last.frame_index) > int(max_gap_frames):
                continue
            iou = bbox_iou(last.bbox, det.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, ti, di))
    candidates.sort(reverse=True)

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for iou, ti, di in candidates:
        if ti in used_tracks or di in used_dets:
            continue
        tracks[ti].add(unmatched[di])
        used_tracks.add(ti)
        used_dets.add(di)

    remaining = [d for i, d in enumerate(unmatched) if i not in used_dets]
    return tracks, remaining


def build_tracks(
    detections_per_frame: list[list[FaceDetection]],
    iou_threshold: float = 0.25,
    max_gap_frames: int = 2,
) -> list[FaceTrack]:
    """Build tracks from per-frame detections."""
    tracks: list[FaceTrack] = []
    for detections in detections_per_frame:
        if not detections:
            continue
        tracks, unmatched = assign_detections_to_tracks(
            tracks,
            detections,
            iou_threshold=iou_threshold,
            max_gap_frames=max_gap_frames,
        )
        for det in unmatched:
            track = FaceTrack(track_id=len(tracks))
            track.add(det)
            tracks.append(track)
    return tracks


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_track(
    track: FaceTrack,
    audio_rms: Optional[np.ndarray],
    frame_width: int,
    frame_height: int,
) -> TrackScore:
    """Score one track for likelihood of being the speaking subject.

    The audio-visual correlation is the strongest cue: the speaking subject's
    mouth motion should follow the clip's speech envelope.  Face size, mouth
    variance, frontality, detector confidence, and centrality are fallbacks and
    tie-breakers.
    """
    dets = track.sorted_detections()
    if not dets:
        return TrackScore(
            track_id=track.track_id,
            score=0.0,
            av_sync=0.0,
            mouth_var=0.0,
            mouth_mean=0.0,
            face_area=0.0,
            frontal=0.0,
            det_score=0.0,
            centrality=0.0,
            n_frames=0,
            av_pairs=0,
        )

    times, mouths = track.mouth_series()
    mouth_var = float(np.var(mouths)) if len(mouths) > 1 else 0.0
    mouth_mean = float(np.mean(mouths)) if len(mouths) else 0.0

    # Audio envelope sampled at the track's frame indices.  ``mouth_series``
    # and ``frame_indices`` must use the exact same filter, otherwise the two
    # arrays silently misalign.
    av_sync = 0.0
    av_pairs = 0
    if audio_rms is not None and len(times) >= 3:
        audio = np.asarray(audio_rms, dtype=np.float32).reshape(-1)
        frame_indices = np.array(
            [
                d.frame_index
                for d in dets
                if d.mouth_open is not None and np.isfinite(d.mouth_open)
            ],
            dtype=np.int64,
        )
        n_pairs = min(len(mouths), len(frame_indices))
        if n_pairs >= 3:
            paired_indices = frame_indices[:n_pairs]
            valid = paired_indices < len(audio)
            if valid.sum() >= 3:
                av_sync = pearson_correlation(
                    mouths[:n_pairs][valid], audio[paired_indices[valid]]
                )
                av_pairs = int(valid.sum())

    areas = np.array([d.area for d in dets], dtype=np.float32)
    median_area = float(np.median(areas)) if areas.size else 0.0
    frame_area = float(max(1, frame_width * frame_height))
    face_area = float(np.clip(median_area / frame_area, 0.0, 1.0))

    frontal = float(np.mean([frontality_from_pose(d.pose) for d in dets]))
    det_score = float(np.mean([d.det_score for d in dets]))
    centrality = float(np.mean([
        centrality_score(d.bbox, frame_width, frame_height) for d in dets
    ]))

    # Normalize the raw cues into roughly [0, 1].
    av_score = float(np.clip((av_sync + 1.0) / 2.0, 0.0, 1.0))
    mouth_score = float(np.clip(mouth_var / 0.02, 0.0, 1.0))
    size_score = float(np.clip(face_area / 0.06, 0.0, 1.0))
    track_len_score = float(np.clip(len(dets) / 6.0, 0.0, 1.0))

    # Correlation from only a handful of frames is noisy.  Shrink it toward
    # neutral until we have enough paired samples to trust it.
    reliability = float(np.clip(av_pairs / 8.0, 0.0, 1.0))
    av_effective = 0.5 + reliability * (av_score - 0.5)

    has_audio = av_pairs >= 3
    if has_audio:
        score = (
            0.50 * av_effective
            + 0.15 * mouth_score
            + 0.10 * size_score
            + 0.08 * frontal
            + 0.07 * det_score
            + 0.05 * centrality
            + 0.05 * track_len_score
        )
    else:
        score = (
            0.30 * mouth_score
            + 0.26 * size_score
            + 0.16 * frontal
            + 0.14 * det_score
            + 0.08 * centrality
            + 0.06 * track_len_score
        )

    # A one-frame track is almost always a false positive or a cutaway.
    if len(dets) < 2:
        score *= 0.5

    return TrackScore(
        track_id=track.track_id,
        score=float(score),
        av_sync=float(av_sync),
        mouth_var=mouth_var,
        mouth_mean=mouth_mean,
        face_area=face_area,
        frontal=frontal,
        det_score=det_score,
        centrality=centrality,
        n_frames=len(dets),
        av_pairs=av_pairs,
    )


def choose_track(
    tracks: list[FaceTrack],
    audio_rms: Optional[np.ndarray],
    frame_width: int,
    frame_height: int,
) -> tuple[Optional[FaceTrack], list[TrackScore]]:
    """Return the best speaking-face track and all track scores."""
    scores = [score_track(t, audio_rms, frame_width, frame_height) for t in tracks]
    if not scores:
        return None, []
    best_idx = max(
        range(len(scores)),
        key=lambda i: (
            scores[i].score,
            scores[i].face_area,
            scores[i].centrality,
            scores[i].n_frames,
        ),
    )
    return tracks[best_idx], scores


# ---------------------------------------------------------------------------
# Model-backed detection
# ---------------------------------------------------------------------------

def _insightface_face(bbox: np.ndarray, kps: Optional[np.ndarray], det_score: float):
    from insightface.app.common import Face

    return Face(bbox=np.asarray(bbox, dtype=np.float32), kps=kps, det_score=float(det_score))


def detect_faces_in_frame(
    app: Any,
    frame_bgr: np.ndarray,
    frame_index: int,
    time_sec: float,
    *,
    max_faces: int = 5,
    min_det_score: float = 0.30,
    min_face_area_ratio: float = 0.001,
) -> list[FaceDetection]:
    """Detect all faces and their 68-point landmarks in one BGR frame.

    Only the detector and the 68-point landmark model are run here.  Recognition
    and gender/age are deferred until the speaking track is known, which keeps
    the per-frame cost low.

    Tiny background faces are dropped before the landmark model runs.  The
    speaking subject is normally one of the largest faces in a VoxCeleb clip,
    and running a 68-point model on every background face is the dominant cost
    on crowded frames.
    """
    # InsightFace's default ``det_thresh`` is 0.5.  VoxCeleb2 frames are small,
    # often shadowed, and frequently motion-blurred; a 0.5 floor drops most
    # frames of the actual subject and leaves only a handful of easy frames.
    # Use our own (lower) floor so tracking has a continuous signal, then apply
    # the caller's confidence filter below.
    try:
        bboxes, kpss = app.det_model.detect(
            frame_bgr, det_thresh=float(min_det_score)
        )
    except TypeError:
        bboxes, kpss = app.det_model.detect(frame_bgr)
    if bboxes is None or len(bboxes) == 0:
        return []

    h, w = frame_bgr.shape[:2]
    frame_area = float(max(1, h * w))
    candidates: list[tuple[float, int]] = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = [float(v) for v in bboxes[i, :4]]
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
        score = float(bboxes[i, 4]) if bboxes.shape[1] > 4 else 0.0
        if score < min_det_score or area_ratio < min_face_area_ratio:
            continue
        candidates.append((area_ratio, i))
    candidates.sort(reverse=True)
    keep = [i for _, i in candidates[: max(1, int(max_faces))]]

    detections: list[FaceDetection] = []
    for i in keep:
        bbox = np.asarray(bboxes[i, :4], dtype=np.float32)
        det_score = float(bboxes[i, 4]) if bboxes.shape[1] > 4 else 0.0
        kps = np.asarray(kpss[i], dtype=np.float32) if kpss is not None else None
        face = _insightface_face(bbox, kps, det_score)
        landmarks_68 = None
        pose = None
        try:
            landmark_model = app.models.get("landmark_3d_68")
            if landmark_model is not None:
                landmark_model.get(frame_bgr, face)
                if getattr(face, "landmark_3d_68", None) is not None:
                    landmarks_68 = np.asarray(face.landmark_3d_68, dtype=np.float32)
                if getattr(face, "pose", None) is not None:
                    pose = np.asarray(face.pose, dtype=np.float32)
        except Exception:
            landmarks_68 = None
            pose = None

        detections.append(
            FaceDetection(
                frame_index=frame_index,
                time_sec=float(time_sec),
                bbox=bbox,
                det_score=det_score,
                kps=kps,
                landmarks_68=landmarks_68,
                pose=pose,
                mouth_open=mouth_openness(landmarks_68),
            )
        )
    return detections


def audio_rms_envelope(
    wav_path: str,
    times_sec: Iterable[float],
    sample_rate: int = 16000,
    window_sec: float = 0.08,
) -> np.ndarray:
    """Sample a normalized audio RMS envelope at the given frame times."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    times = np.asarray(list(times_sec), dtype=np.float32)
    if times.size == 0 or audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    half = max(1, int(round(window_sec * sr / 2.0)))
    values = np.zeros(times.size, dtype=np.float32)
    for i, t in enumerate(times):
        centre = int(round(float(t) * sr))
        lo = max(0, centre - half)
        hi = min(audio.size, centre + half)
        if hi <= lo:
            values[i] = 0.0
            continue
        window = audio[lo:hi]
        values[i] = float(np.sqrt(np.mean(np.square(window)) + 1e-12))

    # z-score so Pearson correlation is not dominated by loudness.
    std = float(values.std())
    if std > 1e-9:
        values = (values - float(values.mean())) / std
    else:
        values[:] = 0.0
    return values


def select_speaking_face(
    frames_bgr: list[np.ndarray],
    frame_times_sec: list[float],
    audio_rms: Optional[np.ndarray],
    app: Any,
    *,
    max_frames: int = 16,
    recognition_frames: int = 4,
    iou_threshold: float = 0.25,
    max_faces: int = 5,
    min_det_score: float = 0.30,
    min_face_area_ratio: float = 0.001,
    max_gap_frames: int = 2,
) -> Optional[SelectedFace]:
    """Select the speaking face and compute its averaged ArcFace embedding.

    Args:
        frames_bgr: sampled frames in temporal order.
        frame_times_sec: timestamp of each frame in seconds.
        audio_rms: audio RMS envelope aligned with ``frames_bgr`` (or None).
        app: InsightFace ``FaceAnalysis`` instance.
        max_frames: maximum frames to analyse (evenly subsampled if needed).
        recognition_frames: number of high-quality track frames to average.
        iou_threshold: IoU threshold for track association.
        max_faces: maximum number of largest faces to run landmarks on.
        min_det_score: detector confidence floor.
        min_face_area_ratio: minimum face area as a fraction of the frame.
        max_gap_frames: maximum frame gap when extending a track.

    Returns:
        ``SelectedFace`` or None when no usable face/embedding is found.
    """
    if not frames_bgr:
        return None
    if len(frames_bgr) != len(frame_times_sec):
        raise ValueError("frames_bgr and frame_times_sec must have equal length")

    # Evenly subsample long clips while preserving temporal order.
    indices = list(range(len(frames_bgr)))
    if len(indices) > max_frames:
        picks = np.linspace(0, len(indices) - 1, max_frames)
        indices = sorted({int(round(p)) for p in picks})

    per_frame: list[list[FaceDetection]] = []
    for idx in indices:
        frame = frames_bgr[idx]
        per_frame.append(
            detect_faces_in_frame(
                app,
                frame,
                idx,
                frame_times_sec[idx],
                max_faces=max_faces,
                min_det_score=min_det_score,
                min_face_area_ratio=min_face_area_ratio,
            )
        )

    tracks = build_tracks(
        per_frame,
        iou_threshold=iou_threshold,
        max_gap_frames=max_gap_frames,
    )
    if not tracks:
        return None

    h, w = frames_bgr[0].shape[:2]
    best_track, scores = choose_track(tracks, audio_rms, w, h)
    if best_track is None:
        return None
    best_score = next(s for s in scores if s.track_id == best_track.track_id)

    # Rank the track's frames by detector confidence, frontality, and mouth
    # openness.  We prefer a few clear, frontal, open-mouth frames for ArcFace.
    ranked = sorted(
        best_track.sorted_detections(),
        key=lambda d: (
            d.det_score
            * frontality_from_pose(d.pose)
            * (0.5 + float(d.mouth_open or 0.0))
        ),
        reverse=True,
    )
    chosen = ranked[: max(1, recognition_frames)]

    embeddings: list[np.ndarray] = []
    genders: list[int] = []
    ages: list[float] = []
    best_det: Optional[FaceDetection] = None
    best_quality = -1.0
    for det in chosen:
        frame = frames_bgr[det.frame_index]
        face = _insightface_face(det.bbox, det.kps, det.det_score)
        try:
            recognition = app.models.get("recognition")
            if recognition is not None:
                recognition.get(frame, face)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is not None:
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if np.isfinite(norm) and norm > 1e-8:
                    embeddings.append((vector / norm).astype(np.float32))
            genderage = app.models.get("genderage")
            if genderage is not None:
                genderage.get(frame, face)
            if getattr(face, "gender", None) is not None:
                genders.append(int(face.gender))
            if getattr(face, "age", None) is not None:
                ages.append(float(face.age))
        except Exception:
            continue

        quality = det.det_score * frontality_from_pose(det.pose)
        if quality > best_quality:
            best_quality = quality
            best_det = det

    if not embeddings:
        return None

    emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(emb))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    emb = (emb / norm).astype(np.float32)

    if best_det is None:
        best_det = chosen[0]

    gender = None
    if genders:
        # Majority vote; InsightFace returns 1 for male, 0 for female.
        gender = int(round(float(np.mean(genders))))
    age = float(np.mean(ages)) if ages else None

    return SelectedFace(
        embedding=emb,
        bbox=best_det.bbox.copy(),
        frame_index=best_det.frame_index,
        time_sec=best_det.time_sec,
        landmarks_68=best_det.landmarks_68,
        pose=best_det.pose,
        gender=gender,
        age=age,
        det_score=best_det.det_score,
        track_id=best_track.track_id,
        track_len=len(best_track.detections),
        n_faces=sum(len(d) for d in per_frame),
        n_tracks=len(tracks),
        av_sync=best_score.av_sync,
        av_pairs=best_score.av_pairs,
        mouth_var=best_score.mouth_var,
        mouth_mean=best_score.mouth_mean,
        face_area=best_score.face_area,
        frontal=best_score.frontal,
        centrality=best_score.centrality,
        score=best_score.score,
        frames_used=len(embeddings),
    )
