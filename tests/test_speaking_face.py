"""Tests for speaking-face tracking and audio-visual selection."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voix.data.speaking_face import (
    FaceDetection,
    audio_rms_envelope,
    bbox_iou,
    build_tracks,
    centrality_score,
    choose_track,
    frontality_from_pose,
    mouth_openness,
    pearson_correlation,
    score_track,
    select_speaking_face,
)


def _landmarks(mouth_gap: float) -> np.ndarray:
    lm = np.zeros((68, 3), dtype=np.float32)
    lm[36, :2] = (0.0, 0.0)   # left eye outer
    lm[45, :2] = (1.0, 0.0)   # right eye outer -> IOC = 1
    lm[62, :2] = (0.5, 0.5)   # upper inner lip
    lm[66, :2] = (0.5, 0.5 + float(mouth_gap))
    return lm


def _detection(
    frame_index: int,
    bbox,
    mouth_gap: float,
    *,
    det_score: float = 0.9,
) -> FaceDetection:
    return FaceDetection(
        frame_index=frame_index,
        time_sec=frame_index * 0.1,
        bbox=np.asarray(bbox, dtype=np.float32),
        det_score=det_score,
        landmarks_68=_landmarks(mouth_gap),
        pose=np.zeros(3, dtype=np.float32),
        mouth_open=float(mouth_gap),
    )


class TestGeometry:
    def test_bbox_iou(self):
        assert bbox_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
        assert bbox_iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0
        # Half-overlap: intersection 50, union 150.
        assert bbox_iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])) == pytest.approx(1 / 3)

    def test_mouth_openness(self):
        assert mouth_openness(_landmarks(0.1)) == pytest.approx(0.1)
        assert mouth_openness(None) is None
        assert mouth_openness(np.zeros((10, 3))) is None
        # Degenerate inter-ocular distance.
        lm = _landmarks(0.1)
        lm[45, :2] = lm[36, :2]
        assert mouth_openness(lm) is None

    def test_frontality_and_centrality(self):
        assert frontality_from_pose(np.zeros(3)) == pytest.approx(1.0)
        assert frontality_from_pose(None) == pytest.approx(0.5)
        assert centrality_score(np.array([220, 140, 420, 340]), 640, 480) == pytest.approx(1.0)
        assert centrality_score(np.array([0, 0, 20, 20]), 640, 480) < 0.2

    def test_pearson_correlation(self):
        assert pearson_correlation(np.array([1, 2, 3]), np.array([2, 4, 6])) == pytest.approx(1.0)
        assert pearson_correlation(np.array([1, 2, 3]), np.array([6, 4, 2])) == pytest.approx(-1.0)
        assert pearson_correlation(np.array([1, 1, 1]), np.array([1, 2, 3])) == 0.0
        assert pearson_correlation(np.array([1, 2]), np.array([1, 2])) == 0.0


class TestTracking:
    def test_build_tracks_keeps_faces_separate(self):
        frames = []
        for i in range(3):
            frames.append(
                [
                    _detection(i, [10, 10, 60, 60], 0.02),
                    _detection(i, [100, 100, 300, 300], 0.02),
                ]
            )
        tracks = build_tracks(frames)
        assert len(tracks) == 2
        assert sorted(len(t.detections) for t in tracks) == [3, 3]

    def test_tracking_respects_max_gap(self):
        frames = [
            [_detection(0, [10, 10, 60, 60], 0.02)],
            [_detection(5, [10, 10, 60, 60], 0.02)],
        ]
        tracks = build_tracks(frames, max_gap_frames=2)
        assert len(tracks) == 2

    def test_speaking_track_beats_larger_static_face(self):
        audio = np.array([0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.2, 0.8], dtype=np.float32)
        frames = []
        for i, amplitude in enumerate(audio):
            frames.append(
                [
                    _detection(i, [10, 10, 60, 60], float(amplitude) * 0.1),
                    _detection(i, [100, 100, 300, 300], 0.02),
                ]
            )
        tracks = build_tracks(frames)
        best, scores = choose_track(tracks, audio, 640, 480)
        assert best is not None
        speaking = next(s for s in scores if s.av_sync > 0.9)
        assert speaking.track_id == best.track_id
        assert speaking.av_pairs == 8
        assert speaking.score > max(s.score for s in scores if s.track_id != best.track_id)

    def test_score_track_without_audio_prefers_face_cues(self):
        track = build_tracks([[_detection(0, [100, 100, 300, 300], 0.02)]])[0]
        score = score_track(track, None, 640, 480)
        assert score.av_pairs == 0
        assert 0.0 <= score.score <= 1.0


class FakeDetector:
    def detect(self, frame):
        bboxes = np.array(
            [
                [10.0, 10.0, 60.0, 60.0, 0.9],
                [100.0, 100.0, 300.0, 300.0, 0.9],
            ],
            dtype=np.float32,
        )
        kpss = np.zeros((2, 5, 2), dtype=np.float32)
        return bboxes, kpss


class FakeLandmarkModel:
    # Must match the audio envelope used by the selection test so the speaking
    # face's mouth motion is actually correlated with speech energy.
    AUDIO_ENVELOPE = (0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.2, 0.8)

    def get(self, frame, face):
        frame_index = int(frame[0, 0, 0])
        width = float(face.bbox[2] - face.bbox[0])
        if width < 100:
            # Speaking face: mouth motion follows the audio envelope.
            gap = 0.01 + 0.08 * self.AUDIO_ENVELOPE[frame_index]
        else:
            gap = 0.02
        face.landmark_3d_68 = _landmarks(gap)
        face.pose = np.zeros(3, dtype=np.float32)


class FakeRecognition:
    def get(self, frame, face):
        width = float(face.bbox[2] - face.bbox[0])
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0 if width < 100 else 1] = 1.0
        # InsightFace exposes ``normed_embedding`` as a read-only property over
        # ``embedding``; real recognition models set the latter.
        face.embedding = embedding


class FakeGenderAge:
    def get(self, frame, face):
        face.gender = 1
        face.age = 30.0


class FakeApp:
    def __init__(self):
        self.det_model = FakeDetector()
        self.models = {
            "landmark_3d_68": FakeLandmarkModel(),
            "recognition": FakeRecognition(),
            "genderage": FakeGenderAge(),
        }


class TestSelectSpeakingFace:
    def test_end_to_end_selection(self):
        # Frame value encodes its index so the fake landmark model can vary
        # mouth openness over time.
        frames = [
            np.full((240, 320, 3), i, dtype=np.uint8) for i in range(8)
        ]
        times = [i * 0.1 for i in range(8)]
        audio = np.array([0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.2, 0.8], dtype=np.float32)
        selected = select_speaking_face(
            frames,
            times,
            audio,
            FakeApp(),
            max_frames=8,
            recognition_frames=4,
        )
        assert selected is not None
        # The small, off-centre face is the speaking track.
        assert selected.bbox[2] - selected.bbox[0] < 100
        assert selected.embedding[0] > 0.9
        assert selected.av_sync > 0.9
        assert selected.av_pairs == 8
        assert selected.gender == 1
        assert selected.age == pytest.approx(30.0)
        assert selected.frames_used == 4


class TestAudioEnvelope:
    def test_rms_envelope_tracks_amplitude(self, tmp_path):
        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        signal = np.concatenate(
            [
                0.05 * np.sin(2 * np.pi * 220 * t),
                0.8 * np.sin(2 * np.pi * 220 * t),
            ]
        )
        wav = tmp_path / "envelope.wav"
        sf.write(wav, signal, sr)
        times = [0.25, 0.75, 1.25, 1.75]
        envelope = audio_rms_envelope(str(wav), times)
        assert envelope.shape == (4,)
        assert envelope[0] < envelope[-1]
        assert abs(float(envelope.mean())) < 1e-5
