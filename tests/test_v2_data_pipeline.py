"""Tests for the FaceFeatureV2 corpus pipeline.

These tests are fully synthetic: they do not require VoxCeleb2, a GPU, or
network access.  They cover the manifest builder, video/audio I/O helpers,
remote ZIP access, the manifest-selected downloader, the V2 HDF5 loader, and
the per-modality fusion normalization that fixed the collapsed V1 features.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from voix.data.voxceleb2_index import (
    ClipRecord,
    assign_speaker_splits,
    build_manifest,
    clips_per_speaker_for_gender,
    load_meta_genders,
    manifest_summary,
    parse_zip_member,
    sample_speaker_clips,
    save_manifest,
    stable_int,
)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _record(
    speaker: str,
    video: str,
    utterance: str,
    *,
    gender: str,
    set_name: str = "dev",
    shard: int = 1,
) -> ClipRecord:
    return ClipRecord(
        set=set_name,
        speaker_id=speaker,
        video_id=video,
        utterance_id=utterance,
        shard=shard,
        gender=gender,
        source_zip="vox2_mp4_1.zip",
    )


def _records() -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for index in range(8):
        gender = "f" if index < 4 else "m"
        speaker = f"id{index:05d}"
        for video in range(3):
            for utterance in range(4):
                records.append(
                    _record(
                        speaker,
                        f"video{video}",
                        f"{utterance:05d}",
                        gender=gender,
                    )
                )
    for index in range(2):
        records.append(
            _record(
                f"id{100 + index:05d}",
                "testvideo",
                "00001",
                gender="f" if index == 0 else "m",
                set_name="test",
                shard=7,
            )
        )
    return records


class TestManifestBuilder:
    def test_stable_int_is_reproducible(self):
        assert stable_int("dev/id00001/a/00001") == stable_int("dev/id00001/a/00001")
        assert stable_int("a") != stable_int("b")

    def test_parse_zip_member_layouts(self):
        dev = parse_zip_member(
            "dev/mp4/id00019/videoId/00001.mp4",
            set_name="dev",
            shard=1,
            source_zip="vox2_mp4_1.zip",
            gender="m",
        )
        assert dev is not None
        assert dev.clip_id == "dev/id00019/videoId/00001"
        assert dev.video_path == "id00019/videoId/00001.mp4"

        official = parse_zip_member(
            "mp4/id00019/videoId/00001.mp4",
            set_name="test",
            shard=7,
            source_zip="vox2_test_mp4.zip",
            gender="f",
        )
        assert official is not None
        assert official.set == "test"
        assert official.gender == "f"

    def test_parse_zip_member_rejects_non_clips(self):
        for name in (
            "dev/mp4/id00019/videoId/readme.txt",
            "dev/mp4/not_a_speaker/videoId/00001.mp4",
            "dev/mp4/id00019/videoId/not_an_utterance.mp4",
            "dev/mp4/id00019/00001.mp4",
            "dev/mp4/id00019/videoId/extra/00001.mp4",
        ):
            assert (
                parse_zip_member(
                    name,
                    set_name="dev",
                    shard=1,
                    source_zip="shard.zip",
                )
                is None
            )

    def test_load_meta_genders_strips_trailing_spaces(self, tmp_path):
        path = tmp_path / "vox2_meta.csv"
        path.write_text(
            "VoxCeleb2 ID ,VGGFace2 ID ,Gender ,Set \n"
            "id00001, n00001, m, train\n"
            "id00002, n00002, F, train\n"
            "id00003, n00003, x, test\n",
            encoding="utf-8",
        )
        genders = load_meta_genders(path)
        assert genders == {"id00001": "m", "id00002": "f", "id00003": "u"}

    def test_assign_speaker_splits_is_deterministic_and_disjoint(self):
        genders = {}
        for index in range(40):
            genders[f"id{index:05d}"] = "f" if index % 2 == 0 else "m"
        first = assign_speaker_splits(genders, val_fraction=0.1)
        second = assign_speaker_splits(genders, val_fraction=0.1)
        assert first == second
        train = {s for s, split in first.items() if split == "train"}
        val = {s for s, split in first.items() if split == "val"}
        assert not train & val
        # Gender-stratified: both groups contribute validation speakers.
        assert any(first[s] == "val" for s in genders if genders[s] == "f")
        assert any(first[s] == "val" for s in genders if genders[s] == "m")

    def test_sample_speaker_clips_is_deterministic_and_spreads_videos(self):
        clips = [r for r in _records() if r.speaker_id == "id00000"]
        first = sample_speaker_clips(clips, 6, seed="test")
        second = sample_speaker_clips(clips, 6, seed="test")
        assert [c.clip_id for c in first] == [c.clip_id for c in second]
        assert len(first) == 6
        assert len({c.video_id for c in first}) == 3

    def test_gender_cap_balances_clip_counts(self):
        caps = clips_per_speaker_for_gender(
            {"f": 100, "m": 200}, target=8, min_clips=4, balance=True
        )
        assert caps["f"] == 8
        assert caps["m"] == 4

    def test_build_manifest_invariants_and_balance(self):
        frame = build_manifest(
            _records(),
            val_fraction=0.25,
            max_clips_per_speaker=6,
            min_clips_per_speaker=2,
            balance_gender=True,
        )
        assert list(frame.columns) == [
            "clip_id",
            "speaker_id",
            "video_id",
            "utterance_id",
            "split",
            "gender",
            "shard",
            "video_path",
            "set",
            "source_zip",
        ]
        assert not frame["clip_id"].duplicated().any()
        assert frame.groupby("speaker_id")["split"].nunique().max() == 1
        assert set(frame["split"]) == {"train", "val", "test"}
        per_speaker = frame.groupby("speaker_id").size()
        assert per_speaker.max() <= 6
        train = frame[frame["split"] == "train"]
        counts = train["gender"].value_counts()
        assert abs(int(counts.get("f", 0)) - int(counts.get("m", 0))) <= 6

    def test_save_manifest_writes_summary_with_hash(self, tmp_path):
        frame = build_manifest(
            _records(),
            val_fraction=0.25,
            max_clips_per_speaker=4,
            min_clips_per_speaker=2,
        )
        manifest = tmp_path / "index.csv"
        summary_path = tmp_path / "summary.json"
        summary = save_manifest(
            frame,
            manifest,
            summary_path,
            val_fraction=0.25,
            max_clips_per_speaker=4,
            balance_gender=True,
        )
        assert manifest.exists() and summary_path.exists()
        assert summary["clips"] == len(frame)
        assert len(summary["manifest_sha256"]) == 64
        assert summary["splits"]["test"]["speakers"] == 2

    def test_manifest_summary_counts_shards(self):
        frame = build_manifest(
            _records(),
            val_fraction=0.25,
            max_clips_per_speaker=4,
            min_clips_per_speaker=2,
        )
        summary = manifest_summary(
            frame,
            val_fraction=0.25,
            max_clips_per_speaker=4,
            balance_gender=True,
        )
        assert summary["shards"]["1"]["clips"] > 0
        assert summary["shards"]["7"]["clips"] == 2


# ---------------------------------------------------------------------------
# Video / audio I/O
# ---------------------------------------------------------------------------


class TestVideoIO:
    def test_sample_indices_cover_first_and_last(self):
        from voix.data.video_io import _sample_indices

        picks = _sample_indices(100, 8)
        assert len(picks) == 8
        assert picks[0] == 0
        assert picks[-1] == 99
        assert picks == sorted(set(picks))

    def test_sample_indices_handle_short_and_single(self):
        from voix.data.video_io import _sample_indices

        assert _sample_indices(3, 8) == [0, 1, 2]
        assert _sample_indices(100, 1) == [0]
        assert _sample_indices(0, 8) == [0]

    def test_audio_duration_and_missing_file(self, tmp_path):
        from voix.data.video_io import audio_duration_sec

        wav = tmp_path / "tone.wav"
        sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
        assert audio_duration_sec(wav) == pytest.approx(1.0, abs=1e-3)
        assert audio_duration_sec(tmp_path / "missing.wav") == 0.0

    def test_missing_video_returns_empty(self, tmp_path):
        from voix.data.video_io import load_sampled_frames

        frames, times, meta = load_sampled_frames(tmp_path / "missing.mp4")
        assert frames == [] and times == [] and meta == {}

    def test_extract_audio_missing_video_returns_false(self, tmp_path):
        from voix.data.video_io import extract_audio

        assert not extract_audio(
            tmp_path / "missing.mp4", tmp_path / "out.wav", timeout_sec=5.0
        )

    @pytest.mark.parametrize("max_frames", [4, 8])
    def test_generated_video_frames_and_audio(self, tmp_path, max_frames):
        from voix.data.video_io import (
            audio_duration_sec,
            extract_audio,
            get_ffmpeg_exe,
            load_sampled_frames,
        )
        import subprocess

        video = tmp_path / "clip.mp4"
        command = [
            get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ]
        result = subprocess.run(command, capture_output=True, timeout=60)
        if result.returncode != 0 or not video.exists():
            pytest.skip("ffmpeg cannot generate the synthetic test video")

        frames, times, meta = load_sampled_frames(
            video, max_frames=max_frames, max_seconds=1.2
        )
        assert 1 <= len(frames) <= max_frames
        assert len(frames) == len(times)
        assert all(frame.shape == (120, 160, 3) for frame in frames)
        assert meta["fps"] == pytest.approx(10.0)
        assert meta["sampled_frames"] == len(frames)
        assert times[0] == pytest.approx(0.0)
        assert times[-1] <= 1.2

        wav = tmp_path / "clip.wav"
        assert extract_audio(video, wav, max_seconds=1.0)
        assert wav.stat().st_size > 44
        assert 0.5 <= audio_duration_sec(wav) <= 1.1


# ---------------------------------------------------------------------------
# Remote ZIP and manifest-selected download
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes, headers: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.text = content[:200].decode("utf-8", errors="replace")

    def close(self) -> None:
        pass


class _FakeRangeSession:
    """A requests.Session look-alike that serves byte ranges from a buffer."""

    def __init__(self, payload: bytes):
        self.payload = payload

    def get(self, url, headers=None, stream=False, timeout=None):
        headers = headers or {}
        range_header = headers.get("Range", "")
        if not range_header.startswith("bytes="):
            return _FakeResponse(200, self.payload)
        start_text, end_text = range_header[len("bytes=") :].split("-", 1)
        start = int(start_text)
        end = int(end_text) if end_text else len(self.payload) - 1
        end = min(end, len(self.payload) - 1)
        if start > end or start >= len(self.payload):
            return _FakeResponse(416, b"")
        body = self.payload[start : end + 1]
        return _FakeResponse(
            206,
            body,
            {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"},
        )

    def close(self) -> None:
        pass


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dev/mp4/id00001/vid/00001.mp4", b"video-one" * 100)
        archive.writestr("dev/mp4/id00001/vid/00002.mp4", b"video-two" * 50)
        archive.writestr("readme.txt", b"not a clip")
    return buffer.getvalue()


class TestRemoteZip:
    def test_read_list_and_download(self, tmp_path, monkeypatch):
        from voix.data import remote_zip

        payload = _zip_bytes()
        monkeypatch.setattr(
            remote_zip.requests,
            "Session",
            lambda: _FakeRangeSession(payload),
        )
        with remote_zip.RemoteZip("https://example.test/shard.zip") as archive:
            assert archive.size == len(payload)
            assert "dev/mp4/id00001/vid/00001.mp4" in archive.namelist()
            assert archive.read("readme.txt") == b"not a clip"
            destination = archive.download(
                "dev/mp4/id00001/vid/00001.mp4", tmp_path / "nested" / "clip.mp4"
            )
            assert destination.read_bytes() == b"video-one" * 100

    def test_remote_file_seek_and_read(self, monkeypatch):
        from voix.data import remote_zip

        payload = bytes(range(256)) * 4
        monkeypatch.setattr(
            remote_zip.requests,
            "Session",
            lambda: _FakeRangeSession(payload),
        )
        handle = remote_zip.RemoteFile("https://example.test/file.bin")
        assert handle.size == len(payload)
        handle.seek(10)
        assert handle.read(5) == payload[10:15]
        handle.seek(-4, 2)
        assert handle.read() == payload[-4:]
        handle.seek(0)
        buffer = bytearray(8)
        assert handle.readinto(buffer) == 8
        assert bytes(buffer) == payload[:8]
        handle.close()


class TestSelectedClipDownloader:
    @staticmethod
    def _module():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "data" / "download_selected_clips.py"
        spec = importlib.util.spec_from_file_location("download_selected_clips", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_member_name_for_dev_and_test(self):
        _member_name = self._module()._member_name

        assert (
            _member_name(
                pd.Series({"video_path": "id00001/vid/00001.mp4", "set": "dev"})
            )
            == "dev/mp4/id00001/vid/00001.mp4"
        )
        assert (
            _member_name(
                pd.Series({"video_path": "id00001/vid/00001.mp4", "set": "test"})
            )
            == "mp4/id00001/vid/00001.mp4"
        )

    def test_fetch_member_stored_and_deflated(self, tmp_path, monkeypatch):
        downloader = self._module()

        payload = _zip_bytes()
        archive = zipfile.ZipFile(io.BytesIO(payload))
        session = _FakeRangeSession(payload)
        monkeypatch.setattr(downloader, "_session", lambda: session)

        for member, expected in (
            ("readme.txt", b"not a clip"),
            ("dev/mp4/id00001/vid/00002.mp4", b"video-two" * 50),
        ):
            info = archive.getinfo(member)
            destination = tmp_path / member
            result = downloader._fetch_member(
                "https://example.test/shard.zip", "token", info, destination
            )
            assert result == destination
            assert destination.read_bytes() == expected

        # A second call with a matching file size skips the network.
        class _NoNetwork:
            def get(self, *args, **kwargs):
                raise AssertionError("network should not be used")

        monkeypatch.setattr(downloader, "_session", lambda: _NoNetwork())
        info = archive.getinfo("readme.txt")
        assert downloader._fetch_member(
            "https://example.test/shard.zip",
            "token",
            info,
            tmp_path / "readme.txt",
        ).read_bytes() == b"not a clip"


# ---------------------------------------------------------------------------
# FaceFeatureV2 HDF5 loader and scaler
# ---------------------------------------------------------------------------


def _write_v2_h5(
    path,
    *,
    rows: list[dict],
    face_dim: int = 560,
    arcface_dim: int = 512,
    voice_dim: int = 192,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    n = len(rows)
    with h5py.File(path, "w") as store:
        dt = h5py.special_dtype(vlen=str)
        for key, values in (
            ("clip_ids", [row["clip_id"] for row in rows]),
            ("speaker_ids", [row["speaker_id"] for row in rows]),
            ("split", [row["split"] for row in rows]),
            ("gender", [row["gender"] for row in rows]),
        ):
            dataset = store.create_dataset(key, shape=(n,), dtype=dt)
            for index, value in enumerate(values):
                dataset[index] = value
        store.create_dataset(
            "face_features", data=rng.standard_normal((n, face_dim)).astype(np.float32)
        )
        store.create_dataset(
            "arcface", data=rng.standard_normal((n, arcface_dim)).astype(np.float32)
        )
        store.create_dataset(
            "speaker_features",
            data=rng.standard_normal((n, voice_dim)).astype(np.float32),
        )


def _v2_rows() -> list[dict]:
    rows = []
    for speaker, split, gender, count in (
        ("id00001", "train", "f", 2),
        ("id00002", "train", "m", 2),
        ("id00003", "val", "f", 1),
        ("id00004", "test", "m", 1),
    ):
        for index in range(count):
            rows.append(
                {
                    "clip_id": f"{split}/{speaker}/vid/{index:05d}",
                    "speaker_id": speaker,
                    "split": split,
                    "gender": gender,
                    "video_path": f"{speaker}/vid/{index:05d}.mp4",
                }
            )
    return rows


class TestV2Dataset:
    def test_load_and_validate_manifest(self, tmp_path):
        from voix.training.data import load_v2_dataset

        rows = _v2_rows()
        features = tmp_path / "features.h5"
        manifest = tmp_path / "index.csv"
        _write_v2_h5(features, rows=rows)
        pd.DataFrame(rows).to_csv(manifest, index=False)

        dataset = load_v2_dataset(features, manifest)
        assert len(dataset) == len(rows)
        assert dataset.face_raw.shape == (len(rows), 560)
        assert dataset.speaker_raw.shape == (len(rows), 192)
        assert set(dataset.split) == {"train", "val", "test"}
        assert len(dataset.subset("train")) == 4

    def test_manifest_speaker_mismatch_is_rejected(self, tmp_path):
        from voix.training.data import load_v2_dataset

        rows = _v2_rows()
        features = tmp_path / "features.h5"
        manifest = tmp_path / "index.csv"
        _write_v2_h5(features, rows=rows)
        frame = pd.DataFrame(rows)
        frame.loc[0, "speaker_id"] = "id99999"
        frame.to_csv(manifest, index=False)
        with pytest.raises(ValueError, match="speaker mismatch"):
            load_v2_dataset(features, manifest)

    def test_duplicate_clip_ids_are_rejected(self, tmp_path):
        from voix.training.data import load_v2_dataset

        rows = _v2_rows()
        rows[1]["clip_id"] = rows[0]["clip_id"]
        features = tmp_path / "features.h5"
        _write_v2_h5(features, rows=rows)
        with pytest.raises(ValueError, match="duplicate clip_ids"):
            load_v2_dataset(features, None)

    def test_non_finite_features_are_rejected(self, tmp_path):
        from voix.training.data import load_v2_dataset

        rows = _v2_rows()
        features = tmp_path / "features.h5"
        _write_v2_h5(features, rows=rows)
        with h5py.File(features, "r+") as store:
            store["face_features"][0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            load_v2_dataset(features, None)

    def test_missing_schema_is_rejected(self, tmp_path):
        from voix.training.data import load_v2_dataset

        features = tmp_path / "features.h5"
        with h5py.File(features, "w") as store:
            store.create_dataset("face_features", data=np.zeros((2, 560), np.float32))
        with pytest.raises(ValueError, match="not a FaceFeatureV2 corpus"):
            load_v2_dataset(features, None)

    def test_arcface_face_key_is_supported(self, tmp_path):
        from voix.training.data import load_v2_dataset

        rows = _v2_rows()
        features = tmp_path / "features.h5"
        _write_v2_h5(features, rows=rows)
        dataset = load_v2_dataset(features, None, face_key="arcface")
        assert dataset.face_raw.shape == (len(rows), 512)

    def test_scaler_uses_train_split_only(self, tmp_path):
        from voix.training.data import build_face_scaler_v2, load_v2_dataset

        rows = _v2_rows()
        features = tmp_path / "features.h5"
        _write_v2_h5(features, rows=rows)
        dataset = load_v2_dataset(features, None)
        output = tmp_path / "scaler.pt"
        payload = build_face_scaler_v2(dataset, output, source="unit-test")
        train = dataset.subset("train")
        assert output.exists()
        assert payload["schema_version"] == "FaceFeatureV2"
        assert payload["face_key"] == "face_features"
        assert payload["feature_dim"] == 560
        assert payload["n_train"] == len(train)
        assert payload["n_train_speakers"] == 2
        assert torch.allclose(
            payload["face_mean"],
            torch.from_numpy(train.face_raw.mean(axis=0)),
            atol=1e-6,
        )
        assert torch.all(payload["face_std"] > 0)
        loaded = torch.load(output, map_location="cpu", weights_only=False)
        assert loaded["source"] == "unit-test"

    def test_arcface_training_uses_separate_scaler(self):
        from argparse import Namespace

        from voix.training.data import DEFAULT_SCALER_V2
        from voix.training.trainer_cfvn import resolve_scaler_path

        fused = Namespace(face_key="face_features", scaler=DEFAULT_SCALER_V2)
        assert resolve_scaler_path(fused) == DEFAULT_SCALER_V2

        arcface = Namespace(face_key="arcface", scaler=DEFAULT_SCALER_V2)
        resolved = resolve_scaler_path(arcface)
        assert resolved != DEFAULT_SCALER_V2
        assert resolved.name == "face_scaler_arcface_v2.pt"

        # A relative spelling of the canonical path must not overwrite it.
        relative = Namespace(
            face_key="arcface",
            scaler=Path("data/processed/voxceleb2_v2/face_scaler_v2.pt"),
        )
        assert resolve_scaler_path(relative).name == "face_scaler_arcface_v2.pt"


class TestScalerProvenance:
    """A scaler is only valid for the exact train split it was fit on."""

    @staticmethod
    def _build(tmp_path, *, face_key="face_features"):
        from voix.training.data import build_face_scaler_v2, load_v2_dataset

        rows = _v2_rows()
        features = tmp_path / "features.h5"
        _write_v2_h5(features, rows=rows)
        dataset = load_v2_dataset(features, None, face_key=face_key)
        output = tmp_path / "scaler.pt"
        build_face_scaler_v2(
            dataset, output, source="unit-test", face_key=face_key
        )
        return dataset, output

    def test_matching_scaler_passes(self, tmp_path):
        from voix.data.feature_contract import load_scaler_payload
        from voix.training.data import validate_scaler_for_dataset

        dataset, output = self._build(tmp_path)
        scaler = load_scaler_payload(
            output, expected_dim=560, face_key="face_features"
        )
        assert validate_scaler_for_dataset(scaler, dataset) is scaler

    def test_stale_row_count_is_rejected(self, tmp_path):
        from voix.data.feature_contract import load_scaler_payload
        from voix.training.data import validate_scaler_for_dataset

        dataset, output = self._build(tmp_path)
        scaler = load_scaler_payload(output, expected_dim=560)
        scaler["n_train"] = int(scaler["n_train"]) + 1
        with pytest.raises(ValueError, match="train rows"):
            validate_scaler_for_dataset(scaler, dataset)

    def test_stale_speaker_count_is_rejected(self, tmp_path):
        from voix.data.feature_contract import load_scaler_payload
        from voix.training.data import validate_scaler_for_dataset

        dataset, output = self._build(tmp_path)
        scaler = load_scaler_payload(output, expected_dim=560)
        scaler["n_train_speakers"] = int(scaler["n_train_speakers"]) - 1
        with pytest.raises(ValueError, match="train speakers"):
            validate_scaler_for_dataset(scaler, dataset)

    def test_wrong_face_key_is_rejected(self, tmp_path):
        from voix.data.feature_contract import load_scaler_payload

        _, output = self._build(tmp_path)
        with pytest.raises(ValueError, match="face_key"):
            load_scaler_payload(output, expected_dim=560, face_key="arcface")

    def test_wrong_dimension_is_rejected(self, tmp_path):
        from voix.data.feature_contract import load_scaler_payload

        _, output = self._build(tmp_path)
        with pytest.raises(ValueError, match="512-D"):
            load_scaler_payload(output, expected_dim=512)

    def test_arcface_scaler_roundtrip(self, tmp_path):
        from voix.data.feature_contract import load_scaler_payload
        from voix.training.data import validate_scaler_for_dataset

        dataset, output = self._build(tmp_path, face_key="arcface")
        scaler = load_scaler_payload(
            output, expected_dim=512, face_key="arcface"
        )
        assert scaler["feature_dim"] == 512
        assert validate_scaler_for_dataset(
            scaler, dataset, face_key="arcface"
        ) is scaler

    def test_cfvn_reuses_valid_scaler(self, tmp_path):
        from argparse import Namespace

        from voix.training.trainer_cfvn import resolve_or_build_scaler

        dataset, output = self._build(tmp_path)
        args = Namespace(
            face_key="face_features",
            scaler=output,
            features=Path("synthetic"),
        )
        path, scaler = resolve_or_build_scaler(args, dataset, 560)
        assert path == output
        # The original build recorded source="unit-test"; a rebuild would
        # replace it with the args.features string.
        assert scaler["source"] == "unit-test"

    def test_cfvn_rebuilds_stale_scaler(self, tmp_path):
        from argparse import Namespace

        from voix.training.trainer_cfvn import resolve_or_build_scaler

        dataset, output = self._build(tmp_path)
        payload = torch.load(output, map_location="cpu", weights_only=False)
        payload["n_train"] = int(payload["n_train"]) - 1
        torch.save(payload, output)

        args = Namespace(
            face_key="face_features",
            scaler=output,
            features=Path("synthetic"),
        )
        path, scaler = resolve_or_build_scaler(args, dataset, 560)
        assert path == output
        assert scaler["n_train"] == 4
        assert scaler["source"] == "synthetic"


# ---------------------------------------------------------------------------
# Image / corpus demographic parity
# ---------------------------------------------------------------------------


class TestDemographicParity:
    def test_aspect_ratio_from_bbox(self):
        from voix.data.demographics import aspect_ratio_from_bbox

        assert aspect_ratio_from_bbox([0, 0, 100, 150]) == 1.5
        assert aspect_ratio_from_bbox(np.array([10, 20, 10, 20])) == 1.0

    def test_crop_face_bbox_is_clamped(self):
        from voix.data.face_extractor import crop_face_bbox

        image = np.zeros((100, 80, 3), dtype=np.uint8)
        crop = crop_face_bbox(image, [-10, -10, 40, 50], margin=0.5)
        assert crop.shape[:2] == (80, 65)

    def test_estimate_forwards_aspect_ratio(self):
        from voix.data.demographics import SoftDemographicEstimator

        estimator = SoftDemographicEstimator(device="cpu")
        direct = estimator.estimate_from_attributes(
            gender=1, age=30.0, aspect_ratio=2.0
        )
        forwarded = estimator.estimate(
            np.zeros(512, dtype=np.float32),
            gender=1,
            age=30.0,
            aspect_ratio=2.0,
        )
        assert forwarded.shape == (16,)
        assert np.allclose(direct, forwarded)
        narrow = estimator.estimate_from_attributes(
            gender=1, age=30.0, aspect_ratio=1.0
        )
        assert not np.allclose(direct, narrow)
        assert np.allclose(direct[:14], narrow[:14])


# ---------------------------------------------------------------------------
# Fusion normalization (the V1 collapse fix)
# ---------------------------------------------------------------------------


class TestFusionNormalization:
    def test_per_modality_normalization_is_scale_invariant(self):
        from voix.models.fusion import FaceFusionLayer

        torch.manual_seed(0)
        layer = FaceFusionLayer().eval()
        identity = torch.randn(4, 512)
        geometry = torch.randn(4, 32) * 100.0
        demographics = torch.randn(4, 16) * 0.01
        with torch.no_grad():
            base = layer(identity, geometry, demographics)
            scaled = layer(
                identity * 500.0,
                geometry * 0.01,
                demographics * 1000.0,
            )
        assert torch.allclose(base, scaled, atol=1e-5)

    def test_fused_features_are_not_constant_across_identities(self):
        from voix.models.fusion import FaceFusionLayer

        torch.manual_seed(1)
        layer = FaceFusionLayer().eval()
        identity = torch.nn.functional.normalize(torch.randn(16, 512), dim=-1)
        geometry = torch.randn(16, 32)
        demographics = torch.randn(16, 16)
        with torch.no_grad():
            fused = torch.nn.functional.normalize(layer(identity, geometry, demographics), dim=-1)
        similarity = fused @ fused.T
        off_diagonal = similarity[~torch.eye(len(fused), dtype=torch.bool)]
        assert float(off_diagonal.max()) < 0.99
        assert float(off_diagonal.std()) > 1e-3

    def test_fusion_is_deterministic(self):
        from voix.models.fusion import FaceFusionLayer

        torch.manual_seed(2)
        inputs = (
            torch.randn(2, 512),
            torch.randn(2, 32),
            torch.randn(2, 16),
        )
        with torch.no_grad():
            first = FaceFusionLayer().eval()(*inputs)
            second = FaceFusionLayer().eval()(*inputs)
        assert torch.allclose(first, second)

class TestDatasetSelect:
    def test_select_preserves_row_alignment(self):
        from voix.training.data import CleanFeatureDataset

        dataset = CleanFeatureDataset(
            face_raw=np.arange(12, dtype=np.float32).reshape(6, 2),
            speaker_raw=np.arange(18, dtype=np.float32).reshape(6, 3),
            speaker_ids=np.asarray(["a", "a", "b", "b", "c", "c"]),
            split=np.asarray(["train"] * 4 + ["val"] * 2),
            clip_ids=np.asarray([f"c{i}" for i in range(6)]),
            hdf5_index=np.arange(6, dtype=np.int64),
            gender=np.asarray(["f", "f", "m", "m", "f", "m"]),
        )
        selected = dataset.select([1, 3, 5])
        assert len(selected) == 3
        assert selected.clip_ids.tolist() == ["c1", "c3", "c5"]
        assert selected.hdf5_index.tolist() == [1, 3, 5]
        assert np.array_equal(selected.face_raw, dataset.face_raw[[1, 3, 5]])
        with pytest.raises(IndexError):
            dataset.select([6])
