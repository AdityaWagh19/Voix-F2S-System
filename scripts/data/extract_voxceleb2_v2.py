"""VoxCeleb2 paired face/voice extraction -- FaceFeatureV2.

This replaces the old ``scripts/extract_voxceleb.py`` face path, which selected
the largest/first frontal face and therefore frequently captured interviewers
instead of the speaking subject.  The new pipeline:

  1. samples frames across the whole clip,
  2. detects and tracks every face,
  3. selects the track whose mouth motion correlates with the audio envelope,
  4. averages ArcFace embeddings over several high-quality frames,
  5. stores raw ArcFace, craniofacial geometry, demographics, fused features,
     and per-clip quality/selection metadata.

The HDF5 is resumable by ``clip_ids`` and safe to append to across shards.

Usage:
    python scripts/data/extract_voxceleb2_v2.py \
        --index data/processed/voxceleb2_v2/index.csv \
        --videos data/raw/voxceleb2/dev/mp4 \
        --output data/processed/features_v2.h5 \
        --device cuda --max-speakers 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Allow direct execution from the repository root.
ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.data.face_extractor import FaceMeshExtractor, crop_face_bbox
from voix.data.feature_contract import FaceFeatureBuilder
from voix.data.morphology import compute_craniofacial_ratios
from voix.data.speaking_face import (
    SelectedFace,
    audio_rms_envelope,
    select_speaking_face,
)
from voix.data.speaker_extractor import ECAPAExtractor
from voix.data.video_io import (
    audio_duration_sec,
    extract_audio,
    last_audio_error,
    load_sampled_frames,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "FaceFeatureV2"
FACE_DIM = 560
ID_DIM = 512
GEO_DIM = 32
DEMO_DIM = 16
SPEAKER_DIM = 192
QUALITY_DIM = 12
MIN_AUDIO_SEC = 2.5
MAX_AUDIO_SEC = 12.0
N_ECAPA_BATCH = 4
PREFETCH_QUEUE = 2

_SENTINEL = object()
_SKIP = object()


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _ensure_datasets(h5_file) -> None:
    import h5py

    if "face_features" in h5_file and "clip_ids" not in h5_file:
        raise ValueError(
            "Output HDF5 uses the old FaceFeatureV1 schema (no clip_ids). "
            "Choose a new output path or pass --overwrite."
        )
    existing_version = h5_file.attrs.get("schema_version")
    if existing_version is not None and str(existing_version) != SCHEMA_VERSION:
        raise ValueError(
            f"Output HDF5 schema is {existing_version!r}, expected {SCHEMA_VERSION!r}"
        )

    def _float(name: str, dim: int, chunks: int = 1000) -> None:
        if name not in h5_file:
            h5_file.create_dataset(
                name,
                shape=(0, dim),
                maxshape=(None, dim),
                dtype="float32",
                chunks=(chunks, dim),
                compression="lzf",
            )

    def _int(name: str, dtype: str = "int32") -> None:
        if name not in h5_file:
            h5_file.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(10000,),
                compression="lzf",
            )

    def _str(name: str) -> None:
        if name not in h5_file:
            dt = h5py.special_dtype(vlen=str)
            h5_file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt)

    _float("face_features", FACE_DIM)
    _float("arcface", ID_DIM)
    _float("geometry", GEO_DIM)
    _float("demographics", DEMO_DIM)
    _float("speaker_features", SPEAKER_DIM)
    _float("face_quality", QUALITY_DIM)
    _float("face_age", 1)
    _int("shard")
    _int("frame_index")
    _int("face_gender", dtype="int8")
    _str("speaker_ids")
    _str("clip_ids")
    _str("split")
    _str("gender")
    _str("face_meta")

    if "schema_version" not in h5_file.attrs:
        h5_file.attrs["schema_version"] = SCHEMA_VERSION
        h5_file.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        h5_file.attrs["source"] = "VoxCeleb2 dev/test"


def _load_checkpoint(h5_path: str | Path) -> set[str]:
    import h5py

    path = Path(h5_path)
    if not path.exists():
        return set()
    with h5py.File(path, "r") as f:
        if "clip_ids" not in f:
            if "face_features" in f:
                raise ValueError(
                    f"{path} uses the old FaceFeatureV1 schema; use a new output path."
                )
            return set()
        return {
            (m.decode() if isinstance(m, bytes) else str(m))
            for m in f["clip_ids"][:]
        }


def _resize_and_set(h5_file, name: str, values: np.ndarray) -> None:
    ds = h5_file[name]
    cur = ds.shape[0]
    n = len(values)
    ds.resize(cur + n, axis=0)
    ds[cur:] = values


def _flush_batch(
    h5_file,
    face_items: list[dict],
) -> None:
    """Append a batch of fully processed clips to the HDF5."""
    if not face_items:
        return

    n = len(face_items)
    face_features = np.stack([x["e_face"] for x in face_items]).astype(np.float32)
    arcface = np.stack([x["e_id"] for x in face_items]).astype(np.float32)
    geometry = np.stack([x["e_geo"] for x in face_items]).astype(np.float32)
    demographics = np.stack([x["e_demo"] for x in face_items]).astype(np.float32)
    speaker_features = np.stack([x["e_spk"] for x in face_items]).astype(np.float32)
    quality = np.stack([x["quality"] for x in face_items]).astype(np.float32)
    age = np.array([x["face_age"] for x in face_items], dtype=np.float32)
    shard = np.array([x["shard"] for x in face_items], dtype=np.int32)
    frame_index = np.array([x["frame_index"] for x in face_items], dtype=np.int32)
    face_gender = np.array([x["face_gender"] for x in face_items], dtype=np.int8)

    _resize_and_set(h5_file, "face_features", face_features)
    _resize_and_set(h5_file, "arcface", arcface)
    _resize_and_set(h5_file, "geometry", geometry)
    _resize_and_set(h5_file, "demographics", demographics)
    _resize_and_set(h5_file, "speaker_features", speaker_features)
    _resize_and_set(h5_file, "face_quality", quality)
    _resize_and_set(h5_file, "face_age", age.reshape(-1, 1))
    _resize_and_set(h5_file, "shard", shard)
    _resize_and_set(h5_file, "frame_index", frame_index)
    _resize_and_set(h5_file, "face_gender", face_gender)

    for name, key in (
        ("speaker_ids", "speaker_id"),
        ("clip_ids", "clip_id"),
        ("split", "split"),
        ("gender", "gender"),
        ("face_meta", "face_meta"),
    ):
        ds = h5_file[name]
        cur = ds.shape[0]
        ds.resize(cur + n, axis=0)
        for j, item in enumerate(face_items):
            ds[cur + j] = item[key]

    h5_file.flush()


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------

def _video_path(row: dict, videos_root: Path) -> Path:
    explicit = row.get("video_path") or row.get("path")
    if explicit:
        p = Path(str(explicit))
        return p if p.is_absolute() else videos_root / p
    return (
        videos_root
        / str(row["speaker_id"])
        / str(row["video_id"])
        / f"{row['utterance_id']}.mp4"
    )


def _queue_put(
    out_q: queue.Queue,
    item,
    stop_event: threading.Event,
    *,
    timeout: float = 0.5,
) -> bool:
    """Put an item, giving the main loop a chance to stop cleanly."""
    while not stop_event.is_set():
        try:
            out_q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def _prefetch_worker(
    rows: list[dict],
    videos_root: Path,
    done_keys: set[str],
    out_q: queue.Queue,
    stop_event: threading.Event,
    *,
    max_frames: int,
    max_seconds: float,
) -> None:
    for row in rows:
        if stop_event.is_set():
            break
        clip_id = str(row["clip_id"])
        if clip_id in done_keys:
            if not _queue_put(out_q, (_SKIP, row, "already_done"), stop_event):
                break
            continue
        path = _video_path(row, videos_root)
        if not path.exists():
            if not _queue_put(out_q, (_SKIP, row, f"missing_video:{path}"), stop_event):
                break
            continue
        try:
            frames, times, vmeta = load_sampled_frames(
                path, max_frames=max_frames, max_seconds=max_seconds
            )
        except Exception as exc:
            if not _queue_put(
                out_q, (_SKIP, row, f"video_read_failed:{exc}"), stop_event
            ):
                break
            continue
        if not frames:
            if not _queue_put(out_q, (_SKIP, row, "no_frames"), stop_event):
                break
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()
        wav_handed_off = False
        try:
            ok = extract_audio(path, wav_path, max_seconds=max_seconds)
            if not ok:
                detail = last_audio_error()
                reason = "audio_extract_failed"
                if detail:
                    reason = f"{reason}:{detail[:180]}"
                if not _queue_put(out_q, (_SKIP, row, reason), stop_event):
                    break
                continue
            duration = audio_duration_sec(wav_path)
            if duration < MIN_AUDIO_SEC:
                if not _queue_put(
                    out_q,
                    (_SKIP, row, f"audio_too_short:{duration:.2f}"),
                    stop_event,
                ):
                    break
                continue
            audio_rms = audio_rms_envelope(wav_path, times)
            item = (row, frames, times, audio_rms, vmeta, wav_path)
            if not _queue_put(out_q, item, stop_event):
                break
            wav_handed_off = True
        except Exception as exc:
            if not _queue_put(out_q, (_SKIP, row, f"audio_failed:{exc}"), stop_event):
                break
        finally:
            # Delete the WAV immediately on every path that did not hand it to
            # the main loop.  The old version only cleaned up when the whole
            # session was stopping, leaking one WAV per skipped clip.
            if not wav_handed_off and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    # Always deliver the sentinel.  The old code dropped it when the queue was
    # full, leaving the main loop waiting forever after the last clip.
    while not stop_event.is_set():
        try:
            out_q.put(_SENTINEL, timeout=0.5)
            break
        except queue.Full:
            continue


# ---------------------------------------------------------------------------
# Per-clip face processing
# ---------------------------------------------------------------------------

def _crop_face(frame: np.ndarray, bbox: np.ndarray, margin: float = 0.20) -> np.ndarray:
    return crop_face_bbox(frame, bbox, margin=margin)


def _compute_geometry(
    frame: np.ndarray,
    selected: SelectedFace,
    mesh_ext: FaceMeshExtractor,
) -> tuple[np.ndarray, bool, str]:
    import cv2

    crop = _crop_face(frame, selected.bbox)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    lm = mesh_ext.extract(rgb)
    source = "crop"
    if lm is None:
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = mesh_ext.extract(rgb_full)
        source = "full_frame"
    if lm is None:
        return np.zeros(GEO_DIM, dtype=np.float32), False, "none"
    return compute_craniofacial_ratios(lm), True, source


def _safe_clip_name(clip_id: str) -> str:
    return str(clip_id).replace("/", "_").replace("\\", "_")


def _save_face_visuals(
    row: dict,
    selected: SelectedFace,
    frame: np.ndarray,
    *,
    crops_dir: Optional[Path],
    frames_dir: Optional[Path],
) -> tuple[Optional[str], Optional[str]]:
    """Save a face crop and a bbox-annotated frame for pilot inspection."""
    import cv2

    stem = _safe_clip_name(str(row["clip_id"]))
    crop_path: Optional[Path] = None
    frame_path: Optional[Path] = None
    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crops_dir / f"{stem}.jpg"
        crop = _crop_face(frame, selected.bbox, margin=0.35)
        cv2.imwrite(str(crop_path), crop)
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / f"{stem}.jpg"
        annotated = frame.copy()
        x1, y1, x2, y2 = [int(round(float(v))) for v in selected.bbox[:4]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{row.get('speaker_id', '')} track={selected.track_id} "
            f"av={selected.av_sync:.2f}",
            (max(0, x1), max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(frame_path), annotated)
    return (
        str(crop_path) if crop_path is not None else None,
        str(frame_path) if frame_path is not None else None,
    )


def _build_face_item(
    row: dict,
    selected: SelectedFace,
    frames: list[np.ndarray],
    vmeta: dict,
    *,
    mesh_ext: FaceMeshExtractor,
    demo_est,
    fusion_builder: FaceFeatureBuilder,
    crops_dir: Optional[Path] = None,
    frames_dir: Optional[Path] = None,
) -> dict:
    selected_frame = frames[selected.frame_index]
    e_id = np.asarray(selected.embedding, dtype=np.float32)
    e_geo, geometry_ok, geometry_source = _compute_geometry(
        selected_frame, selected, mesh_ext
    )
    from voix.data.demographics import aspect_ratio_from_bbox

    aspect_ratio = aspect_ratio_from_bbox(selected.bbox)
    if selected.gender is not None and selected.age is not None:
        e_demo = demo_est.estimate_from_attributes(
            gender=int(selected.gender),
            age=float(selected.age),
            aspect_ratio=aspect_ratio,
        )
    else:
        e_demo = demo_est.estimate(e_id)
    e_demo = np.asarray(e_demo, dtype=np.float32)

    fused = fusion_builder.fuse(
        e_id.reshape(1, -1),
        e_geo.reshape(1, -1),
        e_demo.reshape(1, -1),
    )
    e_face = fused[0].detach().cpu().numpy().astype(np.float32)

    quality = np.array(
        [
            selected.n_faces,
            selected.n_tracks,
            selected.track_len,
            selected.av_sync,
            selected.mouth_var,
            selected.mouth_mean,
            selected.face_area,
            selected.frontal,
            selected.centrality,
            selected.det_score,
            selected.score,
            selected.frames_used,
        ],
        dtype=np.float32,
    )
    crop_path, frame_path = _save_face_visuals(
        row,
        selected,
        selected_frame,
        crops_dir=crops_dir,
        frames_dir=frames_dir,
    )
    meta = {
        "track_id": int(selected.track_id),
        "track_len": int(selected.track_len),
        "n_faces": int(selected.n_faces),
        "n_tracks": int(selected.n_tracks),
        "frames_used": int(selected.frames_used),
        "bbox": [float(v) for v in selected.bbox],
        "pose": [float(v) for v in selected.pose] if selected.pose is not None else None,
        "gender": int(selected.gender) if selected.gender is not None else None,
        "age": float(selected.age) if selected.age is not None else None,
        "av_sync": float(selected.av_sync),
        "av_pairs": int(selected.av_pairs),
        "mouth_var": float(selected.mouth_var),
        "face_area": float(selected.face_area),
        "geometry_ok": bool(geometry_ok),
        "geometry_source": geometry_source,
        "video_fps": float(vmeta.get("fps", 0.0)),
        "video_duration": float(vmeta.get("duration_sec", 0.0)),
        "sampled_frames": int(vmeta.get("sampled_frames", len(frames))),
        "crop_path": crop_path,
        "frame_path": frame_path,
    }
    return {
        "speaker_id": str(row["speaker_id"]),
        "clip_id": str(row["clip_id"]),
        "split": str(row.get("split", "")),
        "gender": str(row.get("gender", "")),
        "shard": int(row.get("shard", -1)),
        "e_id": e_id,
        "e_geo": e_geo,
        "e_demo": e_demo,
        "e_face": e_face,
        "quality": quality,
        "face_age": float(selected.age) if selected.age is not None else -1.0,
        "face_gender": int(selected.gender) if selected.gender is not None else -1,
        "frame_index": int(selected.frame_index),
        "face_meta": json.dumps(meta, separators=(",", ":")),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_face_app(device: str, det_size: int = 640):
    from voix.utils.onnx_runtime import prefer_gpu_onnxruntime

    available = set(prefer_gpu_onnxruntime())
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    available = set(ort.get_available_providers())
    use_cuda = device.startswith("cuda") and "CUDAExecutionProvider" in available
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    ctx_id = 0 if use_cuda else -1
    model_root = str(ROOT / "checkpoints")
    size = (int(det_size), int(det_size))
    print(f"ONNX providers requested: {providers} (available: {sorted(available)})")
    print(f"InsightFace root: {model_root} | det_size: {size}")
    try:
        app = FaceAnalysis(name="buffalo_l", root=model_root, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=size)
    except Exception as exc:
        if use_cuda:
            print(f"[WARN] CUDA ONNX provider failed ({exc}); falling back to CPU.")
            app = FaceAnalysis(
                name="buffalo_l",
                root=model_root,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=size)
        else:
            raise
    return app


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def run_extraction(
    index_path: str | Path,
    videos_root: str | Path,
    output_path: str | Path,
    *,
    device: str = "cuda",
    num_frames: int = 16,
    recognition_frames: int = 4,
    max_seconds: float = MAX_AUDIO_SEC,
    batch_size: int = 200,
    ecapa_batch: int = N_ECAPA_BATCH,
    max_speakers: Optional[int] = None,
    max_clips: Optional[int] = None,
    shard: Optional[int] = None,
    failures_path: str | Path = "data/processed/extraction_failures_v2.csv",
    crops_dir: Optional[str | Path] = None,
    frames_dir: Optional[str | Path] = None,
    det_size: int = 640,
    max_faces: int = 5,
    min_face_area: float = 0.001,
    max_gap_frames: int = 2,
    overwrite: bool = False,
    session_max_hours: float = 24.0,
) -> None:
    import h5py

    output_path = Path(output_path)
    videos_root = Path(videos_root)
    index_path = Path(index_path)
    failures_path = Path(failures_path)
    crops_path = Path(crops_dir) if crops_dir else None
    frames_path = Path(frames_dir) if frames_dir else None

    if overwrite and output_path.exists():
        output_path.unlink()
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Index is empty: {index_path}")

    if shard is not None:
        rows = [r for r in rows if int(r.get("shard", -1)) == int(shard)]
    if max_speakers is not None:
        keep = set(sorted({str(r["speaker_id"]) for r in rows})[: int(max_speakers)])
        rows = [r for r in rows if str(r["speaker_id"]) in keep]
    if max_clips is not None:
        rows = rows[: int(max_clips)]
    if not rows:
        raise ValueError("No rows left after filtering")

    print("=" * 72)
    print("VoxCeleb2 extraction — FaceFeatureV2")
    print("=" * 72)
    print(f"Index: {index_path} ({len(rows):,} clips, "
          f"{len(set(r['speaker_id'] for r in rows)):,} speakers)")
    print(f"Videos: {videos_root}")
    print(f"Output: {output_path}")

    done_keys = _load_checkpoint(output_path)
    pending = [r for r in rows if str(r["clip_id"]) not in done_keys]
    print(f"Already done: {len(done_keys):,} | Pending: {len(pending):,}")
    if not pending:
        print("Nothing to do.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failure_mode = (
        "a"
        if failures_path.exists() and failures_path.stat().st_size > 0
        else "w"
    )
    failure_file = open(failures_path, failure_mode, newline="", encoding="utf-8")
    failure_writer = csv.DictWriter(
        failure_file, fieldnames=["clip_id", "speaker_id", "reason", "timestamp"]
    )
    if failure_mode == "w":
        failure_writer.writeheader()

    print("Loading models...")
    app = _load_face_app(device, det_size=det_size)
    mesh_ext = FaceMeshExtractor(num_faces=1)
    spk_ext = ECAPAExtractor(device=device)
    from voix.data.demographics import SoftDemographicEstimator

    demo_est = SoftDemographicEstimator(device=device)
    fusion_builder = FaceFeatureBuilder(device=device)
    print("Models loaded.")

    h5_file = h5py.File(output_path, "a")
    _ensure_datasets(h5_file)
    h5_file.attrs["extraction_config"] = json.dumps(
        {
            "num_frames": int(num_frames),
            "recognition_frames": int(recognition_frames),
            "max_seconds": float(max_seconds),
            "device": device,
            "det_size": int(det_size),
            "max_faces": int(max_faces),
            "min_face_area": float(min_face_area),
            "max_gap_frames": int(max_gap_frames),
            "schema": SCHEMA_VERSION,
        },
        separators=(",", ":"),
    )

    stop_event = threading.Event()
    prefetch_q: queue.Queue = queue.Queue(maxsize=PREFETCH_QUEUE)
    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(pending, videos_root, done_keys, prefetch_q, stop_event),
        kwargs={"max_frames": num_frames, "max_seconds": max_seconds},
        daemon=True,
    )
    prefetch_thread.start()

    face_micro: list[dict] = []
    wav_micro: list[str] = []
    buf: list[dict] = []

    n_success = 0
    n_failed = 0
    t_start = time.time()
    t_last = time.time()

    def _flush_micro() -> None:
        nonlocal n_success, n_failed
        if not face_micro:
            return
        try:
            spk_vecs = spk_ext.extract_batch_from_paths(
                wav_micro, max_seconds=max_seconds
            )
        except Exception as exc:
            print(f"  [WARN] ECAPA batch failed: {exc}")
            spk_vecs = [None] * len(face_micro)

        for item, spk in zip(face_micro, spk_vecs):
            if spk is None:
                n_failed += 1
                failure_writer.writerow(
                    {
                        "clip_id": item["clip_id"],
                        "speaker_id": item["speaker_id"],
                        "reason": "ecapa_failed",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue
            item["e_spk"] = np.asarray(spk, dtype=np.float32)
            buf.append(item)
            n_success += 1

        for wav in wav_micro:
            try:
                os.unlink(wav)
            except OSError:
                pass
        face_micro.clear()
        wav_micro.clear()
        failure_file.flush()

    def _flush_hdf5() -> None:
        nonlocal buf
        if not buf:
            return
        _flush_batch(h5_file, buf)
        buf = []

    def _drain_queue() -> None:
        """Discard unprocessed prefetch items and delete their WAVs."""
        while True:
            try:
                queued = prefetch_q.get_nowait()
            except queue.Empty:
                return
            if queued is _SENTINEL or queued[0] is _SKIP:
                continue
            queued_wav = queued[-1]
            try:
                os.unlink(queued_wav)
            except OSError:
                pass

    try:
        while True:
            elapsed_h = (time.time() - t_start) / 3600.0
            if elapsed_h >= session_max_hours:
                print(f"[SESSION LIMIT] {elapsed_h:.2f}h >= {session_max_hours}h")
                stop_event.set()
                break

            try:
                item = prefetch_q.get(timeout=60)
            except queue.Empty:
                if not prefetch_thread.is_alive():
                    print(
                        "  [WARN] prefetch worker exited without a sentinel; "
                        "stopping"
                    )
                    break
                print("  [WARN] prefetch timeout")
                continue

            if item is _SENTINEL:
                break
            if item[0] is _SKIP:
                _, row, reason = item
                n_failed += 1
                failure_writer.writerow(
                    {
                        "clip_id": row.get("clip_id", ""),
                        "speaker_id": row.get("speaker_id", ""),
                        "reason": reason,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue

            row, frames, times, audio_rms, vmeta, wav_path = item
            try:
                selected = select_speaking_face(
                    frames,
                    times,
                    audio_rms,
                    app,
                    max_frames=num_frames,
                    recognition_frames=recognition_frames,
                    max_faces=max_faces,
                    min_face_area_ratio=min_face_area,
                    max_gap_frames=max_gap_frames,
                )
                if selected is None:
                    n_failed += 1
                    failure_writer.writerow(
                        {
                            "clip_id": row.get("clip_id", ""),
                            "speaker_id": row.get("speaker_id", ""),
                            "reason": "no_speaking_face",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
                    continue
                face_item = _build_face_item(
                    row,
                    selected,
                    frames,
                    vmeta,
                    mesh_ext=mesh_ext,
                    demo_est=demo_est,
                    fusion_builder=fusion_builder,
                    crops_dir=crops_path,
                    frames_dir=frames_path,
                )
            except Exception as exc:
                n_failed += 1
                failure_writer.writerow(
                    {
                        "clip_id": row.get("clip_id", ""),
                        "speaker_id": row.get("speaker_id", ""),
                        "reason": f"face_failed:{exc}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
                continue

            face_micro.append(face_item)
            wav_micro.append(wav_path)
            if len(face_micro) >= max(1, int(ecapa_batch)):
                _flush_micro()
            if len(buf) >= max(1, int(batch_size)):
                _flush_hdf5()
                now = time.time()
                rate = n_success / max(now - t_start, 1.0)
                remaining = len(pending) - n_success - n_failed
                eta_h = (remaining / rate) / 3600.0 if rate > 0 else 0.0
                print(
                    f"  done={n_success:,} failed={n_failed:,} "
                    f"rate={rate:.2f}/s elapsed={(now - t_start)/3600:.2f}h ETA={eta_h:.2f}h"
                )
                t_last = now

    finally:
        stop_event.set()
        try:
            if face_micro:
                _flush_micro()
            if buf:
                _flush_hdf5()
        finally:
            # Let the prefetch thread notice the stop flag, then drain any
            # queued clips so their temporary WAVs do not leak.
            for _ in range(20):
                _drain_queue()
                prefetch_thread.join(timeout=0.25)
                if not prefetch_thread.is_alive():
                    break
            _drain_queue()
            try:
                h5_file.close()
            finally:
                failure_file.close()

    total_elapsed = time.time() - t_start
    print("=" * 72)
    print(f"Done: {n_success:,} new + {len(done_keys):,} existing = {n_success + len(done_keys):,}")
    print(f"Failed/skipped: {n_failed:,}")
    print(f"Elapsed: {total_elapsed/3600:.2f}h | Output: {output_path}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VoxCeleb2 FaceFeatureV2 extraction")
    p.add_argument("--index", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--output", default="data/processed/features_v2.h5")
    p.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--recognition-frames", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=MAX_AUDIO_SEC)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--ecapa-batch", type=int, default=N_ECAPA_BATCH)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--failures", default="data/processed/extraction_failures_v2.csv")
    p.add_argument("--crops-dir", default=None)
    p.add_argument("--frames-dir", default=None)
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--max-faces", type=int, default=5)
    p.add_argument("--min-face-area", type=float, default=0.001)
    p.add_argument("--max-gap-frames", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--session-max-hours", type=float, default=24.0)
    return p.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    args = parse_args()
    run_extraction(
        index_path=args.index,
        videos_root=args.videos,
        output_path=args.output,
        device=args.device,
        num_frames=args.num_frames,
        recognition_frames=args.recognition_frames,
        max_seconds=args.max_seconds,
        batch_size=args.batch_size,
        ecapa_batch=args.ecapa_batch,
        max_speakers=args.max_speakers,
        max_clips=args.max_clips,
        shard=args.shard,
        failures_path=args.failures,
        crops_dir=args.crops_dir,
        frames_dir=args.frames_dir,
        det_size=args.det_size,
        max_faces=args.max_faces,
        min_face_area=args.min_face_area,
        max_gap_frames=args.max_gap_frames,
        overwrite=args.overwrite,
        session_max_hours=args.session_max_hours,
    )
