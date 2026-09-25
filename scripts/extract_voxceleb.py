"""VoxCeleb2 batch feature extraction pipeline (T2.5 & T2.6).

Reads curated_index.csv, downloads clips in streaming batches,
extracts face (560-D) and speaker (192-D) embeddings, and serialises
them to data/processed/voxceleb2_train.h5.

Checkpoint recovery: progress is saved after every batch so the script
can be interrupted and resumed without re-extracting completed batches.

Usage (local PC):
    python scripts/extract_voxceleb.py \
        --index  data/processed/curated_index.csv \
        --videos /path/to/voxceleb2/dev/mp4/ \
        --output data/processed/voxceleb2_train.h5 \
        --batch  500 \
        --device cuda

Usage (Colab with Google Drive):
    python scripts/extract_voxceleb.py \
        --index  /content/drive/MyDrive/voix/curated_index.csv \
        --videos /content/vox2/dev/mp4/ \
        --output /content/drive/MyDrive/voix/voxceleb2_train.h5 \
        --batch  500

Output HDF5 structure:
    /face_features    (N, 560) float32  -- unified face embedding
    /speaker_features (N, 192) float32  -- ECAPA speaker embedding
    /speaker_ids      (N,)     int32    -- integer speaker label
    /metadata         (N,)     string   -- "spk_id/vid_id/utt_id"
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_DIM:    int = 560
SPEAKER_DIM: int = 192

# Minimum audio duration to accept a clip (seconds)
MIN_AUDIO_SEC: float = 2.5
MAX_AUDIO_SEC: float = 12.0

# Frontal-frame sampling: extract every Nth frame for pose scoring
FRAME_SAMPLE_STEP: int = 5

# Target face crop size for ArcFace
FACE_CROP_SIZE: int = 112


def _ffmpeg_extract_audio(video_path: str, wav_path: str) -> bool:
    """Extract 16 kHz mono WAV from a video file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(wav_path)


def _load_video_frames(video_path: str, step: int = 5) -> list[np.ndarray]:
    """Load sampled BGR frames from a video file using OpenCV.

    Returns every `step`-th frame as a list of HxWx3 uint8 arrays.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def extract_sample(
    video_path: str,
    face_extractor,
    mesh_extractor,
    morph_fn,
    demo_estimator,
    fusion_layer,
    speaker_extractor,
    pose_estimator,
    device: str = "cuda",
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Extract (560-D face, 192-D speaker) embeddings from one video clip.

    Pipeline per clip:
        1. Load sampled frames
        2. Run FaceMesh on each frame -> landmarks list
        3. Select best frontal frame (HeadPoseEstimator)
        4. Extract ArcFace embedding from best frame
        5. Compute craniofacial ratios from best frame landmarks
        6. Get soft demographic prior
        7. Fuse into 560-D vector (numpy, no grad)
        8. Extract 16 kHz audio via ffmpeg
        9. Extract ECAPA speaker embedding

    Returns:
        Tuple of (face_560d, speaker_192d) float32 arrays, or None on failure.
    """
    import cv2
    import torch

    # --- Step 1: Load video frames ----------------------------------------
    frames = _load_video_frames(video_path, step=FRAME_SAMPLE_STEP)
    if not frames:
        return None

    # --- Step 2: FaceMesh on each frame (RGB) --------------------------------
    landmarks_per_frame: list = []
    h, w = frames[0].shape[:2]
    for frame_bgr in frames:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lm = mesh_extractor.extract(frame_rgb)
        landmarks_per_frame.append(lm)

    # --- Step 3: Select best frontal frame ------------------------------------
    best_idx = pose_estimator.select_best_frame(landmarks_per_frame, w, h)
    if best_idx is None:
        # Fallback: use middle frame
        best_idx = len(frames) // 2

    best_frame_bgr = frames[best_idx]
    best_landmarks = landmarks_per_frame[best_idx]

    # --- Step 4: ArcFace identity embedding -----------------------------------
    e_id = face_extractor.extract(best_frame_bgr)
    if e_id is None:
        return None  # No face detected

    # --- Step 5: Craniofacial ratios ------------------------------------------
    if best_landmarks is None:
        # Try to get landmarks for the best frame explicitly
        best_frame_rgb = cv2.cvtColor(best_frame_bgr, cv2.COLOR_BGR2RGB)
        best_landmarks = mesh_extractor.extract(best_frame_rgb)
    if best_landmarks is None:
        return None

    e_geo = morph_fn(best_landmarks)  # (32,) float32

    # --- Step 6: Soft demographics --------------------------------------------
    e_demo = demo_estimator.estimate(e_id)  # (16,) float32

    # --- Step 7: Fuse face features (no gradient) ----------------------------
    with torch.no_grad():
        e_id_t   = torch.from_numpy(e_id).float().unsqueeze(0)
        e_geo_t  = torch.from_numpy(e_geo).float().unsqueeze(0)
        e_demo_t = torch.from_numpy(e_demo).float().unsqueeze(0)
        e_face   = fusion_layer(e_id_t, e_geo_t, e_demo_t)  # (1, 560)
        e_face_np = e_face.squeeze(0).cpu().numpy().astype(np.float32)

    # --- Step 8 & 9: Audio extraction + ECAPA --------------------------------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        ok = _ffmpeg_extract_audio(video_path, wav_path)
        if not ok:
            return None

        # Check audio duration
        import soundfile as sf
        info = sf.info(wav_path)
        if info.duration < MIN_AUDIO_SEC or info.duration > MAX_AUDIO_SEC:
            return None

        e_spk = speaker_extractor.extract_from_path(wav_path)  # (1, 192)
        e_spk_np = e_spk.squeeze(0).cpu().numpy().astype(np.float32)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    assert e_face_np.shape == (FACE_DIM,),    f"Face dim {e_face_np.shape}"
    assert e_spk_np.shape  == (SPEAKER_DIM,), f"Speaker dim {e_spk_np.shape}"

    return e_face_np, e_spk_np


def _load_checkpoint(h5_path: str) -> set[str]:
    """Return set of 'spk/vid/utt' keys already written to HDF5."""
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            if "metadata" not in f:
                return set()
            return set(m.decode() for m in f["metadata"][:])
    except Exception:
        return set()


def run_extraction(
    index_path: str,
    videos_root: str,
    output_path: str,
    device: str = "cuda",
    batch_size: int = 200,
    session_max_hours: float = 11.0,
    progress_callback=None,
) -> None:
    """Main extraction loop with checkpoint recovery and session-aware stopping.

    Args:
        index_path:         Path to curated_index.csv.
        videos_root:        Root directory containing VoxCeleb2 mp4 files.
                            Expected structure: videos_root/spk_id/vid_id/utt_id.mp4
        output_path:        Output HDF5 file path (can be a Google Drive path).
        device:             PyTorch device ('cuda' or 'cpu').
        batch_size:         Clips per batch before flushing to HDF5.
                            Default 200 (T4: ~12 min per checkpoint).
        session_max_hours:  Stop extraction this many hours after function call.
                            Allows safe session timeout management in Colab.
                            Default 11.0 h (30 min safety margin for 12h sessions).
        progress_callback:  Optional callable(n_done: int, last_key: str).
                            Called after every batch flush. Use to update a
                            progress.json file on Google Drive.
    """
    import h5py
    import torch
    from voix.data.face_extractor import ArcFaceExtractor, FaceMeshExtractor
    from voix.data.speaker_extractor import ECAPAExtractor
    from voix.data.demographics import SoftDemographicEstimator
    from voix.data.head_pose import HeadPoseEstimator
    from voix.data.morphology import compute_craniofacial_ratios
    from voix.models.fusion import FaceFusionLayer

    # --- Load models (sequential to stay within 4GB VRAM) -------------------
    print("Loading ArcFace extractor...")
    face_ext = ArcFaceExtractor(device_id=0 if device == "cuda" else -1)

    print("Loading FaceMesh extractor...")
    mesh_ext = FaceMeshExtractor()

    print("Loading ECAPA extractor...")
    spk_ext = ECAPAExtractor(device=device)

    demo_est   = SoftDemographicEstimator(device=device)
    pose_est   = HeadPoseEstimator(max_yaw=25.0, max_pitch=25.0)
    fusion_lay = FaceFusionLayer()
    fusion_lay.eval()
    fusion_lay.freeze()

    # --- Read index ----------------------------------------------------------
    print(f"Reading index from {index_path}...")
    rows: list[dict] = []
    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} entries in index.")

    # --- Build speaker_id -> int mapping -------------------------------------
    speaker_ids = sorted(set(r["speaker_id"] for r in rows))
    spk_to_int  = {s: i for i, s in enumerate(speaker_ids)}

    # --- Checkpoint recovery -------------------------------------------------
    done_keys = _load_checkpoint(output_path)
    print(f"  {len(done_keys)} entries already extracted (checkpoint).")

    pending = [
        r for r in rows
        if f"{r['speaker_id']}/{r['video_id']}/{r['utterance_id']}" not in done_keys
    ]
    print(f"  {len(pending)} entries remaining.")

    if not pending:
        print("All entries extracted. Nothing to do.")
        return

    # --- Open / create HDF5 in append mode -----------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(output_path, "a")

    # Create resizable datasets if they don't exist
    N_existing = len(done_keys)
    if "face_features" not in h5_file:
        h5_file.create_dataset(
            "face_features", shape=(0, FACE_DIM), maxshape=(None, FACE_DIM),
            dtype="float32", chunks=(1000, FACE_DIM), compression="gzip",
            compression_opts=4,
        )
        h5_file.create_dataset(
            "speaker_features", shape=(0, SPEAKER_DIM), maxshape=(None, SPEAKER_DIM),
            dtype="float32", chunks=(1000, SPEAKER_DIM), compression="gzip",
            compression_opts=4,
        )
        h5_file.create_dataset(
            "speaker_ids", shape=(0,), maxshape=(None,),
            dtype="int32", chunks=(10000,), compression="gzip",
        )
        dt = h5py.special_dtype(vlen=str)
        h5_file.create_dataset(
            "metadata", shape=(0,), maxshape=(None,), dtype=dt,
        )

    videos_root_path = Path(videos_root)

    # --- Main extraction loop ------------------------------------------------
    batch_face:    list[np.ndarray] = []
    batch_speaker: list[np.ndarray] = []
    batch_spk_ids: list[int]        = []
    batch_meta:    list[str]        = []

    n_success = 0
    n_failed  = 0
    t_start   = time.time()
    t_keepalive = time.time()
    KEEPALIVE_INTERVAL = 300  # print status every 5 min to prevent idle disconnect

    for i, row in enumerate(pending):
        # Session time limit check (T4 Colab session safety)
        elapsed_h = (time.time() - t_start) / 3600
        if elapsed_h >= session_max_hours:
            print(f"\n[SESSION LIMIT] {elapsed_h:.1f}h elapsed >= {session_max_hours}h limit.")
            print(f"  Stopping safely. Re-run Cell 6 in a new session to continue.")
            break

        # Keepalive print (prevents Colab idle disconnect)
        if time.time() - t_keepalive > KEEPALIVE_INTERVAL:
            total_done = n_success + N_existing
            print(f"  [keepalive] {elapsed_h:.1f}h | {total_done:,} done | "
                  f"{n_success/elapsed_h:.0f} clips/h")
            t_keepalive = time.time()
        spk_id = row["speaker_id"]
        vid_id = row["video_id"]
        utt_id = row["utterance_id"]

        # Locate video file
        video_path = videos_root_path / spk_id / vid_id / f"{utt_id}.mp4"
        if not video_path.exists():
            n_failed += 1
            continue

        try:
            result = extract_sample(
                str(video_path),
                face_extractor=face_ext,
                mesh_extractor=mesh_ext,
                morph_fn=compute_craniofacial_ratios,
                demo_estimator=demo_est,
                fusion_layer=fusion_lay,
                speaker_extractor=spk_ext,
                pose_estimator=pose_est,
                device=device,
            )
        except Exception as exc:
            print(f"  [WARN] {spk_id}/{vid_id}/{utt_id}: {exc}")
            n_failed += 1
            continue

        if result is None:
            n_failed += 1
            continue

        e_face, e_spk = result
        batch_face.append(e_face)
        batch_speaker.append(e_spk)
        batch_spk_ids.append(spk_to_int[spk_id])
        batch_meta.append(f"{spk_id}/{vid_id}/{utt_id}")
        n_success += 1

        # Flush batch to HDF5 (if output is on Drive, this writes directly to Drive)
        if len(batch_face) >= batch_size:
            _flush_batch(h5_file, batch_face, batch_speaker, batch_spk_ids, batch_meta)
            batch_face.clear()
            batch_speaker.clear()
            batch_spk_ids.clear()
            batch_meta.clear()

            elapsed = time.time() - t_start
            rate = n_success / max(elapsed, 1)
            total_done = n_success + N_existing
            remaining = len(pending) - i - 1
            eta_h = (remaining / rate) / 3600 if rate > 0 else 0
            elapsed_h = elapsed / 3600
            print(
                f"  [{i+1}/{len(pending)}] "
                f"done={total_done:,} failed={n_failed} "
                f"rate={rate:.1f}/s elapsed={elapsed_h:.1f}h ETA={eta_h:.1f}h"
            )

            # Call progress callback (e.g. write progress.json to Drive)
            if progress_callback is not None:
                last_k = batch_meta[-1] if batch_meta else None
                try:
                    progress_callback(total_done, last_k)
                except Exception:
                    pass  # Never let callback failure interrupt extraction

    # Flush remaining
    if batch_face:
        _flush_batch(h5_file, batch_face, batch_speaker, batch_spk_ids, batch_meta)

    h5_file.close()

    total = n_success + N_existing
    print(f"\nExtraction complete: {n_success} new + {N_existing} existing = {total} total.")
    print(f"Failed / skipped: {n_failed}")
    print(f"Output: {output_path}")


def _flush_batch(
    h5_file,
    face_list:    list[np.ndarray],
    speaker_list: list[np.ndarray],
    spk_ids_list: list[int],
    meta_list:    list[str],
) -> None:
    """Append a batch of embeddings to the HDF5 file."""
    n = len(face_list)
    if n == 0:
        return

    face_arr = np.stack(face_list,    axis=0)  # (n, 560)
    spk_arr  = np.stack(speaker_list, axis=0)  # (n, 192)
    ids_arr  = np.array(spk_ids_list, dtype=np.int32)

    cur = h5_file["face_features"].shape[0]
    h5_file["face_features"].resize(cur + n, axis=0)
    h5_file["face_features"][cur:] = face_arr

    h5_file["speaker_features"].resize(cur + n, axis=0)
    h5_file["speaker_features"][cur:] = spk_arr

    h5_file["speaker_ids"].resize(cur + n, axis=0)
    h5_file["speaker_ids"][cur:] = ids_arr

    h5_file["metadata"].resize(cur + n, axis=0)
    for j, m in enumerate(meta_list):
        h5_file["metadata"][cur + j] = m

    h5_file.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoxCeleb2 feature extraction pipeline"
    )
    parser.add_argument("--index",   required=True, help="Path to curated_index.csv")
    parser.add_argument("--videos",  required=True, help="Root of VoxCeleb2 mp4 files")
    parser.add_argument("--output",  default="data/processed/voxceleb2_train.h5")
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--batch",   type=int, default=500)
    args = parser.parse_args()

    run_extraction(
        index_path=args.index,
        videos_root=args.videos,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()