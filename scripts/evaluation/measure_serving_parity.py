"""Measure HDF5-to-serving face feature parity on held-out clips.

The corpus stores a four-frame averaged ArcFace embedding plus the fused
FaceFeatureV2 vector.  Serving receives a single image (or a single decoded
frame), so the two representations are not identical.  This script quantifies
the gap on a deterministic, speaker-balanced subset of the official test split
and reports its effect on CFVN face-to-face and face-to-voice retrieval.

Usage:
    python scripts/evaluation/measure_serving_parity.py \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --scaler data/processed/voxceleb2_v2/face_scaler_v2.pt \
        --cfvn results/checkpoints/cfvn_v2_best.pt \
        --split test --limit 240 \
        --output data/processed/voxceleb2_v2/serving_parity_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from voix.data.demographics import (  # noqa: E402
    SoftDemographicEstimator,
    aspect_ratio_from_bbox,
)
from voix.data.face_extractor import FaceMeshExtractor, crop_face_bbox  # noqa: E402
from voix.data.feature_contract import (  # noqa: E402
    FaceFeatureBuilder,
    load_scaler_payload,
)
from voix.data.morphology import compute_craniofacial_ratios  # noqa: E402
from voix.data.speaking_face import (  # noqa: E402
    _insightface_face,
    bbox_iou,
    detect_faces_in_frame,
)
from voix.data.video_io import load_sampled_frames  # noqa: E402
from voix.evaluation.retrieval import (  # noqa: E402
    embedding_health,
    speaker_retrieval,
)
from voix.models.contrastive_net import ContrastiveFaceVoiceNet  # noqa: E402
from voix.training.v2_utils import (  # noqa: E402
    embed_faces,
    normalize_face,
    query_gallery_indices,
)

DEFAULT_FEATURES = ROOT / "data" / "processed" / "voxceleb2_v2" / "features_v2.h5"
DEFAULT_INDEX = ROOT / "data" / "processed" / "voxceleb2_v2" / "index.csv"
DEFAULT_SCALER = ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_v2.pt"
DEFAULT_SCALER_ARCFACE = (
    ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_arcface_v2.pt"
)
DEFAULT_CFVN = ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt"
DEFAULT_CLIPS = ROOT / "data" / "raw" / "voxceleb2" / "clips"


def _decode(values) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ],
        dtype=object,
    )


def select_balanced_indices(
    speakers: np.ndarray,
    limit: int,
    *,
    clips_per_speaker: int = 2,
) -> List[int]:
    """Deterministically take up to ``clips_per_speaker`` clips per speaker.

    Clips are sorted by speaker and clip order, then taken round-robin so the
    subset spans as many speakers as possible before filling the limit.
    """
    groups: Dict[str, List[int]] = {}
    for index, speaker in enumerate(speakers):
        groups.setdefault(str(speaker), []).append(index)
    selected: List[int] = []
    for round_index in range(max(1, clips_per_speaker)):
        for speaker in sorted(groups):
            indices = groups[speaker]
            if round_index < len(indices) and len(selected) < limit:
                selected.append(indices[round_index])
            if len(selected) >= limit:
                return selected
    return selected


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _summary(values: Sequence[float]) -> Dict[str, float]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def resolve_clip_path(clips_root: Path, clip_id: str) -> Optional[Path]:
    """Resolve a manifest clip id to its video file.

    The full corpus keeps the VoxCeleb2 ``dev/`` prefix (``clips/dev/id...``),
    while the 100-speaker pilot downloads stripped it.  Accept both layouts so
    the same parity harness can validate pilot and official corpora.
    """
    direct = clips_root / f"{clip_id}.mp4"
    if direct.exists():
        return direct
    prefix = "dev/"
    if clip_id.startswith(prefix):
        stripped = clips_root / f"{clip_id[len(prefix):]}.mp4"
        if stripped.exists():
            return stripped
    return None


def _load_video_row(
    clip_id: str,
    frame_index: int,
    clips_root: Path,
    *,
    max_frames: int = 16,
    max_seconds: float = 12.0,
) -> Optional[np.ndarray]:
    video_path = resolve_clip_path(clips_root, clip_id)
    if video_path is None:
        return None
    frames, _times, _meta = load_sampled_frames(
        video_path, max_frames=max_frames, max_seconds=max_seconds
    )
    if not frames or frame_index < 0 or frame_index >= len(frames):
        return None
    return frames[frame_index]


def _re_extract_single_frame(
    frame: np.ndarray,
    stored_bbox: np.ndarray,
    frame_index: int,
    *,
    app,
    mesh_ext: FaceMeshExtractor,
    demo_est: SoftDemographicEstimator,
    fusion_builder: FaceFeatureBuilder,
) -> Optional[Dict[str, object]]:
    """Run the serving path on one frame and return its raw modalities."""
    detections = detect_faces_in_frame(
        app,
        frame,
        frame_index,
        0.0,
        max_faces=5,
        min_det_score=0.30,
        min_face_area_ratio=0.001,
    )
    if not detections:
        return None
    chosen = max(detections, key=lambda d: bbox_iou(d.bbox, stored_bbox))
    iou = bbox_iou(chosen.bbox, stored_bbox)

    face = _insightface_face(chosen.bbox, chosen.kps, chosen.det_score)
    recognition = app.models.get("recognition")
    if recognition is None:
        return None
    recognition.get(frame, face)
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
    if embedding is None:
        return None
    e_id = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(e_id))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    e_id = (e_id / norm).astype(np.float32)

    genderage = app.models.get("genderage")
    gender = None
    age = None
    if genderage is not None:
        genderage.get(frame, face)
        if getattr(face, "gender", None) is not None:
            gender = int(face.gender)
        if getattr(face, "age", None) is not None:
            age = float(face.age)

    crop = crop_face_bbox(frame, chosen.bbox, margin=0.20)
    landmarks = mesh_ext.extract(crop[:, :, ::-1].copy())
    geometry_ok = landmarks is not None
    if landmarks is None:
        landmarks = mesh_ext.extract(frame[:, :, ::-1].copy())
        geometry_ok = landmarks is not None
    e_geo = (
        compute_craniofacial_ratios(landmarks)
        if landmarks is not None
        else np.zeros(32, dtype=np.float32)
    )

    if gender is not None and age is not None:
        e_demo = demo_est.estimate_from_attributes(
            gender=gender,
            age=age,
            aspect_ratio=aspect_ratio_from_bbox(chosen.bbox),
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

    return {
        "arcface": e_id,
        "geometry": e_geo,
        "demographics": e_demo,
        "face_features": e_face,
        "bbox_iou": iou,
        "bbox": chosen.bbox.tolist(),
        "geometry_ok": bool(geometry_ok),
        "gender": gender,
        "age": age,
    }


def _retrieval_block(
    face_stored: torch.Tensor,
    face_single: Optional[torch.Tensor],
    voice: torch.Tensor,
    speakers: np.ndarray,
) -> Dict[str, object]:
    query_idx, gallery_idx = query_gallery_indices(speakers)
    if len(query_idx) < 2:
        return {"status": "unavailable", "reason": "need 2+ clips per speaker"}
    query_speakers = speakers[query_idx]
    gallery_speakers = speakers[gallery_idx]
    block: Dict[str, object] = {
        "query_speakers": int(len(set(query_speakers.tolist()))),
        "gallery_clips": int(len(gallery_idx)),
    }
    block["face_stored_to_face_stored"] = speaker_retrieval(
        face_stored[query_idx], query_speakers, face_stored[gallery_idx], gallery_speakers
    )
    if face_single is not None:
        block["face_single_to_face_stored"] = speaker_retrieval(
            face_single[query_idx],
            query_speakers,
            face_stored[gallery_idx],
            gallery_speakers,
        )
    block["face_stored_to_voice"] = speaker_retrieval(
        face_stored[query_idx], query_speakers, voice[gallery_idx], gallery_speakers
    )
    if face_single is not None:
        block["face_single_to_voice"] = speaker_retrieval(
            face_single[query_idx], query_speakers, voice[gallery_idx], gallery_speakers
        )
    return block


def measure(args: argparse.Namespace) -> Dict[str, object]:
    from extract_voxceleb2_v2 import _load_face_app

    device = args.device if (
        args.device != "cuda" or torch.cuda.is_available()
    ) else "cpu"
    with h5py.File(args.features, "r") as store:
        clip_ids = _decode(store["clip_ids"][:])
        speakers_all = _decode(store["speaker_ids"][:])
        splits = _decode(store["split"][:])
        arcface = np.asarray(store["arcface"][:], dtype=np.float32)
        geometry = np.asarray(store["geometry"][:], dtype=np.float32)
        demographics = np.asarray(store["demographics"][:], dtype=np.float32)
        fused = np.asarray(store["face_features"][:], dtype=np.float32)
        voice = np.asarray(store["speaker_features"][:], dtype=np.float32)
        frame_index = np.asarray(store["frame_index"][:], dtype=np.int64)
        face_meta = _decode(store["face_meta"][:])

    mask = splits == args.split
    if not mask.any():
        raise ValueError(f"No clips found for split {args.split!r}")
    indices = np.flatnonzero(mask)
    selected = select_balanced_indices(
        speakers_all[indices], args.limit, clips_per_speaker=args.clips_per_speaker
    )
    selected = indices[np.asarray(selected, dtype=np.int64)]
    speakers = speakers_all[selected]
    print(
        f"Parity subset: {len(selected)} clips / "
        f"{len(set(speakers.tolist()))} speakers from split {args.split!r}"
    )

    app = _load_face_app(device, det_size=args.det_size)
    mesh_ext = FaceMeshExtractor(num_faces=1)
    demo_est = SoftDemographicEstimator(device=device)
    fusion_builder = FaceFeatureBuilder(device=device)

    records: List[Dict[str, object]] = []
    single_arcface = np.zeros((len(selected), 512), dtype=np.float32)
    single_geometry = np.zeros((len(selected), 32), dtype=np.float32)
    single_demographics = np.zeros((len(selected), 16), dtype=np.float32)
    single_fused = np.zeros((len(selected), 560), dtype=np.float32)
    valid = np.zeros(len(selected), dtype=bool)

    for position, row_index in enumerate(selected):
        clip_id = str(clip_ids[row_index])
        meta = json.loads(face_meta[row_index])
        stored_bbox = np.asarray(meta.get("bbox", []), dtype=np.float32)
        if stored_bbox.shape != (4,):
            records.append({"clip_id": clip_id, "status": "no_stored_bbox"})
            continue
        frame = _load_video_row(
            clip_id,
            int(frame_index[row_index]),
            args.clips_root,
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
        )
        if frame is None:
            records.append({"clip_id": clip_id, "status": "frame_unreadable"})
            continue
        result = _re_extract_single_frame(
            frame,
            stored_bbox,
            int(frame_index[row_index]),
            app=app,
            mesh_ext=mesh_ext,
            demo_est=demo_est,
            fusion_builder=fusion_builder,
        )
        if result is None:
            records.append({"clip_id": clip_id, "status": "no_face"})
            continue

        record = {
            "clip_id": clip_id,
            "speaker_id": str(speakers_all[row_index]),
            "status": "ok",
            "bbox_iou": float(result["bbox_iou"]),
            "geometry_ok": bool(result["geometry_ok"]),
            "gender": result["gender"],
            "age": result["age"],
            "arcface_cosine": _cosine(arcface[row_index], result["arcface"]),
            "geometry_cosine": _cosine(geometry[row_index], result["geometry"]),
            "demographics_cosine": _cosine(
                demographics[row_index], result["demographics"]
            ),
            "fused_cosine": _cosine(fused[row_index], result["face_features"]),
        }
        records.append(record)
        valid[position] = True
        single_arcface[position] = result["arcface"]
        single_geometry[position] = result["geometry"]
        single_demographics[position] = result["demographics"]
        single_fused[position] = result["face_features"]

    ok = [r for r in records if r.get("status") == "ok"]
    report: Dict[str, object] = {
        "features": str(args.features),
        "split": args.split,
        "requested": int(len(selected)),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "failure_reasons": {},
        "parity": {
            "bbox_iou": _summary([float(r["bbox_iou"]) for r in ok]),
            "arcface_cosine": _summary(
                [float(r["arcface_cosine"]) for r in ok]
            ),
            "geometry_cosine": _summary(
                [float(r["geometry_cosine"]) for r in ok]
            ),
            "demographics_cosine": _summary(
                [float(r["demographics_cosine"]) for r in ok]
            ),
            "fused_cosine": _summary([float(r["fused_cosine"]) for r in ok]),
        },
        "geometry_ok_rate": (
            float(np.mean([bool(r["geometry_ok"]) for r in ok])) if ok else float("nan")
        ),
        "per_clip": records,
    }
    reasons: Dict[str, int] = {}
    for record in records:
        if record.get("status") != "ok":
            reasons[str(record.get("status"))] = reasons.get(
                str(record.get("status")), 0
            ) + 1
    report["failure_reasons"] = reasons

    if not args.skip_cfvn and ok and args.cfvn.exists():
        cfvn = ContrastiveFaceVoiceNet.from_checkpoint(
            str(args.cfvn), device=device
        ).to(device)
        face_key = "arcface" if cfvn.face_dim == 512 else "face_features"
        stored_raw = arcface if face_key == "arcface" else fused
        single_raw = single_arcface if face_key == "arcface" else single_fused
        scaler_path = Path(args.scaler)
        if face_key == "arcface" and scaler_path.resolve() == Path(
            DEFAULT_SCALER
        ).resolve():
            scaler_path = Path(args.scaler_arcface)
        scaler = load_scaler_payload(
            scaler_path, expected_dim=stored_raw.shape[1], face_key=face_key
        )
        valid_idx = np.flatnonzero(valid)
        face_stored = embed_faces(
            cfvn,
            normalize_face(stored_raw[selected[valid_idx]], scaler),
            device,
        )
        face_single = embed_faces(
            cfvn,
            normalize_face(single_raw[valid_idx], scaler),
            device,
        )
        voice_t = torch.from_numpy(voice[selected[valid_idx]]).float()
        report["cfvn"] = {
            "checkpoint": str(args.cfvn),
            "face_key": face_key,
            "scaler": str(scaler_path),
            "retrieval": _retrieval_block(
                face_stored, face_single, voice_t, speakers[valid_idx]
            ),
            "stored_health": embedding_health(face_stored, speakers[valid_idx]),
            "single_health": embedding_health(face_single, speakers[valid_idx]),
            "stored_single_cosine": _summary(
                F.cosine_similarity(face_stored, face_single, dim=-1).tolist()
            ),
        }
    elif not args.skip_cfvn:
        report["cfvn"] = {"status": "unavailable", "reason": "checkpoint missing"}

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure HDF5/serving face parity")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--scaler-arcface", type=Path, default=DEFAULT_SCALER_ARCFACE)
    parser.add_argument("--cfvn", type=Path, default=DEFAULT_CFVN)
    parser.add_argument("--clips-root", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--clips-per-speaker", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cfvn", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = measure(args)
    text = json.dumps(report, indent=2)
    output = args.output or (
        ROOT / "data" / "processed" / "voxceleb2_v2" / "serving_parity_test.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
