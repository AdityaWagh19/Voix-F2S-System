"""VOIX Evaluation Metrics Suite.

Provides automated quality and consistency scoring for synthesized voices:
  - utmos_score:              Neural MOS prediction (UTMOS 2022)
  - dnsmos_score:             DNS challenge noise+signal MOS
  - speaker_similarity:       ECAPA cosine similarity between two utterances
  - face_voice_consistency:   ContrastiveFaceVoiceNet similarity (face <-> voice)
  - prosody_statistics:       F0 statistics for naturalness analysis
  - evaluate_synthesis:       All-in-one quality report for a synthesis result

All functions degrade gracefully when optional dependencies are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from voix.utils.paths import project_path

logger = logging.getLogger("voix.evaluation")

# Canonical evaluation artifacts.  Every path is project-relative so a clean
# checkout evaluates the same models it trained instead of a workstation path.
DEFAULT_CFVN = project_path("results", "checkpoints", "cfvn_v2_best.pt")
DEFAULT_SCALER = project_path(
    "data", "processed", "voxceleb2_v2", "face_scaler_v2.pt"
)
# The raw-ArcFace diagnostic CFVN consumes 512-D features and therefore needs
# the separately fit ArcFace scaler.  Using the canonical 560-D scaler here
# would fail dimension checks (or, worse, silently mis-normalize if the
# dimension happened to match).
DEFAULT_SCALER_ARCFACE = project_path(
    "data", "processed", "voxceleb2_v2", "face_scaler_arcface_v2.pt"
)
DEFAULT_CHECKPOINT_DIR = project_path("checkpoints")
DEFAULT_ECAPA_SAVEDIR = project_path("checkpoints", "speechbrain", "spkrec-ecapa")


# ---------------------------------------------------------------------------
# Audio loading helper
# ---------------------------------------------------------------------------

def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float32 waveform, with dependency-free fallback.

    Previously a missing ``resampy`` silently skipped resampling while the
    caller still assumed ``target_sr``.  That mislabelled every 24 kHz
    synthesis sample as 16 kHz and corrupted both F0 and ECAPA features.  The
    fallbacks below guarantee the returned array really is at ``target_sr``.
    """
    if sr == target_sr:
        return wav.astype(np.float32, copy=False)
    try:
        import resampy
        return resampy.resample(wav, sr, target_sr).astype(np.float32)
    except ImportError:
        pass
    try:
        import torchaudio
        tensor = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, sr, target_sr)
        return tensor.squeeze(0).numpy().astype(np.float32)
    except Exception:
        pass
    try:
        import librosa
        return librosa.resample(
            wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq"
        ).astype(np.float32)
    except Exception:
        pass
    # Last resort: linear interpolation.  Lower quality than polyphase, but it
    # preserves the correct duration/sample-rate contract instead of lying.
    duration = len(wav) / float(sr)
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _load_wav(
    path: Union[str, Path, np.ndarray, torch.Tensor],
    target_sr: int = 24000,
) -> tuple[np.ndarray, int]:
    """Load a WAV file, numpy array, or torch Tensor to 1-D float32 at target_sr."""
    if isinstance(path, torch.Tensor):
        wav = path.detach().cpu().squeeze().numpy().astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        return wav, target_sr
    if isinstance(path, np.ndarray):
        wav = path.astype(np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        return wav, target_sr
    try:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = _resample(wav, sr, target_sr)
        return wav, target_sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path}: {e}") from e


_ECAPA_CACHE: Dict[tuple[str, str], Any] = {}
_CFVN_CACHE: Dict[tuple[str, str], Any] = {}


def _get_cfvn(cfvn_path: Union[str, Path], device: str):
    """Return a cached ContrastiveFaceVoiceNet for repeated evaluations."""
    from voix.models.contrastive_net import ContrastiveFaceVoiceNet

    resolved = str(Path(cfvn_path).resolve())
    key = (resolved, str(device))
    if key not in _CFVN_CACHE:
        _CFVN_CACHE[key] = ContrastiveFaceVoiceNet.from_checkpoint(
            resolved, device=device
        )
    return _CFVN_CACHE[key]


def _get_ecapa(device: str, savedir: Union[str, Path] = DEFAULT_ECAPA_SAVEDIR):
    """Return a cached SpeechBrain ECAPA extractor."""
    from voix.data.speaker_extractor import ECAPAExtractor

    key = (str(device), str(savedir))
    if key not in _ECAPA_CACHE:
        _ECAPA_CACHE[key] = ECAPAExtractor(
            device=device,
            cache_dir=str(project_path("checkpoints", "speechbrain")),
            savedir=str(savedir),
        )
    return _ECAPA_CACHE[key]


def _embed_wav_16k(
    path: Union[str, Path, np.ndarray, torch.Tensor],
    device: str,
) -> Optional[torch.Tensor]:
    """Embed an utterance with ECAPA at its required 16 kHz sample rate."""
    try:
        wav, _ = _load_wav(path, 16000)
        waveform = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        return _get_ecapa(device).extract(waveform).squeeze(0)
    except Exception as e:
        logger.warning("ECAPA embedding unavailable for %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# 1. Prosody statistics (no extra deps)
# ---------------------------------------------------------------------------

def prosody_statistics(
    wav_path: Union[str, Path],
    sample_rate: int = 24000,
) -> Dict[str, float]:
    """Compute F0 and energy statistics from a WAV file.

    Returns dict with:
        f0_mean_hz, f0_std_hz, f0_range_hz   -- fundamental frequency statistics
        f0_voiced_ratio                        -- fraction of voiced frames
        f0_smoothness                          -- mean abs delta of F0 contour (lower = smoother)
        energy_mean, energy_std               -- RMS energy statistics
        duration_seconds                       -- utterance duration
    """
    wav, sr = _load_wav(wav_path, sample_rate)
    duration = len(wav) / sr

    # F0 extraction via pyin (librosa)
    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            wav.astype(np.float64),
            fmin=50.0,
            fmax=600.0,
            sr=sr,
            frame_length=2048,
        )
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 2:
            f0_mean   = float(np.mean(f0_voiced))
            f0_std    = float(np.std(f0_voiced))
            f0_range  = float(np.max(f0_voiced) - np.min(f0_voiced))
            # Smoothness: lower = more natural gradual transitions
            f0_smooth = float(np.mean(np.abs(np.diff(f0_voiced))))
        else:
            f0_mean = f0_std = f0_range = f0_smooth = 0.0
        voiced_ratio = float(voiced_flag.sum()) / max(1, len(voiced_flag))
    except ImportError:
        logger.warning("librosa not installed; F0 statistics unavailable")
        f0_mean = f0_std = f0_range = f0_smooth = voiced_ratio = 0.0

    # RMS energy
    frame_len = sr // 100  # 10ms frames
    frames = [wav[i: i + frame_len] for i in range(0, len(wav) - frame_len, frame_len)]
    rms = [float(np.sqrt(np.mean(f ** 2))) for f in frames if len(f) == frame_len]
    energy_mean = float(np.mean(rms)) if rms else 0.0
    energy_std  = float(np.std(rms))  if rms else 0.0

    return {
        "duration_seconds":  round(duration, 3),
        "f0_mean_hz":        round(f0_mean,  2),
        "f0_std_hz":         round(f0_std,   2),
        "f0_range_hz":       round(f0_range, 2),
        "f0_voiced_ratio":   round(voiced_ratio, 3),
        "f0_smoothness":     round(f0_smooth, 3),
        "energy_mean":       round(energy_mean, 5),
        "energy_std":        round(energy_std,  5),
    }


# ---------------------------------------------------------------------------
# 2. Speaker similarity (ECAPA cosine distance)
# ---------------------------------------------------------------------------

def speaker_similarity(
    wav_path_a: Union[str, Path],
    wav_path_b: Union[str, Path],
    device: str = "cpu",
) -> Optional[float]:
    """Compute ECAPA-TDNN cosine similarity between two utterances.

    Returns a value in [-1, 1]; >0.5 suggests same speaker.  Returns ``None``
    when the extractor or audio is unavailable -- a failed measurement must not
    be reported as a real similarity of 0.0.
    """
    e_a = _embed_wav_16k(wav_path_a, device)
    e_b = _embed_wav_16k(wav_path_b, device)
    if e_a is None or e_b is None:
        return None
    sim = F.cosine_similarity(e_a.unsqueeze(0), e_b.unsqueeze(0)).item()
    return round(sim, 4)


# ---------------------------------------------------------------------------
# 3. Face-voice consistency
# ---------------------------------------------------------------------------

def _face_embedding_for_cfvn(
    image_path: Union[str, Path],
    net,
    scaler_path: Union[str, Path],
    device: str,
    pipeline=None,
) -> Optional[torch.Tensor]:
    """Build the exact CFVN input for one image (normalized, correct schema)."""
    import cv2

    from voix.data.feature_contract import (
        FaceFeatureBuilder,
        load_scaler_payload,
    )

    face_dim = int(getattr(net, "face_dim", 0))
    if face_dim not in (512, 560):
        logger.warning(
            "Unsupported CFVN face_dim=%s (expected 512 or 560)", face_dim
        )
        return None
    is_arcface = face_dim == 512

    # The pipeline shortcut returns the fused 560-D representation.  It cannot
    # serve the 512-D diagnostic model, which must consume raw ArcFace.
    if (
        not is_arcface
        and pipeline is not None
        and hasattr(pipeline, "extract_face_features")
    ):
        raw, _, _, _ = pipeline.extract_face_features(str(image_path))
        raw = raw.to(device).float()
    else:
        from voix.data.demographics import (
            SoftDemographicEstimator,
            aspect_ratio_from_bbox,
        )
        from voix.data.face_extractor import (
            ArcFaceExtractor,
            FaceMeshExtractor,
            crop_face_bbox,
        )
        from voix.data.morphology import compute_craniofacial_ratios

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Could not read face image: %s", image_path)
            return None

        arcface = ArcFaceExtractor(
            device_id=-1, cache_dir=str(DEFAULT_CHECKPOINT_DIR)
        )
        e_id, gender, age, bbox = arcface.extract_full_with_bbox(img)
        if e_id is None:
            logger.warning("No face detected in image: %s", image_path)
            return None

        if is_arcface:
            # Diagnostic CFVN trained on raw ArcFace only.
            raw = torch.from_numpy(np.asarray(e_id, dtype=np.float32)).unsqueeze(0)
        else:
            task_path = project_path(
                "checkpoints", "mediapipe", "face_landmarker.task"
            )
            facemesh = FaceMeshExtractor(
                model_path=str(task_path) if task_path.exists() else None
            )
            face_crop = crop_face_bbox(img, bbox) if bbox is not None else img
            landmarks = facemesh.extract(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            )
            e_geo = (
                compute_craniofacial_ratios(landmarks)
                if landmarks is not None
                else np.zeros(32, dtype=np.float32)
            )
            demog = SoftDemographicEstimator()
            e_demo = demog.estimate(
                e_id,
                gender=gender,
                age=age,
                aspect_ratio=(
                    aspect_ratio_from_bbox(bbox) if bbox is not None else 1.35
                ),
            )
            builder = FaceFeatureBuilder(device=device)
            raw = builder.fuse(e_id, e_geo, e_demo).cpu()

    # Match the scaler to the representation the CFVN was trained on.  If the
    # caller left the default 560-D path in place, switch to the ArcFace scaler
    # rather than failing on a dimension mismatch.
    resolved_scaler = Path(scaler_path)
    if (
        is_arcface
        and resolved_scaler.resolve() == Path(DEFAULT_SCALER).resolve()
    ):
        resolved_scaler = Path(DEFAULT_SCALER_ARCFACE)
    try:
        scaler = load_scaler_payload(
            resolved_scaler,
            expected_dim=face_dim,
            face_key="arcface" if is_arcface else "face_features",
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("CFVN face scaler unavailable: %s", exc)
        return None

    # The pipeline may return the fused face vector on CUDA while the scaler
    # payload is loaded on CPU.  Move the scaler to the face tensor's device
    # before arithmetic instead of assuming both are co-located.
    target_device = raw.device
    mean = scaler["face_mean"].to(target_device)
    std = torch.clamp(scaler["face_std"].to(target_device), min=1e-6)
    return ((raw - mean) / std).to(device)


def face_voice_consistency(
    image_path: Union[str, Path],
    wav_path: Union[str, Path],
    pipeline=None,
    device: str = "cpu",
    cfvn_path: Union[str, Path] = DEFAULT_CFVN,
    scaler_path: Union[str, Path] = DEFAULT_SCALER,
) -> Optional[float]:
    """Compute ContrastiveFaceVoiceNet similarity between a face image and voice.

    Measures whether the generated voice is acoustically consistent with the
    face biometrics -- the core quality metric for face-to-voice synthesis.

    The face path is the canonical FaceFeatureV2 contract: ArcFace + FaceMesh +
    demographics fused by ``FaceFeatureBuilder``, z-scored with the train-split
    scaler, then mapped by the retrained CFVN.  The old implementation used the
    stale V1 fusion, the old collapsed checkpoint, and a module path that no
    longer existed.

    Args:
        image_path: Path to the subject's face image.
        wav_path:   Path to the synthesized voice WAV file.
        pipeline:   Optional VoixPipeline instance (reuses its extractors).
        device:     Computation device.
        cfvn_path:  CFVN checkpoint (defaults to the V2 checkpoint).
        scaler_path: FaceFeatureV2 scaler checkpoint.

    Returns:
        Cosine similarity in [-1, 1], or ``None`` when the measurement cannot
        be made.  ``None`` is deliberate: silently returning 0.0 made missing
        dependencies look like poor face-voice consistency.
    """
    try:
        cfvn_path = Path(cfvn_path)
        if not cfvn_path.exists():
            logger.warning("CFVN checkpoint not found: %s", cfvn_path)
            return None
        net = _get_cfvn(cfvn_path, device)
        net.eval()

        face = _face_embedding_for_cfvn(
            image_path, net, scaler_path, device, pipeline=pipeline
        )
        if face is None:
            return None

        voice = _embed_wav_16k(wav_path, device)
        if voice is None:
            return None
        voice = F.normalize(voice.unsqueeze(0), dim=-1).to(device)

        with torch.no_grad():
            e_face = F.normalize(net(face), dim=-1)  # (1, 192)
            sim = F.cosine_similarity(e_face, voice).item()
        return round(sim, 4)
    except Exception as e:
        logger.warning("face_voice_consistency unavailable: %s", e)
        return None


# ---------------------------------------------------------------------------
# 4. UTMOS neural MOS prediction
# ---------------------------------------------------------------------------

def utmos_score(wav_path: Union[str, Path], device: str = "cpu") -> float:
    """Predict MOS using UTMOS (Saeki et al., Interspeech 2022).

    Returns predicted MOS in [1, 5]. Returns -1.0 if UTMOS unavailable.
    Install: pip install utmos
    """
    try:
        import utmos
        predictor = utmos.load_model("strong")
        score = predictor.predict(str(wav_path), device=device)
        return round(float(score), 3)
    except ImportError:
        logger.debug("utmos not installed; returning -1.0")
        return -1.0
    except Exception as e:
        logger.debug(f"UTMOS prediction failed: {e}")
        return -1.0


# ---------------------------------------------------------------------------
# 5. DNSMOS (noise + speech quality MOS prediction)
# ---------------------------------------------------------------------------

def dnsmos_score(wav_path: Union[str, Path]) -> Dict[str, float]:
    """Predict DNSMOS scores (OVRL, SIG, BAK) using ONNX model.

    Returns:
        dict with keys: ovrl_mos, sig_mos, bak_mos (all in [1, 5]).
        Returns zeros if DNSMOS not available.

    Install: pip install dnsmos
    """
    try:
        from dnsmos import DNSMOS
        model = DNSMOS()
        result = model(str(wav_path))
        return {
            "ovrl_mos": round(float(result.get("OVRL", 0)), 3),
            "sig_mos":  round(float(result.get("SIG",  0)), 3),
            "bak_mos":  round(float(result.get("BAK",  0)), 3),
        }
    except ImportError:
        return {"ovrl_mos": -1.0, "sig_mos": -1.0, "bak_mos": -1.0}
    except Exception as e:
        logger.debug(f"DNSMOS failed: {e}")
        return {"ovrl_mos": -1.0, "sig_mos": -1.0, "bak_mos": -1.0}


# ---------------------------------------------------------------------------
# 6. All-in-one synthesis evaluation
# ---------------------------------------------------------------------------

def evaluate_synthesis(
    wav_path: Union[str, Path],
    image_path: Optional[Union[str, Path]] = None,
    reference_wav: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    run_utmos: bool = False,
    run_dnsmos: bool = False,
    pipeline=None,
    cfvn_path: Union[str, Path] = DEFAULT_CFVN,
    scaler_path: Union[str, Path] = DEFAULT_SCALER,
) -> Dict[str, Any]:
    """Compute all available quality metrics for a synthesized voice.

    Args:
        wav_path:      Path to synthesized WAV file.
        image_path:    Optional face image for face-voice consistency.
        reference_wav: Optional ground-truth WAV for speaker similarity.
        device:        Computation device.
        run_utmos:     Whether to run UTMOS (slow; requires `pip install utmos`).
        run_dnsmos:    Whether to run DNSMOS (requires onnxruntime).

    Returns:
        Dictionary of all computed metrics.
    """
    results: Dict[str, Any] = {"wav_path": str(wav_path)}

    # Prosody stats (always)
    try:
        results["prosody"] = prosody_statistics(wav_path)
    except Exception as e:
        results["prosody"] = {"error": str(e)}

    # Speaker similarity with reference
    if reference_wav is not None:
        results["speaker_similarity"] = speaker_similarity(wav_path, reference_wav, device)

    # Face-voice consistency
    if image_path is not None:
        results["face_voice_consistency"] = face_voice_consistency(
            image_path,
            wav_path,
            pipeline=pipeline,
            device=device,
            cfvn_path=cfvn_path,
            scaler_path=scaler_path,
        )

    # Neural MOS
    if run_utmos:
        results["utmos"] = utmos_score(wav_path, device)

    if run_dnsmos:
        results["dnsmos"] = dnsmos_score(wav_path)

    return results


# ---------------------------------------------------------------------------
# 7. Batch evaluation for multiple candidates
# ---------------------------------------------------------------------------

def evaluate_candidates(
    candidate_wavs: Union[str, Path, List[Union[str, Path]]],
    image_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    run_utmos: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate a list of candidate WAV files (or directory of WAVs) and return ranked results.

    Args:
        candidate_wavs: List of paths to synthesized WAV files, or directory containing WAVs.
        image_path:     Optional face image for face-voice consistency.
        device:         Computation device.
        run_utmos:      Whether to run UTMOS MOS prediction.

    Returns:
        List of metric dicts, one per candidate, sorted by face_voice_consistency (desc).
    """
    if isinstance(candidate_wavs, (str, Path)):
        p = Path(candidate_wavs)
        if p.is_dir():
            candidate_wavs = sorted([str(f) for f in p.glob("*.wav")])
        else:
            candidate_wavs = [str(p)]

    results = []
    for wav_path in candidate_wavs:
        metrics = evaluate_synthesis(
            wav_path=wav_path,
            image_path=image_path,
            device=device,
            run_utmos=run_utmos,
        )
        metrics["voice_index"] = candidate_wavs.index(wav_path) + 1
        results.append(metrics)

    # Sort by face-voice consistency if available
    has_fvc = all(r.get("face_voice_consistency") is not None for r in results)
    if has_fvc:
        results.sort(key=lambda x: x.get("face_voice_consistency", 0.0), reverse=True)

    return results
