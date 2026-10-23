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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("voix.evaluation")


# ---------------------------------------------------------------------------
# Audio loading helper
# ---------------------------------------------------------------------------

def _load_wav(path: Union[str, Path, np.ndarray, torch.Tensor], target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load a WAV file, numpy array, or torch Tensor to a 1-D float32 numpy array."""
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
            try:
                import resampy
                wav = resampy.resample(wav, sr, target_sr)
            except ImportError:
                logger.warning("resampy not installed; skipping resampling")
        return wav, target_sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path}: {e}") from e


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
) -> float:
    """Compute ECAPA-TDNN cosine similarity between two utterances.

    Returns a value in [-1, 1]; >0.5 suggests same speaker.
    Returns 0.0 if speechbrain is not available.
    """
    try:
        from speechbrain.pretrained import EncoderClassifier
        savedir = "D:/voix/checkpoints/speechbrain/spkrec-ecapa"
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": device},
        )
        import torchaudio
        def embed(path):
            sig, sr = torchaudio.load(str(path))
            if sr != 16000:
                sig = torchaudio.functional.resample(sig, sr, 16000)
            if sig.dim() > 1:
                sig = sig.mean(0, keepdim=True)
            return model.encode_batch(sig).squeeze()

        e_a = embed(wav_path_a)
        e_b = embed(wav_path_b)
        sim = F.cosine_similarity(e_a.unsqueeze(0), e_b.unsqueeze(0)).item()
        return round(sim, 4)
    except Exception as e:
        logger.debug(f"speaker_similarity unavailable: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# 3. Face-voice consistency
# ---------------------------------------------------------------------------

def face_voice_consistency(
    image_path: Union[str, Path],
    wav_path: Union[str, Path],
    pipeline=None,
    device: str = "cpu",
) -> float:
    """Compute ContrastiveFaceVoiceNet similarity between a face image and synthesized voice.

    Measures whether the generated voice is acoustically consistent with the
    face biometrics -- the core quality metric for face-to-voice synthesis.

    Args:
        image_path: Path to the subject's face image.
        wav_path:   Path to the synthesized voice WAV file.
        pipeline:   Optional VoixPipeline instance (reuses face extractor if loaded).
        device:     Computation device.

    Returns:
        Cosine similarity in [-1, 1]. Higher = more face-voice consistent.
        Returns 0.0 if dependencies unavailable.
    """
    try:
        import sys
        sys.path.insert(0, "D:/voix")
        from voix.models.contrastive_net import ContrastiveFaceVoiceNet
        from voix.data.face_extractor import ArcFaceExtractor
        from voix.data.demographics import SoftDemographicEstimator
        from voix.data.morphology import compute_craniofacial_ratios
        import cv2

        # Load contrastive model
        cfvn_path = "D:/voix/results/checkpoints/D_joint_infonce_best.pt"
        if not os.path.exists(cfvn_path):
            return 0.0
        net = ContrastiveFaceVoiceNet.from_checkpoint(cfvn_path, device=device)
        net.eval()

        # Extract face embedding
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        arcface = ArcFaceExtractor(device_id=-1, cache_dir="D:/voix/checkpoints")
        e_id, gender, age = arcface.extract_full(img)
        if e_id is None:
            return 0.0

        # Build 560-D fused face vector (simplified: use ArcFace 512 + zero padding)
        # For a full fused vector use pipeline.extract_face_features()
        e_f_tensor = torch.tensor(e_id, dtype=torch.float32)
        # Pad to 560-D (ArcFace 512 + craniofacial 32 + demog 16)
        e_f_padded = torch.cat([e_f_tensor, torch.zeros(48)])[:560].unsqueeze(0)

        # Extract predicted speaker embedding from face
        with torch.no_grad():
            e_face = net(e_f_padded.to(device))   # (1, 192)

        # Extract voice embedding
        try:
            from speechbrain.pretrained import EncoderClassifier
            import torchaudio
            savedir = "D:/voix/checkpoints/speechbrain/spkrec-ecapa"
            enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=savedir,
                run_opts={"device": device},
            )
            sig, sr = torchaudio.load(str(wav_path))
            if sr != 16000:
                sig = torchaudio.functional.resample(sig, sr, 16000)
            if sig.dim() > 1:
                sig = sig.mean(0, keepdim=True)
            with torch.no_grad():
                e_voice = enc.encode_batch(sig).squeeze()  # (192,)
            e_voice = F.normalize(e_voice.unsqueeze(0), dim=-1)

            sim = F.cosine_similarity(e_face, e_voice.to(device)).item()
            return round(sim, 4)
        except Exception as e:
            logger.debug(f"voice embed failed: {e}")
            return 0.0
    except Exception as e:
        logger.debug(f"face_voice_consistency unavailable: {e}")
        return 0.0


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
        results["face_voice_consistency"] = face_voice_consistency(image_path, wav_path, device=device)

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
