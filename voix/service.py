"""High-Level Application Service and DTO layer for VOIX.

Provides an enterprise-grade, decoupled abstraction between the external interfaces
(CLI, Web UI, REST API) and internal neural models.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch

from voix.inference.pipeline import VoixPipeline
from voix.utils.hardware import detect_hardware


def _compute_quick_metrics(wav: np.ndarray, sample_rate: int = 24000) -> Dict[str, float]:
    """Calculate basic acoustic properties without crashing on silent signals."""
    duration = float(len(wav) / sample_rate) if sample_rate > 0 else 0.0
    peak = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) > 0 else 0.0
    
    # Fast autocorrelation F0 estimate
    f0_mean = 0.0
    if len(wav) > sample_rate // 10:
        try:
            corr = np.correlate(wav[:sample_rate], wav[:sample_rate], mode="full")
            corr = corr[len(corr) // 2 :]
            min_lag = int(sample_rate / 450)
            max_lag = int(sample_rate / 70)
            if max_lag < len(corr):
                peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                if peak_lag > 0:
                    f0_mean = float(sample_rate / peak_lag)
        except Exception:
            f0_mean = 0.0

    return {
        "duration_seconds": round(duration, 3),
        "peak_amplitude": round(peak, 4),
        "rms_energy": round(rms, 4),
        "f0_mean_hz": round(f0_mean, 1),
    }


@dataclass
class SynthesisRequest:
    """Enterprise Data Transfer Object (DTO) for voice synthesis."""
    face_image: str
    text: str
    output_path: Optional[str] = None
    num_voices: int = 1
    temperature: float = 1.0
    seed: Optional[int] = None
    gender: Optional[int] = None  # 0=female, 1=male, None=auto-detect
    age: Optional[float] = None
    craniofacial_sliders: Optional[Dict[str, float]] = None
    accent: Optional[str] = None
    accent_strength: float = 0.5
    emotion: Optional[str] = None
    emotion_intensity: float = 0.5
    language: str = "en"
    speech_difference: Optional[str] = None
    speech_severity: float = 0.5
    watermark: bool = True

    def validate(self) -> None:
        """Validate request invariants before passing to neural models."""
        p = Path(self.face_image)
        if not p.exists():
            raise FileNotFoundError(f"Input face image not found at '{self.face_image}'")
        if not self.text.strip():
            raise ValueError("Synthesis text prompt cannot be empty.")
        if self.num_voices < 1 or self.num_voices > 10:
            raise ValueError(f"num_voices must be between 1 and 10, got {self.num_voices}")
        if self.temperature <= 0.0 or self.temperature > 3.0:
            raise ValueError(f"temperature must be between 0.1 and 3.0, got {self.temperature}")


@dataclass
class VoiceCandidate:
    """Encapsulates a single synthesized voice candidate and its acoustic attributes."""
    index: int
    waveform: np.ndarray
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    is_watermarked: bool = False
    f0_mean_hz: float = 0.0
    rms_energy: float = 0.0
    peak_amplitude: float = 0.0
    file_path: Optional[str] = None

    def save(self, path: Union[str, Path], normalize_peak: bool = True) -> Path:
        """Persist waveform to disk as a 24 kHz WAV file."""
        target_p = Path(path)
        target_p.parent.mkdir(parents=True, exist_ok=True)
        
        wav_data = self.waveform.copy()
        if normalize_peak:
            peak = float(np.max(np.abs(wav_data)))
            if peak > 1e-4:
                wav_data = wav_data / peak * 0.90
                
        sf.write(str(target_p), wav_data, self.sample_rate)
        self.file_path = str(target_p)
        return target_p

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate metadata to JSON-serializable dictionary."""
        return {
            "index": self.index,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "is_watermarked": self.is_watermarked,
            "f0_mean_hz": self.f0_mean_hz,
            "rms_energy": self.rms_energy,
            "peak_amplitude": self.peak_amplitude,
            "file_path": self.file_path,
        }


@dataclass
class SynthesisResponse:
    """Complete structured response object returned by VoixService."""
    request_id: str
    face_image: str
    text: str
    candidates: List[VoiceCandidate] = field(default_factory=list)
    latency_seconds: float = 0.0
    saved_files: List[str] = field(default_factory=list)

    def save_all(self, destination: Union[str, Path]) -> List[Path]:
        """Save all generated candidates to a file or directory."""
        dest = Path(destination)
        out_paths = []
        if len(self.candidates) == 1:
            if dest.suffix.lower() == ".wav":
                p = self.candidates[0].save(dest)
                out_paths.append(p)
            else:
                dest.mkdir(parents=True, exist_ok=True)
                p = self.candidates[0].save(dest / "voice_candidate_1.wav")
                out_paths.append(p)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            for cand in self.candidates:
                p = cand.save(dest / f"voice_candidate_{cand.index}.wav")
                out_paths.append(p)
        self.saved_files = [str(p) for p in out_paths]
        return out_paths

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire response to clean JSON-serializable representation."""
        return {
            "request_id": self.request_id,
            "face_image": self.face_image,
            "text": self.text,
            "latency_seconds": round(self.latency_seconds, 3),
            "candidates_count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
            "saved_files": self.saved_files,
        }


class VoixService:
    """Unified application service managing lifecycle, execution, and diagnostics."""

    _instance: Optional[VoixService] = None

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._pipeline: Optional[VoixPipeline] = None

    @property
    def pipeline(self) -> VoixPipeline:
        """Lazy-load the underlying VoixPipeline on demand."""
        if self._pipeline is None:
            self._pipeline = VoixPipeline(device=self.device)
        return self._pipeline

    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        """Execute single-request voice synthesis."""
        request.validate()

        if request.seed is not None:
            torch.manual_seed(request.seed)
            np.random.seed(request.seed)

        t0 = time.time()
        raw_results = self.pipeline.generate(
            face_input=request.face_image,
            text=request.text,
            num_voices=request.num_voices,
            temperature=request.temperature,
            gender=request.gender,
            age=request.age,
            craniofacial_sliders=request.craniofacial_sliders,
            accent=request.accent,
            accent_strength=request.accent_strength,
            emotion=request.emotion,
            emotion_intensity=request.emotion_intensity,
            language=request.language,
            speech_difference=request.speech_difference,
            speech_severity=request.speech_severity,
            watermark=request.watermark,
        )
        latency = time.time() - t0

        candidates = []
        for raw in raw_results:
            wav_np = raw["waveform"].numpy().squeeze().astype(np.float32)
            metrics = _compute_quick_metrics(wav_np, raw["sample_rate"])
            cand = VoiceCandidate(
                index=raw["voice_index"],
                waveform=wav_np,
                sample_rate=raw["sample_rate"],
                duration_seconds=metrics["duration_seconds"],
                is_watermarked=raw.get("is_watermarked", False),
                f0_mean_hz=metrics["f0_mean_hz"],
                rms_energy=metrics["rms_energy"],
                peak_amplitude=metrics["peak_amplitude"],
            )
            candidates.append(cand)

        req_id = str(uuid.uuid4())[:8]
        response = SynthesisResponse(
            request_id=req_id,
            face_image=request.face_image,
            text=request.text,
            candidates=candidates,
            latency_seconds=latency,
        )

        if request.output_path:
            response.save_all(request.output_path)

        return response

    def synthesize_batch(
        self,
        requests: List[SynthesisRequest],
        progress_callback: Optional[Callable[[int, int, SynthesisResponse], None]] = None,
    ) -> List[SynthesisResponse]:
        """Execute batch synthesis for multiple requests with progress reporting."""
        responses = []
        total = len(requests)
        for idx, req in enumerate(requests):
            resp = self.synthesize(req)
            responses.append(resp)
            if progress_callback:
                progress_callback(idx + 1, total, resp)
        return responses

    def get_capabilities(self) -> Dict[str, Any]:
        """Query available system capabilities (accents, emotions, languages, sliders)."""
        return {
            "accents": [
                "general_american", "british_rp", "australian",
                "indian_english", "irish", "scottish"
            ],
            "emotions": [
                "neutral", "happy", "sad", "angry", "whispered", "surprised"
            ],
            "languages": ["en", "es", "fr", "de", "it"],
            "speech_differences": [
                "stutter", "lisp", "lateral_lisp", "rhotacism", "hypernasal", "cleft_palate", "prolongation"
            ],
            "craniofacial_sliders": [
                "face_height_to_ioc", "face_width_to_ioc", "jaw_width_to_ioc", "mandible_angle_deg"
            ],
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return structured hardware and checkpoint status."""
        profile = detect_hardware()
        root_dir = Path(__file__).resolve().parent.parent

        checkpoints = [
            ("ArcFace (InsightFace)", "checkpoints/models/buffalo_l/w600k_r50.onnx"),
            ("MediaPipe FaceLandmarker", "checkpoints/mediapipe/face_landmarker.task"),
            ("ECAPA-TDNN (SpeechBrain)", "checkpoints/speechbrain/spkrec-ecapa/embedding_model.ckpt"),
            ("CVAE Diversity Model", "checkpoints/cvae_cfvn_face192_latent128_diversity.pt"),
            ("StyleAdapter Checkpoint", "checkpoints/style_adapter_best.pt"),
            ("Environment Config", "configs/environment.yaml"),
        ]

        verified_ckpts = []
        for name, rel_path in checkpoints:
            p = root_dir / rel_path
            verified_ckpts.append({
                "component": name,
                "relative_path": rel_path,
                "exists": p.exists(),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else 0.0,
            })

        return {
            "pytorch_version": torch.__version__,
            "hardware_profile": profile.name,
            "accelerator": profile.accelerator,
            "device": str(profile.device),
            "memory_budget_gb": round(profile.total_memory_gb, 2),
            "precision": profile.precision,
            "pipeline_mode": profile.pipeline_mode,
            "checkpoints": verified_ckpts,
        }
