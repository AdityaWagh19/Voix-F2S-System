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
from voix.utils.config import get_hardware_profile


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
    accent: Optional[str] = None
    accent_strength: float = 1.0
    emotion: Optional[str] = None
    emotion_intensity: float = 1.0
    watermark: bool = True

    def validate(self) -> None:
        """Validate request invariants before passing to neural models."""
        face_ref = str(self.face_image)
        if not Path(face_ref).exists():
            # Preset IDs are valid virtual face inputs and must not be treated
            # as missing files. Import the lightweight registry definitions only.
            from voix.data.presets import CANONICAL_PRESETS_DEF
            preset_ids = {p["preset_id"] for p in CANONICAL_PRESETS_DEF}
            if face_ref not in preset_ids:
                raise FileNotFoundError(
                    f"Input face image not found at '{self.face_image}' and it is "
                    f"not a known demographic preset. Available presets: {sorted(preset_ids)}"
                )
        if not self.text.strip():
            raise ValueError("Synthesis text prompt cannot be empty.")
        if self.num_voices < 1 or self.num_voices > 10:
            raise ValueError(f"num_voices must be between 1 and 10, got {self.num_voices}")
        if self.temperature <= 0.0 or self.temperature > 3.0:
            raise ValueError(f"temperature must be between 0.1 and 3.0, got {self.temperature}")
        from voix.models.accent_control import normalize_accent
        from voix.models.emotion_control import normalize_emotion

        if self.accent:
            normalize_accent(self.accent)
        normalize_emotion(self.emotion)
        if not 0.0 <= float(self.accent_strength) <= 1.0:
            raise ValueError("accent_strength must be between 0.0 and 1.0")
        if not 0.0 <= float(self.emotion_intensity) <= 1.0:
            raise ValueError("emotion_intensity must be between 0.0 and 1.0")


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
    backend: str = ""
    watermark_status: str = "disabled"

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
            "backend": self.backend,
            "watermark_status": self.watermark_status,
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
        """Save all generated candidates to a file or directory.

        A single candidate may target ``out.wav`` directly. Multiple candidates
        require a directory destination; silently creating a directory named
        ``out.wav`` is never acceptable.
        """
        dest = Path(destination)
        out_paths = []
        if len(self.candidates) == 0:
            raise ValueError("No candidates to save.")

        if len(self.candidates) == 1:
            if dest.suffix.lower() == ".wav":
                if dest.exists() and dest.is_dir():
                    raise IsADirectoryError(f"Destination is a directory: {dest}")
                p = self.candidates[0].save(dest)
                out_paths.append(p)
            else:
                if dest.exists() and not dest.is_dir():
                    raise NotADirectoryError(
                        f"Destination exists and is not a directory: {dest}"
                    )
                dest.mkdir(parents=True, exist_ok=True)
                p = self.candidates[0].save(dest / "voice_candidate_1.wav")
                out_paths.append(p)
        else:
            if dest.suffix.lower() == ".wav":
                raise ValueError(
                    f"Cannot save {len(self.candidates)} candidates to the single "
                    f"file path '{dest}'. Pass a directory instead."
                )
            if dest.exists() and not dest.is_dir():
                raise NotADirectoryError(
                    f"Destination exists and is not a directory: {dest}"
                )
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
            accent=request.accent,
            accent_strength=request.accent_strength,
            emotion=request.emotion,
            emotion_intensity=request.emotion_intensity,
            watermark=request.watermark,
        )
        latency = time.time() - t0

        candidates = []
        for raw in raw_results:
            waveform = raw["waveform"]
            if isinstance(waveform, torch.Tensor):
                wav_np = waveform.detach().cpu().numpy().squeeze().astype(np.float32)
            else:
                wav_np = np.asarray(waveform, dtype=np.float32).squeeze()
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
                backend=raw.get("backend", ""),
                watermark_status=raw.get("watermark_status", "disabled"),
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
        """Query available system capabilities (accents and emotions)."""
        from voix.models.accent_control import list_available_accents
        from voix.models.emotion_control import list_available_emotions

        return {
            "accents": [
                a["accent_id"] if isinstance(a, dict) else a
                for a in list_available_accents()
            ],
            "emotions": [
                e["emotion_id"] if isinstance(e, dict) else e
                for e in list_available_emotions()
            ],
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return structured hardware and checkpoint status."""
        profile = get_hardware_profile()
        root_dir = Path(__file__).resolve().parent.parent

        checkpoints = [
            ("ArcFace (InsightFace)", "checkpoints/models/buffalo_l/w600k_r50.onnx"),
            ("MediaPipe FaceLandmarker", "checkpoints/mediapipe/face_landmarker.task"),
            ("ECAPA-TDNN (SpeechBrain)", "checkpoints/speechbrain/spkrec-ecapa/embedding_model.ckpt"),
            ("ContrastiveFaceVoiceNet", "results/checkpoints/D_joint_infonce_best.pt"),
            ("CVAE Diversity Model", "checkpoints/cvae_cfvn_face192_latent128_diversity.pt"),
            ("StyleAdapter Checkpoint", "checkpoints/style_adapter_cfvn_best.pt"),
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
