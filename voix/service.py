"""High-Level Application Service and DTO layer for VOIX.

Provides an enterprise-grade, decoupled abstraction between the external interfaces
(CLI, Web UI, REST API) and internal neural models.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch

from voix.inference.pipeline import VoixPipeline
from voix.utils.artifacts import load_manifest
from voix.utils.config import get_hardware_profile
from voix.utils.model_cache import (
    audioseal_cache_dir,
    cache_file_count,
    cache_has_files,
    styletts2_cache_dir,
)
from voix.utils.paths import project_path


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True when an exception is a CUDA out-of-memory condition."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    if isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda error: out of memory" in message
        or "allocation on device" in message
    )


def validate_output_destination(
    destination: Union[str, Path, None],
    num_voices: int,
) -> None:
    """Validate and prepare an output destination *before* neural synthesis.

    This duplicates only the path-shape checks in ``SynthesisResponse.save_all``.
    Doing them up front means a multi-voice request cannot spend minutes
    synthesizing and then fail because it targeted ``output.wav``.
    """
    if destination is None or str(destination).strip() == "":
        return

    dest = Path(destination)
    is_wav = dest.suffix.lower() == ".wav"

    if num_voices > 1:
        if is_wav:
            raise ValueError(
                f"Cannot save {num_voices} candidates to the single file path "
                f"'{dest}'. Pass a directory instead."
            )
        if dest.exists() and not dest.is_dir():
            raise NotADirectoryError(
                f"Destination exists and is not a directory: {dest}"
            )
        dest.mkdir(parents=True, exist_ok=True)
        return

    if is_wav:
        if dest.exists() and dest.is_dir():
            raise IsADirectoryError(f"Destination is a directory: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        return

    if dest.exists() and not dest.is_dir():
        raise NotADirectoryError(
            f"Destination exists and is not a directory: {dest}"
        )
    dest.mkdir(parents=True, exist_ok=True)


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
        if self.gender is not None and int(self.gender) not in (0, 1):
            raise ValueError(f"gender must be 0 (female) or 1 (male), got {self.gender}")
        if self.age is not None and not 0.0 <= float(self.age) <= 120.0:
            raise ValueError(f"age must be between 0 and 120, got {self.age}")
        validate_output_destination(self.output_path, self.num_voices)


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
    execution_device: str = ""
    fallback_reason: Optional[str] = None

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
            "execution_device": self.execution_device,
            "fallback_reason": self.fallback_reason,
            "candidates": [c.to_dict() for c in self.candidates],
            "saved_files": self.saved_files,
        }


class VoixService:
    """Unified application service managing lifecycle, execution, and diagnostics."""

    _instance: Optional[VoixService] = None

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._pipeline: Optional[VoixPipeline] = None
        self._fallback_reason: Optional[str] = None

    @property
    def pipeline(self) -> VoixPipeline:
        """Lazy-load the underlying VoixPipeline on demand."""
        if self._pipeline is None:
            self._pipeline = VoixPipeline(device=self.device)
        return self._pipeline

    def _configured_device(self) -> str:
        if self.device is not None:
            return str(self.device)
        return str(get_hardware_profile().device)

    def preflight(self, request: Optional[SynthesisRequest] = None) -> Dict[str, Any]:
        """Check artifacts, caches, device, and output paths before synthesis.

        This is intentionally model-free: it reads paths and package metadata so
        it is cheap enough to run on every CLI invocation and before a live demo.
        """
        errors: List[str] = []
        warnings: List[str] = []
        checks: List[Dict[str, Any]] = []

        def check_artifact(
            name: str,
            rel_path: str,
            *,
            required: bool = True,
            role: str = "serving",
        ) -> bool:
            path = project_path(rel_path)
            exists = path.is_file()
            size_mb = round(path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
            checks.append(
                {
                    "name": name,
                    "path": str(path),
                    "exists": exists,
                    "size_mb": size_mb,
                    "required": required,
                    "role": role,
                }
            )
            if required and not exists:
                errors.append(f"Missing required {role} artifact: {name} ({path})")
            elif not required and not exists:
                warnings.append(f"Optional {role} artifact missing: {name} ({path})")
            return exists

        try:
            artifact_specs = load_manifest().artifacts
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            artifact_specs = []
            errors.append(f"Artifact manifest is unavailable or invalid: {exc}")

        for spec in artifact_specs:
            check_artifact(
                spec.name,
                spec.path,
                required=spec.required,
                role=spec.role,
            )

        # Package and cache checks.  A missing cache is a warning during normal
        # operation and an error when offline mode is requested.
        offline = any(
            os.environ.get(key, "").strip().lower() in {"1", "true", "yes"}
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VOIX_OFFLINE")
        )
        tts_installed = importlib.util.find_spec("styletts2") is not None
        tts_cache = styletts2_cache_dir()
        tts_cached = cache_has_files(tts_cache, minimum=6)
        checks.append(
            {
                "name": "StyleTTS2 neural backend",
                "path": str(tts_cache),
                "exists": tts_installed,
                "size_mb": 0.0,
                "required": True,
                "role": "tts",
                "cached": tts_cached,
            }
        )
        if not tts_installed:
            errors.append(
                "StyleTTS2 is not installed. Install the serving extra: "
                "pip install -e '.[serving]'"
            )
        elif not tts_cached:
            message = (
                "StyleTTS2 checkpoints are not cached at "
                f"{tts_cache}; the first neural synthesis will download them."
            )
            if offline:
                errors.append(message + " Offline mode is enabled.")
            else:
                warnings.append(message)

        watermark_requested = bool(request is not None and request.watermark)
        audioseal_installed = importlib.util.find_spec("audioseal") is not None
        audioseal_cache = audioseal_cache_dir()
        audioseal_cached = cache_has_files(audioseal_cache, minimum=2)
        checks.append(
            {
                "name": "AudioSeal watermarker",
                "path": str(audioseal_cache),
                "exists": audioseal_installed,
                "size_mb": 0.0,
                "required": watermark_requested,
                "role": "watermark",
                "cached": audioseal_cached,
            }
        )
        if watermark_requested:
            if not audioseal_installed:
                errors.append(
                    "Watermarking was requested but AudioSeal is not installed. "
                    "Install the serving extra or pass --no-watermark."
                )
            elif not audioseal_cached:
                message = (
                    "Watermarking was requested but the AudioSeal cache at "
                    f"{audioseal_cache} is incomplete; the first run will download it."
                )
                if offline:
                    errors.append(message + " Offline mode is enabled.")
                else:
                    warnings.append(message)

        # Device and memory checks.
        configured = self._configured_device()
        cuda_available = torch.cuda.is_available()
        if configured.startswith("cuda") and not cuda_available:
            errors.append(
                f"Device '{configured}' was requested but CUDA is not available. "
                "Use --device cpu."
            )
        if configured.startswith("mps"):
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                errors.append(
                    "Device 'mps' was requested but the MPS backend is not available."
                )
        if configured.startswith("cuda") and cuda_available:
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                vram_gb = 0.0
            if 0.0 < vram_gb < 5.0:
                warnings.append(
                    f"GPU has {vram_gb:.2f} GB VRAM. Multi-voice synthesis or "
                    "watermarking may exhaust it; the service will retry on CPU "
                    "if CUDA runs out of memory."
                )

        # Request-level validation, including the output destination.
        if request is not None:
            try:
                request.validate()
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                errors.append(f"Invalid request: {exc}")

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
            "device": configured,
            "cuda_available": cuda_available,
            "offline": offline,
            "fallback_available": configured.startswith("cuda"),
        }

    def _synthesize_once(
        self,
        request: SynthesisRequest,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SynthesisResponse:
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
            progress_callback=progress_callback,
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
            execution_device=str(self.pipeline.device),
            fallback_reason=self._fallback_reason,
        )

        if request.output_path:
            response.save_all(request.output_path)

        return response

    def synthesize(
        self,
        request: SynthesisRequest,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SynthesisResponse:
        """Execute single-request voice synthesis with a CPU OOM fallback."""
        request.validate()
        self._fallback_reason = None

        try:
            return self._synthesize_once(request, progress_callback)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            configured = self._configured_device()
            if not configured.startswith("cuda"):
                raise

            if progress_callback:
                try:
                    progress_callback(
                        "CUDA ran out of memory; retrying the same request on CPU."
                    )
                except Exception:
                    pass

            # Drop every CUDA-backed object before rebuilding the pipeline on CPU.
            self._pipeline = None
            self._fallback_reason = f"CUDA OOM: {exc}"
            try:
                from voix.models.tts_backend import StyleTTS2Backend

                StyleTTS2Backend.reset_shared_model()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.device = "cpu"
            return self._synthesize_once(request, progress_callback)

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
        """Return structured hardware, artifact, and cache status."""
        profile = get_hardware_profile()

        try:
            artifact_specs = load_manifest().artifacts
            manifest_error = None
        except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
            artifact_specs = []
            manifest_error = str(exc)

        checkpoints = [
            (spec.name, spec.path, spec.required, spec.role)
            for spec in artifact_specs
        ]
        checkpoints.append(("Environment Config", "configs/environment.yaml", True, "config"))

        verified_ckpts = []
        for name, rel_path, required, role in checkpoints:
            path = project_path(rel_path)
            exists = path.is_file()
            verified_ckpts.append(
                {
                    "component": name,
                    "relative_path": rel_path,
                    "exists": exists,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
                    if exists
                    else 0.0,
                    "required": required,
                    "role": role,
                }
            )

        tts_cache = styletts2_cache_dir()
        audioseal_cache = audioseal_cache_dir()
        offline = any(
            os.environ.get(key, "").strip().lower() in {"1", "true", "yes"}
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VOIX_OFFLINE")
        )

        return {
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hardware_profile": profile.name,
            "accelerator": profile.accelerator,
            "device": str(profile.device),
            "device_requested": self.device,
            "cuda_available": torch.cuda.is_available(),
            "memory_budget_gb": round(profile.total_memory_gb, 2),
            "precision": profile.precision,
            "pipeline_mode": profile.pipeline_mode,
            "offline": offline,
            "checkpoints": verified_ckpts,
            "artifact_manifest_error": manifest_error,
            "caches": {
                "styletts2": {
                    "path": str(tts_cache),
                    "exists": tts_cache.is_dir(),
                    "files": cache_file_count(tts_cache),
                },
                "audioseal": {
                    "path": str(audioseal_cache),
                    "exists": audioseal_cache.is_dir(),
                    "files": cache_file_count(audioseal_cache),
                },
            },
        }
