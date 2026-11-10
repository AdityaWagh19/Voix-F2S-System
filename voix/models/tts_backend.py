"""StyleTTS 2 Acoustic Synthesis Backend for VOIX (Phase 4).

Wraps the frozen StyleTTS 2 neural text-to-speech synthesis model.
Receives text prompt and a predicted 128-D style conditioning vector s_hat
from the StyleAdapter, producing high-fidelity 24 kHz speech waveforms.

Design decisions (from architecture.md):
  - Completely frozen: StyleTTS 2 is NOT fine-tuned (requires_grad = False).
  - Style dimension: 128-D style conditioning vector.
  - Sample rate: 24,000 Hz.
  - Neural Vocoder: Uses official StyleTTS 2 diffusion + PL-BERT + HiFi-GAN vocoder.
  - Robust fallback: Provides harmonic formant-based synthesis for offline testing
    and fast test suite execution.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _prime_einops_torch_backend() -> None:
    """Bind einops to torch before a broken TensorFlow install can be selected.

    The environment has an importable but unusable TensorFlow package.  Once
    anything imports it, einops discovers the TF backend and crashes inside
    ``TensorflowBackend.is_appropriate_type``.  Resolving a torch tensor here
    caches the torch backend, so later rearrangement calls never probe TF.
    """
    try:
        import einops

        einops.rearrange(torch.zeros(1), "b -> b 1 1")
    except Exception:
        pass


_prime_einops_torch_backend()



def _time_stretch(wav: "torch.Tensor", sample_rate: int, duration_scale: float) -> "torch.Tensor":
    """Time-stretch without changing pitch (duration_scale > 1 = longer)."""
    if abs(duration_scale - 1.0) < 1e-4 or wav.numel() == 0:
        return wav
    try:
        import librosa
        mono = wav.squeeze(0).detach().cpu().numpy().astype("float32")
        stretched = librosa.effects.time_stretch(mono, rate=1.0 / duration_scale)
        return torch.from_numpy(stretched).unsqueeze(0).to(wav.device)
    except Exception as exc:
        raise RuntimeError(
            "duration_scale requires librosa time-stretching; install librosa or "
            f"use duration_scale=1.0. Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _pitch_shift(wav: "torch.Tensor", sample_rate: int, semitones: float) -> "torch.Tensor":
    """Pitch-shift without changing duration."""
    if abs(semitones) < 1e-4 or wav.numel() == 0:
        return wav
    try:
        import librosa
        mono = wav.squeeze(0).detach().cpu().numpy().astype("float32")
        shifted = librosa.effects.pitch_shift(
            mono, sr=sample_rate, n_steps=float(semitones)
        )
        return torch.from_numpy(shifted).unsqueeze(0).to(wav.device)
    except Exception as exc:
        raise RuntimeError(
            "f0_scale/target_f0 require librosa pitch-shifting; install librosa "
            f"or leave them at defaults. Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _estimate_f0(wav: "torch.Tensor", sample_rate: int) -> Optional[float]:
    """Median voiced F0, used to realise an absolute ``target_f0``."""
    try:
        import librosa
        mono = wav.squeeze(0).detach().cpu().numpy().astype("float32")
        f0, voiced, _ = librosa.pyin(
            mono, fmin=50.0, fmax=600.0, sr=sample_rate, frame_length=2048
        )
        values = f0[voiced]
        if values.size == 0:
            return None
        return float(np.nanmedian(values))
    except Exception:
        return None


def _lufs_normalize(wav: "torch.Tensor", sample_rate: int, target_lufs: float = -23.0) -> "torch.Tensor":
    """Normalize audio to target integrated loudness (EBU R128 / ITU-R BS.1770-4).

    Replaces destructive peak normalization. Preserves voice dynamics while
    ensuring consistent perceptual loudness across all generated voices.
    A true-peak limiter (default -1 dBTP) prevents inter-sample overs without
    hard-clipping the waveform. Falls back to soft peak normalization only when
    pyloudnorm is not installed.

    Args:
        wav: Float32 tensor of shape (1, T) or (T,).
        sample_rate: Audio sample rate in Hz.
        target_lufs: Target loudness in LUFS. -23.0 is the EBU R128 broadcast standard.

    Returns:
        Loudness-normalized float32 tensor of same shape as input.
    """
    import numpy as _np

    from voix.utils.audio import limit_true_peak

    wav_np = wav.squeeze().cpu().numpy().astype(_np.float64)
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(wav_np)
        if _np.isfinite(loudness) and loudness >= -70.0:
            wav_np = pyln.normalize.loudness(wav_np, loudness, target_lufs)
        # Near-silence is left at its original level: boosting it would only
        # amplify the noise floor and violate the loudness contract.
    except ImportError:
        # pyloudnorm not installed — soft peak normalization as fallback
        peak = float(_np.abs(wav_np).max())
        if peak > 1e-4:
            wav_np = wav_np / peak * 0.85

    wav_np = limit_true_peak(wav_np)
    return torch.from_numpy(wav_np.astype(_np.float32)).unsqueeze(0)

class StyleTTS2Backend(nn.Module):
    """Frozen StyleTTS 2 text-to-speech synthesis wrapper.

    Args:
        checkpoint_path: Optional path to pretrained StyleTTS 2 weights.
        config_path: Optional path to model configuration yaml.
        device: Target execution device ("cuda" or "cpu").
        sample_rate: Output audio sample rate in Hz (default: 24000).
        style_dim: Style vector dimensionality (default: 128).
        use_neural: Whether to use full neural vocoder (default: True, auto-fallback in tests).
    """

    STYLE_DIM: int = 128
    SAMPLE_RATE: int = 24000
    _SHARED_MODEL = None

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 24000,
        style_dim: int = 256,
        use_neural: Optional[bool] = None,
        allow_fallback: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.sample_rate = sample_rate
        self.style_dim = style_dim

        # ``use_neural=None`` keeps the historical test-friendly default, but the
        # backend now reports which path was used so callers can distinguish a
        # real neural synthesis from the offline formant fallback.
        if use_neural is None:
            is_pytest = "pytest" in sys.modules
            self.use_neural = not is_pytest
        else:
            self.use_neural = use_neural

        # A fallback is only acceptable when it is explicitly requested. In
        # production a missing model or a neural runtime error must be visible.
        if allow_fallback is None:
            allow_fallback = not self.use_neural
        self.allow_fallback = bool(allow_fallback)

        self._model = None
        self._loaded: bool = False
        self._use_fallback: bool = not self.use_neural
        self._last_error: Optional[str] = None

    @property
    def backend_name(self) -> str:
        """Return the synthesis path that will be/was used: neural or fallback."""
        if not self._loaded:
            return "unloaded"
        return "fallback" if self._use_fallback else "neural"

    @property
    def last_error(self) -> Optional[str]:
        """Human-readable reason for the most recent neural load/synthesis error."""
        return self._last_error

    @classmethod
    def reset_shared_model(cls) -> None:
        """Drop the process-wide StyleTTS 2 model.

        Used by the service when a CUDA synthesis runs out of memory and the
        request is retried on CPU.  Without this, the CPU backend would reuse
        the CUDA-resident shared model.
        """
        cls._SHARED_MODEL = None
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    def load(self) -> None:
        """Load pretrained StyleTTS 2 checkpoint and freeze all parameters."""
        if self._loaded:
            return

        if not self.use_neural:
            self._use_fallback = True
            self._loaded = True
            return

        try:
            if StyleTTS2Backend._SHARED_MODEL is not None:
                shared = StyleTTS2Backend._SHARED_MODEL
                shared_device = getattr(shared, "device", None)
                if self.device.type == "cpu" and shared_device not in (None, "cpu"):
                    # A previous CUDA load left a GPU-resident shared model.
                    StyleTTS2Backend.reset_shared_model()
                else:
                    self._model = shared
            if self._model is None:
                from styletts2 import tts
                if self.device.type == "cpu":
                    orig_is_avail = torch.cuda.is_available
                    torch.cuda.is_available = lambda: False
                    try:
                        StyleTTS2Backend._SHARED_MODEL = tts.StyleTTS2(
                            model_checkpoint_path=self.checkpoint_path,
                            config_path=self.config_path,
                        )
                    finally:
                        torch.cuda.is_available = orig_is_avail
                else:
                    StyleTTS2Backend._SHARED_MODEL = tts.StyleTTS2(
                        model_checkpoint_path=self.checkpoint_path,
                        config_path=self.config_path,
                    )
                self._model = StyleTTS2Backend._SHARED_MODEL

            self._use_fallback = False
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            if not self.allow_fallback:
                raise RuntimeError(
                    "[StyleTTS2Backend] Failed to load the neural StyleTTS2 model. "
                    "Pass allow_fallback=True (or use_neural=False) to explicitly "
                    "opt into the offline formant fallback. "
                    f"Original error: {self._last_error}"
                ) from exc
            self._use_fallback = True

        self._loaded = True

    def compute_style(self, audio_path: str) -> "torch.Tensor":
        """Compute the 256-D StyleTTS2 reference style from real speech.

        The style and predictor encoders are frozen.  This is the face-free
        identity base used by the expression research track; it is not the
        canonical face-to-voice adapter, which expects a face embedding.
        """
        self.load()
        if self._use_fallback or self._model is None:
            raise RuntimeError(
                "[StyleTTS2Backend] the neural backend is required to compute "
                "a reference style"
            )
        style = self._model.compute_style(str(audio_path))
        if not isinstance(style, torch.Tensor):
            style = torch.as_tensor(style)
        if style.dim() == 1:
            style = style.unsqueeze(0)
        if style.shape[-1] != 256:
            raise RuntimeError(
                "[StyleTTS2Backend] expected a 256-D style vector, got "
                f"{tuple(style.shape)}"
            )
        return style.to(self.device).float()

    @staticmethod
    def _apply_biometric_prosody_bias(
        ref_s: "torch.Tensor",
        gender: int,
        age: float,
        strength: float = 0.15,
    ) -> "torch.Tensor":
        """Add a small face-derived F0-register residual to adapter prosody.

        StyleTTS2 conditions its F0 predictor on ``ref_s[:, 128:]``.  The old
        implementation replaced 80% of that half with a gender/age template and
        renormalised it, which erased the adapter's individual prosody and the
        emotion/accent/craniofacial deltas applied earlier in the pipeline.

        This version adds a bounded residual instead: ``prosody + strength *
        bias``.  At the default strength (0.15) the adapter remains dominant,
        while gender/age still provide a weak physiological prior.  Pass
        ``prosody_bias_strength=0.0`` for a pure adapter/control path.

        Args:
            ref_s:  (B, 256) style tensor from the StyleAdapter.
            gender: 0 = female, 1 = male.
            age:    Estimated age in years.
            strength: Residual magnitude in [0, 1].

        Returns:
            A new tensor with the residual added to the prosody half.
        """
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return ref_s

        ref_s = ref_s.clone()
        dev = ref_s.device
        dtype = ref_s.dtype
        B = ref_s.shape[0]

        # Continuous F0 register target; no hard gender switch in the profile.
        age_offset = max(0.0, float(age) - 20.0)
        if gender == 0:
            f0_target = max(170.0, min(260.0, 220.0 - age_offset * 1.0))
            f0_norm = (f0_target - 170.0) / (260.0 - 170.0)
            dc = 0.85
        else:
            f0_target = max(90.0, min(155.0, 125.0 - age_offset * 0.5))
            f0_norm = (f0_target - 90.0) / (155.0 - 90.0)
            dc = -0.85

        prosody_bias = torch.zeros(B, 128, device=dev, dtype=dtype)
        for i in range(128):
            phase = 2.0 * math.pi * i / 127.0
            prosody_bias[:, i] = (
                dc
                + 0.40 * math.sin(phase + f0_norm * math.pi)
                + 0.20 * math.cos(2.0 * phase)
                + 0.10 * math.sin(4.0 * phase + f0_norm * 0.5)
            )
        target_norm = 3.20 if gender == 0 else 2.20
        cur_norm = prosody_bias.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        prosody_bias = prosody_bias / cur_norm * target_norm

        ref_s[:, 128:] = (ref_s[:, 128:] + strength * prosody_bias).clamp(
            -3.0, 3.0
        )
        return ref_s

    def synthesize(
        self,
        text: str,
        style: torch.Tensor,
        gender: int = 1,
        age: float = 30.0,
        alpha: float = 0.18,
        beta: float = 0.22,
        duration_scale: float = 1.0,
        diffusion_steps: int = 5,
        loudness_lufs: float = -23.0,
        target_f0: float | None = None,
        f0_scale: float = 1.0,
        prosody_bias_strength: float = 0.15,
    ) -> torch.Tensor:
        """Synthesize 24 kHz speech waveform from text and style vector.

        Every argument in this signature has a real effect on the output:
        ``duration_scale`` time-stretches, ``f0_scale``/``target_f0`` pitch-shift,
        ``loudness_lufs`` performs EBU R128 normalization, and
        ``prosody_bias_strength`` controls the gender/age residual.  Parameters
        that could not be implemented (anchor blending, F0-variance scaling)
        were removed rather than silently ignored.

        Args:
            text: Phoneme string or text prompt to speak.
            style: 256-D style conditioning vector from StyleAdapter.
            gender: Face-detected gender (0 = Female, 1 = Male).
            age: Face-detected age in years. Used for the F0-register residual.
            alpha: Timbre diffusion weight.
            beta: Prosody transfer weight.
            duration_scale: Speech pace multiplier (>1 = longer).
            diffusion_steps: Number of ODE diffusion sampling steps.
            loudness_lufs: EBU R128 integrated-loudness target.
            target_f0: Optional absolute median F0 in Hz (neural path only).
            f0_scale: Multiplicative F0 control (neural path only).
            prosody_bias_strength: Gender/age residual magnitude in [0, 1].

        Returns:
            waveform: Float32 Tensor of shape (1, num_samples) at self.sample_rate Hz.
        """
        if not self._loaded:
            self.load()

        if style.dim() == 1:
            style = style.unsqueeze(0)

        assert style.shape[-1] in (128, 256), (
            f"Expected style vector of dimension 128 or 256, got {style.shape[-1]}"
        )

        style = style.to(self.device)

        if not self._use_fallback and self._model is not None:
            try:
                style_proj = style.to(self._model.device).float()
                if style_proj.dim() == 1:
                    style_proj = style_proj.unsqueeze(0)

                # Build 256-D reference style tensor [timbre (128) | prosody (128)]
                if style_proj.shape[-1] == 256:
                    ref_s = style_proj.clone()
                else:
                    # Legacy single-head: duplicate timbre as placeholder prosody
                    ref_s = torch.cat([style_proj, style_proj], dim=-1)

                # Small gender/age F0-register residual; adapter prosody and
                # emotion/accent deltas remain dominant.
                ref_s = self._apply_biometric_prosody_bias(
                    ref_s=ref_s,
                    gender=int(gender),
                    age=float(age),
                    strength=prosody_bias_strength,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    wav = self._model.inference(
                        text=text,
                        ref_s=ref_s,
                        alpha=alpha,
                        beta=beta,
                        diffusion_steps=diffusion_steps,
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not isinstance(wav, torch.Tensor):
                    wav = torch.from_numpy(wav)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)

                # Real, measurable controls: duration then pitch.  Both are
                # applied before loudness normalization so the LUFS target wins.
                wav = _time_stretch(wav, self.sample_rate, duration_scale)
                if target_f0 is not None and float(target_f0) > 0.0:
                    current = _estimate_f0(wav, self.sample_rate)
                    if current is not None and current > 1e-3:
                        semitones = 12.0 * math.log2(float(target_f0) / current)
                        wav = _pitch_shift(
                            wav, self.sample_rate, float(np.clip(semitones, -24.0, 24.0))
                        )
                elif abs(float(f0_scale) - 1.0) > 1e-4:
                    semitones = 12.0 * math.log2(max(1e-3, float(f0_scale)))
                    wav = _pitch_shift(
                        wav, self.sample_rate, float(np.clip(semitones, -24.0, 24.0))
                    )

                # EBU R128 integrated-loudness normalization. The previous
                # implementation peak-normalized here despite claiming LUFS.
                wav = _lufs_normalize(wav, self.sample_rate, target_lufs=loudness_lufs)

                return wav.to(self.device).float()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                if not self.allow_fallback:
                    raise RuntimeError(
                        "[StyleTTS2Backend] Neural synthesis failed. Refusing to "
                        "silently substitute the offline formant fallback. "
                        f"Original error: {self._last_error}"
                    ) from exc
                print(
                    "[StyleTTS2Backend] WARNING: neural synthesis failed; using "
                    f"explicitly-allowed fallback ({self._last_error})"
                )

        # Deterministic acoustic waveform synthesizer fallback
        return self._synthesize_fallback(
            text,
            style,
            duration_scale=duration_scale,
            target_f0=target_f0,
            f0_scale=f0_scale,
            loudness_lufs=loudness_lufs,
        )

    def _synthesize_fallback(
        self,
        text: str,
        style: torch.Tensor,
        duration_scale: float = 1.0,
        target_f0: float | None = None,
        f0_scale: float = 1.0,
        loudness_lufs: float = -23.0,
    ) -> torch.Tensor:
        """Acoustic formant generator driven by style vector features."""
        # Estimate utterance duration from text length: ~0.06s per character
        char_count = max(5, len(text))
        duration_sec = min(8.0, max(0.8, char_count * 0.06 * duration_scale))
        num_samples = int(duration_sec * self.sample_rate)

        # Base fundamental frequency f0 modulated by style vector (e.g. pitch register)
        # style[0] controls gender/pitch register: ~120 Hz (male) to ~220 Hz (female)
        style_pitch = float(torch.tanh(style[0, 0]).item())
        f0 = 160.0 + style_pitch * 50.0  # Range: [110, 210] Hz
        if target_f0 is not None and float(target_f0) > 0.0:
            f0 = float(target_f0)
        else:
            f0 *= float(f0_scale)
        f0 = float(np.clip(f0, 60.0, 400.0))

        # Generate harmonic time series
        t = torch.linspace(0, duration_sec, num_samples, device=self.device)

        # Multi-harmonic vocal tract pulse train
        harmonics = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [1.0, 0.6, 0.4, 0.25, 0.15]
        waveform = torch.zeros(num_samples, device=self.device)

        for h, w in zip(harmonics, weights):
            # Style vector dimensions 1..5 modulate individual formant energy
            formant_mod = 1.0 + 0.2 * torch.tanh(style[0, int(h)]).item()
            waveform += (w * formant_mod) * torch.sin(2.0 * math.pi * f0 * h * t)

        # Smooth envelope (attack, sustain, release) to eliminate clicks
        fade_len = int(0.05 * self.sample_rate)
        envelope = torch.ones(num_samples, device=self.device)
        fade_in = torch.linspace(0.0, 1.0, fade_len, device=self.device)
        fade_out = torch.linspace(1.0, 0.0, fade_len, device=self.device)
        envelope[:fade_len] = fade_in
        envelope[-fade_len:] = fade_out

        waveform = waveform * envelope

        # Same loudness contract as the neural path.  The fallback used to
        # peak-normalize while claiming LUFS, which made loudness tests pass
        # for the wrong reason.
        return _lufs_normalize(
            waveform.unsqueeze(0), self.sample_rate, target_lufs=loudness_lufs
        ).float()

