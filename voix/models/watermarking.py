"""AudioSeal neural watermarking wrapper.

Proactively embeds imperceptible 16-bit neural watermarks into all VOIX
synthesized audio. Every generated waveform is cryptographically attributable
as AI-generated content with AUC=0.97 and sample-level IoU=0.99.

Reference: San Roman et al. (2024). "Proactive Detection of Voice Cloning
with Localized Watermarking." ICML 2024.
"""

from __future__ import annotations

from typing import Tuple


class AudioSealWatermarker:
    """AudioSeal neural watermark generator and detector.

    Embeds a 16-bit message into synthesized audio waveforms with negligible
    perceptual distortion. Embedded watermarks survive MP3 compression,
    cropping, and minor tempo changes.

    Args:
        sample_rate: Audio sample rate in Hz. Default 24000 for StyleTTS 2.
        device: PyTorch device string. "cuda" or "cpu".
        nbits: Number of watermark message bits. Default 16.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        device: str = "cuda",
        nbits: int = 16,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.nbits = nbits
        self._generator = None
        self._detector = None
        self._loaded = False

    def _load(self) -> None:
        """Lazy-load AudioSeal generator and detector models."""
        from audioseal import AudioSeal

        self._generator = AudioSeal.load_generator("audioseal_wm_16bits")
        self._generator = self._generator.to(self.device)
        self._generator.eval()

        self._detector = AudioSeal.load_detector("audioseal_detector_16bits")
        self._detector = self._detector.to(self.device)
        self._detector.eval()

        self._loaded = True

    def embed(self, audio: "torch.Tensor") -> "torch.Tensor":
        """Embed neural watermark into synthesized audio.

        Args:
            audio: Float32 tensor of shape (1, T) or (B, T) at self.sample_rate Hz.

        Returns:
            Watermarked float32 tensor of same shape as input.
            Watermark is imperceptible (SNR degradation < 0.5 dB).
        """
        import torch

        if not self._loaded:
            self._load()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        # Ensure shape is (B, 1, T) for AudioSeal
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        with torch.no_grad():
            watermarked = self._generator(audio, sample_rate=self.sample_rate)

        # Return in (B, T) format
        return watermarked.squeeze(1)

    def detect(self, audio: "torch.Tensor") -> Tuple[bool, float]:
        """Detect presence of AudioSeal watermark in audio.

        Args:
            audio: Float32 tensor of shape (1, T) or (B, T).

        Returns:
            Tuple of (is_watermarked: bool, confidence: float in [0, 1]).
            Confidence > 0.5 indicates synthetic generation.
        """
        import torch

        if not self._loaded:
            self._load()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        with torch.no_grad():
            result, _ = self._detector.detect_watermark(
                audio, sample_rate=self.sample_rate
            )

        confidence = float(result.mean().item())
        return confidence > 0.5, confidence

    def verify_batch(self, audio_list: list["torch.Tensor"]) -> list[bool]:
        """Verify watermark presence across a batch of audio tensors.

        Args:
            audio_list: List of audio tensors.

        Returns:
            List of booleans indicating watermark presence per sample.
        """
        return [self.detect(audio)[0] for audio in audio_list]
