"""SpeechBrain ECAPA-TDNN speaker embedding extractor.

Produces 192-D speaker embeddings in the ECAPA-TDNN acoustic identity space,
trained on VoxCeleb1+2 for speaker verification. All parameters are frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ECAPAExtractor:
    """Frozen ECAPA-TDNN speaker embedding extractor.

    Extracts 192-D speaker embeddings from raw 16 kHz mono waveforms.
    The embedding space is trained for speaker identity discrimination
    via angular margin softmax loss on VoxCeleb1+2.

    Args:
        model_source: HuggingFace model identifier or local path.
        cache_dir: Directory to cache downloaded SpeechBrain model.
        device: PyTorch device string. "cuda" or "cpu".
        savedir: Local directory for SpeechBrain model artifacts.
    """

    EMBEDDING_DIM: int = 192
    SAMPLE_RATE: int = 16000

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        cache_dir: str = "./checkpoints/speechbrain",
        device: str = "cuda",
        savedir: str = "./checkpoints/speechbrain/spkrec-ecapa",
    ) -> None:
        self.model_source = model_source
        self.cache_dir = str(Path(cache_dir).resolve())
        self.device = device
        self.savedir = str(Path(savedir).resolve())
        self._classifier = None
        self._loaded = False

    def _load(self) -> None:
        """Lazy-load SpeechBrain ECAPA-TDNN model on first use."""
        import torch
        from speechbrain.pretrained import EncoderClassifier

        self._classifier = EncoderClassifier.from_hparams(
            source=self.model_source,
            savedir=self.savedir,
            run_opts={"device": self.device},
        )
        # Freeze all parameters
        for param in self._classifier.parameters():
            param.requires_grad = False
        self._classifier.eval()
        self._loaded = True

    def extract(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Extract 192-D speaker embedding from a 16 kHz waveform.

        Args:
            waveform: Float32 tensor of shape (1, T) or (T,) at 16 kHz.
                      T is the number of audio samples.

        Returns:
            Float32 tensor of shape (1, 192) representing the speaker
            embedding in ECAPA-TDNN acoustic identity space.
        """
        import torch

        if not self._loaded:
            self._load()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        with torch.no_grad():
            embedding = self._classifier.encode_batch(waveform)

        # Shape: (1, 1, 192) -> (1, 192)
        embedding = embedding.squeeze(1)
        assert embedding.shape[-1] == self.EMBEDDING_DIM, (
            f"Expected {self.EMBEDDING_DIM}-D embedding, got {embedding.shape[-1]}"
        )
        return embedding.float()

    def extract_from_path(self, audio_path: str) -> "torch.Tensor":
        """Load a WAV file and extract speaker embedding.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.

        Returns:
            Float32 tensor of shape (1, 192).
        """
        import torchaudio

        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return self.extract(waveform)
