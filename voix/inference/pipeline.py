"""End-to-End Face-to-Voice Inference Pipeline for VOIX (Phase 4).

Connects the full generative stack:
    Face Image / Embedding (560-D)
        -> CVAE Probabilistic Mapper (192-D speaker embeddings, K samples)
        -> StyleAdapter (128-D StyleTTS 2 style space)
        -> StyleTTS 2 Neural Vocoder (24 kHz speech synthesis)
        -> AudioSeal Neural Watermarking (provenance embedding)
        -> Auditory Speech Waveforms (Voice 1, Voice 2, ..., Voice K)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from voix.models.adapter import StyleAdapter
from voix.models.cvae import CVAE
from voix.models.tts_backend import StyleTTS2Backend
from voix.models.watermarking import AudioSealWatermarker


class VoixPipeline:
    """Unified end-to-end VOIX synthesis pipeline.

    Args:
        cvae_checkpoint: Path to trained CVAE weights (default: D:/voix/checkpoints/cvae_best.pt).
        adapter_checkpoint: Path to trained StyleAdapter weights (default: D:/voix/checkpoints/style_adapter_best.pt).
        scaler_path: Path to feature normalisation scaler (default: D:/voix/data/processed/feature_scaler.pt).
        device: Execution device ("cuda" or "cpu").
        enable_watermarking: Whether to pass audio through AudioSeal (default: True).
    """

    def __init__(
        self,
        cvae_checkpoint: str = "D:/voix/checkpoints/cvae_best.pt",
        adapter_checkpoint: str = "D:/voix/checkpoints/style_adapter_best.pt",
        scaler_path: str = "D:/voix/data/processed/feature_scaler.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_watermarking: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.enable_watermarking = enable_watermarking

        # 1. Load Scaler
        self.scaler = None
        if os.path.exists(scaler_path):
            self.scaler = torch.load(scaler_path, weights_only=True)
            self.face_mean = self.scaler["face_mean"].to(self.device)
            self.face_std  = self.scaler["face_std"].to(self.device)
            self.spk_mean  = self.scaler["spk_mean"].to(self.device)
            self.spk_std   = self.scaler["spk_std"].to(self.device)

        # 2. Load CVAE
        self.cvae = CVAE(face_dim=560, speaker_dim=192, latent_dim=192).to(self.device)
        if os.path.exists(cvae_checkpoint):
            ckpt = torch.load(cvae_checkpoint, weights_only=False, map_location=self.device)
            self.cvae.load_state_dict(ckpt["model_state_dict"])
        self.cvae.eval()

        # 3. Load StyleAdapter
        self.adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128).to(self.device)
        if os.path.exists(adapter_checkpoint):
            a_ckpt = torch.load(adapter_checkpoint, weights_only=False, map_location=self.device)
            self.adapter.load_state_dict(a_ckpt["model_state_dict"])
        self.adapter.eval()

        # 4. Load Synthesis Backend
        self.tts = StyleTTS2Backend(device=str(self.device))

        # 5. Load Watermarker
        self.watermarker = AudioSealWatermarker(sample_rate=24000, device=str(self.device))

        # 6. Lazy extractors for raw face image input
        self._arcface = None
        self._facemesh = None
        self._demog = None
        self._fusion = None

    def extract_face_features(self, image_input: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """Extract and fuse facial features (ArcFace 512 + FaceMesh 32 + Demographics 16 -> 560-D) from an image.

        Args:
            image_input: File path to image or BGR numpy array.

        Returns:
            torch.Tensor: Fused 560-D face representation of shape (1, 560) on self.device.
        """
        import cv2
        from voix.data.face_extractor import ArcFaceExtractor, FaceMeshExtractor
        from voix.data.morphology import compute_craniofacial_ratios
        from voix.data.demographics import SoftDemographicEstimator
        from voix.models.fusion import FaceFusionLayer

        if self._arcface is None:
            self._arcface = ArcFaceExtractor(
                device_id=0 if self.device.type == "cuda" else -1,
                cache_dir="D:/voix/checkpoints",
            )
        if self._facemesh is None:
            task_path = "D:/voix/checkpoints/mediapipe/face_landmarker.task"
            self._facemesh = FaceMeshExtractor(
                model_path=task_path if os.path.exists(task_path) else None,
            )
        if self._demog is None:
            self._demog = SoftDemographicEstimator()
        if self._fusion is None:
            self._fusion = FaceFusionLayer().to(self.device)

        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise TypeError("image_input must be a file path or numpy array")

        # 1. ArcFace 512-D identity
        e_id = self._arcface.extract(img_bgr)
        if e_id is None:
            raise ValueError(f"No face detected in image: {image_input}")

        # 2. FaceMesh 32-D morphology
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self._facemesh.extract(img_rgb)
        if landmarks is not None:
            e_geo = compute_craniofacial_ratios(landmarks)
        else:
            e_geo = np.zeros(32, dtype=np.float32)

        # 3. Soft Demographics 16-D
        e_demo = self._demog.estimate(e_id)

        # 4. Fusion 560-D
        with torch.no_grad():
            e_id_t = torch.from_numpy(e_id).unsqueeze(0).to(self.device)
            e_geo_t = torch.from_numpy(e_geo).unsqueeze(0).to(self.device)
            e_demo_t = torch.from_numpy(e_demo).unsqueeze(0).to(self.device)
            e_f = self._fusion(e_id_t, e_geo_t, e_demo_t)

        return e_f

    @torch.no_grad()
    def generate(
        self,
        face_input: Union[str, np.ndarray, torch.Tensor],
        text: str = "Welcome to VOIX. This voice was synthesized directly from your facial biometrics.",
        num_voices: int = 3,
        temperature: float = 1.0,
        watermark: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Generate K diverse candidate voice waveforms for a single face representation.

        Args:
            face_input: 560-D face vector (torch.Tensor, np.ndarray) or path to face image.
            text: Text prompt to synthesize.
            num_voices: Number of diverse voice candidates to generate (K).
            temperature: Sampling diversity temperature.
            watermark: Explicit flag to enable/disable AudioSeal watermarking.

        Returns:
            List of dicts, each containing:
              - voice_index: int (1..K)
              - waveform: torch.Tensor of shape (1, T) at 24 kHz
              - sample_rate: int (24000)
              - speaker_embedding: torch.Tensor (192,)
              - style_embedding: torch.Tensor (128,)
              - is_watermarked: bool
        """
        if watermark is None:
            watermark = self.enable_watermarking

        # Process face input into 560-D normalized tensor
        if isinstance(face_input, (str, Path)):
            e_f = self.extract_face_features(str(face_input))
            if self.scaler is not None:
                e_f = (e_f - self.face_mean) / self.face_std
        elif isinstance(face_input, np.ndarray):
            if face_input.ndim == 3:  # HxWx3 image
                e_f = self.extract_face_features(face_input)
                if self.scaler is not None:
                    e_f = (e_f - self.face_mean) / self.face_std
            else:
                e_f = torch.from_numpy(face_input).float()
                if e_f.dim() == 1:
                    e_f = e_f.unsqueeze(0)
                e_f = e_f.to(self.device)
        elif isinstance(face_input, torch.Tensor):
            e_f = face_input.float()
            if e_f.dim() == 1:
                e_f = e_f.unsqueeze(0)
            e_f = e_f.to(self.device)
        else:
            raise TypeError("face_input must be an image path, 560-D tensor, or numpy array")

        # 1. Generate K candidate speaker embeddings via CVAE
        e_s_candidates_norm = self.cvae.sample_voices(
            e_f,
            num_samples=num_voices,
            temperature=temperature,
        )  # (K, 192)

        # Convert to raw ECAPA space if scaler is attached
        if self.scaler is not None:
            e_s_candidates_raw = e_s_candidates_norm * self.spk_std + self.spk_mean
        else:
            e_s_candidates_raw = e_s_candidates_norm

        results = []

        # Free cached VRAM from feature extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. For each candidate voice, project style and synthesize speech
        for k in range(num_voices):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            e_s_k = e_s_candidates_norm[k].unsqueeze(0)  # (1, 192)
            e_s_raw_k = e_s_candidates_raw[k]

            # Module 3: Style Adapter projection
            s_hat_k = self.adapter(e_s_k)  # (1, 128)

            # Module 4: StyleTTS 2 Speech Synthesis
            waveform = self.tts.synthesize(text=text, style=s_hat_k)  # (1, T)

            # Module 5: AudioSeal Watermarking
            is_wm = False
            if watermark:
                try:
                    waveform = self.watermarker.embed(waveform)
                    is_wm = True
                except Exception:
                    is_wm = False

            results.append({
                "voice_index": k + 1,
                "waveform": waveform.cpu(),
                "sample_rate": self.tts.sample_rate,
                "speaker_embedding": e_s_raw_k.cpu(),
                "style_embedding": s_hat_k.squeeze(0).cpu(),
                "is_watermarked": is_wm,
            })

        return results

    def save_audio(
        self,
        results: List[Dict[str, Any]],
        output_dir: str = "D:/voix/artifacts/samples",
        prefix: str = "voice",
    ) -> List[str]:
        """Save generated candidate waveforms to disk as 24 kHz WAV files.

        Args:
            results: List of result dicts from self.generate().
            output_dir: Destination directory.
            prefix: Filename prefix.

        Returns:
            List of written file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for item in results:
            idx = item["voice_index"]
            wav = item["waveform"].squeeze(0).numpy()
            sr = item["sample_rate"]
            path = os.path.join(output_dir, f"{prefix}_{idx}.wav")
            sf.write(path, wav, sr, format="WAV", subtype="PCM_16")
            saved_paths.append(path)

        return saved_paths
