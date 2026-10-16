from __future__ import annotations

from voix.models.speech_simulation import SpeechSimulator
from voix.models.accent_control import AccentTransformer, list_available_accents
from voix.models.emotion_control import EmotionModulator, list_available_emotions
from voix.models.cross_lingual import MultilingualPhoneticAdapter, list_supported_languages
"""End-to-End Face-to-Voice Inference Pipeline for VOIX (Phase 4 & 7).

Connects the full generative stack with dynamic cross-platform hardware auto-configuration:
    Face Image / Embedding (560-D)
        -> CVAE / Contrastive Mapper (192-D speaker embeddings, K samples)
        -> StyleAdapter (128-D StyleTTS 2 style space)
        -> StyleTTS 2 Neural Vocoder (24 kHz speech synthesis, CUDA / MPS / CPU adaptive)
        -> AudioSeal Neural Watermarking (provenance embedding)
        -> Auditory Speech Waveforms (Voice 1, Voice 2, ..., Voice K)
"""


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
from voix.utils.hardware import HardwareProfile, detect_hardware, free_memory


class VoixPipeline:
    """Unified end-to-end VOIX synthesis pipeline with hardware auto-configuration.

    Args:
        cvae_checkpoint: Path to trained CVAE / Mapper weights.
        adapter_checkpoint: Path to trained StyleAdapter weights.
        scaler_path: Path to feature normalisation scaler.
        device: Optional execution device override ("cuda", "mps", "cpu", or None for auto).
        enable_watermarking: Whether to pass audio through AudioSeal (default: True).
        hardware_profile: Optional pre-configured HardwareProfile. If None, auto-detected.
    """

    def __init__(
        self,
        cvae_checkpoint: str = "D:/voix/results/checkpoints/D_joint_infonce_best.pt",
        adapter_checkpoint: str = "D:/voix/checkpoints/style_adapter_best.pt",
        scaler_path: str = "D:/voix/data/processed/face_scaler.pt",
        device: Optional[str] = None,
        enable_watermarking: bool = True,
        hardware_profile: Optional[HardwareProfile] = None,
    ) -> None:
        # 1. Resolve Hardware Auto-Configuration
        self.hardware = hardware_profile or detect_hardware()
        if device is not None:
            self.device = torch.device(device)
            tts_device = device
        else:
            self.device = self.hardware.device
            tts_device = self.hardware.tts_device

        self.enable_watermarking = enable_watermarking

        # 2. Load Scaler (supports both face_scaler.pt and legacy feature_scaler.pt)
        self.scaler = None
        alt_scaler = "D:/voix/data/processed/feature_scaler.pt"
        target_scaler = scaler_path if os.path.exists(scaler_path) else alt_scaler
        if os.path.exists(target_scaler):
            self.scaler = torch.load(target_scaler, weights_only=False)
            if "face_mean" in self.scaler:
                self.face_mean = self.scaler["face_mean"].to(self.device)
                self.face_std  = self.scaler["face_std"].to(self.device)
            elif "mean" in self.scaler:
                self.face_mean = self.scaler["mean"].to(self.device)
                self.face_std  = self.scaler["scale"].to(self.device)
            self.spk_mean  = self.scaler.get("spk_mean", torch.zeros(192)).to(self.device)
            self.spk_std   = self.scaler.get("spk_std", torch.ones(192)).to(self.device)

        # 3. Load CVAE / Mapper
        self.cvae = CVAE(face_dim=560, speaker_dim=192, latent_dim=192).to(self.device)
        if os.path.exists(cvae_checkpoint):
            ckpt = torch.load(cvae_checkpoint, weights_only=False, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.cvae.load_state_dict(state_dict, strict=False)
        self.cvae.eval()

        # 4. Load StyleAdapter
        self.adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=128).to(self.device)
        if os.path.exists(adapter_checkpoint):
            a_ckpt = torch.load(adapter_checkpoint, weights_only=False, map_location=self.device)
            a_state = a_ckpt["model_state_dict"] if "model_state_dict" in a_ckpt else a_ckpt
            self.adapter.load_state_dict(a_state, strict=False)
        self.adapter.eval()

        # 5. Load Synthesis Backend (Adaptive via Hardware Auto-Configuration)
        self.tts = StyleTTS2Backend(device=tts_device)

        # 6. Load Watermarker
        self.watermarker = AudioSealWatermarker(sample_rate=24000, device=str(self.device))

        # 7. Load inference config & pre-load anchors
        from voix.utils.config import load_config
        _cfg = load_config()
        self._inf_cfg = _cfg.get("inference", {})
        self.tts.load_anchors_from_config(_cfg)

        # 8. Lazy extractors for raw face image input
        self._arcface = None
        self._facemesh = None
        self._demog = None
        self._fusion = None

    def extract_face_features(
        self,
        image_input: Union[str, Path, np.ndarray],
        craniofacial_sliders: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
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
            # Use CPUExecutionProvider for ArcFace on budget GPUs to preserve VRAM for vocoder
            use_gpu_arc = (
                self.device.type == "cuda"
                and self.hardware.onnx_execution_provider == "CUDAExecutionProvider"
            )
            self._arcface = ArcFaceExtractor(
                device_id=0 if use_gpu_arc else -1,
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

        # 1. ArcFace 512-D identity and demographic posteriors
        e_id, gender, age = self._arcface.extract_full(img_bgr)
        if e_id is None:
            raise ValueError(f"No face detected in image: {image_input}")

        # 2. FaceMesh 32-D morphology
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self._facemesh.extract(img_rgb)
        if landmarks is not None:
            e_geo = compute_craniofacial_ratios(landmarks)
        else:
            e_geo = np.zeros(32, dtype=np.float32)

        if craniofacial_sliders:
            from voix.data.morphology import apply_craniofacial_sliders
            e_geo = apply_craniofacial_sliders(e_geo, craniofacial_sliders)

        # 3. Soft Demographics 16-D with real age & gender posteriors
        e_demo = self._demog.estimate(e_id, gender=gender, age=age)

        # 4. Fusion 560-D
        with torch.no_grad():
            e_id_t = torch.from_numpy(e_id).unsqueeze(0).to(self.device)
            e_geo_t = torch.from_numpy(e_geo).unsqueeze(0).to(self.device)
            e_demo_t = torch.from_numpy(e_demo).unsqueeze(0).to(self.device)
            e_f = self._fusion(e_id_t, e_geo_t, e_demo_t)

        return e_f, int(gender) if gender is not None else 1, float(age) if age is not None else 30.0

    @torch.no_grad()
    def generate(
        self,
        face_input: Union[str, np.ndarray, torch.Tensor],
        text: str = "Welcome to VOIX. This voice was synthesized directly from your facial biometrics.",
        num_voices: int = 3,
        temperature: float = 1.0,
        watermark: Optional[bool] = None,
        gender: Optional[int] = None,
        age: Optional[float] = None,
        craniofacial_sliders: Optional[Dict[str, float]] = None,
        speech_difference: Optional[str] = None,
        speech_severity: float = 0.5,
        accent: Optional[str] = None,
        accent_strength: float = 0.5,
        emotion: Optional[str] = None,
        emotion_intensity: float = 0.5,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """Generate K diverse candidate voice waveforms for a single face representation.

        Args:
            face_input: 560-D face vector (torch.Tensor, np.ndarray) or path to face image.
            text: Text prompt to synthesize.
            num_voices: Number of diverse voice candidates to generate (K).
            temperature: Sampling diversity temperature.
            watermark: Explicit flag to enable/disable AudioSeal watermarking.

        Returns:
            List of dicts containing voice waveforms, speaker embeddings, and metadata.
        """
        if watermark is None:
            watermark = self.enable_watermarking

        # Process face input into 560-D normalized tensor
        if isinstance(face_input, (str, Path)):
            str_path = str(face_input)
            if not os.path.exists(str_path) and not Path(str_path).is_file():
                # Check if string corresponds to a demographic preset ID
                try:
                    preset = self.get_demographic_preset(str_path)
                    e_f = preset.embedding.to(self.device)
                    if gender is None: gender = preset.gender
                    if age is None: age = preset.age
                    if self.scaler is not None:
                        e_f = (e_f - self.face_mean) / self.face_std
                except KeyError:
                    raise FileNotFoundError(f"Image file or preset not found: {str_path}")
            else:
                e_f, _det_gender, _det_age = self.extract_face_features(
                    str_path,
                    craniofacial_sliders=craniofacial_sliders,
                )
                if gender is None: gender = _det_gender
                if age is None: age = _det_age
                if self.scaler is not None:
                    e_f = (e_f - self.face_mean) / self.face_std
        elif isinstance(face_input, np.ndarray):
            if face_input.ndim == 3:  # HxWx3 image
                e_f, _det_gender, _det_age = self.extract_face_features(
                    face_input,
                    craniofacial_sliders=craniofacial_sliders,
                )
                if gender is None: gender = _det_gender
                if age is None: age = _det_age
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
            if self.scaler is not None and e_f.abs().mean() > 1.5:
                e_f = (e_f - self.face_mean) / self.face_std
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

        # Staged Memory Isolation: Release visual extractors to free VRAM before vocoding
        if self.hardware.pipeline_mode == "staged_memory_isolated":
            self._arcface = None
            self._facemesh = None
            self._demog = None
            self._fusion = None
            free_memory(self.device)

        # 2. Text preprocessing: Cross-lingual adaptation, Accent, Speech Difference, and Emotion modulation
        lang_mods = {"duration_scale": 1.0, "pitch_dynamics_factor": 1.0}
        text_lang = text
        if language and language.lower() != "en":
            text_lang, lang_mods = MultilingualPhoneticAdapter.transform(text, language=language)

        text_accented = text_lang
        accent_mods = {"duration_mult": 1.0, "pitch_mult": 1.0, "prosody_mult": 1.0}
        if accent:
            text_accented, accent_mods = AccentTransformer.transform(
                text=text_lang,
                accent=accent,
                strength=accent_strength,
            )

        duration_mult = 1.0
        synth_text = text_accented
        if speech_difference and speech_severity > 0.001:
            synth_text, duration_mult = SpeechSimulator.process_text(
                text=text_accented,
                difference_type=speech_difference,
                severity=speech_severity,
            )

        emotion_mods = EmotionModulator.get_acoustic_modifiers(
            emotion=emotion,
            intensity=emotion_intensity,
        )
        # 3. For each candidate voice, project style and synthesize speech
        _inf = self._inf_cfg
        diffusion_steps = _inf.get("diffusion_steps", self.hardware.diffusion_steps)

        for k in range(num_voices):
            if self.hardware.torch_empty_cache_per_sample:
                free_memory(self.device)

            e_s_k = e_s_candidates_norm[k].unsqueeze(0)
            e_s_raw_k = e_s_candidates_raw[k]

            # Style Adapter projection (192-D -> 128-D)
            s_hat_k = self.adapter(e_s_raw_k.unsqueeze(0))

            _gender = gender if gender is not None else 1
            _age    = age if age is not None else 30.0
            from voix.utils.config import get_anchor_path, select_anchor_key
            _anchor_key = select_anchor_key(_gender, _age)
            try:
                _anchor_path = get_anchor_path(_anchor_key)
            except (KeyError, FileNotFoundError):
                _anchor_path = None

            base_dur = _inf.get("duration_scale", 1.0)
            final_duration = base_dur * duration_mult * accent_mods["duration_mult"] * emotion_mods["duration_scale"] * lang_mods["duration_scale"]
            base_beta = _inf.get("beta", 0.70)
            final_beta = float(np.clip(base_beta * accent_mods["prosody_mult"] * emotion_mods["beta_mult"] * lang_mods["pitch_dynamics_factor"], 0.20, 0.95))
            waveform = self.tts.synthesize(
                text=synth_text,
                style=s_hat_k,
                gender=_gender,
                blend_weight=_inf.get("blend_weight", 0.35),
                beta=final_beta,
                alpha=_inf.get("alpha", 0.25),
                duration_scale=final_duration,
                diffusion_steps=diffusion_steps,
                anchor_path=_anchor_path,
            )

            # Apply post-synthesis acoustic filters for speech differences
            if speech_difference and speech_severity > 0.001:
                waveform = SpeechSimulator.process_waveform(
                    waveform=waveform,
                    sample_rate=self.tts.sample_rate,
                    difference_type=speech_difference,
                    severity=speech_severity,
                )

            # Apply post-synthesis acoustic filters for emotion
            if emotion and emotion_intensity > 0.01:
                waveform = EmotionModulator.apply_acoustic_filter(
                    waveform=waveform,
                    sample_rate=self.tts.sample_rate,
                    emotion=emotion,
                    intensity=emotion_intensity,
                )
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

    def list_demographic_presets(self) -> List[Dict[str, Any]]:
        """Return list of available reference demographic presets."""
        from voix.data.presets import list_demographic_presets
        return list_demographic_presets()

    def get_demographic_preset(self, preset_id: str):
        """Retrieve a specific demographic preset by ID."""
        from voix.data.presets import get_demographic_preset
        return get_demographic_preset(preset_id)

    def generate_from_preset(
        self,
        preset_id: str,
        text: str = "Welcome to VOIX. This voice was synthesized directly from your facial biometrics.",
        num_voices: int = 3,
        temperature: float = 1.0,
        watermark: Optional[bool] = None,
        craniofacial_sliders: Optional[Dict[str, float]] = None,
        speech_difference: Optional[str] = None,
        speech_severity: float = 0.0,
        accent: Optional[str] = None,
        accent_strength: float = 0.5,
        emotion: Optional[str] = None,
        emotion_intensity: float = 0.5,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """Synthesize candidate voices directly from a demographic preset ID."""
        preset = self.get_demographic_preset(preset_id)
        return self.generate(
            face_input=preset.embedding,
            text=text,
            num_voices=num_voices,
            temperature=temperature,
            watermark=watermark,
            gender=preset.gender,
            age=preset.age,
            craniofacial_sliders=craniofacial_sliders,
            speech_difference=speech_difference,
            speech_severity=speech_severity,
            accent=accent,
            accent_strength=accent_strength,
            emotion=emotion,
            emotion_intensity=emotion_intensity,
            language=language,
        )

    def list_supported_speech_differences(self) -> List[Dict[str, Any]]:
        """Return list of supported speech difference simulation modes."""
        from voix.models.speech_simulation import SpeechSimulator
        return SpeechSimulator.list_supported_differences()

    def save_audio(
        self,
        results: List[Dict[str, Any]],
        output_dir: str = "D:/voix/artifacts/samples",
        prefix: str = "voice",
    ) -> List[str]:
        """Save generated candidate waveforms to disk as 24 kHz WAV files."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for item in results:
            idx = item["voice_index"]
            wav = item["waveform"].squeeze().numpy()
            sr = item["sample_rate"]
            path = os.path.join(output_dir, f"{prefix}_{idx}.wav")
            sf.write(path, wav, sr)
            saved_paths.append(path)
        return saved_paths

    def list_accents(self) -> List[Dict[str, Any]]:
        """Return list of available regional accents and dialects."""
        return list_available_accents()

    def list_emotions(self) -> List[Dict[str, Any]]:
        """Return list of available emotional tone profiles."""
        return list_available_emotions()

    def list_languages(self) -> List[Dict[str, Any]]:
        """Return list of supported languages for cross-lingual transfer."""
        return list_supported_languages()
