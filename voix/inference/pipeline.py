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
from voix.models.craniofacial_style import apply_craniofacial_to_style
from voix.models.cvae import CVAE
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
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
        cvae_checkpoint: str = "D:/voix/checkpoints/cvae_cfvn_face192_latent128_diversity.pt",
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

        # Initialize visual extractors lazily
        self._arcface = None
        self._facemesh = None
        self._demog = None
        self._fusion = None

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

        # Canonical identity projection from ArcFace (512-D) to ECAPA space (192-D)
        torch.manual_seed(42)
        _w = torch.randn(512, 192)
        _q, _ = torch.linalg.qr(_w)
        self.id_proj_q = _q.to(self.device)
        self._last_e_id = None

        # 3. Load CVAE / Mapper (face_dim=192: CFVN-produced ECAPA-space embedding)
        self.cvae = CVAE(face_dim=192, speaker_dim=192, latent_dim=128).to(self.device)
        if os.path.exists(cvae_checkpoint):
            ckpt = torch.load(cvae_checkpoint, weights_only=False, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.cvae.load_state_dict(state_dict, strict=False)
        self.cvae.eval()

        # 3b. Load ContrastiveFaceVoiceNet (560-D fused face -> 192-D ECAPA space)
        self.cfvn = ContrastiveFaceVoiceNet.from_checkpoint(
            "D:/voix/results/checkpoints/D_joint_infonce_best.pt",
            device=str(self.device),
        ).to(self.device)
        self.cfvn.eval()

        # 4. Load StyleAdapter
        self.adapter = StyleAdapter(in_dim=192, hidden_dim=256, out_dim=256).to(self.device)
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

        # Pre-compute mean ECAPA from a diverse set of face images for deviation amplification
        self.mean_ecapa = None
        self._compute_mean_ecapa()

    def _compute_mean_ecapa(self):
        """Compute mean ECAPA embedding from a diverse set of face images.
        
        Used as reference for amplifying face-specific deviations in ECAPA space.
        This helps differentiate voices for same-gender faces whose ECAPA embeddings
        are naturally very similar (cosine similarity > 0.99).
        """
        # Diverse face images for computing mean ECAPA
        mean_ecapa_faces = [
            f"D:/voix/data/test_subjects/{f}"
            for f in [
                "C1_01_young_female_Zendaya.jpg",
                "C1_04_adult_male_George_Clooney.jpg",
                "C2_08_south_asian_female_Priyanka_Chopra.jpg",
                "C2_11_black_male_Barack_Obama.jpg",
                "C2_15_latino_male_Pedro_Pascal.jpg",
                "C3_17_prominent_jaw_Arnold_Schwarzenegger.jpg",
                "SD_07_rhotacism_Barbara_Walters.jpg",
            ]
        ]
        # Filter to existing files
        existing = [p for p in mean_ecapa_faces if os.path.exists(p)]
        if len(existing) < 2:
            # Not enough face images, use zero mean (no amplification)
            self.mean_ecapa = torch.zeros(192, device=self.device)
            return
        
        # Extract ECAPA embeddings from all faces
        embs = []
        with torch.no_grad():
            for f_path in existing:
                try:
                    e_f, _, _, _ = self.extract_face_features(f_path)
                    e_f_voice = self.cfvn(e_f, normalize=True)  # (1, 192)
                    embs.append(e_f_voice.squeeze(0))
                except Exception:
                    pass
        
        if len(embs) > 0:
            self.mean_ecapa = torch.stack(embs).mean(dim=0)  # (192,)
        else:
            self.mean_ecapa = torch.zeros(192, device=self.device)

    def extract_face_features(
        self,
        image_input: Union[str, Path, np.ndarray],
        craniofacial_sliders: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """Extract and fuse facial features (ArcFace 512 + FaceMesh 32 + Demographics 16 -> 560-D) from an image.

        Args:
            image_input: File path to image or BGR numpy array.

        Returns:
            Tuple of (e_f, gender, age, e_geo):
                e_f: Fused 560-D face representation of shape (1, 560) on self.device.
                gender: Detected biological sex (1 = Male, 0 = Female).
                age: Estimated chronological age in years.
                e_geo: Craniofacial morphology vector of shape (32,).
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
        self._last_e_id = e_id

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

        return e_f, int(gender) if gender is not None else 1, float(age) if age is not None else 30.0, e_geo

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
        speaker_embedding: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
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

        # Det gender/age defaults (set when face features are extracted)
        _det_gender = None
        _det_age = None
        e_geo = None

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
                e_geo = None  # Presets don't have craniofacial data
            else:
                e_f, _det_gender, _det_age, e_geo = self.extract_face_features(
                    str_path,
                    craniofacial_sliders=craniofacial_sliders,
                )
                if gender is None: gender = _det_gender
                if age is None: age = _det_age
                if self.scaler is not None:
                    e_f = (e_f - self.face_mean) / self.face_std
        elif isinstance(face_input, np.ndarray):
            if face_input.ndim == 3:  # HxWx3 image
                e_f, _det_gender, _det_age, e_geo = self.extract_face_features(
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
                e_geo = None
        elif isinstance(face_input, torch.Tensor):
            e_f = face_input.float()
            if e_f.dim() == 1:
                e_f = e_f.unsqueeze(0)
            e_f = e_f.to(self.device)
            if self.scaler is not None and e_f.abs().mean() > 1.5:
                e_f = (e_f - self.face_mean) / self.face_std
            if e_f.shape[-1] >= 544:
                e_geo_base = e_f[:, 512:544]
                if craniofacial_sliders:
                    from voix.data.morphology import apply_craniofacial_sliders
                    base_np = e_geo_base.detach().cpu().numpy().squeeze()
                    mod_np = apply_craniofacial_sliders(base_np, craniofacial_sliders)
                    e_geo = torch.from_numpy(mod_np).float().to(self.device)
                else:
                    e_geo = e_geo_base
            else:
                e_geo = None
        else:
            raise TypeError("face_input must be an image path, 560-D tensor, or numpy array")

        # Map 560-D fused face -> 192-D ECAPA-space embedding via CFVN
        with torch.no_grad():
            e_f_voice = self.cfvn(e_f, normalize=True)  # (1, 192)

        # Extract ArcFace visual identity projection for consistency
        e_id_val = getattr(self, "_last_e_id", None)
        if e_id_val is None and e_f.shape[-1] >= 512:
            e_id_val = e_f[:, :512]

        if speaker_embedding is not None:
            # Use explicit fixed speaker embedding (e.g. Primary candidate identity)
            spk_t = speaker_embedding.to(self.device).float()
            if spk_t.dim() == 1:
                spk_t = spk_t.unsqueeze(0)
            if spk_t.shape[0] < num_voices:
                spk_t = spk_t.repeat(num_voices, 1)
            e_s_candidates_norm = F.normalize(spk_t, dim=-1)
        elif temperature <= 0.001:
            # Deterministic mode: prior mean mode without stochastic noise
            mu_prior, _ = self.cvae.prior_net(e_f_voice)
            e_s_base = self.cvae.decoder(mu_prior)
            if e_id_val is not None:
                if isinstance(e_id_val, np.ndarray):
                    eid_t = torch.from_numpy(e_id_val).float().to(self.device)
                else:
                    eid_t = e_id_val.float().to(self.device)
                if eid_t.dim() == 1:
                    eid_t = eid_t.unsqueeze(0)
                id_delta = F.normalize(eid_t @ self.id_proj_q, dim=-1)
                e_s_candidates_norm = F.normalize(0.55 * e_s_base + 0.45 * id_delta, dim=-1)
            else:
                e_s_candidates_norm = F.normalize(e_s_base, dim=-1)
            if e_s_candidates_norm.shape[0] < num_voices:
                e_s_candidates_norm = e_s_candidates_norm.repeat(num_voices, 1)
        else:
            if seed is not None:
                torch.manual_seed(seed)
            e_s_candidates = self.cvae.sample_voices(
                e_f_voice,
                num_samples=num_voices,
                temperature=temperature,
            )
            if e_id_val is not None:
                if isinstance(e_id_val, np.ndarray):
                    eid_t = torch.from_numpy(e_id_val).float().to(self.device)
                else:
                    eid_t = e_id_val.float().to(self.device)
                if eid_t.dim() == 1:
                    eid_t = eid_t.unsqueeze(0)
                id_delta = F.normalize(eid_t @ self.id_proj_q, dim=-1)
                e_s_candidates_norm = F.normalize(0.55 * e_s_candidates + 0.45 * id_delta, dim=-1)
            else:
                e_s_candidates_norm = F.normalize(e_s_candidates, dim=-1)

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

            # Dual-Head Style Adapter projection (192-D -> 256-D [timbre | prosody])
            s_hat_k = self.adapter(e_s_raw_k.unsqueeze(0))

            # Apply craniofacial morphology modulation to style vector (oral/nasal resonance)
            if e_geo is not None:
                from voix.models.craniofacial_style import apply_craniofacial_to_style
                s_hat_k = apply_craniofacial_to_style(
                    s_hat_k,
                    e_geo,
                    strength=1.0,
                )

            # Apply style-space accent delta to prosody half of s_hat_k
            if accent and accent != "general_american":
                from voix.models.accent_control import apply_accent_style_delta
                s_hat_k = apply_accent_style_delta(
                    ref_s=s_hat_k,
                    accent_id=accent,
                    strength=accent_strength,
                )

            # Apply style-space emotion delta to prosody half of s_hat_k
            if emotion and emotion != "neutral":
                from voix.models.emotion_control import apply_emotion_style_delta
                s_hat_k = apply_emotion_style_delta(
                    ref_s=s_hat_k,
                    emotion_id=emotion,
                    intensity=emotion_intensity,
                )

            _gender = gender if gender is not None else 1

            base_dur = _inf.get("duration_scale", 1.0)
            final_duration = base_dur * duration_mult * accent_mods["duration_mult"] * emotion_mods["duration_scale"] * lang_mods["duration_scale"]
            base_beta = _inf.get("beta", 0.80)
            final_beta = float(np.clip(base_beta * accent_mods["prosody_mult"] * emotion_mods["beta_mult"] * lang_mods["pitch_dynamics_factor"], 0.55, 0.95))
            eff_alpha = _inf.get("alpha", 0.30)
            waveform = self.tts.synthesize(
                text=synth_text,
                style=s_hat_k,
                gender=_gender,
                blend_weight=0.0,
                beta=final_beta,
                alpha=eff_alpha,
                duration_scale=final_duration,
                diffusion_steps=diffusion_steps,
                anchor_path=None,
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
