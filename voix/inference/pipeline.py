"""End-to-End Face-to-Voice Inference Pipeline for VOIX.

Connects the full generative stack with dynamic cross-platform hardware auto-configuration:
    Face Image / Embedding (560-D)
        -> CVAE / Contrastive Mapper (192-D speaker embeddings, K samples)
        -> StyleAdapter (128-D StyleTTS 2 style space)
        -> Optional accent/emotion style residuals + phonemizer
        -> StyleTTS 2 Neural Vocoder (24 kHz speech synthesis, CUDA / MPS / CPU adaptive)
        -> AudioSeal Neural Watermarking (provenance embedding)
        -> Auditory Speech Waveforms (Voice 1, Voice 2, ..., Voice K)
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from voix.models.accent_control import (
    AccentTransformer,
    list_available_accents,
    normalize_accent,
)
from voix.models.accent_residual import apply_accent_residual
from voix.models.adapter import StyleAdapter
from voix.models.cvae import CVAE
from voix.models.emotion_control import (
    EmotionModulator,
    list_available_emotions,
    normalize_emotion,
)
from voix.models.emotion_residual import (
    apply_style_residual,
    load_artifact,
    residual_from_artifact,
)
from voix.models.contrastive_net import ContrastiveFaceVoiceNet
from voix.models.tts_backend import StyleTTS2Backend
from voix.models.watermarking import AudioSealWatermarker
from voix.inference.candidate_selection import select_diverse_embeddings
from voix.data.feature_contract import (
    FACE_DIM,
    SCHEMA_VERSION,
    FaceFeatureBuilder,
    canonicalize,
    load_face_scaler,
)
from voix.utils.audio import limit_true_peak
from voix.utils.hardware import HardwareProfile, free_memory
from voix.utils.paths import project_path
from voix.utils.provenance import checkpoint_provenance

logger = logging.getLogger("voix.pipeline")

# Serving-side application strengths.  These are the scales at which the
# residuals were validated, not the larger exploratory demo scales:
# * E1 emotion class-mean prosody residual: 0.28.
# * A3 accent class-contrast residual: 0.3 with a 0.25 timbre weight.
EMOTION_RESIDUAL_SCALE = 0.28
ACCENT_RESIDUAL_SCALE = 0.3
ACCENT_TIMBRE_WEIGHT = 0.25


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
        cvae_checkpoint: str = str(
            project_path("checkpoints", "cvae_cfvn_face192_latent128_diversity.pt")
        ),
        adapter_checkpoint: str = str(
            project_path("checkpoints", "style_adapter_cfvn_best.pt")
        ),
        scaler_path: str = str(
            project_path("data", "processed", "voxceleb2_v2", "face_scaler_v2.pt")
        ),
        cfvn_checkpoint: str = str(
            project_path("results", "checkpoints", "cfvn_v2_best.pt")
        ),
        device: Optional[str] = None,
        enable_watermarking: bool = True,
        hardware_profile: Optional[HardwareProfile] = None,
        tts_use_neural: Optional[bool] = None,
        tts_allow_fallback: Optional[bool] = None,
        validate_artifacts: bool = True,
    ) -> None:
        # 1. Resolve Hardware Auto-Configuration.  Use the YAML-aware resolver:
        # calling detect_hardware() with no path silently ignored
        # configs/hardware.yaml and fell back to conservative defaults (5
        # diffusion steps instead of the budget-GPU profile's 20).
        if hardware_profile is not None:
            self.hardware = hardware_profile
        else:
            from voix.utils.config import get_hardware_profile

            self.hardware = get_hardware_profile()
        if device is not None:
            self.device = torch.device(device)
            tts_device = device
        else:
            self.device = self.hardware.device
            tts_device = self.hardware.tts_device

        self.enable_watermarking = enable_watermarking

        # 0. Fail fast when required learned artifacts are missing. Silent
        # fallbacks to random weights make every downstream metric meaningless.
        self.cvae_checkpoint = str(cvae_checkpoint)
        self.adapter_checkpoint = str(adapter_checkpoint)
        self.scaler_path = str(scaler_path)
        self.cfvn_checkpoint = str(cfvn_checkpoint)
        if validate_artifacts:
            required = {
                "CVAE checkpoint": self.cvae_checkpoint,
                "StyleAdapter checkpoint": self.adapter_checkpoint,
                "face scaler": self.scaler_path,
                "CFVN checkpoint": self.cfvn_checkpoint,
            }
            missing = [f"{name}: {path}" for name, path in required.items() if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(
                    "[VoixPipeline] Missing required model artifact(s):\n  - "
                    + "\n  - ".join(missing)
                    + "\nRetrain or restore these artifacts before starting the pipeline."
                )

        # Initialize visual extractors lazily
        self._arcface = None
        self._facemesh = None
        self._demog = None
        self._fusion = None
        self._emotion_residual_payload: Optional[Dict[str, Any]] = None
        self._accent_residual_payload: Optional[Dict[str, Any]] = None

        # 2. Load the canonical FaceFeatureV2 scaler (speaker-disjoint train split)
        self.scaler = load_face_scaler(self.scaler_path)
        self.face_mean = self.scaler["face_mean"].to(self.device)
        self.face_std = self.scaler["face_std"].to(self.device)
        self._feature_builder: Optional[FaceFeatureBuilder] = None

        # 3. Load CVAE / Mapper.  Architecture comes from the checkpoint
        # metadata, not from a hardcoded serving assumption, so a retrained
        # latent size or face dimension cannot be silently misloaded.
        if not os.path.exists(self.cvae_checkpoint):
            raise FileNotFoundError(
                f"[VoixPipeline] CVAE checkpoint not found: {self.cvae_checkpoint}\n"
                f"Run trainer_cvae_v3_diversity.py to generate it."
            )
        try:
            ckpt = torch.load(self.cvae_checkpoint, weights_only=True, map_location=self.device)
        except Exception:
            ckpt = torch.load(self.cvae_checkpoint, weights_only=False, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        cvae_face_dim = int(ckpt.get("face_dim", 192))
        cvae_speaker_dim = int(ckpt.get("speaker_dim", 192))
        cvae_latent_dim = int(ckpt.get("latent_dim", 128))
        self.cvae = CVAE(
            face_dim=cvae_face_dim,
            speaker_dim=cvae_speaker_dim,
            latent_dim=cvae_latent_dim,
        ).to(self.device)
        try:
            self.cvae.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"[VoixPipeline] CVAE checkpoint is incompatible with the "
                f"serving architecture: {self.cvae_checkpoint}\n{exc}"
            ) from exc
        self.cvae.eval()

        # 3b. Load ContrastiveFaceVoiceNet (560-D fused face -> 192-D ECAPA space)
        if not os.path.exists(self.cfvn_checkpoint):
            raise FileNotFoundError(
                f"[VoixPipeline] CFVN checkpoint not found: {self.cfvn_checkpoint}"
            )
        try:
            self.cfvn = ContrastiveFaceVoiceNet.from_checkpoint(
                self.cfvn_checkpoint,
                device=str(self.device),
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"[VoixPipeline] Failed to load CFVN from {self.cfvn_checkpoint}: {e}") from e
        self.cfvn.eval()
        if self.cvae.face_dim != self.cfvn.out_dim:
            raise RuntimeError(
                f"[VoixPipeline] CVAE expects {self.cvae.face_dim}-D CFVN "
                f"embeddings but the CFVN produces {self.cfvn.out_dim}-D; "
                f"retrain them as a matching pair."
            )

        # 4. Load StyleAdapter using the architecture recorded at training time.
        if not os.path.exists(self.adapter_checkpoint):
            raise FileNotFoundError(
                f"[VoixPipeline] StyleAdapter checkpoint not found: {self.adapter_checkpoint}"
            )
        try:
            a_ckpt = torch.load(self.adapter_checkpoint, weights_only=True, map_location=self.device)
        except Exception:
            a_ckpt = torch.load(self.adapter_checkpoint, weights_only=False, map_location=self.device)
        a_state = a_ckpt["model_state_dict"] if "model_state_dict" in a_ckpt else a_ckpt
        hparams = a_ckpt.get("hparams", {})
        self.adapter = StyleAdapter(
            in_dim=int(hparams.get("in_dim", self.cfvn.out_dim)),
            hidden_dim=int(hparams.get("hidden_dim", 256)),
            out_dim=int(hparams.get("out_dim", 256)),
            dropout=float(hparams.get("dropout", 0.0)),
        ).to(self.device)
        try:
            self.adapter.load_state_dict(a_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"[VoixPipeline] StyleAdapter checkpoint is incompatible with "
                f"the serving architecture: {self.adapter_checkpoint}\n{exc}"
            ) from exc
        self.adapter.eval()

        # 5. Load Synthesis Backend (Adaptive via Hardware Auto-Configuration)
        self.tts = StyleTTS2Backend(
            device=tts_device,
            use_neural=tts_use_neural,
            allow_fallback=tts_allow_fallback,
        )

        # 6. Watermarker is lazy: constructing AudioSeal is expensive and must
        # never happen when watermarking is disabled.
        self.watermarker = None
        if self.enable_watermarking:
            self._ensure_watermarker()

        # 7. Load inference config
        from voix.utils.config import load_config
        _cfg = load_config()
        self._inf_cfg = _cfg.get("inference", {})

        # 8. Structured provenance: every synthesis result records the exact
        # representation schema, checkpoint hashes, and runtime backend.
        self.provenance = checkpoint_provenance(
            {
                "face_scaler": self.scaler_path,
                "cfvn": self.cfvn_checkpoint,
                "cvae": self.cvae_checkpoint,
                "style_adapter": self.adapter_checkpoint,
            },
            device=str(self.device),
            backend=self.tts.backend_name,
            schema_version=SCHEMA_VERSION,
        )
        logger.info(
            "VOIX pipeline ready: schema=%s device=%s backend=%s",
            SCHEMA_VERSION,
            self.device,
            self.tts.backend_name,
        )

    def _ensure_watermarker(self) -> AudioSealWatermarker:
        """Lazily construct the AudioSeal watermarker only when requested."""
        if self.watermarker is None:
            self.watermarker = AudioSealWatermarker(
                sample_rate=24000,
                device=str(self.device),
            )
        return self.watermarker

    def _emotion_residual(self, emotion: str) -> Optional[torch.Tensor]:
        """Return the validated E1 prosody residual for an emotion, if available."""
        path = project_path("checkpoints", "emotion_residuals", "class_mean.pt")
        if not os.path.exists(path):
            return None
        if self._emotion_residual_payload is None:
            self._emotion_residual_payload = load_artifact(path)
        try:
            return residual_from_artifact(
                self._emotion_residual_payload, emotion, variant="prosody_only"
            )
        except KeyError:
            return None

    def _accent_residual(self, accent: str) -> Optional[torch.Tensor]:
        """Return the A3 class-contrast residual for an accent, if available."""
        path = project_path("checkpoints", "accent_residuals", "class_mean.pt")
        if not os.path.exists(path):
            return None
        if self._accent_residual_payload is None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("residuals"), dict
            ):
                raise ValueError(f"invalid accent residual artifact: {path}")
            self._accent_residual_payload = payload
        raw = self._accent_residual_payload["residuals"].get(accent)
        if raw is None:
            return None
        return torch.as_tensor(raw).float().reshape(-1)

    def _ensure_feature_builder(self) -> FaceFeatureBuilder:
        """Lazily construct the canonical FaceFeatureV2 builder."""
        if self._feature_builder is None:
            self._feature_builder = FaceFeatureBuilder(device=self.device)
        return self._feature_builder

    def normalize_face(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the canonical FaceFeatureV2 z-score normalization."""
        return canonicalize(raw, self.face_mean, self.face_std)

    def extract_face_features(
        self,
        image_input: Union[str, Path, np.ndarray],
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
        from voix.data.face_extractor import (
            ArcFaceExtractor,
            FaceMeshExtractor,
            crop_face_bbox,
        )
        from voix.data.morphology import compute_craniofacial_ratios
        from voix.data.demographics import SoftDemographicEstimator

        if self._arcface is None:
            # Use CPUExecutionProvider for ArcFace on budget GPUs to preserve VRAM for vocoder
            use_gpu_arc = (
                self.device.type == "cuda"
                and self.hardware.onnx_execution_provider == "CUDAExecutionProvider"
            )
            self._arcface = ArcFaceExtractor(
                device_id=0 if use_gpu_arc else -1,
                cache_dir=str(project_path("checkpoints")),
            )
        if self._facemesh is None:
            task_path = str(project_path("checkpoints", "mediapipe", "face_landmarker.task"))
            self._facemesh = FaceMeshExtractor(
                model_path=task_path if os.path.exists(task_path) else None,
            )
        if self._demog is None:
            self._demog = SoftDemographicEstimator()

        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise TypeError("image_input must be a file path or numpy array")

        # 1. ArcFace 512-D identity and demographic posteriors
        e_id, gender, age, bbox = self._arcface.extract_full_with_bbox(img_bgr)
        if e_id is None:
            raise ValueError(f"No face detected in image: {image_input}")

        # 2. FaceMesh 32-D morphology on the same bbox crop as the corpus
        # extractor. Running the mesh on the full frame can pick a different
        # person when the image contains more than one face.
        face_crop = crop_face_bbox(img_bgr, bbox) if bbox is not None else img_bgr
        img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        landmarks = self._facemesh.extract(img_rgb)
        if landmarks is not None:
            e_geo = compute_craniofacial_ratios(landmarks)
        else:
            e_geo = np.zeros(32, dtype=np.float32)

        # 3. Soft Demographics 16-D with real age & gender posteriors.
        # The corpus extractor uses the selected face bbox aspect ratio; use the
        # same value here so image and HDF5 inputs share one contract.
        from voix.data.demographics import aspect_ratio_from_bbox

        aspect_ratio = (
            aspect_ratio_from_bbox(bbox) if bbox is not None else 1.35
        )
        e_demo = self._demog.estimate(
            e_id,
            gender=gender,
            age=age,
            aspect_ratio=aspect_ratio,
        )

        # 4. Canonical FaceFeatureV2 fusion — identical to the HDF5 corpus path.
        builder = self._ensure_feature_builder()
        e_f = builder.fuse(e_id, e_geo, e_demo)

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
        accent: Optional[str] = None,
        accent_strength: float = 1.0,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
        speaker_embedding: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        face_input_is_normalized: bool = False,
        selection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate K diverse candidate voice waveforms for a single face representation.

        Args:
            face_input: 560-D *raw fused* face vector (torch.Tensor, np.ndarray),
                a face image path, or a demographic preset ID.
            face_input_is_normalized: Set True only when the tensor/array has
                already been normalized with the canonical FaceFeatureV2 scaler.
            text: Text prompt to synthesize.
            num_voices: Number of diverse voice candidates to generate (K).
            temperature: Sampling diversity temperature.
            watermark: Explicit flag to enable/disable AudioSeal watermarking.
            accent: One of ``us``, ``england``, ``indian_english``,
                ``australian`` (legacy aliases are accepted).
            accent_strength: Accent intensity in [0, 1].  Applied to the
                conservative G2P rule set and the A3 style residual.
            emotion: ``neutral``, ``happy``, ``sad``, or ``angry``.
            emotion_intensity: Emotion intensity in [0, 1].  Scales the E1
                prosody residual and the bounded duration/energy shaping.
            selection: Optional candidate-selection strategy.  ``"diverse"``
                samples a larger pool and returns one central mode plus the
                most dissimilar alternatives; ``None`` keeps the historical
                independent sampling.

        Returns:
            List of dicts containing voice waveforms, speaker embeddings, and metadata.
        """
        if selection not in (None, "diverse"):
            raise ValueError(
                f"selection must be None or 'diverse', got {selection!r}"
            )
        if watermark is None:
            watermark = self.enable_watermarking

        accent_id = normalize_accent(accent) if accent else None
        emotion_id = normalize_emotion(emotion)
        accent_strength = float(np.clip(accent_strength, 0.0, 1.0))
        emotion_intensity = float(np.clip(emotion_intensity, 0.0, 1.0))

        # Det gender/age defaults (set when face features are extracted)
        _det_gender = None
        _det_age = None

        # Process face input into the canonical 560-D normalized tensor.
        # All tensors/arrays are interpreted as *raw fused* FaceFeatureV2 vectors
        # unless the caller explicitly marks them as already normalized.
        raw_face: torch.Tensor
        if isinstance(face_input, (str, Path)):
            str_path = str(face_input)
            if not os.path.exists(str_path) and not Path(str_path).is_file():
                # String corresponds to a demographic preset ID.
                try:
                    preset = self.get_demographic_preset(str_path)
                except KeyError as exc:
                    raise FileNotFoundError(f"Image file or preset not found: {str_path}") from exc
                raw_face = preset.embedding.to(self.device).float()
                if gender is None: gender = preset.gender
                if age is None: age = preset.age
            else:
                raw_face, _det_gender, _det_age, _ = self.extract_face_features(
                    str_path
                )
                if gender is None: gender = _det_gender
                if age is None: age = _det_age
            face_input_is_normalized = False
        elif isinstance(face_input, np.ndarray) and face_input.ndim == 3:
            raw_face, _det_gender, _det_age, _ = self.extract_face_features(
                face_input
            )
            if gender is None: gender = _det_gender
            if age is None: age = _det_age
            face_input_is_normalized = False
        elif isinstance(face_input, (np.ndarray, torch.Tensor)):
            raw_face = torch.as_tensor(face_input, dtype=torch.float32, device=self.device)
            if raw_face.dim() == 1:
                raw_face = raw_face.unsqueeze(0)
            if raw_face.ndim != 2 or raw_face.shape[-1] != FACE_DIM:
                raise ValueError(
                    f"face_input must have shape (B, {FACE_DIM}) or be an image path; "
                    f"got {tuple(raw_face.shape)}"
                )
        else:
            raise TypeError("face_input must be an image path, preset ID, 560-D tensor, or numpy array")

        if face_input_is_normalized:
            e_f = raw_face.to(self.device)
        else:
            e_f = self.normalize_face(raw_face.to(self.device))

        # Map 560-D fused face -> 192-D ECAPA-space embedding via CFVN
        with torch.no_grad():
            e_f_voice = self.cfvn(e_f, normalize=True)  # (1, 192)

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
            e_s_candidates_norm = F.normalize(e_s_base, dim=-1)
            if e_s_candidates_norm.shape[0] < num_voices:
                e_s_candidates_norm = e_s_candidates_norm.repeat(num_voices, 1)
        else:
            if seed is not None:
                torch.manual_seed(seed)
            if selection == "diverse":
                # Sample a wider pool, then keep the central mode plus the
                # most dissimilar alternatives.  Independent K-sample draws
                # can collapse to near-duplicates (ECAPA cosine 0.939 in the
                # 15-asset demo), which defeats one-to-many generation.
                pool_size = max(num_voices * 3, 8)
                pool = self.cvae.sample_voices(
                    e_f_voice,
                    num_samples=pool_size,
                    temperature=temperature,
                )
                mu_prior, _ = self.cvae.prior_net(e_f_voice)
                primary_mode = self.cvae.decoder(mu_prior)
                e_s_candidates = select_diverse_embeddings(
                    pool,
                    num_voices,
                    reference=primary_mode,
                )
            else:
                e_s_candidates = self.cvae.sample_voices(
                    e_f_voice,
                    num_samples=num_voices,
                    temperature=temperature,
                )
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

        # 2. Text preprocessing: accent prosody, emotion, and duration/energy shaping.
        text_accented = text
        accent_mods = {"duration_mult": 1.0, "pitch_mult": 1.0, "prosody_mult": 1.0}
        if accent_id:
            text_accented, accent_mods = AccentTransformer.transform(
                text=text,
                accent=accent_id,
                strength=accent_strength,
            )

        synth_text = text_accented
        emotion_mods = EmotionModulator.get_acoustic_modifiers(
            emotion=emotion_id,
            intensity=emotion_intensity,
        )
        # 3. For each candidate voice, project style and synthesize speech
        _inf = self._inf_cfg
        # null/absent -> the detected hardware profile decides; a fixed YAML
        # value previously overrode the CPU and budget-GPU profiles.
        diffusion_steps = _inf.get("diffusion_steps") or self.hardware.diffusion_steps

        for k in range(num_voices):
            if self.hardware.torch_empty_cache_per_sample:
                free_memory(self.device)

            e_s_k = e_s_candidates_norm[k].unsqueeze(0)
            e_s_raw_k = e_s_candidates_raw[k]

            # Dual-Head Style Adapter projection (192-D -> 256-D [timbre | prosody])
            s_hat_k = self.adapter(e_s_raw_k.unsqueeze(0))

            # Apply the A3 class-contrast accent residual.  Accent identity is
            # carried mostly by prosody, so the timbre half is down-weighted to
            # protect the face-derived speaker identity.
            if accent_id:
                accent_residual = self._accent_residual(accent_id)
                if accent_residual is not None:
                    s_hat_k = apply_accent_residual(
                        s_hat_k,
                        accent_residual,
                        scale=ACCENT_RESIDUAL_SCALE * accent_strength,
                        timbre_weight=ACCENT_TIMBRE_WEIGHT,
                    )

            # Apply the E1 prosody-only emotion residual.  Emotion intensity
            # scales the validated 0.28 residual; neutral is an exact no-op.
            if emotion_id and emotion_id != "neutral":
                emotion_residual = self._emotion_residual(emotion_id)
                if emotion_residual is not None:
                    s_hat_k = apply_style_residual(
                        s_hat_k,
                        emotion_residual,
                        intensity=emotion_intensity,
                        scale=EMOTION_RESIDUAL_SCALE,
                    )

            _gender = gender if gender is not None else 1
            _age = float(age) if age is not None else 30.0

            # Anchor-free: no reference-file blending is exposed or reported.
            base_dur = _inf.get("duration_scale", 1.0)
            final_duration = (
                base_dur
                * accent_mods["duration_mult"]
                * emotion_mods["duration_scale"]
            )
            base_beta = _inf.get("beta", 0.22)
            final_beta = float(
                np.clip(
                    base_beta
                    * accent_mods["prosody_mult"]
                    * emotion_mods["beta_mult"],
                    0.10,
                    0.90,
                )
            )
            eff_alpha = _inf.get("alpha", 0.18)

            # Emotion/accent pitch multipliers are realised as a real
            # post-synthesis pitch shift.  Pitch *variance* is controlled by
            # StyleTTS2's beta (prosody transfer), not a fake output parameter.
            eff_f0_scale = float(emotion_mods.get("pitch_factor", 1.0)) * float(
                accent_mods.get("pitch_mult", 1.0)
            )

            # Accent phonemization is applied only on the neural backend.  The
            # fallback backend has no phoneme stage and would silently ignore it.
            phonemizer_ctx = nullcontext()
            if accent_id and self.tts.backend_name == "neural":
                from voix.models.accent_g2p import use_accent_phonemizer

                phonemizer_ctx = use_accent_phonemizer(
                    self.tts,
                    accent_id,
                    strength=accent_strength,
                )

            with phonemizer_ctx:
                waveform = self.tts.synthesize(
                    text=synth_text,
                    style=s_hat_k,
                    gender=_gender,
                    age=_age,
                    beta=final_beta,
                    alpha=eff_alpha,
                    duration_scale=final_duration,
                    diffusion_steps=diffusion_steps,
                    f0_scale=eff_f0_scale,
                )

            # Emotion energy is a real post-synthesis gain, applied after the
            # backend's LUFS normalization so it is not normalized away.  Use a
            # true-peak limiter instead of a hard clamp: clamping would distort
            # loud emotions and add harmonics that look like noise.
            energy_scale = float(emotion_mods.get("energy_scale", 1.0))
            if abs(energy_scale - 1.0) > 1e-4:
                wav_np = waveform.detach().cpu().numpy().squeeze() * energy_scale
                waveform = (
                    torch.from_numpy(limit_true_peak(wav_np))
                    .unsqueeze(0)
                    .to(waveform.device)
                )

            # Emotion is steered natively in style/prosody space (no post-hoc DSP filtering)
            is_wm = False
            wm_status = "disabled"
            if watermark:
                wm = self._ensure_watermarker()
                try:
                    waveform = wm.embed(waveform)
                except Exception as exc:
                    raise RuntimeError(
                        "[VoixPipeline] Watermarking was requested but failed. "
                        "Refusing to return an unwatermarked waveform while "
                        f"reporting success. Original error: {type(exc).__name__}: {exc}"
                    ) from exc
                is_wm = True
                wm_status = "embedded"

            results.append({
                "voice_index": k + 1,
                "waveform": waveform.cpu(),
                "sample_rate": self.tts.sample_rate,
                "speaker_embedding": e_s_raw_k.cpu(),
                "style_embedding": s_hat_k.squeeze(0).cpu(),
                "is_watermarked": is_wm,
                "watermark_status": wm_status,
                "backend": self.tts.backend_name,
                "provenance": self.provenance,
            })

        self.provenance["runtime"]["backend"] = self.tts.backend_name
        logger.info(
            "synthesized %d candidate(s): backend=%s schema=%s",
            len(results),
            self.tts.backend_name,
            SCHEMA_VERSION,
        )
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
        accent: Optional[str] = None,
        accent_strength: float = 1.0,
        emotion: Optional[str] = None,
        emotion_intensity: float = 1.0,
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
            accent=accent,
            accent_strength=accent_strength,
            emotion=emotion,
            emotion_intensity=emotion_intensity,
        )

    def save_audio(
        self,
        results: List[Dict[str, Any]],
        output_dir: str = str(project_path("artifacts", "samples")),
        prefix: str = "voice",
    ) -> List[str]:
        """Save generated candidate waveforms to disk as 24 kHz WAV files."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for item in results:
            idx = item["voice_index"]
            waveform = item["waveform"]
            if isinstance(waveform, torch.Tensor):
                wav = waveform.detach().cpu().squeeze().numpy()
            else:
                wav = np.asarray(waveform).squeeze()
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
