# architecture.md
> Full system design — modules, data flow, component interfaces, mathematical formulations, and engineering decisions.
> Last updated: 2026-10-16. All claims verified against source code in `voix/`.

---

## Status Legend

| Symbol | Meaning |
|---|---|
| **Implemented & Verified** | Code exists; unit tests pass; empirically measured |
| **Implemented; under-evaluated** | Code exists; limited or no formal benchmark yet |
| **Future Scope** | Designated out-of-scope for current release |

---

## System Pipeline (End-to-End)

```
Face Image + Text Prompt
    |
    +-- ArcFace ResNet-50 (frozen, InsightFace Buffalo_L)   --> 512-D identity embedding (e_id)
    +-- MediaPipe FaceMesh (468 3D landmarks)               -->  32-D craniofacial ratios  (e_geo)
    +-- Soft Demographic Estimator                          -->  16-D age/sex posteriors   (e_demo)
                                |
                FaceFusionLayer  (Linear 560->560 + LayerNorm + GELU)
                                |
                         560-D  e_f
                 +--------------+------------------------------+
                 |                                            |
                 v                                            v
      CVAE Conditional Prior                    Exemplar Retrieval Gallery
      p_psi(z | e_f)  -->  sample z in R^192    3,029 speaker centroids (192-D)
      CVAEDecoder  p_theta(e_s | z, e_f)        Gender-conditioned kNN (k=5 or k=10)
                 |                                            |
                 +--------------+------------------------------+
                                |  Hybrid Blending
                                |  e_s_hybrid = alpha*e_s_hat + (1-alpha)*sum(w_k*c_k)
                                |  (alpha = 0.70 by default)
                                v
                      StyleAdapter (MLP 192->256->256->128)
                      Projects ECAPA space --> StyleTTS 2 style space (128-D)
                                |
                      StyleTTS 2 (completely frozen)
                      Inputs: phoneme tokens + style vector s_hat (128-D)
                                |
                      LUFS Loudness Normalization (EBU R128, -23 LUFS)
                                |
                      AudioSeal Watermarker (16-bit neural watermark)
                                |
                      24 kHz Speech Waveforms (Voice 1, Voice 2, ..., Voice K)
```

---

## Module 1: Face Representation  [Implemented & Verified]

### 1a. ArcFace Identity Embedding (512-D)
- **Model:** Pretrained ArcFace ResNet-50 (`buffalo_l`, InsightFace) — completely frozen.
- **Input:** Single frontal face image; pose-filtered frames (|pitch| < 15°, |yaw| < 15°).
- **Output:** L2-normalized 512-D identity vector $\mathbf{e}_{\text{id}} \in \mathbb{S}^{511}$.
- **Source:** `voix/data/face_extractor.py` — `ArcFaceExtractor.extract_full()` returns `(e_id, gender, age)`.

### 1b. Craniofacial Ratio Features (32-D)  [Implemented & Verified]
- **Source:** `voix/data/morphology.py` — `compute_craniofacial_ratios(landmarks)`.
- **Landmarks:** MediaPipe FaceMesh 468-point 3D mesh.
- **Normalization:** All distances divided by inter-ocular distance (IOC, landmarks 33 ↔ 263) to remove camera scale.

Key landmark constants (canonical MediaPipe indices):

| Constant | Index | Anatomical Point |
|---|---|---|
| IOC_LEFT | 33 | Left eye outer canthus |
| IOC_RIGHT | 263 | Right eye outer canthus |
| FOREHEAD_TOP | 10 | Trichion |
| CHIN_BOTTOM | 152 | Gnathion / menton |
| JAW_LEFT / RIGHT | 172 / 397 | Mandibular gonion |
| NOSE_TIP | 4 | Pronasale |
| UPPER_LIP_TOP | 13 | Labrale superius |
| LOWER_LIP_BOTTOM | 14 | Labrale inferius |

### 1c. Soft Demographic Estimator (16-D)  [Implemented & Verified]
- **Source:** `voix/data/demographics.py` — `SoftDemographicEstimator.estimate_from_attributes()`.
- **Output layout (16-D):**
  - Indices 0–7: Age bin soft distribution (8 bins: ~0–10, 11–20, ..., 71+).
  - Indices 8–9: Biological sex distribution [p_male, p_female].
  - Indices 10–13: Vocal age grouping [child, young, adult, senior].
  - Indices 14–15: Apparent body build proxy [slender, broad].

### 1d. FaceFusionLayer (560-D Multimodal Fusion)  [Implemented & Verified]
- **Source:** `voix/models/fusion.py` — `FaceFusionLayer`.
- **Architecture:** `Linear(560, 560) → LayerNorm(560) → GELU`.
- **Input:** Concatenation $[\mathbf{e}_{\text{id}} \| \mathbf{e}_{\text{geo}} \| \mathbf{e}_{\text{demo}}]$ in $\mathbb{R}^{560}$.
- **Note:** Weights are NOT auto-loaded at import time; callers must explicitly call `load_canonical_weights()`.

---

## Module 2: Probabilistic Voice Mapper  [Implemented & Verified]

### 2a. CVAE v2  (`voix/models/cvae.py`)

Three subnetworks:

| Network | Role | Architecture |
|---|---|---|
| `CVAEPriorNetwork` $p_\psi(z \mid \mathbf{e}_f)$ | Conditional prior from face features | MLP (560→512→384→256) + μ/logvar heads |
| `CVAERecognitionNetwork` $q_\phi(z \mid \mathbf{e}_s, \mathbf{e}_f)$ | Joint posterior during training | MLP (752→512→384→256) + μ/logvar heads |
| `CVAEDecoder` $p_\theta(\mathbf{e}_s \mid z, \mathbf{e}_f)$ | Reconstruct 192-D speaker embeddings | MLP (384→384→256) → Linear(192) |

Key engineering choices:
- **GroupNorm** instead of BatchNorm: works at `batch_size=1` during inference.
- **Free-bits KL** ($\lambda_{fb}=0.5$ nats/dim): prevents posterior collapse.
- **$\beta(t)$ annealing**: smooth KL regularization warm-up during training.
- **`sample_voices(e_f, K, temperature)`**: generates K diverse candidate embeddings at inference.

### 2b. Metric Contrastive Mapper  [Implemented & Verified]
- **Architecture:** 560→512→384→256→192-D MLP + L2-normalize output.
- **Loss:** Supervised InfoNCE ($\tau=0.07$) over precomputed speaker centroids.
- **Auxiliary gender head:** `Linear(256, 1)` with $\lambda_{\text{gender}}=0.50$.
- **Best checkpoint:** `results/checkpoints/D_joint_infonce_best.pt` (Model D), val_cos=0.1288.

### 2c. Exemplar Retrieval Gallery  [Implemented & Verified]
- **Gallery:** 3,029 speaker ECAPA-TDNN identity centroids precomputed from 75,720 VoxCeleb2 utterances.
- **Retrieval:** Gender-conditioned cosine kNN ($k=5$ or $k=10$).
- **Hybrid blending:** $\mathbf{e}_{s,\text{hybrid}} = \alpha \hat{\mathbf{e}}_s + (1-\alpha) \sum_{k=1}^K w_k \mathbf{c}_{(k)}$ with $\alpha=0.70$.
- **Pareto-optimal configuration:** Model F (k=10, α=0.70) — val_cos=0.1299, gender_acc=100%.

---

## Module 3: Acoustic Synthesis Stack  [Implemented & Verified]

### 3a. StyleAdapter  (`voix/models/adapter.py`)
- **Architecture:** `Linear(192,256) → LayerNorm → GELU → Linear(256,256) → LayerNorm → GELU → Linear(256,128)`.
- **Purpose:** Projects ECAPA-TDNN speaker embedding space (192-D) into StyleTTS 2 style conditioning space (128-D).
- **Trainable parameters:** ~200k. Only trainable component beyond the mapper.

### 3b. StyleTTS 2 Backend  (`voix/models/tts_backend.py`)
- **Status:** Completely frozen. `requires_grad=False` on all parameters.
- **Input:** Phoneme tokens + 128-D style vector.
- **Output:** 24 kHz speech waveform.
- **LUFS Normalization:** `_lufs_normalize()` applies EBU R128 (-23 LUFS) via `pyloudnorm`. Falls back to soft peak normalization if pyloudnorm unavailable.
- **Reference Anchors:** Gender/age-matched demographic anchor audio paths resolved via `select_anchor_key()` in `voix/utils/config.py`.

### 3c. AudioSeal Watermarker  (`voix/models/watermarking.py`)
- **Model:** AudioSeal generator (16-bit message) + detector.
- **Properties:** AUC=0.97, sample-level IoU=0.99, >40 dB SNR.
- **Default:** Enabled in `VoixPipeline`; can be disabled with `enable_watermarking=False`.

---

## Module 4: Controllable Expressivity Engine  [Implemented & Verified]

### 4a. Craniofacial Morphology Sliders  (`voix/data/morphology.py`)
- `apply_craniofacial_sliders(embedding, sliders_dict)`: adjusts the 32-D ratio sub-vector by weighted differential perturbation.
- Supported dimensions: `jaw_width`, `lip_fullness`, `nose_width`, `midface_length`, `cheekbone_prominence`, `eye_separation`, `face_width_to_height`.

### 4b. Regional Accent Control  (`voix/models/accent_control.py`)
- `AccentTransformer.transform(text, accent, strength)`: Returns phonemically transformed text + prosodic modifiers dict.
- Six supported accents via `AccentType` enum:

| Accent | Duration Scale | Pitch Factor | Prosody Dynamics |
|---|---|---|---|
| General American | 1.00 | 1.00 | 1.00 |
| British RP | 0.96 | 1.03 | 1.12 |
| Australian | 1.04 | 1.02 | 1.15 |
| Indian English | 1.08 | 1.05 | 0.92 |
| Irish | 1.06 | 1.04 | 1.20 |
| Scottish | 1.02 | 0.98 | 1.10 |

### 4c. Emotion & Affective Tone Modulation  (`voix/models/emotion_control.py`)
- `EmotionModulator.get_acoustic_modifiers(emotion, intensity)`: Returns modifiers dict.
- `EmotionModulator.apply_acoustic_filter(waveform, sr, emotion, intensity)`: Post-synthesis spectral filter.
- Six emotional states via `EmotionType` enum: Neutral, Happy, Sad, Angry, Whispered, Surprised.

### 4d. Clinical Speech Difference Simulation  (`voix/models/speech_simulation.py`)
- `SpeechSimulator.process_text(text, difference_type, severity)`: Pre-synthesis phonetic text transform.
- `SpeechSimulator.process_waveform(waveform, sr, difference_type, severity)`: Post-synthesis acoustic filter.
- Six `SpeechDifferenceType` values:

| Type | Clinical Characteristic | Empirically Verified Effect |
|---|---|---|
| `interdental_lisp` | $[s] → [\theta]$ substitution | Sibilant softening at mild/moderate; ASR decoding shift at severe |
| `lateral_lisp` | Lateral airflow with slushy sibilants | Distinct lateral friction coloration |
| `rhotacism` | $[r] → [w]$ liquid gliding | Verified: "clinical twile" / "Wi-Fi's speech defiance" |
| `stutter_clonic` | Syllable-repetition blocks | Duration dilated 9.05 s → 13.27 s; clonic syllables in ASR output |
| `hypernasality` | Velopharyngeal resonance coupling | Nasal formant enhancement post-synthesis |
| `prolonged_phonemes` | Extended phoneme durations | Duration monotonically expanded 8.85 s → 9.85 s |

### 4e. Cross-Lingual Synthesis  (`voix/models/cross_lingual.py`)
- `MultilingualPhoneticAdapter.transform(text, language)`: Language-adapted phonetic text + pacing modifiers.
- Five `LanguageType` values: English (en), Spanish (es), French (fr), German (de), Italian (it).

---

## Module 5: Inference Pipeline  (`voix/inference/pipeline.py`)  [Implemented & Verified]

### `VoixPipeline.generate()` — Primary API

```
face_input:        str path | np.ndarray | torch.Tensor (560-D pre-extracted)
text:              str
num_voices:        int (default: 3)
temperature:       float (default: 1.0)
gender:            Optional[int] (0=Female, 1=Male; auto-detected if None)
age:               Optional[float] (years; auto-detected if None)
craniofacial_sliders: Optional[Dict[str, float]]
speech_difference: Optional[str] (SpeechDifferenceType value)
speech_severity:   float [0.0, 1.0]
accent:            Optional[str] (AccentType value)
accent_strength:   float [0.0, 1.0]
emotion:           Optional[str] (EmotionType value)
emotion_intensity: float [0.0, 1.0]
language:          str (LanguageType code; default "en")
watermark:         Optional[bool] (default: pipeline default)
```

Processing stages inside `generate()`:
1. **Face feature extraction:** ArcFace → FaceMesh → Demographics → FaceFusion → 560-D `e_f`.
2. **CVAE prior sampling:** K diverse `e_s` candidates via temperature-scaled reparameterization.
3. **Staged memory isolation:** Releases visual extractors from VRAM before vocoding (GTX 1650 mode).
4. **Text preprocessing chain:** Cross-lingual → Accent → Speech Difference → final `synth_text`.
5. **Style projection:** For each voice k: `e_s_k → StyleAdapter → s_hat_k (128-D)`.
6. **Speech synthesis:** `StyleTTS2Backend.synthesize(text, style, gender, ...)`.
7. **Post-synthesis filters:** Speech difference waveform filter → Emotion acoustic filter.
8. **Watermarking:** AudioSeal embed (optional).

---

## Module 6: Hardware Auto-Configuration  (`voix/utils/hardware.py`)  [Implemented & Verified]

`detect_hardware()` resolves cross-platform `HardwareProfile`:

| Tier | Device | Pipeline Mode | Precision | Diffusion Steps |
|---|---|---|---|---|
| Budget CUDA (≤4 GB) | GTX 1650 | `staged_memory_isolated` | fp16 | 5 |
| Mid CUDA (6–12 GB) | RTX 3060/4060 | `concurrent_gpu` | fp16 | 10 |
| High CUDA (≥16 GB) | RTX 3090/4090 | `batched_gpu` | fp16 | 20 |
| Apple MPS | M1/M2/M3/M4 | `concurrent_gpu` | fp32 | 10 |
| CPU | Any | `multithreaded_cpu` | fp32 | 3 |

---

## Dataset & Training  [Implemented & Verified]

### VoxCeleb2 HDF5 Feature Store
- **Samples:** 75,720 utterances × (face_features=560-D, speaker_features=192-D, metadata).
- **Identities:** 3,029 unique speakers.
- **Split:** 2,629 train / 400 zero-shot validation (fully disjoint speaker identities).
- **Implementation:** `voix/data/dataset.py` — `VoxCelebPairedDataset` with in-memory preloading.

### Loss Functions  (`voix/training/losses.py`)
- `CosineReconstructionLoss`: $1 - \cos(\hat{\mathbf{e}}_s, \mathbf{e}_s)$, directional optimization on hypersphere.
- `FreeBitsKLLoss`: Dimension-wise free-bits KL ($\lambda_{fb}=0.5$ nats/dim) preventing posterior collapse.
- `CVAELoss`: Combined cosine reconstruction + $\beta(t) \cdot$ KL with annealing schedule.

---

## Known Limitations & Design Decisions

| Decision | Rationale |
|---|---|
| GroupNorm over BatchNorm in CVAE | Batch-size-1 safety at inference; identical behavior during training |
| StyleTTS 2 is completely frozen | Adapter-only training (<200k params) sufficient; full vocoder fine-tuning is expensive and out-of-scope |
| ArcFace Buffalo_L onnxruntime backend | CUDA provider availability depends on system; CPU fallback is automatic |
| Pose filtering (±15°) | Eliminates expression bleed and camera distortion from feature extraction |
| Indian F2S Benchmark — Future Scope | VoxCeleb2 alone is sufficient for demonstrating the core biometric alignment pipeline; regional corpus curation is post-publication work |
