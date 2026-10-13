# architecture.md
> Full system design — modules, data flow, component interfaces, mathematical formulations, and engineering decisions.
> Last updated: 2026-10-13. All claims verified against source code in `voix/`.

---

## Status Legend

| Symbol | Meaning |
|---|---|
| **Implemented & Verified** | Code exists; unit tests pass; empirically measured |
| **Implemented; under-evaluated** | Code exists; limited or no formal benchmark yet |
| **Planned** | Specification written; code not yet committed |
| **Known Gap / Mismatch** | Discrepancy between doc and code flagged explicitly |

---

## System Pipeline (End-to-End)

```
Face Image + Text Prompt
    |
    +-- ArcFace ResNet-50 (frozen, InsightFace Buffalo_L)   --> 512-D identity embedding (e_id)
    +-- MediaPipe FaceMesh (468 3D landmarks)               --> 32-D craniofacial ratios  (e_geo)
    +-- Soft Demographic Estimator                          --> 16-D age/sex posteriors   (e_demo)
                                    |
                    FaceFusionLayer  (Linear 560->560 + LayerNorm + GELU)
                                    |
                             560-D  e_f
                     +--------------+------------------------------+
                     |                                            |
                     v                                            v
          CVAE Conditional Prior                    Exemplar Retrieval Gallery
          p_psi(z | e_f)  -->  sample z in R^192    3,029 speaker centroids (192-D)
          CVAEDecoder  p_theta(e_s | z)              Gender-conditioned kNN (k=5 or k=10)
                     |                                            |
                     +--------------+------------------------------+
                                    |  Hybrid Blending
                                    |  e_s_hybrid = alpha*e_s_hat + (1-alpha)*sum(w_k*c_k)
                                    |  (alpha = 0.70 by default)
                                    v
                          StyleAdapter (MLP 192->256->256->128)
                          Projects ECAPA space --> StyleTTS 2 style space (128-D)
                                    |
                          VocoderStyleBridge
                          s_target = beta*s_hat + (1-beta)*s_anchor  (beta = 0.30)
                                    |
                          StyleTTS 2 (frozen)
                          Inputs: phoneme tokens + style vector s_target (128-D)
                                    |
                          AudioSeal Watermarker (16-bit neural watermark)
                                    |
                          24 kHz Speech Waveforms (Voice 1, Voice 2, ..., Voice K)
```

---

## Module 1: Face Representation  [IMPLEMENTED & VERIFIED]

### 1a. ArcFace Identity Embedding (512-D)
- **Model:** Pretrained ArcFace ResNet-50 (`buffalo_l`, InsightFace) -- completely frozen.
- **Input:** Single frontal face image; pose-filtered frames (|pitch| < 15 deg, |yaw| < 15 deg).
- **Output:** L2-normalized 512-D identity vector e_id in S^511.
- **Source:** `voix/data/face_extractor.py` -- `ArcFaceExtractor.extract_full()` returns `(e_id, gender, age)`.

### 1b. Craniofacial Ratio Features (32-D)  [IMPLEMENTED]
- **Source:** `voix/data/morphology.py` -- `compute_craniofacial_ratios(landmarks)`.
- **Landmarks:** MediaPipe FaceMesh 468-point 3D mesh.
- **Normalization:** All distances divided by inter-ocular distance (IOC, landmarks 33 <-> 263) to remove camera scale.

Key landmark constants (canonical MediaPipe indices):

| Constant | Index | Anatomical Point |
|---|---|---|
| IOC_LEFT | 33 | Left eye outer canthus |
| IOC_RIGHT | 263 | Right eye outer canthus |
| FOREHEAD_TOP | 10 | Trichion |
| CHIN_BOTTOM | 152 | Gnathion / menton |
| JAW_LEFT / RIGHT | 172 / 397 | Mandibular gonion |
| NOSE_TIP | 4 | Pronasale |
| NOSE_BRIDGE | 6 | Nasion |
| UPPER_LIP_TOP | 13 | Labiale superius |

Selected anatomical hypotheses (32 features total):

| Feature | Vocal Correlate Hypothesis |
|---|---|
| Face height/width ratio | Head volume -> overall vocal tract resonance scale |
| Jaw width (IOC-normalized) | Mandibular resonance cavity volume |
| Jaw gonial angle | Oral cavity volume and acoustic radiation impedance |
| Bizygomatic width | Pharyngeal cavity width proxy |
| Nose width / height | Nasal resonance aperture and airway path length |
| Lower face ratio | Oral articulation cavity volume |
| Philtrum length | Lip articulation range |
| Lip width / height | Oral radiation aperture |

> [KNOWN GAP] Biomechanical Note (RQ2): Craniofacial ratio-to-voice correlations are hypotheses grounded in published allometry literature. Their incremental contribution over ArcFace identity alone has not yet been formally ablated in this codebase.

### 1c. Soft Demographic Indicators (16-D)  [IMPLEMENTED]
- **Source:** `voix/data/demographics.py` -- `SoftDemographicEstimator.estimate_from_attributes()`.
- **Output vector layout (16-D):**
  - [0:8]   -- Age bin soft distribution (8 bins: ~0-10, 11-20, ..., 71+), Gaussian centered at detected age (sigma=7.0 yr).
  - [8:10]  -- Biological sex posterior [p_male, p_female] from InsightFace genderage detector.
  - [10:14] -- Vocal age grouping [child, young, adult, senior] -- Gaussian over vocal centers [8, 22, 42, 68] yr.
  - [14:16] -- Apparent build proxy [slender, broad] -- sigmoid on face aspect ratio (threshold ~1.35).
- If InsightFace gender/age is unavailable, falls back to ArcFace embedding projection heuristic.

### 1d. Feature Fusion (FaceFusionLayer)  [IMPLEMENTED]
- **Source:** `voix/models/fusion.py` -- `FaceFusionLayer`.

```
ArcFace Identity          512-D  (e_id,   L2-normalized, in S^511)
FaceMesh Ratios            32-D  (e_geo,  IOC-normalized ratios)
Soft Demographics          16-D  (e_demo, probability distributions in [0,1]^16)
-------------------------------------------
Concatenated              560-D
          |
    Linear(560 -> 560, bias=True)
          |
      LayerNorm(560)
          |
        GELU
          |
Unified Face Representation e_f (560-D)
```

- **Design:** Uses LayerNorm (not BatchNorm/GroupNorm) for training stability across heterogeneous scale ranges.
- **Weights:** NOT auto-loaded on import. Call `load_canonical_weights(path)` explicitly.
- **Mathematical formulation:**
  e_f = GELU( LayerNorm( W_f * [e_id || e_geo || e_demo] + b_f ) )

---

## Module 2: Mapper Architecture

### 2a. CVAE v2 Conditional Generative Mapper  [IMPLEMENTED -- Production Model]
- **Source:** `voix/models/cvae.py`
- **Status:** Primary generative model used in inference pipeline (`pipeline.py`).

> [IMPORTANT CORRECTION] Earlier architecture.md versions described a MetricContrastiveMapper (MLP + InfoNCE) as the primary network. The actual production pipeline uses the CVAE v2 with a StyleAdapter. The contrastive mapper was a training/evaluation paradigm (ablation Models A-F), not the deployed inference architecture.

#### CVAE v2 Components

**Prior Network p_psi(z | e_f) -- CVAEPriorNetwork:**
```
Input: e_f (560-D)
  |
  +-- Linear(560->512) + GroupNorm + GELU
  +-- Linear(512->384) + GroupNorm + GELU
  +-- Linear(384->256) + GroupNorm + GELU
       |              |
  fc_mu(256->192)  fc_logvar(256->192)  [clamped in [-10, 10]]
       |              |
  mu_prior         log_sigma2_prior   (both in R^192)
```

**Recognition Network q_phi(z | e_s, e_f) -- CVAERecognitionNetwork (training only):**
```
Input: [e_s || e_f]  (192 + 560 = 752-D)
  |
  +-- Linear(752->512) + GroupNorm + GELU
  +-- Linear(512->384) + GroupNorm + GELU
  +-- Linear(384->256) + GroupNorm + GELU
       |              |
  fc_mu(256->192)  fc_logvar(256->192)
       |              |
  mu_z             log_sigma2_z    (both in R^192)
```
> At inference, e_s is unavailable; the pipeline samples from the prior p_psi(z | e_f) only.

**Decoder p_theta(e_s | z) -- CVAEDecoder:**
```
Input: z in R^192  (NO e_f skip connection -- by design)
  |
  +-- Linear(192->512) + GroupNorm + GELU
  +-- Linear(512->384) + GroupNorm + GELU
  +-- Linear(384->256) + GroupNorm + GELU
       |
  output_proj(256->192)
       |
  e_s_hat in R^192  (optionally L2-normalized)
```
> Why no e_f skip? Passing e_f directly to the decoder creates a shortcut allowing the decoder to bypass z (posterior collapse via skip connection). Removing the skip forces the decoder to rely entirely on z, encouraging meaningful latent utilization.

**Reparameterization:**
  z = mu + eps * sigma * T,   eps ~ N(0, I),   T in R+ (temperature)

**Voice diversity via sample_voices(e_f, K, temperature):**
- Predicts prior (mu_prior, sigma_prior) from e_f once.
- Samples K independent eps ~ N(0, I), producing K distinct z vectors.
- Decodes each z to K candidate speaker embeddings e_s_hat in R^192.

#### CVAE Training Loss (voix/training/losses.py)

**1. Cosine Reconstruction Loss:**
  L_recon = (1/B) * sum_i [1 - cos_sim(e_s_hat_i, e_s_i)]
  Range: [0, 2] where 0=perfect alignment, 2=anti-parallel.
  Rationale: ECAPA embeddings are angular; cosine distance directly optimizes directional alignment.

**2. Free-Bits KL Divergence (FreeBitsKLLoss):**
  KL_d = 0.5 * [log_sigma2_prior_d - log_sigma2_d - 1 + (sigma2_d / sigma2_prior_d) + (mu_d - mu_prior_d)^2 / sigma2_prior_d]
  L_KL = (1/B) * sum_i sum_d max(lambda_fb, KL_d)
  lambda_fb = 0.5 nats/dim -- prevents posterior collapse for dims encoding < 0.5 nats.
  Active unit metric: dims with KL > 0.1 nats; healthy CVAE expects >= 32 active units.

**3. Combined Objective with beta-Annealing:**
  L_total = L_recon + beta(t) * L_KL
  beta(t) = beta_max * min(1, t / T_warmup)
  beta_max = 1.0, T_warmup = 5,000 steps.

### 2b. Contrastive Mapper Paradigm (Ablation Models B-D)  [EVALUATION / ABLATION ONLY]

> Status: The InfoNCE contrastive mapper was evaluated as Models B, C, D in the ablation benchmark. It is NOT deployed in the production VoixPipeline but IS the architecture whose checkpoint (D_joint_infonce_best.pt) was benchmarked. The CVAE is the active generative model for multi-voice sampling.

Architecture:
```
Input: e_f (560-D)
  +-- Linear(560->512) + GroupNorm + GELU
  +-- Linear(512->384) + GroupNorm + GELU
  +-- Linear(384->256) + GroupNorm + GELU
       |                         |
  Linear(256->192)          Linear(256->1)
  L2 Normalize               Sigmoid
       |                         |
  e_s_hat in S^191          y_gender in [0,1]
```

Supervised InfoNCE Loss (Models C, D):
  L_InfoNCE = -(1/B) sum_i log[ exp(e_s_hat_i . c_yi / tau) / sum_j exp(e_s_hat_i . c_j / tau) ]
  Temperature tau = 0.07; targets = precomputed speaker centroids c_y.

Auxiliary Gender Loss:
  L_gender = -(1/B) sum_i [y_i * log(y_hat_i) + (1-y_i) * log(1-y_hat_i)]
  lambda_gender = 0.50

Combined:
  L_total = L_InfoNCE + 0.50 * L_gender

### 2c. Exemplar Retrieval Gallery (Non-Parametric)  [IMPLEMENTED]
- **Gallery:** `data/processed/retrieval_gallery.pt` -- 3,029 speaker centroids (c_i in S^191).
- **Query:** Given predicted embedding e_s_hat and predicted gender, filter gallery to matching gender partition, retrieve top-K by cosine similarity.
- **Centroid construction:** c_y = normalize( sum_{u in U_y} e_{s,u} ) in S^191.

### 2d. Hybrid Manifold Blending  [IMPLEMENTED]
  e_s_hybrid = alpha * e_s_hat + (1-alpha) * sum_k [w_k * c_k]
  alpha=0.70, K=10. Weights w_k are softmax-normalized cosine similarity scores.
  Rationale: alpha*e_s_hat carries facial identity specificity; (1-alpha)*sum(w_k*c_k) anchors on real human acoustic manifold.

---

## Module 3: Style Adapter  [IMPLEMENTED]
- **Source:** `voix/models/adapter.py` -- StyleAdapter.
- **Role:** Bridges 192-D ECAPA speaker embedding space to 128-D StyleTTS 2 style latent space.

```
Input: e_s in R^192  (CVAE output or hybrid blend)
  |
  Linear(192->256) + LayerNorm(256) + GELU
  |
  Linear(256->256) + LayerNorm(256) + GELU
  |
  Linear(256->128)
  |
  s_hat in R^128  (StyleTTS 2 conditioning vector)
```

- **Parameters:** ~163k (< 200k) -- the only trainable module during Phase 4.
- **Training Loss:** MSE + Cosine alignment:
  L_adapter = MSE(s_hat, s_target) + lambda_cos * (1 - cos_sim(s_hat, s_target)), lambda_cos=1.0
- **Target:** Real StyleTTS 2 style vectors extracted from LibriSpeech reference audio.

---

## Module 4: Vocoder Style Bridge  [IMPLEMENTED]
- **Source:** `voix/models/tts_backend.py` -- VocoderStyleBridge within StyleTTS2Backend.

The Latent Space Mismatch Problem:
  ECAPA-TDNN is trained for speaker verification (angular margin discrimination).
  StyleTTS 2 style latent encodes acoustic timbre, pitch, duration, spectral envelope.
  Direct injection of ECAPA vectors causes artifacts.

Solution: Demographic Reference Anchoring:
```
Hybrid Speaker Embedding  e_s_hybrid in R^192
          |
    StyleAdapter  (192->128)
          |
      s_hat in R^128
          |
    beta*s_hat + (1-beta)*s_anchor   [beta = 0.30]
          |
    s_target in R^128  (final StyleTTS 2 conditioning)
```
- **Anchors:** Curated male/female reference audio from `data/raw/librispeech_ref/`. Style vectors extracted offline.
- **Blending:** beta=0.30 (30% adapter prediction, 70% calibrated demographic anchor).
- **Loudness norm:** EBU R128 LUFS via pyloudnorm (soft peak fallback if unavailable).

---

## Module 5: StyleTTS 2 (Frozen Neural Vocoder)  [IMPLEMENTED]
- **Role:** Zero-shot TTS conditioned on style vector s_target.
- **Input:** Text prompt (phonemized) + style vector s_target (128-D).
- **Output:** 24 kHz float32 speech waveform.
- **State:** Completely frozen (requires_grad = False). Zero fine-tuning.
- **Robust fallback:** Harmonic formant synthesizer for offline / CI testing.

---

## Module 6: AudioSeal Neural Watermarking  [IMPLEMENTED]
- **Source:** `voix/models/watermarking.py` -- AudioSealWatermarker.
- **Method:** AudioSeal (San Roman et al., ICML 2024) -- imperceptible 16-bit in-band neural watermark.
- **SNR:** > 40 dB (inaudible). Detection accuracy > 98% under MP3 / resampling.
- **Payload:** 16-bit cryptographic provenance marker in all generated waveforms.

---

## Module 7: Hardware Auto-Configuration  [IMPLEMENTED]
- **Source:** `voix/utils/hardware.py` -- detect_hardware(), HardwareProfile.
- **Config:** `configs/hardware.yaml` -- 5 canonical profiles.

| Profile | VRAM | Device | batch_voices | Mode |
|---|---|---|---|---|
| tier_1_budget_gpu | <= 4.5 GB | CUDA | false (sequential) | staged_memory_isolated |
| tier_2_mid_gpu | 4.5-12.5 GB | CUDA | true (parallel) | full_parallel |
| tier_3_high_gpu | >= 16 GB | CUDA | true | high_throughput |
| tier_apple_silicon | Unified RAM | MPS | true | mps_optimized |
| tier_cpu_only | N/A | CPU | false | cpu_fallback |

- **Detection:** Queries platform, CUDA/MPS/ROCm/XPU availability, VRAM at runtime.
- **OOM safety:** Tier 1 uses torch.cuda.empty_cache() between voices; ONNX ArcFace on CPU.

---

## Inference: VoixPipeline API  [IMPLEMENTED]
- **Source:** `voix/inference/pipeline.py` -- VoixPipeline.

Constructor:
```python
VoixPipeline(
    cvae_checkpoint: str = "D:/voix/checkpoints/cvae_best.pt",
    adapter_checkpoint: str = "D:/voix/checkpoints/style_adapter_best.pt",
    scaler_path: str = "D:/voix/data/processed/face_scaler.pt",
    device: Optional[str] = None,          # None = auto-detect
    enable_watermarking: bool = True,
    hardware_profile: Optional[HardwareProfile] = None,
)
```

> [KNOWN GAP] No from_pretrained() class method exists. Earlier docs showed this API but it does not exist in the current codebase. Planned for Phase 7 hardening.

Usage:
```python
from voix.inference.pipeline import VoixPipeline

pipeline = VoixPipeline(
    cvae_checkpoint="D:/voix/checkpoints/cvae_best.pt",
    adapter_checkpoint="D:/voix/checkpoints/style_adapter_best.pt",
)

results = pipeline.generate_voices(
    face_image_path="test_face.jpg",
    text_prompt="The voice reflects both physical anatomy and lived culture.",
    num_voices=3,
    strategy="hybrid",
    output_dir="outputs/",
)
```

Inference flow:
1. extract_face_features() -> e_f (560-D), gender, age
2. CVAE.sample_voices(e_f, K, temperature) -> K candidate e_s_hat
3. Exemplar gallery retrieval + hybrid blending -> e_s_hybrid (per voice)
4. StyleAdapter(e_s_hybrid) -> s_hat (128-D per voice)
5. VocoderStyleBridge: s_target = beta*s_hat + (1-beta)*s_anchor
6. StyleTTS2(text, s_target) -> 24 kHz waveform
7. AudioSeal(waveform) -> watermarked output

---

## Training Pipelines

### Phase 3: CVAE Training  [IMPLEMENTED]
- **Trainer:** `voix/training/trainer_adapter.py`
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR
- **Mixed Precision:** AMP FP16
- **Loss:** CVAELoss -- Cosine reconstruction + free-bits KL with beta annealing
- **Checkpoint:** `checkpoints/cvae_best.pt`

### Phase 4: Style Adapter Supervised Training  [IMPLEMENTED]
- **Trainer:** `voix/training/trainer_adapter_supervised.py`
- **Loss:** SupervisedAdapterLoss -- MSE + Cosine alignment (lambda_cos=1.0)
- **Target:** Real StyleTTS 2 style vectors from LibriSpeech reference audio
- **Checkpoint:** `checkpoints/style_adapter_best.pt`

### Optimization Hyperparameters (Verified)

| Parameter | Value | Source |
|---|---|---|
| Optimizer | AdamW | trainer_adapter.py |
| Learning rate | 1e-3 | Config |
| Weight decay | 1e-4 | Config |
| Batch size | 64 | cvae_training.yaml |
| InfoNCE temperature tau | 0.07 | losses.py |
| Gender loss weight lambda | 0.50 | losses.py |
| Free-bits threshold lambda_fb | 0.5 nats/dim | losses.py |
| beta warmup steps | 5,000 | losses.py |
| Mixed precision | AMP FP16 | Trainer |
| Epochs | 40 | Config |

---

## Datasets  [IMPLEMENTED]

### Primary Feature Store (data/processed/voxceleb2_features.h5)
- **Total Utterances:** 75,720 multimodal pairs.
- **Unique Speakers:** 3,029 identities.
- **Training Split:** 2,629 speakers (65,720 utterance pairs) -- identity-disjoint.
- **Zero-Shot Validation:** 400 speakers (10,000 utterance pairs) -- strictly unseen.
- **HDF5 Keys:** face_features (N,560), speaker_embeddings (N,192), speaker_centroids (3029,192), speaker_ids (N,), gender_labels (N,).

### Indian F2S Benchmark  [PLANNED]
- Target: ~20 hours, 40-60 speakers, 5 Indian languages.
- Code-switching: ~25% of utterances.
- Release: Privacy-preserving embeddings only.
- Status: Specification complete; curation not yet started.

---

## Empirical Benchmarks (Verified from results/benchmarks/summary.csv)

> [IMPORTANT] The summary.csv reports different metric values than earlier documentation.
> The CSV uses: val_cos, recall_at_1, recall_at_3, recall_at_10, gender_accuracy, secs, secs_std
> (NOT recall_at_5 as previously documented). Verified values from CSV are used below.

### Benchmark Results (n_synthesized=10 per model unless noted)

| Model | val_cos | R@1 | R@3 | R@10 | Gender Acc | SECS Mean | SECS Std |
|---|---|---|---|---|---|---|---|
| A: Baseline Cosine Regression | 0.1372 | 0.0010 | 0.0025 | 0.0050 | 62.6% | 0.0207 | 0.0673 |
| B: SpkAvg + Gender Loss | 0.1370 | 0.0005 | 0.0020 | 0.0050 | 62.6% | 0.0171 | 0.0722 |
| D: Joint InfoNCE Centroids | 0.1288 | 0.0010 | 0.0020 | 0.0060 | 62.6% | 0.0177 | 0.0600 |
| E: Exemplar Retrieval k=5 | 0.0642 | 0.0010 | 0.0020 | 0.0060 | 100.0% | 0.0252 | 0.0542 |
| F: Hybrid k=10 | 0.1299 | 0.0010 | 0.0020 | 0.0060 | 100.0% | 0.0117 | 0.0608 |
| F: Hybrid k=10 (n=20 samples) | 0.1299 | 0.0010 | 0.0040 | 0.0085 | 100.0% | 0.0324 | 0.0728 |

> Note: WER, FAD, and MOS columns all -1.0 (not measured). Model C not present in CSV.

### Key Empirical Findings
1. Gender fidelity: Models E and F achieve 100.0% gender accuracy vs 62.6% for regression models -- gallery gender conditioning is the key driver.
2. SECS quality: Model F with n=20 achieves highest mean SECS (0.0324). More synthesis samples improve reliability.
3. Val Cosine: Models A and B (0.137) exceed Model D (0.129) in this evaluation run, reflecting the retrieval vs regression trade-off.
4. Recall: All models show low absolute recall values (<0.01) -- this reflects the difficulty of zero-shot speaker retrieval from 400 identities with only 10-20 audio samples.

---

## Software Stack (Verified)

| Component | Library | Version | Source File |
|---|---|---|---|
| Deep Learning | PyTorch | 2.5.1+cu121 | requirements.txt |
| Face Detection | InsightFace / RetinaFace | 2.0 | voix/data/face_extractor.py |
| Face Biometrics | ArcFace ResNet-50 | Buffalo_L | ArcFaceExtractor |
| Morphometry | MediaPipe FaceMesh | 1.0.1 | voix/data/morphology.py |
| Speaker Modeling | SpeechBrain ECAPA-TDNN | 1.1.1 | voix/data/speaker_extractor.py |
| Speech Synthesis | StyleTTS 2 | Frozen | voix/models/tts_backend.py |
| Style Bridging | Custom StyleAdapter | -- | voix/models/adapter.py |
| Audio I/O | Librosa / SoundFile | 1.0.0 / 0.14.0 | requirements.txt |
| Loudness Norm | pyloudnorm (optional) | -- | tts_backend.py |
| HDF5 Storage | h5py | 3.12.x | voix/data/dataset.py |
| Configuration | PyYAML | 6.0.x | voix/utils/config.py |
| Watermarking | AudioSeal | -- | voix/models/watermarking.py |

---

## Hardware & Compute (Verified)

| Resource | Specification | Role |
|---|---|---|
| Local GPU (dev) | NVIDIA GeForce GTX 1650 (4.0 GB VRAM) | Feature extraction, debugging, Tier 1 tests |
| Training GPU | NVIDIA RTX 4050 / Colab T4 | CVAE + adapter training |
| Peak Training VRAM | ~1.8 GB @ batch 64 | Fits within 4 GB constraint |
| Inference VRAM (Tier 1) | ~1.41 GB peak | Sequential, ONNX on CPU |
| Inference VRAM (Tier 2) | ~3.8 GB peak | Parallel batch synthesis |
| Epoch Duration | ~35 s on RTX 4050 | 40 epochs ~= 24 min |

---

## Known Gaps & Open Issues

| Gap | Severity | Status |
|---|---|---|
| from_pretrained() class method missing from VoixPipeline | Medium | Open -- planned Phase 7 |
| Craniofacial ratio contribution (RQ2) not formally ablated | Medium | Planned -- Indian F2S benchmark |
| WER, FAD, MOS metrics not measured in current evaluation | Medium | Planned -- needs ASR + listening study |
| Recall metric is R@3 in CSV vs R@5 in earlier docs | Low | Fixed in this doc |
| Indian F2S Benchmark curation (Phase 5) not started | High | Pending -- core research deliverable |
| CVAE training logs / loss curves not preserved | Low | Recommend W&B/TensorBoard logging |

---

*Read next: [research.md](file:///D:/voix/project-context/research.md)*
