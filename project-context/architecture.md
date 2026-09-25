# architecture.md
> Full system design — modules, data flow, component interfaces, and design decisions.

---

## System Pipeline (End-to-End)

```
Face Image
    |
    +-- ArcFace (frozen)          --> 512-D identity embedding
    +-- MediaPipe FaceMesh        --> 32-D craniofacial ratio features
    +-- Soft Demographic Estimator --> 16-D soft age/sex distributions
    |
    +--> Linear Projection + LayerNorm
    |
    v
Fused Face Representation (560-D)
    |
    v
CVAE Probabilistic Mapper
    |   Encoder: 560-D --> mu (192-D) + log_sigma^2 (192-D)
    |   Reparameterization: z = mu + sigma * epsilon
    |   Decoder: z (192-D) --> reconstructed speaker embedding e_s_hat (192-D)
    |
    v
Style Projection Adapter (trainable MLP)
    |   e_s_hat (192-D ECAPA space) --> s_hat (StyleTTS2 style dim)
    |
    v
StyleTTS2 (frozen)
    |   Input: text (phoneme sequence) + style embedding s_hat
    |
    v
Synthesized Speech Waveform (Voice_1, Voice_2, ..., Voice_K)
```

---

## Module 1: Face Representation

### 1a. ArcFace Identity Embedding (512-D)
- **Model:** Pretrained ArcFace ResNet-50 (InsightFace) — frozen throughout
- **Input:** Single face image — most frontal, most neutral frame per speaker clip (selected via MediaPipe pose angle estimation)
- **Output:** L2-normalized 512-D identity vector
- **What it encodes:** Discriminative facial geometry for identity — bone structure, inter-feature distances

### 1b. Craniofacial Ratio Features (32-D)
- **Source:** MediaPipe FaceMesh 468-point landmarks
- **Processing:** Normalize to canonical face space by inter-ocular distance to remove perspective/scale artifacts
- **Features (30-32 dimensions):**

| Feature | Vocal Correlate Hypothesis |
|---|---|
| Face height / width ratio | Head volume -> resonance cavity size |
| Jaw width | Mandibular resonance |
| Jaw angle | Oral resonance shape |
| Cheekbone width | Facial cavity width |
| Forehead ratio | Skull dimensions |
| Chin length | Subglottal anatomy proxy |
| Nose width | Nasal resonance |
| Nose height | Nasal tract length |
| Eye separation | Interorbital distance |
| Lower face ratio | Oral cavity proportion |
| Mandible width | Jaw acoustic coupling |
| Philtrum length | Lip/oral articulation |
| Upper/lower lip height | Articulatory range |
| Lip width | Oral aperture |
| Neck width proxy | Pharyngeal resonance |

> These features are explicit hypotheses for ablation (RQ2). If SHAP shows they add no value, that is a publishable negative result.

### 1c. Soft Demographic Indicators (16-D)
- **Variables:** Apparent age (Gaussian mean + variance), apparent sex (softmax 2-D), body build proxy, vocal age grouping
- **Key constraint:** Soft probability distributions — NOT hard demographic labels
- **Gradient gating:** Learned gate down-weights demographic gradient contribution during training
- **Excluded features:** Expression features (neutrality, smile, eye openness) — transient signals that destabilize identity-based prediction

### 1d. Feature Fusion
```
ArcFace Identity       512-D
FaceMesh Ratios         32-D
Soft Demographics       16-D
-------------------------------
Concatenated           560-D
         |
    Linear -> LayerNorm -> GELU
         |
Unified Face Representation (560-D)
```
**Fusion choice:** Simple learned linear projection — no transformer fusion. Justified by fixed 560-D input size and compute constraints.

---

## Module 2: CVAE Probabilistic Mapper

### Architecture Choice Rationale

| Approach | Decision | Reason |
|---|---|---|
| Simple Gaussian Head (MLP -> mu, sigma) | Rejected | No learned prior; sigma is unconstrained noise; mode collapse likely |
| Normalizing Flow | Rejected | Infeasible compute on Colab T4 for 192-D target |
| CVAE | Selected | Principled KL regularization; prevents collapse; established training recipes |

### Encoder q_phi(z | e_f)
```
Input:   560-D face representation
Layer 1: Linear(560 -> 512) + BatchNorm + GELU
Layer 2: Linear(512 -> 384) + BatchNorm + GELU
Layer 3: Linear(384 -> 256) + BatchNorm + GELU
Layer 4: Split ->
    mu_head:       Linear(256 -> 192)
    log_sigma2_head: Linear(256 -> 192)
Reparameterization: z = mu + exp(0.5 * log_sigma^2) * epsilon,  epsilon ~ N(0,I)
```

### Decoder p_theta(e_s_hat | z)
```
Input:   192-D sampled z
Layer 1: Linear(192 -> 256) + BatchNorm + GELU
Layer 2: Linear(256 -> 192)   --> reconstructed speaker embedding e_s_hat
```

### Prior
- `p(z) = N(0, I)` — standard isotropic Gaussian

### Posterior Collapse Prevention
- **Beta-VAE KL annealing:** beta increases linearly from 0 to beta_max over first 30% of training steps
- **Free bits:** Minimum KL per dimension = 0.5 nats (lambda_fb) — prevents selective collapse on individual dimensions
- **Monitoring:** If KL < 0.01 nats after warmup period, stop and diagnose

---

## Module 3: Style Projection Adapter

**Problem:** CVAE decoder outputs in ECAPA-TDNN space (trained for speaker verification). StyleTTS2 style conditioning is in its own internally learned space. Direct injection produces poor synthesis.

**Solution:**
```
Input:   e_s_hat (192-D, ECAPA space)
Layer 1: Linear(192 -> 256) + GELU
Layer 2: Linear(256 -> 256) + GELU
Layer 3: Linear(256 -> StyleTTS2_style_dim)   [confirm dim from codebase: ~128-D or 256-D]
Output:  s_hat (StyleTTS2 style embedding)
```

**Training signal:** Pass s_hat through frozen StyleTTS2 and compute SECS against reference audio. Only adapter parameters update — StyleTTS2 remains frozen.

---

## Module 4: StyleTTS2 (Frozen Synthesis Backend)

- **Role:** Zero-shot TTS conditioned on predicted style embedding
- **Input:** Phoneme sequence (from text) + style embedding s_hat
- **Output:** Synthesized speech waveform (24 kHz)
- **Training status:** Completely frozen — not fine-tuned
- **Integration point:** The adapter (Module 3) is the only trainable bridge

---

## Training

### Loss Function
```
L_total = L_recon + beta(t) * L_KL + lambda_adapt * L_style
```

**L_recon** — Cosine similarity loss (not MSE — ECAPA embeddings are direction-based):
```
L_recon = 1 - cosine_similarity(e_s_hat, e_s_gt)
```

**L_KL** — KL divergence with free bits per dimension:
```
L_KL = sum_d max(lambda_fb, KL(q_phi(z_d | e_f) || p(z_d)))
lambda_fb = 0.5 nats
```

**L_style** — SECS loss on StyleTTS2 output (trains adapter):
```
L_style = SECS(StyleTTS2(T, s_hat), ECAPA(reference_audio))
```

**Beta annealing schedule:**
```
beta(t) = beta_max * min(1, t / t_warmup)
t_warmup = 30% of total training steps
beta_max in {0.1, 0.5, 1.0}  [hyperparameter search]
```

### Hyperparameter Grid
| Parameter | Search Range | Default |
|---|---|---|
| beta_max | {0.1, 0.5, 1.0} | 0.5 |
| lambda_adapt | {0.1, 0.5, 1.0} | 0.5 |
| lambda_fb (free bits) | {0.1, 0.5, 1.0} nats | 0.5 |
| Learning rate | {1e-4, 5e-4} | 1e-4 |
| Batch size | {64, 128} | 64 |
| KL warmup % | {20%, 30%, 40%} | 30% |

### Training Data Structure
- **Dataset:** VoxCeleb2 — 2,000 speakers x 50 utterances = 100K training pairs (initial)
- **Pairs:** (e_face_560D, e_speaker_192D) — utterance-level ECAPA targets, not speaker-averaged
- **Rationale for utterance-level:** Speaker-averaged targets cause sigma to collapse to 0. Utterance-level targets expose within-speaker variation, teaching the model a meaningful sigma.

---

## Inference: Multiple Voice Generation

```python
def generate_voices(face_image, text, K=3):
    # Feature extraction
    e_arcface = arcface_encoder(face_image)         # 512-D
    e_morph   = facemesh_ratios(face_image)         # 32-D
    e_demog   = demographic_estimator(face_image)   # 16-D
    e_face    = fusion_layer(concat([e_arcface, e_morph, e_demog]))  # 560-D

    # Predict distribution
    mu, log_var = cvae_encoder(e_face)
    sigma = exp(0.5 * log_var)

    # Sample K voices
    voices = []
    for k in range(K):
        z_k      = mu + sigma * randn_like(sigma)
        e_s_k    = cvae_decoder(z_k)
        style_k  = adapter(e_s_k)
        speech_k = styletts2(text, style_k)
        voices.append(speech_k)

    return voices, mu, sigma
```

---

## Datasets

### Training: VoxCeleb2
| Attribute | Value |
|---|---|
| Identities | 6,112 speakers |
| Utterances | ~1.1 million clips |
| Duration | ~2,442 hours |
| Usage | Extract ArcFace + ECAPA + FaceMesh embeddings; form 100K training pairs |

### In-Domain Evaluation: VoxCeleb1
| Attribute | Value |
|---|---|
| Identities | 1,251 speakers |
| Utterances | ~153,000 clips |
| Usage | Recall@K, SECS, speaker identity generalization |

### Out-of-Domain Evaluation: India F2S Benchmark
| Attribute | Target |
|---|---|
| Duration | ~20 hours |
| Speakers | 40-60 |
| Languages | Hindi, Tamil, Telugu, Bengali, Marathi |
| Code-switching | ~25% of clips |
| Gender ratio | 50:50 M:F |
| Audio | 16 kHz mono WAV |
| Video | 224x224 face crop |
| Sources | NPTEL (CC-licensed), AI4Bharat, IndicTTS |
| Release format | Embeddings only (no raw A/V) |

**Data pipeline:**
```
Source video (NPTEL / YouTube stream)
    -> yt-dlp (streaming mode, no full download)
    -> Audio demux (ffmpeg 16kHz) + Face detect (RetinaFace 224x224)
    -> Active Speaker Detection (SyncNet > 4.5)
    -> Identity verification (ArcFace cosine purity)
    -> ASR transcript (Whisper large-v3)
    -> Language ID (IndicLID for code-switch detection)
    -> Quality filters: SNR > 20dB, clip duration 3-10s, no background music
    -> Save: face_crops/, audio/, transcripts/, embeddings/
```

---

## Evaluation Metrics

### Objective
| Metric | Formula | Purpose |
|---|---|---|
| SECS | Mean cosine sim(ECAPA(Speech_k), e_s_gt) over K samples | Speaker similarity |
| Recall@K | Fraction of correct retrievals at K from N-candidate pool | Identity retrieval |
| Diversity Score (DS) | Mean pairwise cosine distance between K sampled embeddings | Voice diversity |
| ECE | |P_predicted - P_actual| coverage calibration | Uncertainty calibration |
| FSD | Frechet distance between predicted and real embedding distributions | Distribution quality |

> DS is only meaningful paired with SECS. High DS + low SECS = diverse but wrong voices.

### Subjective
| Metric | Scale | Purpose |
|---|---|---|
| MOS naturalness | 1-5 (P.808) | Voice quality |
| Face-voice consistency MOS | 1-5 | Perceptual plausibility |
| A/B preference test | % win/loss/tie | vs. deterministic baseline |
| Diversity preference test | % natural vs. random | Perceptual diversity quality |

### Baselines
| ID | Description | Purpose |
|---|---|---|
| B1 | Random speaker embedding | Lower bound |
| B2 | Nearest-neighbor face retrieval | Non-generative strong baseline |
| B3 | Deterministic mapper (beta=0, sigma ignored) | Tests value of probabilistic component |
| B4 | ArcFace-only input CVAE | Feature ablation A1 |
| B5 | Zero-Shot F2S (2026 arXiv) | Closest prior work |
| B6 | GMM-style implicit generation | Comparison to implicit probabilistic baseline |

---

## Ablation Studies

### Feature Ablation (RQ2)
| Experiment | Input | Tests |
|---|---|---|
| A1 | ArcFace 512-D only | Latent baseline |
| A2 | FaceMesh 32-D only | Morphology standalone |
| A3 | Demographics 16-D only | Demographics standalone |
| A4 | ArcFace + FaceMesh 544-D | Does morphology add to latent? |
| A5 | ArcFace + Demographics 528-D | Does demographics add to latent? |
| A6 | Full 560-D | Full model |

### Architecture Ablation
| Experiment | Modification |
|---|---|
| B1 | beta=0 throughout (deterministic) |
| B2 | KL without free bits |
| B3 | KL without annealing (beta_max from step 0) |
| B4 | Full CVAE |

---

## Software Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch 2.x |
| Face Detection | RetinaFace (pytorch) |
| Face Embedding | ArcFace (InsightFace) |
| Landmark Extraction | MediaPipe 0.10.x |
| Speaker Embedding | ECAPA-TDNN (SpeechBrain) |
| Speech Synthesis | StyleTTS2 |
| Active Speaker Detection | SyncNet (VGG implementation) |
| ASR | OpenAI Whisper large-v3 |
| Language ID | IndicLID (AI4Bharat) |
| Audio Processing | librosa, SoundFile, ffmpeg |
| Video Streaming | yt-dlp, OpenCV |
| Experiment Tracking | Weights & Biases (free tier) |
| Reproducibility | torch.manual_seed(42), 3-seed reporting |
| Model Hosting | HuggingFace Hub |

---

## Hardware & Compute

| Resource | Specification | Role |
|---|---|---|
| Primary GPU | RTX 4050 (6GB VRAM) | CVAE training, ArcFace extraction |
| Backup GPU | Colab T4 (15GB VRAM) | Large batch extraction, StyleTTS2 inference |
| Storage | Google Drive (15GB) | Processed embeddings + Indian dataset |

**Compute estimates:**
- VoxCeleb2 ArcFace extraction (100K): ~8 hrs on Colab T4
- VoxCeleb2 ECAPA extraction (100K): ~10 hrs on Colab T4
- MediaPipe FaceMesh extraction: ~12 hrs on CPU/RTX 4050
- Indian dataset collection: ~3-5 days
- CVAE mapper training (100K pairs, 50 epochs): ~1-2 days on RTX 4050
- StyleTTS2 adapter fine-tuning: ~6 hrs on Colab T4
- Total compute: ~10-14 days

---

*Read next: research.md*
