# VOIX: Probabilistic Face-to-Voice Generation Through Cross-Modal Identity Modeling
## Master Research Document — Single Source of Truth
### Version 1.0 | 2026-09-22 | Status: Pre-Literature-Survey Draft

---

> [!IMPORTANT]
> This document is the **authoritative specification** for the Voix research project. Every design decision recorded here was resolved through the Phase 1–2 analysis and critical review process. It supersedes all prior drafts. The literature survey will be conducted after this document is finalized and may trigger targeted revisions to the novelty claims and related work section.

---

## Executive Summary

Voix is a single research paper proposing a **probabilistic framework for Face-to-Voice (F2V) generation** that models the inherently one-to-many relationship between facial appearance and vocal identity. The system predicts not a single voice from a face, but a **conditional distribution over plausible speaker representations** — capturing the biological and psychological reality that a given face constrains but does not uniquely determine a voice.

The paper makes **three tightly integrated contributions**:

1. **A probabilistic F2V model** using a Conditional Variational Autoencoder (CVAE)-based mapper that predicts Gaussian parameters (μ, σ) over the ECAPA-TDNN speaker embedding space, enabling multiple distinct but perceptually consistent voices to be sampled from a single face image.

2. **An India-specific F2S benchmark dataset** — a curated, linguistically diverse audio-visual corpus (~20 hours, 40–60 speakers, 5 Indian languages, 25% code-switching) — constructed via a storage-efficient streaming pipeline, serving as an out-of-domain evaluation benchmark demonstrating demographic generalization failure in Western-trained F2V systems and improvement through fine-tuning.

3. **A systematic ablation study** of facial feature types (ArcFace latent identity vs. craniofacial morphological ratios vs. soft demographic indicators) establishing which facial information actually contributes to voice prediction — a scientific question with genuine research value independent of synthesis quality.

The system is trained on VoxCeleb2 (primary), evaluated on VoxCeleb1 (in-domain), and further evaluated on the India-specific benchmark (out-of-domain, cross-linguistic generalization). A frozen StyleTTS2 TTS engine, conditioned via a learned projection adapter, synthesizes the final speech from text and predicted speaker embeddings. The NPC voice generation demo is a qualitative showcase only, not a research claim.

**Infrastructure**: Solo researcher, consumer GPU (RTX 4050 / Colab T4), Google Drive for data offload, 16-week timeline.

---

## 1. Research Motivation

### 1.1 The Determinism Problem in Face-to-Voice Synthesis

Every major Face-to-Voice and Face-to-Speech system published to date — Face-TTS, Face-StyleSpeech, FaceSpeak, Vclip, Zero-Shot F2S — implicitly or explicitly models the face-to-voice relationship as a **deterministic function**:

```
f : Face Image → Single Speaker Embedding → Single Voice
```

This formulation is computationally convenient but scientifically incorrect. The human voice is shaped by two fundamentally different categories of factors:

**Physiological factors** — partially inferable from facial appearance:
- Vocal tract length and shape (correlated with craniofacial geometry)
- Subglottal resonance characteristics (correlated with body mass, neck structure)
- Laryngeal anatomy (partially correlated with sex and age)
- Overall head and jaw structure (correlated with lower-frequency resonance)

**Learned and environmental factors** — completely invisible in facial appearance:
- Accent and dialect (product of geographic/social origin)
- Speaking style and prosody (product of education, personality, context)
- Code-switching habits (product of multilingual upbringing)
- Emotional expression patterns (product of personality and culture)
- Vocal fry, breathiness, nasality habits (product of learned patterns)

Therefore, the correct computational formulation is:

```
P(voice | face) : Face Image → Distribution over Speaker Embeddings → Multiple Plausible Voices
```

A face provides *evidence* about voice, not *specification* of voice. Treating this as a probability distribution is not a stylistic modeling choice — it is the scientifically accurate formulation.

### 1.2 The Indian Representation Gap

Every established F2V/F2S dataset and model is trained on Western, primarily English-speaking faces and voices:
- VoxCeleb1/2: Predominantly Western celebrities, English and European languages
- LRS3: English TED talks
- EM²TTS: Primarily Western speakers

Indian faces and voices present unique challenges that are systematically out-of-distribution for these systems:
- Diverse craniofacial morphology across 5 major regional genetic clusters
- Viseme (visual phoneme) patterns specific to retroflex, dental, and other Indian consonants absent in English
- Harmonic prosodic structures from tonal regional languages (Tamil, Telugu)
- Code-switching — fluid intra-sentence mixing of Indian languages and English (Hinglish, Tanglish, Benglish) — a linguistic phenomenon almost entirely absent in Western datasets
- Gender presentation norms that differ from Western training distribution baselines

No existing F2S dataset or system addresses this gap at all.

---

## 2. Problem Statement

### 2.1 Formal Problem Definition

Given a face image **I** and optional text input **T**, learn a generative model:

```
P(e_s | e_f) : Speaker Embedding Distribution conditioned on Face Embedding
```

Where:
- `e_f ∈ R^D_f` — face representation (multi-feature, fused)
- `e_s ∈ R^192` — speaker embedding in ECAPA-TDNN space
- `P(e_s | e_f) = N(μ_θ(e_f), σ_θ²(e_f))` — predicted Gaussian over speaker embedding space

At inference, generate K plausible voices:
```
z_k ~ N(μ_θ(e_f), σ_θ²(e_f)),  k = 1,...,K
Speech_k = StyleTTS2(text=T, style=P_adapter(z_k))
```

Where `P_adapter` is a learned projection from ECAPA space to StyleTTS2 style space.

### 2.2 Primary Research Questions

**RQ1 (Probabilistic Modeling)**:
> Can a Conditional VAE-based mapper learn a meaningful distribution over speaker embeddings conditioned on facial appearance, such that sampled voices are both perceptually plausible (relative to the face) and mutually diverse?

**RQ2 (Feature Contribution)**:
> Does incorporating explicit craniofacial morphological features (biometric ratios from MediaPipe FaceMesh) alongside ArcFace latent identity embeddings improve face-to-voice prediction over using latent embeddings alone?

**RQ3 (Demographic Generalization)**:
> Do F2V models trained on Western (VoxCeleb2) data exhibit measurable performance degradation on Indian faces/voices, and does fine-tuning on an India-specific benchmark recover this performance gap?

### 2.3 Core Hypotheses

**H1**: A CVAE-based face-to-speaker mapper trained with utterance-level embedding pairs will learn a non-trivial posterior distribution `q(z|e_f)` that, when sampled, produces speaker embeddings with higher face-voice perceptual consistency scores than a noise-added deterministic baseline.

**H2**: Craniofacial ratio features extracted from MediaPipe FaceMesh contribute incrementally predictive information about vocal characteristics (specifically pitch range F0 and formant structure) beyond what ArcFace latent embeddings alone capture. This will be demonstrated through SHAP feature attribution and ablation.

**H3**: Zero-shot evaluation of a VoxCeleb2-trained F2V model on the India-specific benchmark will show statistically significant performance degradation (≥10% drop in SECS and Recall@5) compared to VoxCeleb1 evaluation, demonstrating the demographic distribution gap. Fine-tuning on the Indian dataset will partially recover this performance gap.

---

## 3. Literature Context & Research Gaps

> [!NOTE]
> This section contains the **pre-survey framework**. The literature survey conducted after this document will populate the specific citation grid, challenge or confirm the novelty claims, and may revise specific positions.

### 3.1 Key Prior Work (Pre-Survey Summary)

| Paper | Year | Venue | Core Contribution | Limitation |
|---|---|---|---|---|
| Face-StyleSpeech | 2023 | arXiv | Face → speaker embedding → TTS (deterministic) | Deterministic, single voice, Western-only |
| FaceSpeak | 2025 | AAAI | Face + emotion → expressive TTS | Emotion disentanglement, but deterministic, no uncertainty |
| Vclip | 2026 | arXiv | CLIP face features + GMM speaker generation | GMM is implicit uncertainty — not a learned conditional distribution; retrieval-based |
| Zero-Shot F2S | 2026 | arXiv | Face Adapter → StyleTTS2 style space (frozen) | Deterministic, latent features only, no morphology |
| VoxCeleb1/2 | 2017/2018 | INTERSPEECH/BMVC | Audiovisual dataset creation methodology | Western-only, no Indian representation |

### 3.2 Research Gaps This Work Addresses

| Gap | Status in Literature | This Work's Contribution |
|---|---|---|
| Explicit probabilistic F2V (learned μ, σ) | Absent (Vclip GMM is implicit, not conditioned) | CVAE-based conditional distribution |
| Craniofacial morphology features in F2V | Absent | Engineered biometric ratios from FaceMesh |
| One-to-many voice sampling from one face | Absent | K-sample generation with diversity metric |
| Calibrated uncertainty quantification for F2V | Absent | Calibration error metric + diversity score |
| Indian demographic/linguistic representation | Absent in all F2S datasets | India-specific benchmark dataset |
| Code-switching (Hinglish, Tanglish) in F2S | Absent | 25% code-switching data in Indian benchmark |
| Cross-demographic generalization study | Absent | Zero-shot + fine-tune on India benchmark |

### 3.3 Novelty Positioning (Post-Literature-Survey to Verify)

The primary novelty claim is:
> **First work to explicitly model Face-to-Voice synthesis as a learned conditional Gaussian distribution, predicting per-face distributional parameters (μ, σ) from a multi-feature facial representation that includes both latent identity embeddings and handcrafted craniofacial morphological features.**

This claim's validity will be confirmed or refined by the literature survey. The fallback position if prior probabilistic F2V work is found: demonstrate that our approach is superior in calibration, interpretability (morphology features), and cross-demographic generalization (Indian benchmark).

---

## 4. Objectives

1. **O1**: Design and implement a CVAE-based probabilistic mapper that learns `P(e_s | e_f)` from VoxCeleb2 face-voice pairs.
2. **O2**: Construct a multi-feature face representation integrating ArcFace identity (512-D), FaceMesh craniofacial ratios (32-D), and soft demographic indicators (16-D) as a fused 560-D input.
3. **O3**: Design and train a StyleTTS2 projection adapter that maps predicted speaker embeddings from ECAPA space to StyleTTS2 style conditioning space.
4. **O4**: Build and release the India-specific F2S benchmark dataset (~20 hours, 40–60 speakers, 5 languages, under appropriate licensing).
5. **O5**: Conduct systematic ablation on feature type contributions and probabilistic vs. deterministic modeling.
6. **O6**: Evaluate cross-demographic generalization via zero-shot and fine-tuning on the Indian benchmark.

---

## 5. System Architecture

### 5.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         FACE REPRESENTATION                      │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ ArcFace      │   │ MediaPipe    │   │ Soft Demographics  │  │
│  │ (Frozen)     │   │ FaceMesh     │   │ Estimator          │  │
│  │ 512-D        │   │ → 32-D Ratios│   │ 16-D (Age + Sex    │  │
│  │              │   │              │   │  distributions)    │  │
│  └──────┬───────┘   └──────┬───────┘   └────────┬───────────┘  │
│         └──────────────────┴───────────────────── ┘             │
│                            ↓                                     │
│              Linear Projection + LayerNorm → 560-D              │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                    CVAE PROBABILISTIC MAPPER                      │
│                                                                   │
│   Encoder: 560-D → [4-layer MLP] → μ (192-D) + log_σ² (192-D) │
│   Prior:   N(0, I) with KL annealing (β-VAE schedule)           │
│   Reparameterization: z = μ + σ * ε,  ε ~ N(0, I)              │
│   Decoder: z (192-D) → [2-layer MLP] → ê_s (192-D)             │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                    STYLE PROJECTION ADAPTER                       │
│                                                                   │
│   Input:  ê_s (192-D, ECAPA space)                              │
│   Output: ŝ (StyleTTS2 style dim)                               │
│   Architecture: 3-layer MLP, trainable jointly with mapper      │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                       StyleTTS2 (FROZEN)                          │
│                                                                   │
│   Input:  Text (phoneme sequence) + Style Embedding ŝ           │
│   Output: Synthesized speech waveform                            │
└──────────────────────────────────────────────────────────────────┘
                               ↓
                  Voice_1, Voice_2, ..., Voice_K
```

### 5.2 Face Representation Module (Detailed)

#### 5.2.1 ArcFace Identity Embedding (512-D)

- **Model**: Pretrained ArcFace (ResNet-50 backbone), frozen throughout training
- **Input**: Single face image, selected via head-pose-quality scoring (most frontal, most neutral frame per speaker clip — see Section 7.3)
- **Output**: L2-normalized 512-D identity vector
- **What it encodes**: Discriminative facial geometry for identity verification — bone structure, inter-feature distances, unique identity geometry

#### 5.2.2 Craniofacial Ratio Features (32-D)

- **Source**: MediaPipe FaceMesh 468-point landmark set
- **Processing**: Normalize landmarks to canonical face space (by inter-ocular distance) to remove perspective/scale artifacts
- **Engineered features** (30–32 dimensions):

| Feature | Landmark Points | Vocal Correlate Hypothesis |
|---|---|---|
| Face Height / Width Ratio | Chin → Forehead / Cheek-to-cheek | Overall head volume → resonance cavity size |
| Jaw Width | Left jaw angle → Right jaw angle | Mandibular resonance |
| Jaw Angle (degrees) | Gonion-to-Menton-to-Gonion | Jaw shape → oral resonance |
| Cheekbone Width | Left/right zygomatic arch landmarks | Facial cavity width |
| Forehead Ratio | Forehead height / Face height | Skull dimensions |
| Chin Length | Subnasal → Menton | Subglottal anatomy proxy |
| Nose Width | Left/right alar base | Nasal resonance |
| Nose Height | Nasion → Tip | Nasal tract length |
| Eye Separation | Inner/outer canthi | Interorbital distance |
| Lower Face Ratio | Nose base to chin / total face height | Oral cavity proportion |
| Mandible Width | Left/right mandible edge | Jaw acoustic coupling |
| Philtrum Length | Subnasal → Labrale superior | Lip/oral articulation |
| Upper Lip Height | Labrale superior → Stomion | Articulatory range |
| Lower Lip Height | Stomion → Labrale inferior | Articulatory range |
| Lip Width | Left/right cheilion | Oral aperture |
| Neck Width proxy | Mastoid-to-mastoid approximation | Pharyngeal resonance |

> [!NOTE]
> These features are framed explicitly as **hypotheses for ablation testing** (RQ2), not as proven correlates. The ablation studies will determine whether they add predictive value. If SHAP shows them to be noise, this negative result is itself a publishable scientific finding: "Craniofacial morphological ratios extracted from 2D images do not contribute measurable F2V predictive power beyond latent identity embeddings."

#### 5.2.3 Soft Demographic Indicators (16-D)

- **Estimated variables** (as probability distributions, not hard labels):
  - Apparent age: Gaussian mean + variance over [0, 100] → 2-D per feature
  - Apparent sex: Softmax-based probability distribution over binary categories → 2-D
  - Additional voice-relevant proxies: body build proxy (face width-to-height), vocal age grouping
- **Total**: 16-D vector of soft probabilistic estimates
- **Key constraint**: These are **soft conditioning signals** used as regularizing priors, not primary discriminative features. They are NOT used as hard demographic classifiers. Their gradient contribution is down-weighted via a learned gate during training.
- **Ethical note**: Explicit limitations of facial demographic inference will be documented in the paper (see Section 13.4)

> [!WARNING]
> Expression features (neutrality, smile intensity, eye openness) are explicitly **excluded** from the main mapper inputs. They are transient signals that would introduce instability into identity-based voice prediction. They are documented in Section 15 (Future Work) as a potential conditioning signal for a separate prosody/emotion branch.

#### 5.2.4 Feature Fusion

```
ArcFace Identity      512-D
FaceMesh Ratios        32-D
Soft Demographics      16-D
─────────────────────────────
Concatenated          560-D
       ↓
  Linear → LayerNorm → GELU
       ↓
  Unified Face Representation (560-D)
```

The fusion is a simple learned linear projection with layer normalization — no transformer fusion. Justification: The three feature types are already well-specified and independently extracted; a transformer fusion adds parameter overhead without clear benefit on a 560-D input at this scale. An MLP fusion is more appropriate for the compute constraints of this project.

### 5.3 CVAE Probabilistic Mapper (Detailed)

#### 5.3.1 Architecture Choice: CVAE (not Normalizing Flow, not simple Gaussian head)

**Decision rationale**:

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Simple Gaussian Head (MLP → μ, σ) | Simplest, lowest params | No learned prior structure; mode collapse likely; σ is unconstrained noise | Rejected — too fragile |
| Normalizing Flow | Richest distributions, invertible | High param count, slow on Colab T4, complex training, overkill for 192-D target | Rejected — infeasible compute |
| **CVAE (Conditional VAE)** | Principled probabilistic framework; KL regularization prevents collapse; latent z interpretable; well-established training recipes | Requires careful β annealing | **Selected** |

**CVAE Structure**:

```
Encoder q_φ(z | e_f):
  Input: 560-D face representation
  Layer 1: Linear(560 → 512) + BatchNorm + GELU
  Layer 2: Linear(512 → 384) + BatchNorm + GELU
  Layer 3: Linear(384 → 256) + BatchNorm + GELU
  Layer 4: Split → μ_head: Linear(256 → 192)
                   log_σ²_head: Linear(256 → 192)
  Reparameterization: z = μ + exp(0.5 * log_σ²) * ε,  ε ~ N(0,I)

Decoder p_θ(ê_s | z):
  Input: 192-D sampled z
  Layer 1: Linear(192 → 256) + BatchNorm + GELU
  Layer 2: Linear(256 → 192) — outputs reconstructed speaker embedding ê_s
```

**Prior**: `p(z) = N(0, I)` — standard isotropic Gaussian

**Reparameterization**: Standard VAE reparameterization trick — gradients flow through μ and σ; ε is sampled noise.

**Posterior collapse prevention**:
- β-VAE KL annealing: β increases linearly from 0 → β_max over first 30% of training, then held constant. β_max is a hyperparameter (search range: 0.1 – 1.0).
- Free bits strategy: Minimum KL per dimension (λ = 0.5 nats) to prevent selective posterior collapse on individual latent dimensions.
- Monitoring: Track KL divergence per batch during training; if KL < 0.01 nats after warm-up, stop and diagnose.

#### 5.3.2 Style Projection Adapter

**Problem**: The CVAE decoder outputs ê_s in the ECAPA-TDNN embedding space (192-D, trained for speaker verification). StyleTTS2's style conditioning operates in its own internally learned style space. Direct injection of ê_s into StyleTTS2 without adaptation will produce poor-quality synthesis because the spaces are not aligned.

**Solution**: A trainable 3-layer MLP projection adapter:

```
Adapter P_adapter:
  Input:  ê_s (192-D, ECAPA space)
  Layer 1: Linear(192 → 256) + GELU
  Layer 2: Linear(256 → 256) + GELU
  Layer 3: Linear(256 → StyleTTS2_style_dim)
  Output: ŝ (StyleTTS2 style embedding dimension)
```

The StyleTTS2 style encoder outputs an embedding from its internal style space. We extract the style dimension from the StyleTTS2 codebase (typically 128-D or 256-D depending on model variant — to be confirmed during implementation). The adapter is trained jointly with the CVAE mapper.

**Training signal for the adapter**: Pass ŝ through frozen StyleTTS2 and compute speech-level reconstruction losses against the reference audio from VoxCeleb2. This end-to-end fine-tuning of the adapter (with StyleTTS2 frozen) teaches the adapter to produce style embeddings that actually generate recognizable voices.

### 5.4 Inference: Multiple Voice Generation

```python
# Inference pseudocode
def generate_voices(face_image, text, K=3):
    # Extract face features
    e_arcface = arcface_encoder(face_image)   # 512-D
    e_morph = facemesh_ratios(face_image)     # 32-D
    e_demog = demographic_estimator(face_image) # 16-D
    
    e_face = fusion_layer(concat([e_arcface, e_morph, e_demog]))  # 560-D
    
    # Predict distribution
    mu, log_var = cvae_encoder(e_face)        # 192-D each
    sigma = exp(0.5 * log_var)
    
    # Sample K voices
    voices = []
    for k in range(K):
        z_k = mu + sigma * randn_like(sigma)  # Reparameterization
        e_s_k = cvae_decoder(z_k)             # 192-D
        style_k = adapter(e_s_k)              # StyleTTS2 style dim
        speech_k = styletts2(text, style_k)   # Waveform
        voices.append(speech_k)
    
    return voices, mu, sigma
```

---

## 6. Training Methodology

### 6.1 Training Data: VoxCeleb2

- **Dataset**: VoxCeleb2 (Nagrani et al., 2018) — 6,112 speakers, ~1.1M utterances
- **Data pairs used**: `(face_frame, audio_utterance)` per clip
- **Scale for solo researcher**: Full VoxCeleb2 is feasible for embedding extraction; mapper training subset: **2,000 speakers × 50 utterances = 100K training pairs** (initial); scale to full dataset if Colab quota allows.

### 6.2 Ground Truth: Utterance-Level Speaker Embeddings

**Critical design decision**: Training targets are **utterance-level** ECAPA-TDNN embeddings, not speaker-averaged embeddings.

**Rationale**: Using a single mean embedding per speaker would cause the CVAE to learn `σ ≈ 0` (delta distribution) since all face images of a speaker map to the same target. The probabilistic modeling collapses. Instead:

- For each speaker, multiple audio utterances produce multiple ECAPA embeddings: `{e_s^1, e_s^2, ..., e_s^N}`
- These form the empirical distribution `P_empirical(e_s | speaker)` that the CVAE must learn to approximate
- Training pairs: `(face_frame_i, e_s^j)` where i indexes face frames and j indexes audio utterances from the **same speaker** (not necessarily same clip — allowing within-speaker variation to teach σ)

This structure ensures the encoder learns a meaningful σ that reflects genuine within-speaker vocal variation across utterances and recording conditions.

### 6.3 Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \beta(t) \cdot \mathcal{L}_{KL} + \lambda_{adapt} \cdot \mathcal{L}_{style}$$

#### 6.3.1 Reconstruction Loss

Single cosine similarity loss (not MSE + cosine — conflicting gradients eliminated):

$$\mathcal{L}_{recon} = 1 - \frac{\hat{e}_s \cdot e_s^{gt}}{||\hat{e}_s|| \cdot ||e_s^{gt}||}$$

Cosine loss is chosen because:
- ECAPA-TDNN embeddings are trained with cosine-margin objectives (ArcFace-family losses)
- Speaker similarity is directional, not magnitude-dependent
- Eliminates the scale conflict inherent in MSE + cosine combinations

#### 6.3.2 KL Divergence with Free Bits

$$\mathcal{L}_{KL} = \sum_{d=1}^{192} \max(\lambda_{fb}, \text{KL}(q_\phi(z_d|e_f) \| p(z_d)))$$

Where:
- `λ_fb = 0.5` nats (free bits threshold per dimension)
- KL per dimension = `0.5 * (μ_d² + σ_d² - 1 - log(σ_d²))`

#### 6.3.3 KL Annealing Schedule (β-VAE)

$$\beta(t) = \beta_{max} \cdot \min\left(1, \frac{t}{t_{warmup}}\right)$$

- `t` = current training step
- `t_warmup` = 30% of total training steps
- `β_max` ∈ {0.1, 0.5, 1.0} — hyperparameter to tune
- Start with β = 0 to ensure reconstruction quality before imposing distributional constraint

#### 6.3.4 Style Adapter Loss

$$\mathcal{L}_{style} = \mathcal{L}_{SECS}(\text{StyleTTS2}(T, \hat{s}), \text{ECAPA}(\text{Reference Audio}))$$

Speaker embedding cosine similarity between StyleTTS2 output (conditioned on ŝ) and the reference ECAPA embedding. Computed with StyleTTS2 frozen — only the adapter parameters are updated.

#### 6.3.5 Hyperparameter Grid

| Parameter | Search Range | Default |
|---|---|---|
| β_max | {0.1, 0.5, 1.0} | 0.5 |
| λ_adapt | {0.1, 0.5, 1.0} | 0.5 |
| λ_fb (free bits) | {0.1, 0.5, 1.0} nats | 0.5 |
| Learning rate | {1e-4, 5e-4} | 1e-4 |
| Batch size | {64, 128} | 64 |
| KL warmup % | {20%, 30%, 40%} | 30% |

---

## 7. Datasets

### 7.1 Primary Training Dataset: VoxCeleb2

| Attribute | Value |
|---|---|
| Identities | 6,112 speakers |
| Utterances | ~1.1 million clips |
| Duration | ~2,442 hours |
| Format | MP4 video + WAV audio |
| Language | English (primarily) + multilingual |
| Access | Public, academic use permitted |

**Usage in this project**:
- Extract ArcFace embeddings per speaker face frame
- Extract ECAPA-TDNN embeddings per audio utterance
- Extract MediaPipe FaceMesh craniofacial ratios per face frame
- Extract soft demographic estimates per face frame
- Form training pairs `(e_face, e_speaker_utterance)` for CVAE mapper training

### 7.2 In-Domain Evaluation Dataset: VoxCeleb1

| Attribute | Value |
|---|---|
| Identities | 1,251 speakers |
| Utterances | ~153,000 clips |
| Format | MP4 video + WAV audio |
| Access | Public, academic use permitted |

**Usage**: Retrieval evaluation (Recall@K, SECS), speaker identity generalization testing.

### 7.3 Out-of-Domain Evaluation: India-Specific F2S Benchmark

#### 7.3.1 Dataset Goals

A curated, publication-quality benchmark demonstrating:
1. The cross-demographic generalization gap of Western-trained F2V models
2. Improvement achievable via domain-specific fine-tuning
3. The unique challenge of Indian linguistic diversity (code-switching, retroflex phonemes, tonal prosody)

#### 7.3.2 Target Specifications

| Attribute | Target |
|---|---|
| Total duration | ~20 hours (cleaned) |
| Unique speakers | 40–60 speakers |
| Per-speaker duration | 15–25 minutes |
| Languages | Hindi, Tamil, Telugu, Bengali, Marathi |
| Code-switching proportion | ~25% of clips |
| Gender ratio | 50:50 (M:F) |
| Audio format | 16 kHz, mono, .wav |
| Video format | 224×224 face crop, .mp4 |

#### 7.3.3 Language Distribution

| Language | Speakers | Target Duration |
|---|---|---|
| Hindi | 12 speakers | ~4 hours |
| Tamil | 12 speakers | ~4 hours |
| Telugu | 10 speakers | ~3.3 hours |
| Bengali | 8 speakers | ~2.7 hours |
| Marathi | 8 speakers | ~2.7 hours |
| Code-switching clips | drawn from above | ~5 hours |
| **Total** | **~50 speakers** | **~20 hours** |

#### 7.3.4 Data Source Strategy

**Primary sources** (legally lower risk, publicly accessible):
- **NPTEL (National Programme on Technology Enhanced Learning)**: IIT/IISc lecture videos — Creative Commons licensed, stable frontal camera, professional audio, diverse linguistic backgrounds of faculty
- **AI4Bharat project data**: Publicly released Indian language corpora — explicit academic licensing
- **IndicTTS / IIT Madras TTS lab**: Released audiovisual resources with academic permissions
- **Prasar Bharati Archive**: Government broadcaster, research access may be obtained through formal request

**Secondary sources** (conditional on legal review):
- **YouTube long-form podcasts** (e.g., The Ranveer Show, Cyrus Says, regional academic talks): Usable only if:
  - Content creators are approached for explicit permission via email/DM
  - Or the content is under Creative Commons (CC-BY/CC-BY-SA) license
  - The raw video is NOT stored — only face crops and audio extractions
  - The dataset is distributed via feature embeddings only (not raw clips) to minimize redistribution concerns

#### 7.3.5 Legal Framework

**Summary of legal position**:
- Indian Copyright Act Section 52 permits "fair dealing" for private study and research — applicable for internal use but legally uncertain for large-scale scraping
- YouTube ToS prohibits automated scraping regardless of use purpose
- The DPDP Act (2023) imposes obligations on datasets containing personally identifiable individuals
- Raw video redistribution has high legal risk; **embedding-only dataset distribution** substantially reduces risk

**Adopted framework**:
1. **Preferred path**: NPTEL + AI4Bharat + IIT-licensed sources (CC-licensed, clear research permissions)
2. **Conditional path**: YouTube scraping of face crops only (raw video never stored or redistributed) — use yt-dlp streaming pipeline, not download
3. **Public figures**: Include only public figures who have demonstrably consented to public video sharing; add explicit data ethics statement to paper
4. **Dataset release**: Distribute metadata + features only (ArcFace embeddings, ECAPA embeddings, metadata.csv), not raw audio or video, to comply with copyright and privacy obligations
5. **IRB/Ethics**: Add a data ethics section to the paper declaring the collection methodology and legal basis

#### 7.3.6 Data Collection Pipeline

```
Target Source Video (NPTEL / YouTube Stream)
                 ↓
         yt-dlp (streaming mode — no full download)
                 ↓
          OpenCV Memory Buffer
                 ↓
    ┌────────────┴────────────┐
    ↓                         ↓
Audio Demux (ffmpeg)    Face Detect (RetinaFace)
    ↓                         ↓
16 kHz .wav             Crop 224×224 face track
    ↓                         ↓
    └────────────┬────────────┘
                 ↓
    Active Speaker Detection (SyncNet)
         Threshold: score > 4.5
    [Calibrated per source type — see below]
                 ↓
    ┌────────────┴────────────┐
    ↓ YES                     ↓ NO
Save: face crop + audio      Discard from memory
                 ↓
    Identity Verification (ArcFace cosine similarity
    across all frames of clip — purity check)
                 ↓
    ASR Transcript (OpenAI Whisper)
    Language ID (IndicLID for code-switch detection)
                 ↓
    Quality Filters:
    - SyncNet > 4.5 (or calibrated threshold)
    - Face detection confidence > 0.9
    - Audio SNR > 20 dB
    - Clip duration: 3–10 seconds
    - No background music (energy spectral ratio check)
                 ↓
    Save to Google Drive:
    face_crops/{speaker_id}/clip_{n}.mp4
    audio/{speaker_id}/clip_{n}.wav
    transcripts/{speaker_id}/clip_{n}.txt
```

**SyncNet threshold calibration**: Run pilot batch of 100 clips per source type; inspect SyncNet score distributions; adjust threshold if Indian content shows systematic bias (e.g., if 4.5 discards >80% of good clips). Document final thresholds per source type in the paper.

**Code-switching annotation**:
- IndicLID language ID model applied per 3-second segment of Whisper transcript
- Clip flagged as code-switching if ≥2 language IDs present within a single sentence (not just across sentences)
- 10% random sample of code-switching clips undergo human verification

#### 7.3.7 Dataset Structure

```
indian_f2s_benchmark/
├── metadata.csv                     # Master registry
├── speakers.json                    # Speaker metadata
├── face_crops/
│   ├── spk_001/
│   │   ├── clip_001.mp4             # 224×224, max 10 sec
│   │   └── clip_002.mp4
│   └── ...
├── audio/
│   ├── spk_001/
│   │   ├── clip_001.wav             # 16 kHz mono
│   │   └── clip_002.wav
│   └── ...
├── transcripts/
│   ├── spk_001/
│   │   ├── clip_001.txt             # Whisper output
│   │   └── clip_001.lang.json       # IndicLID language tags
│   └── ...
└── embeddings/                      # For public release (no raw A/V)
    ├── arcface/spk_001/clip_001.npy
    ├── ecapa/spk_001/clip_001.npy
    └── facemesh/spk_001/clip_001.npy
```

**metadata.csv fields**:
```
clip_id, speaker_id, gender, primary_language, secondary_language,
code_switching (bool), accent_region, syncnet_score, audio_snr_db,
duration_sec, whisper_transcript, source_type (nptel/youtube/other),
face_crop_path, audio_path, transcript_path
```

---

## 8. Evaluation Framework

### 8.1 Objective Metrics

#### 8.1.1 Speaker Embedding Cosine Similarity (SECS)

$$\text{SECS} = \frac{1}{K} \sum_{k=1}^{K} \cos(\text{ECAPA}(\text{Speech}_k), e_s^{gt})$$

Measures how similar the synthesized speech's speaker embedding is to the ground truth speaker embedding. Higher is better. Reported as mean ± std across K=3 samples.

#### 8.1.2 Face-Voice Retrieval: Recall@K

**Task**: Given a predicted speaker embedding distribution, retrieve the correct speaker from a pool of N candidates (N = 100, 1000, full test set).

- Recall@1, Recall@5, Recall@10
- Evaluated by sampling μ (mean prediction) and comparing to ground truth ECAPA embedding rankings by cosine similarity
- Higher is better

#### 8.1.3 Diversity Score (DS)

$$\text{DS} = \frac{1}{C(K,2)} \sum_{i \neq j} \cos\text{-dist}(z_i, z_j)$$

Average pairwise cosine **distance** (1 − cosine similarity) between K sampled speaker embeddings for the same face input. Measures how spread out the generated voices are in embedding space.

> [!NOTE]
> Diversity Score is only meaningful when paired with SECS. A high DS with low SECS indicates the model is generating diverse but wrong voices. Report both together.

#### 8.1.4 Expected Calibration Error (ECE) — Distribution Calibration

For each speaker in the test set:
1. Sample 100 z from predicted N(μ, σ²)
2. Compute empirical coverage: what fraction of real utterance embeddings fall within the predicted confidence intervals (50%, 75%, 95%)
3. Calibration error = |predicted coverage − actual coverage|

$$\text{ECE} = \sum_p |P_{predicted} - P_{actual}|$$

Lower ECE = better calibrated uncertainty. Computed per embedding dimension and averaged.

#### 8.1.5 Fréchet Speaker Distance (FSD)

Fréchet distance between the distribution of predicted speaker embeddings and the distribution of real ECAPA embeddings over the test set. Borrowed from Fréchet Inception Distance (FID) — adapted for 192-D speaker embedding space. Lower is better.

$$\text{FSD} = ||\mu_{pred} - \mu_{real}||_2^2 + \text{Tr}(\Sigma_{pred} + \Sigma_{real} - 2(\Sigma_{pred}\Sigma_{real})^{1/2})$$

### 8.2 Subjective Metrics

#### 8.2.1 MOS — Voice Naturalness

- **Scale**: 1–5 (ITU-T P.808 standard)
- **Evaluators**: 20–25 evaluators (primary goal); minimum 15 for workshop submission
- **Platform**: Prolific (demographic filtering for Indian evaluators for Indian benchmark)
- **Samples**: 50 synthesized clips evaluated per condition
- **Inter-annotator agreement**: Krippendorff's α reported for all MOS scores

#### 8.2.2 Face-Voice Consistency MOS

- **Scale**: 1–5
- **Instruction**: "Given this face image and this voice clip, how well do you feel this voice fits this person's appearance?" (1=Not at all, 5=Very well)
- **Evaluators**: Same set as naturalness MOS, evaluated independently
- **Note**: This metric acknowledges the inherent subjectivity — low inter-annotator agreement is itself a publishable finding about the ambiguity of face-voice matching

#### 8.2.3 A/B Preference Test

- **Task**: Compare Voix probabilistic output vs. deterministic baseline on same face input
- **Metric**: % of evaluators preferring Voix output
- **Reported**: Win/Loss/Tie percentages with binomial confidence intervals

#### 8.2.4 Diversity Preference Test

- **Task**: Present 3 voices generated from same face; evaluators rate whether diversity "feels natural and plausible" or "feels like random noise"
- **Metric**: % natural vs. % random across 20 evaluators

### 8.3 Cross-Demographic Evaluation (Indian Benchmark)

| Condition | Dataset | Metric |
|---|---|---|
| Zero-shot Western model | India benchmark | SECS, Recall@5 |
| Fine-tuned on India data | India benchmark | SECS, Recall@5 |
| Zero-shot Western model | VoxCeleb1 | SECS, Recall@5 |
| Fine-tuned on India data | VoxCeleb1 | SECS, Recall@5 (check for catastrophic forgetting) |

Hypothesis: ≥10% SECS drop from VoxCeleb1 → India benchmark in zero-shot condition (H3).

---

## 9. Baselines

| Baseline | Description | Purpose |
|---|---|---|
| **B1: Random Speaker** | Sample random ECAPA embedding from VoxCeleb2 distribution, ignore face | Lower bound — proves face carries signal |
| **B2: Speaker Mean Retrieval** | Retrieve nearest neighbor face from gallery; use that speaker's embedding | Strong non-generative baseline |
| **B3: Deterministic Mapper (μ-only)** | Same CVAE architecture but training with β=0 (no KL) and σ ignored at inference | Tests value of probabilistic component |
| **B4: ArcFace-Only Mapper** | Input: ArcFace 512-D only (no morphology, no demographics) | Tests whether added features help (Ablation A1) |
| **B5: Zero-Shot F2S (2026)** | Reproduce Face Adapter + StyleTTS2 from arXiv:2607.26742 | Direct comparison to closest prior work |
| **B6: Vclip-inspired GMM** | Implement GMM speaker generation from face features | Comparison to implicit probabilistic prior work |

---

## 10. Ablation Studies

### 10.1 Feature Ablation (Testing RQ2)

| Experiment | Input Features | Hypothesis |
|---|---|---|
| **A1: ArcFace only** | 512-D latent | Baseline latent performance |
| **A2: Morphology only** | 32-D craniofacial ratios | Interpretable features standalone |
| **A3: Demographics only** | 16-D soft indicators | Demographic proxies standalone |
| **A4: ArcFace + Morphology** | 544-D | Does morphology add to latent? |
| **A5: ArcFace + Demographics** | 528-D | Does demographic add to latent? |
| **A6: Full (ArcFace + Morph + Demog)** | 560-D | Full model |

**Analysis tool**: SHAP values computed for each feature dimension on test set. Identify which specific craniofacial ratios (if any) have non-zero attribution to SECS improvement.

### 10.2 Architecture Ablation

| Experiment | Modification | Metric |
|---|---|---|
| **B1: No KL (deterministic)** | β = 0 throughout | SECS, DS, ECE |
| **B2: KL no free bits** | Remove λ_fb | Posterior collapse rate |
| **B3: KL no annealing** | β = β_max from step 0 | Training stability |
| **B4: Full CVAE (ours)** | Complete model | All metrics |

### 10.3 Vocal Trait Predictability Analysis

For each vocal trait below, compute the Pearson correlation between the predicted μ and the actual per-speaker value:

| Trait | Extraction Method | Expected Predictability |
|---|---|---|
| F0 mean (pitch) | CREPE or WORLD vocoder | High (sex-correlated) |
| F0 range | CREPE | Medium |
| F1 formant (vowel quality) | Praat/LibROSA | Low-Medium |
| F2 formant | Praat/LibROSA | Low |
| Spectral centroid (timbre) | LibROSA | Medium |
| Speaking rate (syllables/sec) | Whisper alignment | Low |
| Voice quality (HNR) | Praat | Low |

This analysis directly answers H2 and provides the scientific contribution: quantifying which vocal traits are face-predictable.

---

## 11. Infrastructure & Resource Plan

### 11.1 Hardware

| Resource | Specification | Role |
|---|---|---|
| Primary GPU | RTX 4050 (6GB VRAM) if available | CVAE mapper training, ArcFace extraction |
| Backup GPU | Google Colab T4 (15GB VRAM) | Large batch extraction, StyleTTS2 inference |
| Storage | Google Drive (15 GB free) | Processed embeddings + Indian dataset |
| CPU | Consumer laptop | Preprocessing, MediaPipe (CPU-only), IndicLID |

### 11.2 Compute Budget

| Task | Hardware | Estimated Time |
|---|---|---|
| VoxCeleb2 ArcFace embedding extraction (100K clips) | Colab T4 | ~8 hours |
| VoxCeleb2 ECAPA-TDNN embedding extraction (100K clips) | Colab T4 | ~10 hours |
| MediaPipe FaceMesh extraction (100K frames) | CPU / RTX 4050 | ~12 hours |
| Indian dataset collection + processing | RTX 4050 + Colab | ~3–5 days |
| CVAE mapper training (100K pairs, 50 epochs) | RTX 4050 | ~1–2 days |
| StyleTTS2 adapter fine-tuning | Colab T4 | ~6 hours |
| Evaluation + ablations | Colab T4 | ~1 day |
| **Total** | | **~10–14 days compute** |

### 11.3 Storage Plan

| Data | Size | Location |
|---|---|---|
| VoxCeleb2 ArcFace embeddings (100K × 512-D) | ~200 MB | Google Drive |
| VoxCeleb2 ECAPA embeddings (100K × 192-D) | ~75 MB | Google Drive |
| VoxCeleb2 FaceMesh ratios (100K × 32-D) | ~12 MB | Google Drive |
| Indian face crops (50 speakers × 15 min, 224×224) | ~8 GB | Google Drive |
| Indian audio (50 speakers × 15 min, 16 kHz WAV) | ~4 GB | Google Drive |
| Model checkpoints | ~500 MB | Google Drive |
| **Total** | **~13 GB** | **Google Drive (free tier)** |

### 11.4 Software Stack

| Component | Tool | Version |
|---|---|---|
| Deep Learning | PyTorch | 2.x |
| Face Detection | RetinaFace (pytorch) | Latest |
| Face Embedding | ArcFace (InsightFace) | Latest |
| Landmark Extraction | MediaPipe | 0.10.x |
| Speaker Embedding | ECAPA-TDNN (SpeechBrain) | Latest |
| Speech Synthesis | StyleTTS2 | Official repo |
| Active Speaker Detection | SyncNet | Official VGG impl |
| ASR | OpenAI Whisper | large-v3 |
| Language ID | IndicLID (AI4Bharat) | Latest |
| Audio Processing | librosa, SoundFile, ffmpeg | - |
| Video Streaming | yt-dlp, OpenCV | - |
| Experiment Tracking | Weights & Biases (free tier) | - |
| Code Repository | GitHub (public on completion) | - |

---

## 12. Implementation Roadmap (16 Weeks)

### Phase 1 — Environment + Data (Weeks 1–4)

**Week 1**: Environment setup + VoxCeleb2 preprocessing pilot
- [ ] Set up Python environment (torch, speechbrain, insightface, mediapipe)
- [ ] Download VoxCeleb2 subset (1,000 speakers sample) 
- [ ] ArcFace embedding extraction — verify output quality
- [ ] ECAPA-TDNN embedding extraction via SpeechBrain
- [ ] MediaPipe FaceMesh extraction — verify landmark quality + ratio computation
- [ ] Frame selection strategy: implement head-pose frontal filter (using MediaPipe face detector with pose angle estimation)

**Week 2**: Full VoxCeleb2 embedding extraction (100K pairs)
- [ ] Batch extraction pipeline for ArcFace + ECAPA + FaceMesh
- [ ] Build training pairs dataset: `(e_face_560D, e_speaker_192D)` 
- [ ] Verify distribution of embeddings (PCA/UMAP visualization)
- [ ] Store all embeddings to Google Drive

**Week 3**: Indian dataset — pilot collection
- [ ] Identify 5 speakers per language (25 total) from NPTEL + AI4Bharat sources
- [ ] Run streaming pipeline on 2 speakers as proof-of-concept
- [ ] Calibrate SyncNet threshold on Indian content
- [ ] Validate Whisper + IndicLID code-switching detection

**Week 4**: Indian dataset — full collection
- [ ] Process all 40–60 target speakers
- [ ] Apply quality filters, manual spot-check 10%
- [ ] Build metadata.csv and speakers.json
- [ ] Compute ArcFace + ECAPA embeddings for Indian dataset

---

### Phase 2 — Model Development (Weeks 5–8)

**Week 5**: CVAE Mapper Implementation
- [ ] Implement encoder q_φ(z|e_f) — MLP with μ/log_σ² heads
- [ ] Implement decoder p_θ(ê_s|z) — MLP reconstruction head
- [ ] Implement reparameterization trick
- [ ] Implement KL loss with free bits + β annealing schedule
- [ ] Unit test: verify gradient flow through reparameterization
- [ ] Unit test: verify KL approaches 0 during warmup (β=0)

**Week 6**: StyleTTS2 Integration + Adapter
- [ ] Install + verify StyleTTS2 inference pipeline locally / on Colab
- [ ] Identify StyleTTS2 style embedding dimension from codebase
- [ ] Implement 3-layer MLP projection adapter (ECAPA dim → StyleTTS2 style dim)
- [ ] Implement end-to-end inference: face → CVAE → adapter → StyleTTS2 → speech
- [ ] Smoke test: generate speech from ArcFace embedding of known speaker

**Week 7**: Feature Fusion Module + Soft Demographics
- [ ] Implement linear fusion layer (ArcFace 512 + FaceMesh 32 + Demographics 16 → 560)
- [ ] Implement soft demographic estimator (age/sex probability head on top of ArcFace features)
- [ ] Implement learned gating for demographic soft conditioning
- [ ] Integrate full feature extraction → fusion → CVAE pipeline end-to-end

**Week 8**: Training Infrastructure
- [ ] Implement training loop with W&B logging
- [ ] Implement model checkpointing (save best SECS on val set)
- [ ] Implement Recall@K evaluation function
- [ ] Implement Diversity Score computation
- [ ] Run initial 5-epoch smoke test on 10K pairs to verify loss behavior

---

### Phase 3 — Training & Evaluation (Weeks 9–12)

**Week 9**: Baseline Training
- [ ] Train B3 (Deterministic mapper, ArcFace only) — establish baseline SECS/Recall@K
- [ ] Train B4 (ArcFace-only input, CVAE) — feature ablation A1
- [ ] Train B2 (Retrieval baseline) — implement nearest-neighbor face-voice retrieval

**Week 10**: Full Model Training + Hyperparameter Search
- [ ] Run hyperparameter grid: β_max × λ_adapt × learning rate
- [ ] Select best checkpoint by val SECS + low ECE
- [ ] Train full model (ArcFace + Morph + Demog, CVAE) — best config

**Week 11**: Ablation Studies
- [ ] Run feature ablation experiments A1–A6 (Weeks 9 ablations continued)
- [ ] Run architecture ablations B1–B4
- [ ] Vocal trait correlation analysis (F0, formants, spectral centroid)
- [ ] SHAP feature attribution analysis on best model

**Week 12**: Indian Benchmark Evaluation
- [ ] Zero-shot evaluation on India benchmark
- [ ] Fine-tune (5–10 epochs) on Indian dataset
- [ ] Evaluate fine-tuned model on India benchmark + re-evaluate on VoxCeleb1 (catastrophic forgetting check)
- [ ] Compute all objective metrics: SECS, Recall@K, DS, ECE, FSD

---

### Phase 4 — MOS Study + Writing (Weeks 13–16)

**Week 13**: Subjective Evaluation Setup + Launch
- [ ] Select 50 clips for MOS evaluation
- [ ] Design rating interface (simple web form or Prolific study)
- [ ] Recruit 20–25 evaluators on Prolific (English native + Indian language filters)
- [ ] Launch MOS study + A/B preference test
- [ ] While MOS runs: Begin writing Introduction + Related Work sections

**Week 14**: Results Analysis + Writing
- [ ] Collect MOS results, compute Krippendorff's α
- [ ] Compile all objective + subjective metrics into results tables
- [ ] Run statistical significance tests (paired t-test or Wilcoxon)
- [ ] Write Methodology + Experiments + Results sections

**Week 15**: Paper Completion
- [ ] Write Discussion + Ablation analysis narrative
- [ ] Write Conclusion + Future Work
- [ ] Create system architecture figure (LaTeX/TikZ or draw.io export)
- [ ] Create results tables and plots (Python/matplotlib)
- [ ] Prepare qualitative demo examples (audio samples)
- [ ] Prepare NPC voice demo (appendix/supplementary video only)

**Week 16**: Revision + Submission Preparation
- [ ] Internal revision pass: check all claims are supported by results
- [ ] Update Related Work post-literature-survey (revise novelty claims if needed)
- [ ] Format to target venue template
- [ ] Prepare supplementary materials, code, and data release
- [ ] Submit

---

## 13. Risk Analysis

### 13.1 Research Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Novelty claim weakened by literature survey | Medium | High | Fallback: emphasize morphology ablation + Indian benchmark as contributions even if probabilistic F2V has partial precedent |
| CVAE posterior collapse | Medium | High | β annealing + free bits; monitor KL per batch; switch to β_max=0.1 if collapse observed |
| Morphology features add noise (H2 rejected) | Medium | Medium | Negative result is publishable; keep ablation regardless of direction |
| SECS gains are statistically insignificant | Low-Medium | High | Increase training data to full VoxCeleb2; consider contrastive pre-training of mapper |
| StyleTTS2 adapter fails to align spaces | Low-Medium | High | Fallback: use Vclip-style retrieval-based speaker selection for synthesis instead of generating |

### 13.2 Engineering Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Colab T4 quota exhausted mid-training | Medium | Medium | Save checkpoints to Drive every epoch; use Colab Pro ($10/month) if needed |
| RTX 4050 6GB VRAM insufficient for batch size | Medium | Low | Reduce batch size to 32; use gradient accumulation |
| VoxCeleb2 face extraction failures (low quality frames) | High | Low | Filter frames with RetinaFace confidence > 0.9; use multiple frames per speaker |
| Indian dataset pipeline failures (SyncNet bias) | Medium | Medium | Calibrate threshold per source; fallback to manual selection for small pilot |

### 13.3 Timeline Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Data collection takes longer than 2 weeks | Medium | Medium | Reduce Indian dataset to 10 hours (30 speakers) if needed — still publication-worthy |
| MOS study takes longer than 1 week | Low | Low | Buffer: MOS launched in Week 13 overlaps with writing |
| Literature survey reveals major competing work | Medium | High | Revise claims in Week 16; adjust contribution framing |

### 13.4 Ethical Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Venue rejects paper for data ethics issues | Low-Medium | High | Use NPTEL/AI4Bharat sources primarily; add explicit ethics statement |
| Demographic inference features criticized | Medium | Medium | Frame as soft priors; include limitations section; use distributions not labels |
| Synthesized voices of public figures raise concerns | Low | High | Limit synthesis to VoxCeleb speakers; do not generate mimicry of named individuals in demos |

---

## 14. Reproducibility Plan

### 14.1 Code Release

- **Repository**: GitHub (public at time of paper submission)
- **Structure**: `voix/` with modules: `data/`, `models/`, `training/`, `evaluation/`, `scripts/`
- **Environment**: `requirements.txt` + `environment.yml` (conda)
- **Documentation**: README with step-by-step reproduction instructions

### 14.2 Training Reproducibility

- All random seeds fixed: `torch.manual_seed(42)`, `numpy.random.seed(42)`, CUDA determinism flags
- All hyperparameters logged to W&B run configs (exportable to JSON)
- All results reported as **mean ± std over 3 seeds**
- Model checkpoints hosted on HuggingFace Hub

### 14.3 Dataset Release

- **VoxCeleb2 embeddings**: Will NOT be released (redistributing derived features may violate VoxCeleb license). Instead, release extraction scripts so anyone can reproduce embeddings.
- **Indian F2S benchmark**: Release as embedding-only package (ArcFace + ECAPA features + metadata) via HuggingFace Datasets. Raw audio/video NOT redistributed.
- Provide verification scripts to check embedding extraction matches paper's statistics.

---

## 15. Future Work (Out of Scope for This Paper)

The following are explicitly deferred and documented as future directions:

1. **Expression-Conditioned Emotion Branch**: Multi-branch architecture (Timbre + Emotion + Style branches) using expression features for prosody-aware synthesis. The full multi-branch system adds significant complexity and scope beyond this paper's focus.

2. **Normalizing Flow mapper**: Replace CVAE with a normalizing flow for richer, non-Gaussian speaker embedding distributions.

3. **Real-time inference pipeline**: Optimize the full pipeline for <100ms latency for interactive NPC generation.

4. **Illustrated/artistic character portraits**: Extend ArcFace and FaceMesh to work on non-photorealistic images (anime, fantasy art) for NPC applications. Current models fail on non-photorealistic inputs.

5. **Longitudinal voice modeling**: Predict how a speaker's voice changes with age from facial age cues.

6. **Cross-lingual voice generation**: Generate Indian language speech (not just Indian-accented English) from Indian faces using multilingual TTS.

---

## 16. Publication Strategy

### 16.1 Venue Considerations (To Be Decided Post-Literature-Survey)

| Venue | Track | Fit | Notes |
|---|---|---|---|
| **INTERSPEECH** | Main | ⭐⭐⭐ High | Premier speech synthesis venue; strong F2S community; ideal for both model + dataset contributions |
| **ICASSP** | Main | ⭐⭐⭐ High | Signal processing + speech; appropriate scope |
| **NeurIPS Datasets & Benchmarks** | Datasets | ⭐⭐ Medium | Strong for Indian benchmark specifically; less focus on the model |
| **ICLR** | Main | ⭐⭐ Medium | If probabilistic modeling angle is the dominant contribution |
| **ACL/EMNLP** | Main | ⭐ Low | Only if code-switching / multilingual angle is significantly expanded |
| **CVPR** | Main | ⭐ Low | Vision community less focused on speech synthesis specifically |

**Recommended primary target**: **INTERSPEECH** (next available deadline after 16-week window)

### 16.2 Contribution Framing Options

**Framing 1 (Model-centric)**: "Probabilistic Face-to-Voice Generation via Cross-Modal Identity Modeling"
- Lead with CVAE mapper, probabilistic distribution, uncertainty quantification
- Indian dataset as an evaluation contribution (Section 6)
- Suitable for: INTERSPEECH, ICASSP, ICLR

**Framing 2 (Benchmark-centric)**: "Bridging the Demographic Gap in Face-to-Voice Synthesis: An Indian Benchmark and Probabilistic Baseline"
- Lead with the dataset gap and Indian benchmark
- Probabilistic model as the baseline experiment
- Suitable for: NeurIPS Datasets & Benchmarks

**Decision**: Defer final framing until after literature survey reveals the competition landscape. If strong probabilistic F2V work exists, pivot to Framing 2.

---

## 17. Appendix: NPC Voice Generation Demo

> [!NOTE]
> This section documents the NPC demo workflow. It is a **qualitative demonstration only** — not a research claim, not an evaluation metric, not evidence of scientific validity. It will appear in the supplementary materials as a video demo.

**What it demonstrates**: The user experience of generating multiple voices from a character portrait + optional metadata (role, temperament, age). The demo uses real photograph inputs (public figures from VoxCeleb test set) rather than artistic illustrations, since ArcFace and MediaPipe are trained on photorealistic images.

**Why it's included**: Illustrates the practical value of probabilistic multi-voice generation for creative applications. Shows the "creative selection" workflow: the system generates 3 voices, user picks the most appropriate one.

**Why it's NOT a research claim**:
- Fantasy character portraits (orcs, elves) are not in the model's training distribution
- There is no ground truth voice for fictional characters — no quantitative evaluation is possible
- The demo is purely subjective and illustrative

---

*Master Document Status: **READY FOR LITERATURE SURVEY***
*Next action: Conduct systematic literature survey (Google Scholar / Semantic Scholar / arXiv) on probabilistic F2V, VAE-based speaker generation, and cross-modal uncertainty quantification. Update Section 3 with verified citations and revise novelty claims accordingly.*

---
*Document version: 1.0 | Created: 2026-09-22 | Author: Solo Researcher | Confidential Pre-Submission Draft*
