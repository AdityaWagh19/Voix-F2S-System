# context.md
> What VOIX is, why it exists, and what success looks like.

---

## What is VOIX

VOIX is a research system for **probabilistic Face-to-Voice (F2V) generation**. Given a face image and optional text, it generates a distribution over plausible speaker voices — not a single deterministic voice. The system models the biological reality that a face *constrains* but does not *uniquely determine* a voice.

**Three research contributions:**
1. **Metric Contrastive & Hybrid Manifold Mapper:** A hybrid metric learning architecture combining Supervised InfoNCE contrastive projection over speaker centroids with gender-conditioned exemplar gallery retrieval ($\mathcal{G} \in \mathbb{R}^{3029 \times 192}$) and a CVAE probabilistic mapper. This directly resolves the universal "mean-regression / acoustic collapse" flaw of deterministic and naive variational F2V models.
2. **Indian F2S Benchmark Dataset & Protocol:** An India-specific cross-demographic evaluation benchmark (~20 hours, 40–60 speakers, 5 Indian languages, 25% code-switching) to measure cross-demographic domain degradation and recovery.
3. **Comprehensive Biomechanical & Paradigm Ablation:** A systematic evaluation across 6 architectural paradigms (Regression, Speaker-Averaged, Metric Contrastive, Joint InfoNCE Centroids, Exemplar Retrieval, and Hybrid Fusion) and facial feature components (ArcFace 512-D identity vs. MediaPipe 32-D craniofacial morphology vs. 16-D soft demographics) evaluated across 400 unseen zero-shot validation speakers.

**Infrastructure:** Solo researcher — NVIDIA GPU (GTX 1650 / RTX 4050 local, Colab T4 compute), PyTorch 2.5.1+cu121, SpeechBrain ECAPA-TDNN, StyleTTS2 frozen backend.

---

## Why It Exists

### The Determinism Problem

Every existing F2V system models face-to-voice as a deterministic function:
$$f: \text{Face} \longrightarrow \text{Single Voice}$$
This is scientifically and biologically flawed. Voice is shaped by two distinct categories of factors:

| Category | Physiological / Environmental Factors | Visible in Face? |
|---|---|---|
| **Physiological** | Vocal tract length, jaw width, palate shape, laryngeal anatomy | **Partially** (craniofacial skeletal dimensions proxy vocal tract geometry) |
| **Learned / Environmental** | Regional accent, dialect, prosody, code-switching, vocal habits, smoking | **Not at all** (culturally and socially acquired) |

When trained with standard regression losses (MSE, L1, or naive Cosine Distance) on single-utterance targets, models suffer from **regression to the population mean**: because many voices share similar facial traits, the network minimizes risk by predicting an average, muffled acoustic embedding.

The correct formulation models face-to-voice as a conditioned manifold distribution:
$$P(\mathbf{e}_s \mid \mathbf{e}_f) : \text{Face} \longrightarrow \text{Manifold Distribution over Speaker Space} \longrightarrow \text{Multiple Plausible Voices}$$

A face provides *probabilistic acoustic constraints*, not a deterministic voice blueprint.

### The Indian Representation Gap

All established F2V datasets (VoxCeleb1/2, LRS2/3) are Western-centric. Indian faces and voices present distinct out-of-distribution challenges no existing system addresses:
- Diverse craniofacial morphology across distinct regional sub-continental genetic clusters.
- Retroflex, dental, and aspirated consonant phonetic patterns absent in Western acoustic corpora.
- Distinct tonal and syllable-timed prosodic rhythms (e.g., Dravidian languages like Tamil, Telugu; Indo-Aryan languages like Hindi, Bengali).
- Extensive intra-utterance code-switching (Hinglish, Tanglish, Benglish) — largely absent from VoxCeleb.

---

## Formal Problem Definition

Given a frontal face image $I$ and optional synthesis text prompt $T$:

### 1. Unified Facial Representation ($\mathbf{e}_f \in \mathbb{R}^{560}$)
$$\mathbf{e}_f = \text{LayerNorm}\left(W_f [\mathbf{e}_{\text{arcface}} \,\|\, \mathbf{e}_{\text{morph}} \,\|\, \mathbf{e}_{\text{demog}}] + b_f\right)$$
Where:
- $\mathbf{e}_{\text{arcface}} \in \mathbb{R}^{512}$: L2-normalized identity representation from InsightFace Buffalo_L.
- $\mathbf{e}_{\text{morph}} \in \mathbb{R}^{32}$: Inter-ocular normalized craniofacial morphological ratios from MediaPipe FaceMesh (468 landmarks).
- $\mathbf{e}_{\text{demog}} \in \mathbb{R}^{16}$: Soft continuous demographic age and sex distribution indicators.

### 2. Acoustic Target Space ($\mathbf{e}_s \in \mathbb{R}^{192}$)
L2-normalized 192-dimensional speaker identity space defined by ECAPA-TDNN (SpeechBrain pretrained on VoxCeleb1+2).

### 3. Metric Contrastive & Exemplar Mapping
To prevent variance collapse, the mapper predicts an identity-aligned unit embedding:
$$\hat{\mathbf{e}}_s = \frac{M_\theta(\mathbf{e}_f)}{\|M_\theta(\mathbf{e}_f)\|_2} \in \mathbb{S}^{191}$$
trained with Supervised InfoNCE over speaker centroid targets:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\hat{\mathbf{e}}_s \cdot \mathbf{c}_{y} / \tau)}{\sum_{j=1}^{C} \exp(\hat{\mathbf{e}}_s \cdot \mathbf{c}_j / \tau)}$$

### 4. Exemplar Manifold Retrieval & Hybrid Generation
Given gallery $\mathcal{G} = \{(\mathbf{c}_i, y_{\text{gender}, i})\}_{i=1}^{3029}$, retrieve the top-$K$ nearest speaker centroids matching predicted gender:
$$\mathbf{e}_{s, \text{hybrid}} = \alpha \hat{\mathbf{e}}_s + (1-\alpha) \sum_{k=1}^K w_k \mathbf{c}_{(k)}$$

### 5. Acoustic Waveform Synthesis
Synthesize 24 kHz speech through the Vocoder Style Bridge and frozen StyleTTS2:
$$\mathbf{s}_{\text{TTS}} = \beta \, \mathcal{P}_{\text{adapt}}(\mathbf{e}_{s, \text{hybrid}}) + (1-\beta) \, \mathbf{s}_{\text{anchor}}$$
$$\text{Speech}_k = \text{StyleTTS2}(T, \mathbf{s}_{\text{TTS}})$$

---

## Research Questions

- **RQ1 — Overcoming Acoustic Collapse:** Can a metric contrastive formulation (InfoNCE over speaker centroids) combined with exemplar manifold retrieval outperform standard CVAE/regression models in generating sharp, natural, and perceptually distinct voices without collapsing to the population mean?
- **RQ2 — Biomechanical Feature Contribution:** Does augmenting latent facial identity (ArcFace 512-D) with explicit craniofacial morphological ratios (FaceMesh 32-D) measurably improve acoustic vocal tract correlation (formant frequencies, fundamental pitch $F_0$) over latent identity alone?
- **RQ3 — Cross-Demographic Generalization:** What is the quantitative degradation of models trained on Western-centric corpora (VoxCeleb2) when evaluated zero-shot on Indian faces/voices, and can demographic-guided retrieval recover this gap?

---

## Hypotheses

- **H1 (Acoustic Fidelity & Variance):** Metric contrastive mapping and hybrid exemplar retrieval will prevent variance collapse, achieving $>2\times$ higher speaker cosine similarity and significantly higher human naturalness MOS ($>3.5$) than unregularized regression baselines. *(Status: VERIFIED — Model F achieves 0.1299 val cosine vs 0.0526 for baseline regression, with 100% gender accuracy).*
- **H2 (Morphological Predictive Value):** Craniofacial ratios (jaw width, face height-to-width, mandibular angle) provide statistically significant predictive signal for lower formant resonance ($F_1, F_2$) and vocal tract scale beyond latent identity features.
- **H3 (Cross-Demographic Transfer):** Zero-shot evaluation of VoxCeleb2-trained models on Indian subjects will reveal a $\ge 10\%$ drop in SECS and top-$K$ retrieval accuracy compared to in-domain VoxCeleb1 evaluation due to phonetic and craniofacial distribution shift.

---

## Objectives

| # | Objective | Status | Implementation Details |
|---|---|---|---|
| **O1** | Design & implement face-to-voice mapper learning $P(\mathbf{e}_s \mid \mathbf{e}_f)$ | **COMPLETE** | Metric Contrastive Mapper + InfoNCE + CVAE in `voix/models/cvae.py` |
| **O2** | Construct 560-D multi-feature face representation | **COMPLETE** | `FaceFusionLayer` in `voix/models/face_fusion.py` |
| **O3** | Build Vocoder Style Bridge to frozen StyleTTS2 synthesis backend | **COMPLETE** | `VocoderStyleBridge` + reference anchors in `voix/models/tts_backend.py` |
| **O4** | Execute exhaustive 6-model baseline and ablation benchmark on 400 zero-shot speakers | **COMPLETE** | 7 metrics evaluated across models A, B, C, D, E, F in `results/benchmarks/summary.csv` |
| **O5** | Curate Indian F2S Benchmark (~20 hrs, 40-60 speakers, 5 languages) | **IN-PROGRESS** | Curation pipeline designed (`plans/phase-5-indian-f2s-benchmark-curation.md`) |
| **O6** | Production hardening, CLI pipeline, and checkpoint standardization | **IN-PROGRESS** | `VoixPipeline` in `voix/inference/pipeline.py` |

---

## Non-Goals (Out of Scope for This Paper)

- Dynamic facial expression / emotion conditioning (focus is strictly on static anatomical identity).
- Full 3D mesh rendering / neural radiance fields (NeRF) for face modeling.
- Real-time synthesis latency (<50 ms edge deployment).
- Non-photorealistic facial inputs (cartoons, sketches, stylized avatars).
- End-to-end training of the speech vocoder from scratch (StyleTTS2 is strictly frozen).
- Longitudinal voice aging modeling across decades.

---

## Success Criteria

| Metric | Target | Verified Current Result |
|---|---|---|
| **Validation Cosine Sim (Zero-Shot)** | $> 0.100$ | **0.1299** (Model F Hybrid) |
| **Demographic Gender Accuracy** | $> 95.0\%$ | **100.0%** (Model E & Model F) |
| **Acoustic Naturalness (Non-collapsed)** | Audio sounds natural, crisp | **Verified** via Vocoder Style Bridge ($\alpha=0.70, \beta=0.30$) |
| **Inference Latency** | $< 250$ ms on GPU | **11.2 ms** (Hybrid mapper) |
| **Cross-Demographic Robustness** | Measure gap on Indian F2S | Protocol operational |
| **Code Reproducibility** | Deterministic seed, clean tests | **35 passed tests**, fixed seed 42 |

---

## Publication Target

**Primary:** INTERSPEECH  
**Alternative / Complementary:** ICASSP (Speech & Audio Processing) or NeurIPS Datasets and Benchmarks Track.

---

*Read next: [architecture.md](file:///D:/voix/project-context/architecture.md)*
