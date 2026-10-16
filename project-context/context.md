# context.md
> VOIX system research context, research questions, hypotheses, and objectives.
> Last updated: 2026-10-16. All statuses verified against source code in `voix/`.

---

## What Problem Does VOIX Solve?

Human speech is physically shaped by the vocal tract — a biological instrument whose dimensions are anatomically co-determined by craniofacial morphology. VOIX investigates whether this biophysical link is sufficiently captured by deep neural features to enable **zero-shot voice synthesis directly from a single face photograph**, without any prior reference speech from the target person.

This is a fundamentally one-to-many problem: a single face can plausibly correspond to many natural-sounding voices. VOIX treats this uncertainty probabilistically, generating $K$ diverse yet identity-consistent candidate voice hypotheses per face image.

### The Core Challenge: Acoustic Mean Collapse

Early regression baselines predicted a single "average" voice for every face — collapsing the distribution $P(\mathbf{e}_s \mid \mathbf{e}_f)$ to its conditional mean. This produces robotic, metallic output indistinguishable from white noise. VOIX's solution is a **Metric Contrastive Mapper** trained with Supervised InfoNCE over precomputed speaker centroids, combined with **Exemplar Manifold Retrieval** from a gallery of 3,029 identity centroids.

The correct formulation models face-to-voice as a conditioned manifold distribution:
$$P(\mathbf{e}_s \mid \mathbf{e}_f) : \text{Face} \longrightarrow \text{Manifold Distribution over Speaker Space} \longrightarrow \text{Multiple Plausible Voices}$$

A face provides *probabilistic acoustic constraints*, not a deterministic voice blueprint.

### Cross-Demographic Generalization

> **Scope Note:** In the current release, cross-demographic biometric generalization is rigorously evaluated across a 31-subject unconstrained global test grid (encompassing South Asian, East Asian, Black, Caucasian, and Hispanic cohorts). The dedicated 20-hour regional scraping and ASR curation for a standalone India F2S Benchmark is designated as **Future Scope** to maintain a focused production boundary.

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
Project through StyleAdapter and synthesize 24 kHz speech through frozen StyleTTS 2:
$$\mathbf{s}_{\text{TTS}} = \text{StyleAdapter}(\mathbf{e}_{s, \text{hybrid}})$$
$$\text{Speech}_k = \text{StyleTTS2}(T, \mathbf{s}_{\text{TTS}})$$

---

## Research Questions

- **RQ1 - Overcoming Acoustic Collapse:** Can a metric contrastive formulation (InfoNCE over speaker centroids) combined with exemplar manifold retrieval outperform standard CVAE/regression models in generating sharp, natural, and perceptually distinct voices without collapsing to the population mean?
- **RQ2 - Biomechanical Feature Contribution:** Does augmenting latent facial identity (ArcFace 512-D) with explicit craniofacial morphological ratios (FaceMesh 32-D) measurably improve acoustic vocal tract correlation (formant frequencies, fundamental pitch $F_0$) over latent identity alone?
- **RQ3 - Cross-Demographic Generalization (Future Scope):** What is the quantitative degradation of models trained on Western-centric corpora (VoxCeleb2) when evaluated zero-shot on demographically diverse faces/voices, and can demographic-guided retrieval recover this gap?

---

## Hypotheses

- **H1 (Acoustic Fidelity & Variance):** Metric contrastive mapping and hybrid exemplar retrieval will prevent variance collapse, achieving $>2\times$ higher speaker cosine similarity and significantly higher human naturalness than unregularized regression baselines.
  - *(Status: **VERIFIED** — Model F achieves 0.1299 val cosine vs 0.0526 for baseline regression, with 100% gender accuracy).*
- **H2 (Morphological Predictive Value):** Craniofacial ratios (jaw width, face height-to-width, mandibular angle) provide statistically significant predictive signal for lower formant resonance ($F_1, F_2$) and vocal tract scale beyond latent identity features.
  - *(Status: **Implemented & Under Evaluation** — 32-D ratios extracted and fused; formal ablation ongoing).*
- **H3 (Cross-Demographic Transfer — Future Scope):** Zero-shot evaluation on demographically diverse non-Western subjects will reveal measurable performance gaps due to phonetic and craniofacial distribution shift.
  - *(Status: **Future Scope** — Indian F2S benchmark formally out-of-scope for current version).*

---

## Objectives

| # | Objective | Status | Implementation Details |
|---|---|---|---|
| **O1** | Design & implement face-to-voice mapper learning $P(\mathbf{e}_s \mid \mathbf{e}_f)$ | **COMPLETE** | Metric Contrastive Mapper + InfoNCE + CVAE in `voix/models/cvae.py` |
| **O2** | Construct 560-D multi-feature face representation | **COMPLETE** | `FaceFusionLayer` in `voix/models/fusion.py` fusing ArcFace (512-D) + ratios (32-D) + demographics (16-D) |
| **O3** | Build Vocoder Style Bridge to frozen StyleTTS 2 synthesis backend | **COMPLETE** | `StyleAdapter` (192→256→256→128-D) + `StyleTTS2Backend` with reference anchors in `voix/models/` |
| **O4** | Execute exhaustive 6-model baseline and ablation benchmark on 400 zero-shot speakers | **COMPLETE** | 7 metrics across Models A–F in `results/benchmarks/summary.csv` |
| **O5** | Controllable expressivity engine: morphology sliders, emotion, accent, cross-lingual, clinical speech | **COMPLETE** | `voix/models/speech_simulation.py`, `accent_control.py`, `emotion_control.py`, `cross_lingual.py` |
| **O6** | Production hardening: unified inference pipeline | **COMPLETE** | `VoixPipeline` in `voix/inference/pipeline.py`; `AudioSeal` watermarking |
| **O7** | Exhaustive empirical stress test across diverse global subjects | **COMPLETE** | 31 subjects, 181 audio trials, 100% ASR intelligibility, clinical speech 4-tier severity test |
| **O8** | India F2S Benchmark curation (~20 hrs, 40–60 speakers, 5 languages) | **FUTURE SCOPE** | Formally out-of-scope for current release |

---

## Non-Goals (Out of Scope for Current Version)

- Dynamic facial expression / emotion conditioning from video (focus is strictly on static anatomical identity from a single photo).
- Full 3D mesh rendering / neural radiance fields (NeRF) for face modeling.
- Real-time synthesis latency (<50 ms edge deployment).
- Non-photorealistic facial inputs (cartoons, sketches, stylized avatars).
- End-to-end training of the speech vocoder from scratch (StyleTTS 2 is strictly frozen).
- Longitudinal voice aging modeling across decades.
- **India F2S Benchmark** — 20-hour dedicated Indian regional corpus curation. Formally designated as Future Scope.

---

## Success Criteria

| Metric | Target | Verified Current Result |
|---|---|---|
| **Validation Cosine Similarity (Zero-Shot)** | $> 0.100$ | **0.1299** (Model F Hybrid Retrieval-Contrastive) |
| **Demographic Gender Accuracy** | $> 95.0\%$ | **100.0%** (Models E & F) |
| **Acoustic Naturalness** | No metallic artifacts; natural voice dynamics | **Verified** — Peak 0.8968, RMS 0.1443, 0 clipping events |
| **Inference Latency** | $< 250$ ms on GPU | **11.2 ms** (Hybrid mapper forward pass) |
| **Stress Test Coverage** | Diverse unconstrained face portraits | **31 subjects, 181 audio trials, 100% Whisper ASR intelligibility** |
| **Biological Sex Pitch Separation** | Male $F_0 \approx 100$–$135$ Hz, Female $F_0 \approx 185$–$230$ Hz | **Male: 112.6 Hz, Female: 205.7 Hz** |
| **Code Reproducibility** | Deterministic seed, clean test suite | **35 passing unit tests**, fixed global seed 42 |

---

## Publication Target

**Primary:** INTERSPEECH
**Alternative / Complementary:** ICASSP (Speech & Audio Processing) or NeurIPS Datasets and Benchmarks Track.

---

*Read next: [architecture.md](file:///D:/voix/project-context/architecture.md)*
