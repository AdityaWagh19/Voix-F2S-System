# VOIX Project-Context Recreation Specification & Blueprint

> **Notice:** This document serves as the single authoritative blueprint for recreating all five project-context documents (rchitecture.md, context.md, mvp.md, 
esearch.md, and 	asks.md) from scratch at the conclusion of the VOIX research and engineering project.
> It defines the exact required structures, mathematical formulations, empirical metrics, architectural boundaries, and reference sources.

---

## 1. Executive Summary & Purpose

The project-context/ directory provides high-fidelity, production-grade documentation for the VOIX Zero-Shot Face-to-Voice Synthesis System. Previous revisions of these documents became outdated as the architecture transitioned from early naive regression baselines to a robust, contrastive-probabilistic pipeline incorporating:
1. Multi-modal facial biometrics (ArcFace identity + MediaPipe craniofacial morphology + Soft demographics).
2. Probabilistic latent modeling via CVAE with identity-direction preservation to prevent same-gender collapse.
3. Decoupled acoustic and prosodic style synthesis via StyleTTS2.
4. Imperceptible deepfake watermarking via AudioSeal.

When the project concludes, five dedicated markdown documents must be authored from scratch inside D:/voix/project-context/. Each section below details the exact requirements, contents, tables, and metrics for each file.

---

## 2. Document 1: rchitecture.md Specification

### 2.1 Document Purpose
rchitecture.md must present the complete technical blueprint, component interfaces, mathematical formulations, dimensional contracts, and dataflow of the VOIX system.

### 2.2 Required Document Structure & Sections
1. **System Pipeline Diagram**: End-to-end ASCII and Mermaid dataflow diagram from raw input image and text prompt to synthesized 24 kHz WAV audio.
2. **Component 1: Visual Biometrics Extractor (oix/data/face_extractor.py, morphology.py, demographics.py)**:
   - **ArcFace ResNet-50** (Buffalo_L backbone): Extract 512-D identity vector {id} \in \mathbb{S}^{511}$ (L2-normalized). Completely frozen.
   - **MediaPipe FaceMesh v1** (468 3D landmarks): Compute 32-D inter-ocular distance (IOC) normalized craniofacial ratio vector {geo} \in \mathbb{R}^{32}$. Must document key ratios: total face height to IOC (VTL proxy), bizygomatic width to IOC, mandibular jaw width to IOC (oral cavity resonance), and mandibular vertex angle.
   - **Soft Demographic Estimator**: Compute 16-D continuous age and gender probability distribution {demo} \in \mathbb{R}^{16}$.
   - **FaceFusionLayer (oix/models/fusion.py)**: $\text{Linear}(560 \to 560) + \text{LayerNorm} + \text{GELU}$ unifying visual streams into  \in \mathbb{R}^{560}$.
3. **Component 2: Contrastive & Probabilistic Voice Mapping (oix/models/cvae.py, oix/models/cfvn.py)**:
   - **CFVN (Contrastive Face-Voice Network)**: Multi-layer projection head mapping  \in \mathbb{R}^{560} \to \hat{e}_s \in \mathbb{S}^{191}$ (ECAPA speaker embedding space), trained using Supervised InfoNCE over speaker identity centroids.
   - **CVAE Diversity Engine (cvae_cfvn_face192_latent128_diversity.pt)**:
     - Latent dimension  = 128$.
     - Prior network: \psi(z \mid e_f) = \mathcal{N}(\mu_\psi(e_f), \Sigma_\psi(e_f))$.
     - Decoder network: \theta(e_s \mid z, e_f)$. Takes input concatenation $[z, e_f]$ and predicts 192-D speaker representation.
     - ArcFace identity injection: Decoder projects 512-D identity direction to modulate acoustic style, avoiding same-gender acoustic collapse.
   - **Exemplar Retrieval Gallery (oix/models/retrieval.py)**:
     - Precomputed gallery of 3,029 speaker centroids (192-D) from VoxCeleb2.
     - Gender-constrained kNN search (=5$ or $).
     - Blending equation: {s\text{, hybrid}} = \alpha \hat{e}_s + (1-\alpha) \sum_{k=1}^K w_k c_k$ (default $\alpha = 0.70$).
4. **Component 3: Acoustic Synthesis Bridge & StyleTTS2 (oix/models/adapter.py, oix/tts/)**:
   - **StyleAdapter (oix/models/adapter.py)**: Maps 192-D ECAPA representation to 256-D StyleTTS2 latent style vector  \in \mathbb{R}^{256}$ via Linear(192 $\to$ 256) + LayerNorm + GELU.
   - **Acoustic/Prosody Decoupling**: Splits 256-D style into:
     - Acoustic timbre vector: {\text{acoustic}} = s[:128] \in \mathbb{R}^{128}$ (controls vocal tract resonances, formant envelope).
     - Prosodic dynamic vector: {\text{prosodic}} = s[128:] \in \mathbb{R}^{128}$ (controls pitch modulation, speaking cadence).
   - **StyleTTS2 Vocoding Engine**: Pretrained StyleTTS2 diffusion/pitch/duration predictors + HiFi-GAN neural vocoder, generating 24 kHz high-fidelity speech.
5. **Component 4: Watermarking & Provenance (oix/audio/watermark.py)**:
   - Embedded AudioSeal 16-bit payload watermarking at 24 kHz for provenance and deepfake defense.
6. **Component 5: Parametric Sliders (oix/models/craniofacial_style.py)**:
   - Continuous acoustic adjustment for pitch ($), vocal tract length (formant shift), breathiness, rate, emotion, and accent offsets.
7. **Hardware & Latency Budgets**:
   - Minimum GPU budget: NVIDIA GTX 1650 (4 GB VRAM).
   - Peak VRAM measured: 1431.6 MB (well under 4 GB).
   - Latency: Sub-second visual extraction (< 0.4s), Real-Time Factor (RTF) < 0.3 for synthesis.

---

## 3. Document 2: context.md Specification

### 3.1 Document Purpose
context.md provides the scientific rationale, biophysical grounding, theoretical formulation, and research questions that justify VOIX.

### 3.2 Required Document Structure & Sections
1. **The Biophysical Link: Craniofacial Morphology to Vocal Tract Acoustics**:
   - Anatomical coupling: Vocal tract length (VTL) is bounded by total cranial height and mandible depth.
   - Pharyngeal and oral cavity resonant volumes shape low-order formants (, F_2, F_3$).
   - Vocal fold mass and laryngeal framework scale with facial sexual dimorphism, governing baseline fundamental frequency ($).
2. **The Fundamental Challenge: Acoustic Mean Collapse**:
   - Mathematical explanation of why MSE/L1 regression on (e_s \mid e_f)$ collapses to the conditional mean $\mathbb{E}[e_s \mid e_f]$.
   - Result of mean collapse: Robotic, metallic, muffled speech where all male speakers sound identical (~0.99 cosine similarity) and all female speakers sound identical.
   - VOIX Solution: Reformulate face-to-voice as a conditioned manifold distribution (e_s \mid e_f)$ modeled by contrastive hyperspherical projection + CVAE diversity sampling.
3. **Formal Problem Definition**:
   - Visual input: Frontal image  \in \mathbb{R}^{H \times W \times 3}$ and text string $.
   - Output: Waveform audio  \in \mathbb{R}^{N}$ at 24 kHz such that identity matches biometric constraints.
   - Mathematical equations for the fused face vector $ and contrastive InfoNCE objective:
     \mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(\text{sim}(e_f, e_s) / \tau)}{\sum_{j} \exp(\text{sim}(e_f, e_{s,j}) / \tau)}
4. **Primary Research Questions (RQs)**:
   - **RQ1**: Can 3D craniofacial ratios extract meaningful speaker timbre variance beyond simple gender/age classification?
   - **RQ2**: Does contrastive hyperspherical alignment outperform direct reconstruction in cross-modal retrieval?
   - **RQ3**: Can stochastic latent sampling achieve high intra-gender acoustic diversity (cosine similarity $< 0.70$) while maintaining biometric fidelity?
5. **Feature Store Specifications**:
   - Dataset: VoxCeleb2 (3,029 unique identities, 75,720 face-audio pairs).
   - Storage format: High-throughput HDF5 file (oxceleb2_train_val.h5) with disjoint identity splits (2,629 train / 400 zero-shot val).

---

## 4. Document 3: mvp.md Specification

### 4.1 Document Purpose
mvp.md documents the deliverable product definition, user interface, operational workflows, and empirical verification benchmarks of the shipped system.

### 4.2 Required Document Structure & Sections
1. **MVP Definition & Delivered Capabilities**:
   - Single-image zero-shot voice generation (any RGB face $\to$ 24 kHz WAV).
   - Multi-candidate generation: Generate $ distinct voice variations (=1..5$) from a single face with adjustable sampling temperature ($\tau = 0.8 - 1.2$).
   - Parametric slider controls for pitch, vocal tract, emotion, and accent.
   - Cross-lingual synthesis across 5 languages: English, Spanish, French, German, Italian.
   - Long-form utterance synthesis with chunking and cross-fade stitching (RMS standard deviation $\le 0.012$).
   - Imperceptible AudioSeal watermarking with 16-bit payload.
2. **Empirical Benchmarks & Verification Summary**:
   | Evaluation Metric | Baseline / Initial | Acceptance Target | Shipped Final Result | Status |
   |---|---|---|---|---|
   | Same-Gender Cosine Similarity | 0.999+ (Collapsed) | < 0.700 | **0.582 - 0.648** | PASS |
   | Cross-Gender Cosine Distance | > 0.60 | > 0.50 | **0.62 - 0.78** | PASS |
   | Cross-Lingual Speaker Consistency | N/A | > 0.850 | **1.0000** | PASS |
   | Long Utterance Stability (RMS Std) | N/A (drifted) | < 0.050 | **0.0100** (27.85s) | PASS |
   | Peak VRAM Footprint | > 3.8 GB | < 4.0 GB | **1431.6 MB** | PASS |
   | Visual Extraction Latency | ~1.2s | < 0.80s | **0.38s** | PASS |
3. **Public API & CLI Usage Guide**:
   - Python code example invoking VoixPipeline:
     `python
     from voix.inference.pipeline import VoixPipeline
     pipeline = VoixPipeline.from_pretrained('configs/pipeline.yaml')
     result = pipeline.generate(
         face_image='demo/faces/face1.jpg',
         text='Welcome to VOIX zero-shot synthesis.',
         temperature=1.0,
         num_voices=3,
     )
     result.save('output.wav')
     `
   - CLI commands, available presets (gender, age, accent), and argument flags.

---

## 5. Document 4: 
esearch.md Specification

### 5.1 Document Purpose

esearch.md synthesizes the literature review across the 25 research papers (stored in D:/voix/Papers/) and analyzes how VOIX adopts, modifies, or surpasses existing paradigms.

### 5.2 Required Document Structure & Sections
1. **Taxonomy of Surveyed Literature (5 Core Buckets)**:
   - **Bucket 1: Cross-Modal Speech/Face Baselines**:
     - *Speech2Face* (Oh et al., CVPR 2019): Established voice-to-face mapping; suffered from conditional mean collapse. VOIX inverts the direction and uses contrastive hyperspherical mapping.
     - *Seeing Voices and Hearing Faces* (Nagrani et al., CVPR 2018): 1:N forced choice cross-modal retrieval protocol.
     - *ImageBind* (Girdhar et al., CVPR 2023): Dual-backbone frozen alignment via metric projection heads.
     - *Imaginary Voice* (Lee et al., ICASSP 2023): Diffusion-based F2V synthesis; high latency (>2s). VOIX achieves <12ms mapping latency via CVAE + StyleAdapter.
   - **Bucket 2: Visual Biometrics & 3D Craniofacial Analysis**:
     - *InsightFace / ArcFace* (Deng et al., CVPR 2019): Geodesic distance on hypersphere $\mathbb{S}^{511}$ for identity.
     - *MediaPipe FaceMesh* (Lugaresi et al., 2019): 468 3D landmarks for real-time craniofacial ratio computation.
     - Anthropometric literature linking bizygomatic width, jaw angle, and cranial height to vocal tract acoustic properties.
   - **Bucket 3: Neural Acoustic Synthesis & Zero-Shot TTS**:
     - *StyleTTS2* (Li et al., NeurIPS 2023): Style diffusion and adversarial vocoding with 256-D latent style vector.
     - HiFi-GAN (Kong et al., NeurIPS 2020) and VITS (Kim et al., ICML 2021).
   - **Bucket 4: Metric Learning & Conditional Variational Autoencoders**:
     - *Supervised Contrastive Learning* (Khosla et al., NeurIPS 2020): Centroid clustering in hyperspherical metric space.
     - CVAE formulation: Addressing posterior collapse through KL-annealing, free bits, and explicit identity direction conditioning.
   - **Bucket 5: Audio Provenance & Watermarking**:
     - *AudioSeal* (San Roman et al., 2024): Localized, imperceptible 16-bit watermarking for AI audio attribution.
2. **SOTA Comparison Matrix**:
   - Comprehensive comparison table comparing VOIX with Speech2Face, Imaginary Voice, Face2Speech, and standard zero-shot voice cloning (XTTSv2, VoiceCraft) across: input modality, inference latency, diversity mechanism, VRAM footprint, and watermarking.

---

## 6. Document 5: 	asks.md Specification

### 6.1 Document Purpose
	asks.md acts as the complete phase-by-phase chronological engineering log and post-project technical backlog.

### 6.2 Required Document Structure & Sections
1. **Project Execution Retrospective (Phases 0 through 12)**:
   - **Phase 0: Baseline Reproduction & Problem Identification**: Audit of the 5 foundational flaws:
     1. Representation collapse in simple MLP regressor.
     2. Dimension mismatch in StyleAdapter (in_dim != speaker_dim).
     3. Unscaled MediaPipe coordinates leading to unstable IOC ratios.
     4. Extreme bias toward dominant training demographic centroids.
     5. Lack of synthetic speech provenance watermarking.
   - **Phase 1: Environment & Foundation Setup**: Python 3.12, CUDA 12.1, Buffalo_L, MediaPipe Tasks, SpeechBrain ECAPA-TDNN, StyleTTS2.
   - **Phase 2: Feature Pipeline Construction**: Multi-modal extractor (ArcFace 512 + FaceMesh 32 + Demog 16 $\to$ 560-D fused) and VoxCeleb2 HDF5 feature store.
   - **Phase 3: Contrastive Face-Voice Network (CFVN)**: Training with Supervised InfoNCE loss over 3,029 speaker centroids.
   - **Phase 4: StyleAdapter Bridge**: Mapping ECAPA 192-D to StyleTTS2 256-D style vector; acoustic/prosody split.
   - **Phase 5: Exemplar Retrieval & Hybrid Blending**: Gallery indexing and kNN blending ($\alpha = 0.70$).
   - **Phase 6: Parametric Voice Sliders**: Craniofacial geometry sliders, emotion modulation, accent shifts.
   - **Phase 7: Same-Gender Voice Differentiation**: Debugging and resolving representation collapse; fine-tuning CVAE with diversity conditioning and identity projection; reducing same-gender similarity from 0.999+ to 0.60.
   - **Phase 8: Subjective & Objective Evaluation**: MOS evaluation pipeline (scripts/run_mos_evaluation.py) and ECAPA cosine distance tests.
   - **Phase 9: Cross-Lingual Evaluation**: Validating speaker consistency across English, Spanish, French, German, Italian (consistency = 1.0000).
   - **Phase 10: Long Utterance Stability**: Chunking and cross-fade synthesis (RMS stability std = 0.0100 on 27.85s utterance).
   - **Phase 11: Hardware Profiling & VRAM Budgeting**: Validating memory footprint (1431.6 MB on GTX 1650 4 GB GPU).
   - **Phase 12: Production Hardening & Repository Hygiene**: Final cleanup of temporary test scripts and consolidation of project specifications.
2. **Post-Project Roadmap & Future Research**:
   - End-to-end joint fine-tuning of StyleTTS2 decoder on regional multi-accented speech (e.g., dedicated Indian F2S benchmark).
   - Ultra-low latency streaming neural vocoder for interactive real-time avatar synthesis.
   - Video-to-speech temporal co-articulation tracking (lips, jaw dynamic kinematics).

---

## 7. Re-Creation Execution Protocol

When authoring the 5 markdown files at project conclusion, follow these rules:
1. **Source of Truth**: All claims, dimensions, and interfaces must match the finalized codebase in oix/ (pipeline.py, models/, data/).
2. **Benchmark Figures**: Extract empirical numbers directly from:
   - 	est_ecapa_diversity.py
   - scripts/benchmark.py
   - scripts/run_mos_evaluation.py
   - BASELINE_REPORT_PHASE_0.md
3. **Format Standards**: Maintain GitHub Flavored Markdown, valid LaTeX math blocks ($...$ and ...), clean ASCII/Mermaid flowcharts, and structured status tables.
