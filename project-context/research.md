# research.md
> What the surveyed papers mean for VOIX — organized by engineering relevance.
> Last updated: 2026-10-16.

Full survey details: `../Papers/` directory (25 reference papers).

---

## What VOIX Borrows from Each Bucket

### Bucket 1 — Cross-Modal Baselines (Papers 1–4)

**Speech2Face (Oh et al., CVPR 2019)**
- **Contribution:** Establishes voice-to-face cross-modal signal is learnable without explicit demographic supervision.
- **Flaw VOIX Fixes:** Deterministic L1/MSE regression produces blurred "mean" faces — the exact same problem exists in the reverse F2V direction (acoustic mean collapse). VOIX solves this via Supervised InfoNCE metric contrastive mapping over speaker centroids and hybrid exemplar manifold retrieval.

**Seeing Voices & Hearing Faces (Nagrani et al., CVPR 2018)**
- **Contribution:** Provides the cross-modal evaluation protocol VOIX adopts: 1:N forced-choice retrieval (Recall@K).
- **Application:** Demonstrates face-voice cross-modal matching significantly exceeds random chance under contrastive hyperspherical alignment, directly justifying VOIX's InfoNCE objective.

**ImageBind (Girdhar et al., CVPR 2023)**
- **Contribution:** Demonstrates visual and acoustic modalities can be aligned via a shared metric manifold without requiring end-to-end multi-modal retraining.
- **Application:** Informs VOIX's frozen dual-backbone design: ArcFace biometric space ($\mathbb{S}^{511}$) and ECAPA speaker space ($\mathbb{S}^{191}$) are aligned via a compact metric projection head without altering pretrained representations.

**Imaginary Voice (Lee et al., ICASSP 2023) — Closest Prior Work**
- **Contribution:** Face image → diffusion model → TTS acoustic generation.
- **VOIX Differences:**
  1. Lightweight metric contrastive & exemplar retrieval (<12 ms latency) rather than iterative diffusion denoising.
  2. Multi-feature 560-D input incorporating explicit 3D craniofacial morphology and soft demographics rather than 2D latent identity alone.
  3. Direct feature and paradigm ablation study on 400 zero-shot held-out speakers.
  4. Controllable expressivity engine (accent, emotion, clinical speech differences, cross-lingual) — not present in Imaginary Voice.

---

### Bucket 2 — Visual Representation (Papers 5–8)

**ArcFace (Deng et al., CVPR 2019) — Used Directly**
- **Contribution:** Additive Angular Margin (AAM-Softmax) loss enforces hyperspherical clustering of facial identities.
- **Application:** VOIX uses InsightFace's pretrained ArcFace ResNet-50 as the frozen 512-D identity extractor. Features are normalized onto $\mathbb{S}^{511}$, ensuring robustness against lighting and camera variation.

**FLAME (Li et al., SIGGRAPH 2017) — Biomechanical Context**
- **Contribution:** 3D morphable head model disentangling identity shape $\beta$ (300-D) from expression $\psi$ and pose $\theta$.
- **Application:** Inspires VOIX's pose filtering (isolating pure neutral frontal frames) and sets the foundation for future full-3D head mesh integrations.

**DECA (Feng et al., TOG 2021) — Methodology Reference**
- **Contribution:** Identity-consistency supervision via a pretrained recognition network rather than unconstrained pixel/geometric loss.
- **Application:** VOIX adopts this pattern: acoustic supervision is enforced in ECAPA-TDNN feature space, rather than raw STFT or waveform loss.

**EMOCA (Danecek et al., CVPR 2022) — Methodology Reference**
- **Contribution:** Disentangling emotion and expression from identity geometry using perceptual feature distances.
- **Application:** Guides VOIX's strict exclusion of transient facial expressions (smiling, squinting) to prevent prosodic overfitting on static facial snapshots. Also informs the expressivity engine design.

---

### Bucket 3 — Acoustic Representation & Synthesis (Papers 9–12)

**ECAPA-TDNN (Desplanques et al., Interspeech 2020) — Used Directly**
- **Contribution:** Squeeze-and-Excitation channel-attention architecture for robust speaker verification embeddings.
- **Application:** Serves as VOIX's 192-dimensional target space ($\mathbb{S}^{191}$), providing clean speaker identity clustering invariant to lexical content and linguistic variations.

**StyleTTS 2 (Li et al., NeurIPS 2024) — Used Directly (Frozen)**
- **Contribution:** Diffusion-based style transfer for high-fidelity TTS with style conditioning vector.
- **Application:** VOIX feeds the projected 128-D style vector from the StyleAdapter into the completely frozen StyleTTS 2 vocoder, producing 24 kHz natural speech. Only the StyleAdapter (< 200k parameters) is trained.

**AudioSeal (San Roman et al., ICML 2024) — Used Directly**
- **Contribution:** Localizable neural audio watermarking with AUC=0.97 and sample-level IoU=0.99.
- **Application:** Every VOIX synthesized waveform is embedded with a 16-bit imperceptible neural watermark for AI provenance attribution. Survives MP3 compression, cropping, and tempo changes.

---

### Bucket 4 — Training Datasets (Papers 21–24)

**VoxCeleb2 (Chung et al., Interspeech 2018) — Training Data**
- 75,720 utterances from 3,029 unique identities used to train CVAE and Metric Contrastive Mapper.
- Speaker-disjoint 2,629/400 train/validation split (zero identity overlap).

**Svarah (Joshi et al., LREC-COLING 2024) — Future Scope Reference**
- **Contribution:** 117 speakers, 9.6 hours, 19 Indian accents evaluated on English ASR systems.
- **Application (Future Scope):** Referenced as a benchmark for the India F2S out-of-domain evaluation protocol. The dedicated Indian corpus curation is formally designated as Future Scope for the current release.

---

### Bucket 5 — Fairness & Ethics (Paper 23)

**Demographic Fairness in Multimodal Biometrics (Rozsa et al., 2023)**
- **Contribution:** Identifies systematic disparities in cross-modal biometric systems across demographic subgroups.
- **Application:** Motivates VOIX's gender-conditioned retrieval (not blind cosine NN) and demographic anchor-based style conditioning, ensuring $F_0$ pitch registers are biologically appropriate across all subjects.

---

## Key Engineering Insights from Literature Survey

| Insight | Source | VOIX Implementation |
|---|---|---|
| MSE/L1 regression collapses to mean | Speech2Face, Imaginary Voice | InfoNCE + Exemplar Retrieval |
| Hyperspherical embeddings → cosine loss | ArcFace, ECAPA | `CosineReconstructionLoss`, all embeddings on $\mathbb{S}^{N-1}$ |
| Frozen backbone alignment via metric projection | ImageBind | Frozen ArcFace + ECAPA, trainable projection only |
| Pose must be filtered for static identity | FLAME, EMOCA | `|pitch| < 15°, |yaw| < 15°` pose gate |
| Speaker centroids ≫ utterance-level targets | VoxCeleb2 ablation | Precomputed centroid gallery, InfoNCE over centroids |
| Acoustic watermarking is safety-critical | AudioSeal | Mandatory by default in `VoixPipeline` |
