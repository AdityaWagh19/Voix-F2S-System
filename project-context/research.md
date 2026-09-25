# research.md
> What the 25 surveyed papers mean for VOIX — organized by engineering relevance, not bibliography.

Full survey details: ../Initial Docs/voix_lit_survey_outline.md
Detailed matrix: ../Initial Docs/Voix F2S lit survey.csv

---

## What VOIX Borrows from Each Bucket

### Bucket 1 — Cross-Modal Baselines (Papers 1-4)

**Speech2Face (Oh et al., CVPR 2019)**
- Establishes voice-to-face cross-modal signal is learnable without explicit demographic labels
- Flaw VOIX fixes: deterministic L1 regression produces blurred "mean" faces — same problem exists in F2V direction; CVAE solves this

**Seeing Voices & Hearing Faces (Nagrani et al., CVPR 2018)**
- Provides the cross-modal evaluation protocol VOIX adopts: 1:N forced-choice retrieval (Recall@K)
- Shows face-voice matching exceeds human performance with contrastive training

**ImageBind (Girdhar et al., CVPR 2023)**
- Demonstrates visual + acoustic spaces can be aligned via a shared metric manifold without direct paired supervision
- Informs VOIX's design: ArcFace identity embedding and ECAPA speaker embedding can be aligned in a shared latent space

**Imaginary Voice (Lee et al., ICASSP 2023) — Closest prior work**
- Most direct baseline: face image -> diffusion model -> TTS
- VOIX differences: (1) CVAE not diffusion; (2) 3D biomechanical craniofacial prior not 2D identity; (3) morphology ablation study; (4) Indian benchmark
- Key metric to beat: 5-way cross-modal matching accuracy 38.0% (face-conditioned, unseen speakers)

---

### Bucket 2 — Visual Representation (Papers 5-8)

**ArcFace (Deng et al., CVPR 2019) — Used directly**
- VOIX uses ArcFace ResNet-50 as the frozen 512-D identity extractor
- AAM-Softmax margin training produces hyperspherical embeddings optimal for cosine-similarity downstream tasks
- Implementation: InsightFace library

**FLAME (Li et al., SIGGRAPH 2017) — Background context**
- 3D morphable model separating identity shape beta (300-D) from expression psi and pose theta
- Relevant if VOIX extends to full 3D craniofacial priors in future work
- Current VOIX: 2D FaceMesh ratios, not full FLAME reconstruction

**DECA (Feng et al., TOG 2021) — Methodology reference**
- Detail-consistency loss recipe: supervise identity features via a pretrained recognition network rather than raw geometric loss
- VOIX adopts this pattern: SECS loss via ECAPA rather than MSE on raw waveform features

**EMOCA (Danecek et al., CVPR 2022) — Methodology reference**
- Perceptual consistency loss approach: L_emo computed as L2 distance between emotion-network features of input vs. rendered output
- Pattern VOIX adopts for adapter training: perceptual SECS loss rather than raw reconstruction loss

---

### Bucket 3 — Acoustic Representation & Synthesis (Papers 9-12)

**ECAPA-TDNN (Desplanques et al., Interspeech 2020) — Used directly**
- VOIX's ground-truth acoustic target space: 192-D speaker embedding
- The CVAE mapper is trained to predict distributions over this exact space
- EER 0.87% on VoxCeleb1 — proven discriminative speaker representation
- Implementation: SpeechBrain library

**Descript Audio Codec / DAC (Kumar et al., NeurIPS 2023) — Evaluation reference**
- Informs decision: VOIX targets continuous ECAPA embeddings (192-D) rather than discrete RVQ tokens
- RVQ architecture (Snake activations, quantizer dropout) relevant if VOIX extends to discrete token generation

**StyleTTS 2 (Li et al., NeurIPS 2023) — Used directly**
- VOIX's frozen speech synthesis backend
- Zero-shot TTS conditioned on speaker style embedding — no reference audio needed at inference
- Style diffusion sampler + SLM discriminator architecture informs why SECS loss is the right training signal for the adapter
- Implementation: Official StyleTTS2 repo

**NaturalSpeech 3 (Ju et al., arXiv 2024) — Design justification**
- Factorizing speech into content/prosody/timbre/detail subspaces proves timbre must be isolated before biometric conditioning
- Justifies VOIX targeting ECAPA timbre embedding specifically, not raw waveform or entangled representations

---

### Bucket 4 — Probabilistic Modeling (Papers 13-16)

**CVAE (Sohn et al., NeurIPS 2015) — Core framework**
- VOIX's probabilistic mapper is directly built on this formulation
- Conditional ELBO: L = -KL(q_phi(z|x,y) || p(z|x)) + E[log p_theta(y|x,z)]
- x = face features, y = speaker embedding, z = latent voice factors
- Train/test discrepancy (reconstruction vs. sampling) is a known issue — addressed by beta annealing

**beta-VAE (Higgins et al., ICLR 2017) — Training recipe**
- VOIX uses beta-annealed KL weighting to prevent posterior collapse
- beta schedule: linear warmup from 0 to beta_max over first 30% of training steps
- Free bits strategy (lambda_fb = 0.5 nats) prevents selective per-dimension collapse

**Flow Matching (Lipman et al., arXiv 2022) — Ablation comparison**
- Faster, more stable alternative to CVAE via straight optimal-transport paths
- VOIX evaluates flow-matching adapter as an architecture ablation variant
- Provides upgrade path if CVAE underperforms

**Latent Diffusion Models (Rombach et al., CVPR 2022) — Design justification**
- Validates VOIX's core decision: predict compact 192-D embedding latent, not raw waveform
- Cross-attention conditioning pattern directly informs the adapter architecture

---

### Bucket 5 — Biomechanical Foundations (Papers 17-19)

**Fitch & Giedd (JASA 1999) — Physical grounding**
- MRI study of 129 subjects: VTL-height correlation r=0.926, VTL-log body mass r=0.941
- Sex differences in VTL emerge at puberty — justifies soft demographic indicators in VOIX
- Establishes the biological basis for craniofacial geometry -> vocal tract geometry -> voice correlation

**Macari et al. (Journal of Voice 2014) — Correlation bounds**
- Jaw length correlates with F0 at r=-0.528 to -0.577 (moderate, not strong)
- Angular cephalometric measures poorly correlated (r<0.3)
- **Critical implication for VOIX:** Moderate correlations justify probabilistic modeling. If correlations were strong, deterministic mapping would be appropriate. They are not — so sigma must be learned, not ignored.

**Li et al. ACM MM 2023 — Direct validation**
- Rethinking Voice-Face Correlation: A Geometry View
- Nasal cavity and cranial structure show strongest geometric correlations with voice acoustics
- Many jaw-width AMs not statistically predictable — confirms limits of face-to-voice inference
- Directly validates VOIX's FaceMesh feature selection (nose width, nose height, forehead ratio)
- Guards against spurious semantic shortcuts in cross-modal correlation

---

### Bucket 6 — Corpora, Fairness, Ethics (Papers 20-25)

**Learnable PINs (Nagrani et al., ECCV 2018) — Evaluation protocol**
- Self-supervised cross-modal retrieval without identity labels
- VOIX's primary evaluation: Learnable PINs retrieval protocol (face->voice and voice->face AUC)
- Curriculum hard-negative mining strategy relevant for VOIX adapter training

**VoxCeleb2 (Chung et al., 2018) — Primary training data**
- 1M+ utterances, 6,112 speakers — VOIX's main training dataset
- EER 3.95% with ResNet-50 contrastive — sets quality floor for speaker embedding representations

**Looking to Listen / AVSpeech (Ephrat et al., 2018)**
- AVSpeech quality filtering criteria (active speaker verification, SNR>20dB, single speaker) directly adopted in VOIX's Indian dataset curation pipeline
- SyncNet threshold (>4.5) is the baseline — calibrated per source for Indian content

**Fenu & Marras (Procedia CS 2022) — Fairness design**
- Model-level fusion reduces gender disparity gap from 5.8 to 0.3 EER points
- Justifies VOIX's demographic fairness evaluation and Indian cohort construction

**SVARAH (Javed et al., arXiv 2023)**
- Whisper-large achieves 7.2% WER on Indian accents vs. much lower on native English
- Quantifies the distribution shift VOIX's Indian benchmark is designed to expose
- Informs which model families generalize to Indian accents

**AudioSeal (San Roman et al., arXiv 2024) — Ethics integration**
- AUC 0.97, IoU 0.99, 485x faster detection than WavMark
- VOIX integrates AudioSeal watermarking on all synthesized outputs
- All generated audio carries imperceptible neural watermarks — detectable as AI-generated

---

## Key Design Decisions Grounded in Literature

| Decision | Grounded By |
|---|---|
| CVAE over normalizing flow | Compute constraints; CVAE sufficient for 192-D target (Sohn 2015) |
| Utterance-level ECAPA targets (not speaker-averaged) | Required to learn non-trivial sigma (beta-VAE insight) |
| Cosine loss not MSE | ECAPA trained with angular margin objectives (ArcFace, ECAPA papers) |
| Free bits + beta annealing | Prevents posterior collapse (beta-VAE 2017) |
| ECAPA not DAC tokens | Continuous embeddings sufficient for identity; DAC for compression use cases |
| Soft demographics not hard labels | Ethical framing; Fenu & Marras fairness analysis |
| 560-D fusion via linear MLP | Transformer overkill at this scale with fixed feature types |
| SyncNet threshold 4.5 | AVSpeech pipeline (Ephrat 2018) — calibrated for Indian content |

---

*Read next: mvp.md*
