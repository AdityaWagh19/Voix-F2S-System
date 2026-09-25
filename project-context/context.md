# context.md
> What VOIX is, why it exists, and what success looks like.

---

## What is VOIX

VOIX is a research system for **probabilistic Face-to-Voice (F2V) generation**. Given a face image and optional text, it generates a distribution over plausible speaker voices — not a single deterministic voice. The system models the biological reality that a face *constrains* but does not *uniquely determine* a voice.

**Three research contributions:**
1. A CVAE-based probabilistic mapper predicting a Gaussian distribution over ECAPA-TDNN speaker embeddings conditioned on a multi-feature facial representation
2. An India-specific F2S benchmark dataset (~20 hours, 40-60 speakers, 5 Indian languages, 25% code-switching)
3. A systematic ablation of facial feature types (latent identity vs. craniofacial morphology vs. soft demographics) establishing which facial information actually predicts voice

**Infrastructure:** Solo researcher — RTX 4050 / Colab T4, Google Drive, 16-week timeline.

---

## Why It Exists

### The Determinism Problem

Every existing F2V system models face-to-voice as a deterministic function `f: Face -> Single Voice`. This is scientifically wrong.

Voice is shaped by two categories of factors:

| Category | Examples | Visible in Face? |
|---|---|---|
| Physiological | Vocal tract length, jaw structure, laryngeal anatomy | Partially |
| Learned/environmental | Accent, prosody, code-switching, vocal habits | Not at all |

The correct formulation is:
```
P(voice | face) : Face -> Distribution over Speaker Embeddings -> Multiple Plausible Voices
```
A face provides *evidence* about voice, not *specification* of it.

### The Indian Representation Gap

All established F2V datasets (VoxCeleb1/2, LRS3) are Western-centric. Indian faces and voices present out-of-distribution challenges no existing system addresses:
- Diverse craniofacial morphology across regional genetic clusters
- Retroflex, dental, and other consonant patterns absent in English training data
- Tonal harmonic prosody from Tamil, Telugu
- Code-switching (Hinglish, Tanglish, Benglish) — almost entirely absent in Western datasets

---

## Formal Problem Definition

Given face image I and optional text T, learn:
```
P(e_s | e_f) : Speaker Embedding Distribution conditioned on Face Embedding
```
Where:
- e_f in R^560 — fused face representation (ArcFace 512-D + FaceMesh ratios 32-D + soft demographics 16-D)
- e_s in R^192 — speaker embedding in ECAPA-TDNN space
- P(e_s | e_f) = N(mu_theta(e_f), sigma_theta^2(e_f)) — predicted Gaussian over speaker embedding space

At inference, generate K plausible voices:
```
z_k ~ N(mu_theta(e_f), sigma_theta^2(e_f)),  k = 1,...,K
Speech_k = StyleTTS2(text=T, style=P_adapter(z_k))
```

---

## Research Questions

**RQ1 — Probabilistic Modeling:**
Can a CVAE-based mapper learn a meaningful distribution over speaker embeddings conditioned on facial appearance, such that sampled voices are both perceptually plausible and mutually diverse?

**RQ2 — Feature Contribution:**
Does incorporating explicit craniofacial morphological features (MediaPipe FaceMesh ratios) alongside ArcFace latent embeddings improve face-to-voice prediction over latent embeddings alone?

**RQ3 — Demographic Generalization:**
Do F2V models trained on VoxCeleb2 exhibit measurable performance degradation on Indian faces/voices, and does fine-tuning on the Indian benchmark recover this gap?

---

## Hypotheses

**H1:** A CVAE mapper trained on utterance-level embedding pairs will learn a non-trivial posterior that, when sampled, produces speaker embeddings with higher face-voice perceptual consistency than a noise-added deterministic baseline.

**H2:** Craniofacial ratio features from MediaPipe FaceMesh contribute incrementally predictive information (specifically F0 range and formant structure) beyond ArcFace alone, demonstrated via SHAP attribution and ablation.

**H3:** Zero-shot evaluation of a VoxCeleb2-trained model on the Indian benchmark will show >=10% drop in SECS and Recall@5 vs. VoxCeleb1. Fine-tuning on Indian data will partially recover this gap.

---

## Objectives

| # | Objective |
|---|---|
| O1 | Design and implement a CVAE mapper learning P(e_s | e_f) from VoxCeleb2 face-voice pairs |
| O2 | Construct a 560-D multi-feature face representation (ArcFace 512 + FaceMesh 32 + demographics 16) |
| O3 | Design and train a StyleTTS2 projection adapter (ECAPA space -> StyleTTS2 style space) |
| O4 | Build and release the India-specific F2S benchmark (~20 hrs, 40-60 speakers, 5 languages) |
| O5 | Conduct systematic feature and architecture ablation studies |
| O6 | Evaluate cross-demographic generalization zero-shot and fine-tuned on the Indian benchmark |

---

## Non-Goals (Out of Scope for This Paper)

- Expression-conditioned emotion branch for prosody-aware synthesis
- Normalizing Flow mapper (future upgrade path)
- Real-time inference (<100ms latency)
- Non-photorealistic inputs (anime, fantasy art character portraits)
- Cross-lingual Indian language generation beyond Indian-accented English
- Longitudinal voice aging modeling

---

## Success Criteria

| Metric | Target |
|---|---|
| SECS (VoxCeleb1, in-domain) | Greater than deterministic baseline B3 |
| Recall@5 (VoxCeleb1) | Greater than random B1 and retrieval baseline B2 |
| Diversity Score | Meaningful (>0), not random noise |
| ECE | Lower than deterministic baseline |
| H3 gap | >=10% SECS drop confirmed; fine-tune recovery demonstrated |
| MOS naturalness | >=3.5 on 5-point scale |

---

## Publication Target

**Primary:** INTERSPEECH (next deadline after 16-week window)
**Fallback:** ICASSP or NeurIPS Datasets & Benchmarks track (if Indian benchmark is the dominant contribution)

---

*Read next: architecture.md*
