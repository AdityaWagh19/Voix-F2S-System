# VOIX — Face-to-Speech Synthesis System

> **Predict how someone sounds, just from their face.**

VOIX (Voice + Inference from craniofacial geometry) is a research system for **probabilistic voice synthesis conditioned on 3D craniofacial geometry** — generating a plausible speaker voice from a face image alone, without any enrollment audio.

---

## Overview

Given a face image, VOIX:
1. Extracts a 2D identity embedding (ArcFace 512-d) and a 3D craniofacial shape prior (FLAME β via DECA/EMOCA)
2. Samples a plausible ECAPA-TDNN speaker embedding from a CVAE adapter conditioned on the facial features
3. Synthesizes natural speech in that predicted voice using StyleTTS 2 (zero-shot TTS)

The probabilistic CVAE adapter models the inherent **one-to-many ambiguity** of face-to-voice mapping — the same face can plausibly sound many ways due to soft tissue and habitual prosody unobservable from appearance alone.

---

## Architecture

`
Face Image
    ├── ArcFace (512-d identity embedding)
    └── DECA/EMOCA → FLAME β (3D craniofacial shape)
            ↓
     CVAE Face Adapter  (β-annealed ELBO)
            ↓
     Sampled ECAPA-TDNN Speaker Embedding (192-d)
            ↓
     StyleTTS 2 (Zero-Shot TTS)
            ↓
     Synthesized Speech Waveform
`

---

## Literature Survey

This repository includes a 25-paper literature survey grounding VOIX across six research domains:

| Bucket | Domain |
|---|---|
| 1 | Cross-modal biometric matching and synthesis baselines |
| 2 | Visual biometrics and 3D craniofacial geometry |
| 3 | Speaker embeddings, neural codecs and zero-shot TTS |
| 4 | Probabilistic generative modeling (CVAE, Flow Matching, LDM) |
| 5 | Craniofacial biomechanics and vocal tract physics |
| 6 | Audiovisual corpora, fairness and audio forensics |

---

## Key Papers

- **Speech2Face** (Oh et al., CVPR 2019) — cross-modal baseline
- **FLAME** (Li et al., SIGGRAPH 2017) — 3D craniofacial prior
- **DECA / EMOCA** (Feng et al. 2021; Daněček et al., CVPR 2022) — monocular 3D reconstruction
- **ECAPA-TDNN** (Desplanques et al., Interspeech 2020) — speaker embedding target space
- **StyleTTS 2** (Li et al., NeurIPS 2023) — zero-shot speech synthesis backend
- **AudioSeal** (San Roman et al., 2024) — ethical deepfake watermarking

---

## Ethical Safeguards

All synthesized audio is watermarked using **AudioSeal** (Meta, 2024) — a proactive localized watermarking system that embeds imperceptible neural watermarks into every generated waveform, making VOIX outputs verifiably traceable and detectable as AI-generated.

---

## Status

> 🔬 Research / Pre-implementation phase — literature survey and system design complete.

---

## License

MIT
