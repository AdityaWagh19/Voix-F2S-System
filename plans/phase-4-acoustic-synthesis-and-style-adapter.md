# Phase 4: Acoustic Synthesis & Vocoder Style Bridge

> **Phase Identifier:** `PHASE-4`  
> **Target Duration:** Weeks 7–8  
> **Status:** Complete (Executed & Verified)  
> **Prerequisites:** `PHASE-3` Complete  

---

## 1. Objective

Integrate the trained face-to-voice mapper with the frozen StyleTTS 2 acoustic synthesis backend. Bridge the metric ECAPA-TDNN speaker verification embedding space ($\mathbb{R}^{192}$) with StyleTTS 2's expressive style conditioning space using the `VocoderStyleBridge` and demographic acoustic reference anchors. Deliver a zero-shot, multi-voice waveform generation engine that outputs natural, speaker-consistent, AudioSeal-watermarked 24 kHz speech waveforms from a single face image and text prompt without vocoder distortion.

---

## 2. Scope

### In Scope
- Frozen StyleTTS 2 synthesis backend integration (`voix/models/tts_backend.py`).
- Resolution of the acoustic representation mismatch between ECAPA verification embeddings and vocoder style representations.
- `VocoderStyleBridge` architecture with dual-tier style blending:
  $$\mathbf{e}_{s, \text{hybrid}} = \alpha \, \hat{\mathbf{e}}_s + (1-\alpha) \sum_{k=1}^K w_k \mathbf{c}_{(k)}$$
  $$\mathbf{s}_{\text{target}} = \beta \, \mathcal{P}_{\text{adapt}}(\mathbf{e}_{s, \text{hybrid}}) + (1-\beta) \, \mathbf{s}_{\text{anchor}}$$
- Demographic acoustic reference anchor integration (`data/raw/librispeech_ref/`) for male and female baseline trajectories.
- Perceptual hyperparameter sweep calibrating optimal blending parameters ($\alpha=0.70, \beta=0.30, \text{blend\_weight}=0.75$).
- Proactive neural audio watermarking via AudioSeal (San Roman et al., 2024).
- End-to-end multi-voice waveform synthesis API.

### Out of Scope
- End-to-end retraining or fine-tuning of the StyleTTS 2 vocoder weights (strictly frozen).
- Real-time speech streaming (<100 ms latency).
- Cross-lingual phonemizer adaptation beyond accented English synthesis.

---

## 3. Design Decisions & Rationale

### 3.1 The Verification-Vocoder Latent Gap
ECAPA-TDNN embeddings are trained with angular margin loss (AAM-Softmax) to maximize inter-speaker angular distance and minimize intra-speaker distance. In contrast, StyleTTS 2 style conditioning vectors capture expressive acoustic dynamics (prosody, duration, energy, and pitch contour). Directly feeding raw ECAPA vectors into StyleTTS 2 produces severe acoustic artifacts, including metallic buzzing, raspiness, and phoneme dropping.

### 3.2 Dual-Tier Blending via the Vocoder Style Bridge
To resolve this gap, VOIX decouples speaker conditioning into two complementary tiers:
1. **Tier 1 (Speaker Identity Manifold):** Blends the parametric facial prediction $\hat{\mathbf{e}}_s$ with empirical nearest-neighbor centroids from the retrieval gallery:
   $$\mathbf{e}_{s, \text{hybrid}} = \alpha \hat{\mathbf{e}}_s + (1-\alpha) \sum_{k=1}^K w_k \mathbf{c}_{(k)}$$
2. **Tier 2 (Vocoder Trajectory Anchoring):** Projects $\mathbf{e}_{s, \text{hybrid}}$ into style space and blends it with clean demographic reference anchors:
   $$\mathbf{s}_{\text{target}} = \beta \mathcal{P}_{\text{adapt}}(\mathbf{e}_{s, \text{hybrid}}) + (1-\beta) \mathbf{s}_{\text{anchor}}$$
   - **Empirically Tuned Weights:** $\alpha = 0.70, \beta = 0.30, \text{blend\_weight} = 0.75$.
   - **Result:** Provides facial identity fidelity while ensuring the neural vocoder receives an acoustically stable, natural style trajectory.

### 3.3 Ethical Audio Attribution via AudioSeal
All synthesized speech is passed through AudioSeal (`voix/models/watermarking.py`). An imperceptible 16-bit watermark is embedded directly into the 24 kHz waveform, providing cryptographic provenance and deepfake attribution with zero audible distortion.

---

## 4. Sequential Implementation Tasks

### T4.1: Frozen StyleTTS2 Backend Setup
- [x] Configure StyleTTS 2 model loading and phonemizer execution in `voix/models/tts_backend.py`.
- [x] Enforce `requires_grad = False` on all vocoder parameters.

### T4.2: VocoderStyleBridge Implementation
- [x] Implement `VocoderStyleBridge` class supporting projection and weighted style interpolation.
- [x] Implement robust fallback handling for CPU and single-GPU environments.

### T4.3: Demographic Reference Anchor Curation
- [x] Curate clean male and female LibriSpeech reference audio clips in `data/raw/librispeech_ref/`.
- [x] Register anchor audio file paths in `configs/environment.yaml`.

### T4.4: Perceptual Hyperparameter Tuning
- [x] Conduct systematic parameter sweep across blending weights ($\alpha \in [0.5, 0.8]$, $\beta \in [0.2, 0.4]$, $\text{blend\_weight} \in [0.5, 0.9]$).
- [x] Calibrated optimal configuration: $\alpha = 0.70$, $\beta = 0.30$, $\text{blend\_weight} = 0.75$.
- [x] Verified synthesized audio across test subjects: eliminated distortion and robotic artifacts.

### T4.5: AudioSeal Watermarking Integration
- [x] Integrated `AudioSealWatermarker` in `voix/models/watermarking.py`.
- [x] Verified watermark persistence under MP3 compression and resampling.

---

## 5. Acceptance Criteria & Test Matrix

| Test Suite / Metric | Target Criteria | Actual Verified Result | Status |
|---|---|---|---|
| Audio Sampling Rate | 24 kHz mono WAV | Exact 24,000 Hz WAV | Passed |
| Synthesis Naturalness | Clean, natural speech without metallic buzzing | Human audit confirmed crisp vocal timbre | Passed |
| Gender Alignment | Synthesized voice matches facial gender | 100.0% gender consistency in hybrid mode | Passed |
| Watermark Detection | AudioSeal detector $>95\%$ confidence | $>98\%$ detection rate | Passed |
| End-to-End Pipeline Test | `pytest tests/test_pipeline.py` | 100% Passed | Passed |

---

## 6. Deliverables & Production Artifacts

1. `voix/models/tts_backend.py`: `StyleTTS2Backend` and `VocoderStyleBridge`.
2. `voix/models/watermarking.py`: `AudioSealWatermarker`.
3. `data/raw/librispeech_ref/`: Curated male and female acoustic reference anchor WAVs.
4. `configs/environment.yaml`: Validated synthesis configuration with anchor paths.

---

*Read next: [phase-5-indian-f2s-benchmark-curation.md](phase-5-indian-f2s-benchmark-curation.md)*
