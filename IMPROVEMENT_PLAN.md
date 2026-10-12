# Voix: Face-to-Voice Synthesis — Comprehensive Improvement Plan

## Publishability Assessment

### Is This Publishable?

**Short answer: Yes — at the right venue, with the right framing.**

| SECS Range | Venue | Condition |
|------------|-------|-----------|
| 0.10–0.15 (current regressor) | Workshop / Preprint | Not sufficient alone |
| 0.18–0.25 (ArcFace + InfoNCE) | **INTERSPEECH, ICASSP** | With ablation study ✅ |
| 0.25–0.32 (+ retrieval) | **INTERSPEECH oral, ACL Findings** | Strong paper ✅ |
| 0.32–0.40 (+ E2E fine-tuning) | **CVPR, ECCV, NeurIPS** | Top-tier ✅ |

> **The novel contribution:** A systematic, component-wise ablation study of face-to-voice
> synthesis — showing quantitatively which architectural choices (CVAE vs. regressor vs.
> contrastive), which face representations (morphological vs. identity), and which inference
> strategies (direct prediction vs. retrieval-augmented) contribute most to speaker identity
> transfer. No existing paper provides this clean comparison.

---

## Evaluation Metrics (Tracked at Every Checkpoint)

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **SECS** | Speaker identity cosine similarity (primary) | > 0.25 |
| **val_cos** | Training proxy for SECS | > 0.22 |
| **Recall@1/3/10** | Face→voice retrieval accuracy | > random baseline |
| **Gender Accuracy** | Does synthesized voice match face gender? | > 85% |
| **MOS-N** | Speech naturalness, human 1–5 scale | > 3.5 |
| **FAD** | Fréchet Audio Distance (distributional quality) | < 2.0 |
| **WER** | Word Error Rate (intelligibility) | < 10% |

---

## Checkpoint A — Baseline (Complete ✅)

**State:** Direct cosine regressor, morphological face features (560D)

| Metric | Value |
|--------|-------|
| val_cos | 0.1366 |
| Expected SECS | 0.08–0.12 |
| Expected Gender Acc | ~70% |
| Training time | 30 epochs / 30 min |

**Significance:** 16.3× better than CVAE (0.0084). Proved the face features ARE informative.
Plateau hit at epoch 1 → demographic ceiling. Individual identity requires better approach.

---

## Phase 1 — Data Quality Wins

**Goal:** Improve training signal quality with zero architectural cost.

### Task 1.1 — Speaker-Averaged ECAPA Targets (20 min)
- Compute mean ECAPA across all utterances per speaker
- Use as regression target instead of noisy per-utterance ECAPA
- Reduces utterance-level variation (speaking style, recording conditions)
- Code: group by `speaker_id`, average `speaker_features`, store in new H5 key

### Task 1.2 — Gender Auxiliary Loss (1 hr)
- Add binary cross-entropy on gender prediction (M/F)
- CSV already has `gender` column for all 71k clips
- `Loss = cosine_distance + 0.3 × gender_BCE`
- Forces face encoder to learn voice-correlated demographic structure

### Task 1.3 — Clip Quality Filtering (1 hr)
- For each speaker, compute mean ECAPA and per-clip deviation
- Remove clips where ECAPA is > 2σ from speaker mean (outliers)
- Cleaner training distribution, more consistent supervision

### Checkpoint B — After Phase 1
**Expected:** val_cos 0.15–0.18, SECS ~0.10–0.14, Gender Acc ~78–83%

---

## Phase 2 — ArcFace Identity Features (HIGHEST ROI)

**Goal:** Replace morphological features with identity-focused ArcFace embeddings.

**Why this is the biggest single change:**
Current features (560D): face geometry — nose shape, jaw width, eye spacing.
These encode WHAT a face looks like. Voice identity correlates with WHO the person is.
ArcFace (512D): encodes person identity — the same representation achieving 99%+ accuracy
on face verification. Trained to distinguish millions of identities.

### Task 2.1 — Extract ArcFace Embeddings (2 hrs)
- Use already-installed `w600k_r50.onnx` from `checkpoints/models/buffalo_l/`
- Re-run face extraction on all training frames → 512D ArcFace per clip
- Store as `arcface_features` (80832, 512) in features.h5
- Fit new face scaler for 512D space

### Task 2.2 — Retrain Regressor with ArcFace (30 min training)
- Replace face_dim=560 → face_dim=512
- Keep identical architecture and hyperparameters
- 50-epoch training run
- Compare directly vs Checkpoint B (same everything except face features)

### Task 2.3 — Combined Best: ArcFace + Speaker-Avg + Gender Loss (30 min training)
- Stack the best elements from Phase 1 + Phase 2
- This is the definitive "simple regressor" ceiling

### Checkpoint C — After Phase 2
**Expected:** val_cos 0.20–0.27, SECS ~0.15–0.22, Gender Acc ~85–90%

---

## Phase 3 — InfoNCE Contrastive Training

**Goal:** Learn a joint face-voice embedding space, surpass regression ceiling.

**Why regression has a hard ceiling:**
Direct regressor learns E[voice | face] — the average voice for a face type.
It cannot represent distributional structure (one face type → many plausible voices).
InfoNCE treats face-voice matching as a discrimination task, learning richer structure.

### Task 3.1 — InfoNCE Loss (3 hrs)
```
For a batch of N face-voice pairs:
- Compute cosine similarity matrix (N×N)
- Diagonal = positive pairs (same person)
- Off-diagonal = negatives (different people)
- Loss = NTXent cross-entropy over rows and columns
- Temperature τ = 0.07
```
- Project face and voice to shared 128D unit sphere via linear projection heads
- Separate projection heads from the backbone encoders

### Task 3.2 — Hard Negative Mining (2 hrs)
- Pre-compute per-speaker mean ECAPA embeddings
- For each batch, also include top-5 most ECAPA-similar speakers as extra hard negatives
- These are speakers who SOUND similar but LOOK different → hardest case
- Significantly improves discriminability vs easy random negatives

### Task 3.3 — Two-Stage: Contrastive Pretrain → Regressor Fine-tune (4 hrs)
- Stage 1: Train InfoNCE for 30 epochs → aligned face-voice space
- Stage 2: Freeze backbone, add regression head, fine-tune 20 epochs with cosine loss
- Best of both: aligned space (contrastive) + direct prediction (regression)

### Checkpoint D — After Phase 3
**Expected:** val_cos 0.22–0.30, SECS ~0.18–0.25, Recall@3 ~10–18%

---

## Phase 4 — Retrieval-Augmented Inference

**Goal:** Stop predicting ECAPA — retrieve REAL ECAPA embeddings from similar faces.

**Why this fundamentally changes the ceiling:**
Predicted ECAPA has prediction error embedded — even the best model (val_cos ~0.28)
means 72% of the direction is wrong. Retrieved REAL ECAPA from a training speaker who
looks like the target has ZERO prediction error. The voice is always a real, natural voice.

### Inference at Test Time:
```
Input: target face image
  ↓
Step 1: Extract ArcFace embedding (512D)
  ↓
Step 2: FAISS search → find top-K most similar faces in training gallery
        (using the aligned face-voice space from Phase 3)
  ↓
Step 3: Retrieve paired real ECAPA embeddings for top-K neighbors
  ↓
Step 4: Weighted average → blended ECAPA (weights = cosine similarities)
  ↓
Step 5: XTTS conditioning → synthesized speech
```

### Task 4.1 — Build FAISS Gallery (2 hrs)
- Index all 55,448 training face embeddings (Phase 3 representation)
- Store paired ECAPA embeddings for retrieval
- Use FAISS IVF-Flat index for fast cosine search

### Task 4.2 — Evaluate Retrieval Quality (1 hr)
- On validation set: given a face, retrieve K voices, compute SECS vs ground truth
- Compare retrieval SECS vs direct prediction SECS
- Shows empirically how much headroom retrieval unlocks

### Task 4.3 — Optimal K and Hybrid α Sweep (1 hr)
- Test K ∈ {1, 3, 5, 10, 20}
- Test blending α × predicted + (1-α) × retrieved, α ∈ {0, 0.25, 0.5, 0.75, 1.0}
- Select optimal (K, α) on validation set, report on test set only once

### Checkpoint E — After Phase 4
**Expected:** SECS ~0.25–0.33, Recall@3 ~18–28%, Gender Acc ~90–94%
**This is the primary publishable result.**

---

## Phase 5 — End-to-End Acoustic Fine-Tuning (Optional / Bonus)

**Goal:** Optimize directly for audio quality rather than ECAPA proxy.

> ⚠️ Only attempt if Phase 4 achieves SECS > 0.22 on test set.
> This is the most complex phase. Required for top-tier venue submission.

### Task 5.1 — XTTS Conditioning Space Analysis (2 hrs)
- Map how audio-level speaker characteristics change with ECAPA perturbations
- Identify the principal directions in ECAPA space that affect: pitch, timbre, tempo
- Required to design meaningful audio-level loss

### Task 5.2 — Differentiable Speaker Conditioning (4 hrs)
- Expose XTTS speaker conditioning as differentiable operation
- Allow gradient flow from spectrogram-level loss back to face encoder
- Freeze XTTS vocoder, only update face→ECAPA mapping

### Task 5.3 — Joint Training (1 day)
```
Loss = 0.5 × SECS_loss + 0.3 × mel_spectrogram_L1 + 0.2 × speaker_CE
```
- Small learning rate (1e-5) to avoid catastrophic forgetting
- Early stopping on validation SECS

### Checkpoint F — After Phase 5
**Expected:** SECS ~0.28–0.38, MOS-N ~3.7–4.1

---

## Phase 6 — Ablation Study & Paper Preparation

**Goal:** Generate the complete results table for publication.

### Task 6.1 — Full Ablation Table

Run EVERY combination and record all 7 metrics:

| ID | Model Configuration | SECS | Recall@3 | Gender Acc | MOS-N | FAD | WER |
|----|---------------------|------|----------|------------|-------|-----|-----|
| A | CVAE (prev approach) | ~0.04 | | | | | |
| B | Direct Regressor (morphology) | | | | | | |
| C | + Speaker-avg targets | | | | | | |
| D | + Gender aux loss | | | | | | |
| E | + ArcFace features | | | | | | |
| F | + InfoNCE contrastive | | | | | | |
| G | + Hard negatives | | | | | | |
| H | + Retrieval (K=5) | | | | | | |
| I | + Hybrid blend (optimal α) | | | | | | |
| J | + E2E fine-tuning (if done) | | | | | | |

### Task 6.2 — Human Evaluation (2–3 days for data collection)
- **MOS-N:** 20 raters, 50 clips, naturalness 1–5 scale
- **Face-Voice Match:** "Does this voice fit this face?" — binary forced choice, 200 pairs
- **MUSHRA:** Quality comparison against: random baseline, gender-matched baseline, our best model

### Task 6.3 — Comparison Against Prior Work
Include in paper:
- Speech2Face (Oh et al., CVPR 2019) — reported numbers
- Nagrani et al. cross-modal matching (CVPR 2018) — reported numbers
- Most recent INTERSPEECH 2024 face-voice baseline

### Task 6.4 — Qualitative Analysis
- t-SNE of learned face-voice embedding space (colored by speaker)
- Attention maps: which facial regions most predict voice?
- Failure case analysis: which speakers are hardest to predict and why?

---

## Timeline

| Phase | Description | Est. Time |
|-------|-------------|-----------|
| Phase 1 | Data quality (speaker-avg, gender, filtering) | 4 hrs |
| Phase 2 | ArcFace features + retraining | 6 hrs |
| Phase 3 | InfoNCE contrastive + hard negatives | 8 hrs |
| Phase 4 | FAISS retrieval + hybrid inference | 4 hrs |
| Phase 5 | E2E fine-tuning (optional) | 1–2 days |
| Phase 6 | Full ablation + human eval + writing | 2–3 days |
| **Total (without Phase 5)** | | **~3 days** |
| **Total (with Phase 5)** | | **~5 days** |

---

## Realistic Final Numbers

```
Conservative (Phases 1–4 only):
  SECS:          0.20–0.26
  Recall@3:      12–20%
  Gender Acc:    88–93%
  MOS-N:         3.4–3.8
  WER:           5–8%
  Target venue:  INTERSPEECH / ICASSP ✅

Optimistic (Phases 1–5):
  SECS:          0.28–0.35
  Recall@3:      18–28%
  Gender Acc:    90–95%
  MOS-N:         3.6–4.1
  WER:           4–7%
  Target venue:  INTERSPEECH oral / ACL Findings / ECCV ✅
```

---

## Implementation Order (Start Here)

```
1. Compute speaker-averaged ECAPA targets (preprocessing, 20 min)
2. Retrain regressor with speaker-avg targets + gender loss → Checkpoint B
3. Extract ArcFace embeddings for all clips (2 hrs)
4. Retrain regressor with ArcFace → Checkpoint C
5. Implement InfoNCE loss → Checkpoint D
6. Build FAISS gallery → Checkpoint E (primary publishable result)
7. Full ablation table → Checkpoint F
8. Human evaluation → paper submission
```

> **Rule:** Every phase gets its own benchmark run logged to `results/benchmarks/`.
> Never modify a checkpoint after it has been benchmarked.
> All test-set numbers are computed ONCE at the end — val set is used for all decisions.
