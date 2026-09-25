# mvp.md
> The minimum working VOIX system — what is in scope for v1, what is deferred, and how to know it worked.

The entire master document scope is the MVP. This document defines execution boundaries and validation gates.

---

## MVP Scope

The MVP is a complete, single-paper-worthy research system with:

1. **Probabilistic CVAE mapper** trained on VoxCeleb2 — face embedding -> distribution over speaker embeddings
2. **India-specific F2S benchmark** — curated, quality-filtered, released as embeddings
3. **Ablation study** — feature types and architecture variants
4. **Evaluation** — objective (SECS, Recall@K, DS, ECE, FSD) + subjective (MOS, A/B, diversity)
5. **Cross-demographic evaluation** — zero-shot + fine-tuned on Indian benchmark

All 6 objectives (O1-O6 in context.md) are in scope.

---

## Phased Delivery

### Phase 1 — Environment + Data (Weeks 1-4)

**Deliverables:**
- Python environment fully configured (torch, speechbrain, insightface, mediapipe)
- VoxCeleb2 embedding extraction pipeline working: ArcFace 512-D + ECAPA 192-D + FaceMesh 32-D
- 100K training pairs (e_face_560D, e_speaker_192D) extracted and stored on Google Drive
- Frame selection strategy implemented: frontal pose filter via MediaPipe pose angle
- Indian dataset pilot: 5 speakers per language (25 total), SyncNet threshold calibrated
- Indian dataset full: 40-60 speakers, quality-filtered, metadata.csv + speakers.json built
- All Indian embeddings (ArcFace + ECAPA + FaceMesh) extracted

**Validation gates:**
- PCA/UMAP visualization shows speaker clusters in embedding space
- SyncNet calibration plot documented per source type
- metadata.csv passes schema validation
- Whisper + IndicLID correctly flags code-switching clips on 10% manual spot-check

---

### Phase 2 — Model Development (Weeks 5-8)

**Deliverables:**
- CVAE encoder (q_phi), decoder (p_theta), reparameterization implemented
- KL loss with free bits (lambda_fb=0.5 nats) + beta annealing schedule
- StyleTTS2 inference pipeline verified locally / on Colab
- 3-layer MLP style projection adapter (192-D ECAPA -> StyleTTS2 style dim)
- End-to-end inference: face image -> CVAE -> adapter -> StyleTTS2 -> speech waveform
- Linear feature fusion layer (ArcFace 512 + FaceMesh 32 + demographics 16 -> 560-D)
- Soft demographic estimator (age/sex probability heads)
- Training loop with W&B logging, model checkpointing, Recall@K evaluation function

**Validation gates:**
- Unit test: KL = 0 during warmup (beta=0)
- Unit test: gradient flows through reparameterization trick
- Smoke test: 5-epoch run on 10K pairs, loss curves are reasonable (L_recon decreasing, KL non-zero after warmup)
- Smoke test: end-to-end generates intelligible speech from known VoxCeleb speaker

---

### Phase 3 — Training + Evaluation (Weeks 9-12)

**Deliverables:**
- All 6 baseline models trained (B1-B6)
- Hyperparameter grid complete: beta_max x lambda_adapt x learning rate
- Best checkpoint selected by val SECS + low ECE
- Full model trained (ArcFace + Morph + Demog, full CVAE)
- Feature ablation experiments A1-A6 complete
- Architecture ablation experiments B1-B4 complete
- SHAP feature attribution computed on best model
- Vocal trait correlation analysis (F0, F1/F2, spectral centroid)
- Zero-shot evaluation on Indian benchmark (SECS, Recall@5)
- Fine-tuned model (5-10 epochs on Indian data) evaluated on Indian benchmark + VoxCeleb1

**Validation gates:**
- Full model SECS > B3 (deterministic baseline) on VoxCeleb1 test set
- Diversity Score > 0 (non-trivial) and < 1 (not random noise) on same-face K=3 samples
- H3: >=10% SECS drop confirmed zero-shot on Indian benchmark vs. VoxCeleb1
- No catastrophic forgetting: VoxCeleb1 SECS after fine-tuning within 5% of pre-fine-tuning

---

### Phase 4 — MOS Study + Writing (Weeks 13-16)

**Deliverables:**
- 50 clips selected for MOS evaluation across conditions
- MOS study launched on Prolific (20-25 evaluators)
- A/B preference test vs. deterministic baseline B3
- Diversity preference test (3 voices same face — natural vs. random?)
- All metrics compiled: SECS, Recall@K, DS, ECE, FSD, MOS, Krippendorff's alpha
- Statistical significance: paired t-test / Wilcoxon on key comparisons
- Paper written: Intro, Related Work, Methodology, Experiments, Results, Discussion, Conclusion
- Architecture figure, results tables, qualitative audio samples
- NPC voice demo video (supplementary only)
- Final submission to INTERSPEECH

---

## Explicitly Deferred (Not MVP)

| Item | Status |
|---|---|
| Expression/emotion branch for prosody-aware synthesis | Future work |
| Normalizing Flow mapper | Architecture alternative — tested in ablation only if CVAE fails |
| Real-time inference pipeline (<100ms latency) | Future work |
| Anime/artistic portrait inputs | Future work |
| Cross-lingual Indian language generation | Future work |
| Full VoxCeleb2 (all 6,112 speakers) training | Scale-up if compute allows; not required for paper |

---

## Fallback Positions

| Risk | Fallback |
|---|---|
| Novelty claim weakened by literature survey | Pivot to Framing 2: benchmark-centric (NeurIPS Datasets track) |
| CVAE posterior collapse persists | Reduce beta_max to 0.1; add KL monitoring alarm at KL < 0.01 nats |
| H2 rejected (morphology adds no value) | Publish as negative result — scientifically valid contribution |
| StyleTTS2 adapter fails to align spaces | Fall back to retrieval-based speaker selection (Vclip-style) for synthesis |
| Colab T4 quota exhausted | Save checkpoints every epoch; use Colab Pro ($10/month) |
| Indian dataset collection exceeds 2 weeks | Reduce to 10 hours / 30 speakers — still publication-worthy |
| RTX 4050 6GB VRAM insufficient | Reduce batch size to 32; use gradient accumulation |

---

## Reproducibility Requirements (All MVP)

- All random seeds fixed: torch.manual_seed(42), numpy.random.seed(42), CUDA determinism
- All hyperparameters logged to W&B run configs
- All results reported as mean +/- std over 3 seeds
- Model checkpoints hosted on HuggingFace Hub
- Indian benchmark released as embedding-only package (ArcFace + ECAPA + metadata) via HuggingFace Datasets
- VoxCeleb2 embeddings NOT released — release extraction scripts instead

---

*Read next: tasks.md*
