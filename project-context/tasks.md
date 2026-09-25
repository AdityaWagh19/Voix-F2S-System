# tasks.md
> Implementation checklist — updated as work progresses. Add dates and notes inline.

Status key: [ ] todo | [~] in-progress | [x] done | [!] blocked

---

## Phase 1 — Environment + Data (Weeks 1-4)

### Week 1: Environment + VoxCeleb2 Pilot
- [ ] Set up Python environment (torch, speechbrain, insightface, mediapipe, librosa, ffmpeg)
- [ ] Download VoxCeleb2 subset (1,000 speaker sample)
- [ ] ArcFace embedding extraction — verify 512-D output quality on sample
- [ ] ECAPA-TDNN embedding extraction via SpeechBrain — verify 192-D output
- [ ] MediaPipe FaceMesh extraction — verify 468 landmarks + 32-D ratio computation
- [ ] Implement frontal frame selection: head-pose filter via MediaPipe pose angles
- [ ] PCA/UMAP visualization of ArcFace + ECAPA embeddings — confirm speaker clusters

### Week 2: Full VoxCeleb2 Extraction
- [ ] Batch ArcFace extraction pipeline (100K clips)
- [ ] Batch ECAPA-TDNN extraction pipeline (100K clips)
- [ ] Batch MediaPipe FaceMesh ratio extraction (100K frames)
- [ ] Soft demographic estimator: age/sex probability heads on ArcFace features
- [ ] Build training pairs dataset (e_face_560D, e_speaker_192D)
- [ ] Store all embeddings to Google Drive
- [ ] Validate pair alignment: confirm same-speaker face and audio clips match correctly

### Week 3: Indian Dataset Pilot
- [ ] Identify 5 speakers per language (25 total) from NPTEL + AI4Bharat sources
- [ ] Set up yt-dlp streaming pipeline (no full downloads)
- [ ] Run streaming pipeline on 2 NPTEL speakers as proof-of-concept
- [ ] Calibrate SyncNet threshold on Indian content (pilot batch of 100 clips)
- [ ] Validate Whisper large-v3 transcription on Indian English
- [ ] Validate IndicLID code-switching detection on sample clips
- [ ] Document SyncNet threshold per source type

### Week 4: Indian Dataset Full Collection
- [ ] Process all 40-60 target speakers across 5 languages
- [ ] Apply quality filters (SyncNet, SNR >20dB, duration 3-10s, no background music)
- [ ] Manual spot-check of 10% of code-switching clips
- [ ] Build metadata.csv and speakers.json
- [ ] Compute ArcFace + ECAPA + FaceMesh embeddings for all Indian clips
- [ ] Verify metadata.csv schema completeness
- [ ] Store Indian dataset to Google Drive

---

## Phase 2 — Model Development (Weeks 5-8)

### Week 5: CVAE Mapper Implementation
- [ ] Implement encoder q_phi(z|e_f): 4-layer MLP with mu + log_sigma^2 heads
- [ ] Implement decoder p_theta(e_s_hat|z): 2-layer MLP reconstruction head
- [ ] Implement reparameterization trick
- [ ] Implement KL loss with free bits (lambda_fb=0.5 nats)
- [ ] Implement beta annealing schedule (linear warmup, 30% of steps)
- [ ] Unit test: KL = 0 during warmup (beta=0)
- [ ] Unit test: gradient flows through reparameterization trick

### Week 6: StyleTTS2 Integration + Adapter
- [ ] Install StyleTTS2 and verify inference pipeline locally or on Colab
- [ ] Identify StyleTTS2 style embedding dimension from codebase
- [ ] Implement 3-layer MLP projection adapter (192-D ECAPA -> StyleTTS2 style dim)
- [ ] Implement end-to-end inference: face -> CVAE -> adapter -> StyleTTS2 -> speech
- [ ] Smoke test: generate speech from ArcFace embedding of known VoxCeleb speaker
- [ ] Verify output is intelligible and identity-consistent

### Week 7: Feature Fusion + Soft Demographics
- [ ] Implement linear fusion layer (ArcFace 512 + FaceMesh 32 + demographics 16 -> 560)
- [ ] Implement soft demographic estimator (age + sex probability heads)
- [ ] Implement learned gating for demographic gradient down-weighting
- [ ] Integrate full pipeline: feature extraction -> fusion -> CVAE -> adapter -> StyleTTS2

### Week 8: Training Infrastructure
- [ ] Implement training loop with W&B logging (loss curves, KL per batch, SECS on val)
- [ ] Implement model checkpointing (save best val SECS)
- [ ] Implement Recall@K evaluation function
- [ ] Implement Diversity Score computation
- [ ] Implement ECE calibration evaluation
- [ ] Run 5-epoch smoke test on 10K pairs — verify loss behavior (L_recon decreasing, KL non-zero post-warmup)

---

## Phase 3 — Training + Evaluation (Weeks 9-12)

### Week 9: Baseline Training
- [ ] Train B1: Random speaker baseline (measure SECS lower bound)
- [ ] Train B2: Nearest-neighbor face retrieval baseline
- [ ] Train B3: Deterministic mapper (beta=0, sigma ignored at inference)
- [ ] Train A1: ArcFace-only CVAE (feature ablation baseline)
- [ ] Record all baseline SECS / Recall@5 / DS scores

### Week 10: Full Model Training + Hyperparameter Search
- [ ] Run hyperparameter grid: beta_max {0.1, 0.5, 1.0} x lambda_adapt {0.1, 0.5, 1.0} x LR {1e-4, 5e-4}
- [ ] Select best checkpoint by val SECS + low ECE
- [ ] Train full model (ArcFace + Morph + Demog, full CVAE) with best config
- [ ] Compute SECS, Recall@K, DS, ECE on VoxCeleb1 test set

### Week 11: Ablation Studies
- [ ] Feature ablation A2: FaceMesh 32-D only
- [ ] Feature ablation A3: Demographics 16-D only
- [ ] Feature ablation A4: ArcFace + FaceMesh 544-D
- [ ] Feature ablation A5: ArcFace + Demographics 528-D
- [ ] Feature ablation A6: Full 560-D (same as full model — cross-check)
- [ ] Architecture ablation B2: KL without free bits
- [ ] Architecture ablation B3: KL without annealing (beta_max from step 0)
- [ ] SHAP feature attribution analysis on best model
- [ ] Vocal trait correlation analysis: F0, F1/F2 formants, spectral centroid, HNR, speaking rate

### Week 12: Indian Benchmark Evaluation
- [ ] Zero-shot evaluation on India benchmark: SECS, Recall@5
- [ ] Fine-tune full model (5-10 epochs) on Indian training split
- [ ] Evaluate fine-tuned model on India benchmark
- [ ] Evaluate fine-tuned model on VoxCeleb1 (catastrophic forgetting check)
- [ ] Compute FSD on all evaluation conditions
- [ ] Compile all results into comparison table

---

## Phase 4 — MOS Study + Writing (Weeks 13-16)

### Week 13: Subjective Evaluation Setup
- [ ] Select 50 clips for MOS: sample across conditions (full model, B3 baseline, Indian, Western)
- [ ] Design MOS rating interface (web form or Prolific)
- [ ] Design A/B preference test (Voix vs. B3, same face + text)
- [ ] Design diversity preference test (3 voices from same face — natural vs. random?)
- [ ] Recruit 20-25 evaluators on Prolific (Indian language filter for Indian benchmark clips)
- [ ] Launch MOS + A/B + diversity preference studies
- [ ] Begin writing: Introduction, Related Work

### Week 14: Results Analysis + Writing
- [ ] Collect MOS results, compute Krippendorff's alpha
- [ ] Statistical significance: paired t-test or Wilcoxon on SECS, Recall@5, MOS
- [ ] Compile all objective + subjective metrics into final results tables
- [ ] Write Methodology section
- [ ] Write Experiments + Results sections

### Week 15: Paper Completion
- [ ] Write Discussion + Ablation analysis narrative
- [ ] Write Conclusion + Future Work
- [ ] Create system architecture figure (draw.io or LaTeX/TikZ)
- [ ] Create results tables and plots (matplotlib)
- [ ] Prepare qualitative audio sample demonstrations
- [ ] Prepare NPC voice demo video (supplementary only)

### Week 16: Revision + Submission
- [ ] Internal revision: verify all claims are supported by reported results
- [ ] Update Related Work if needed (post-survey novelty claim revisions)
- [ ] Format to INTERSPEECH template
- [ ] Prepare supplementary materials, code release, HuggingFace dataset upload
- [ ] Final proofread
- [ ] Submit

---

## Ongoing / Cross-Phase

- [ ] W&B experiment runs saved with full config export (JSON)
- [ ] All random seeds documented: torch.manual_seed(42), numpy.random.seed(42)
- [ ] 3-seed training runs for all reported metrics
- [ ] Model checkpoints pushed to HuggingFace Hub
- [ ] Indian benchmark released as embedding-only package on HuggingFace Datasets
- [ ] VoxCeleb2 extraction scripts released (not embeddings)

---

*Last updated: [add date when you start]*
*Current phase: Phase 1*
*Current focus: [update as you work]*
