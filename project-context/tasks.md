# tasks.md
> Implementation checklist — updated as work progresses. Tracks deliverables, completion dates, and validation artifacts.

Status key: `[ ]` todo | `[~]` in-progress | `[x]` done | `[!]` blocked

---

## Phase 1 — Environment & Pretrained Backbones (COMPLETE - 2026-09-25)

### T1.1: Environment Isolation & Core Dependencies
- [x] Create Python virtual environment (Python 3.12.1, `D:\voix\.venv`)
- [x] Install PyTorch 2.5.1+cu121 with CUDA 12.1 support
- [x] Install all backbone dependencies (`mediapipe 1.0.1`, `speechbrain 1.1.1`, `insightface 2.0`, `transformers 5.17.0`, `librosa 1.0.0`, `soundfile 0.14.0`)
- [x] Generate locked `requirements.txt`
- [x] Verify: `torch.cuda.is_available()=True`, GPU detected, VRAM footprint calibrated

### T1.2: Repository Scaffolding & Configuration Schema
- [x] Directory structure: `voix/`, `tests/`, `configs/`, `checkpoints/`, `data/processed/`, `results/`
- [x] `configs/environment.yaml`: seed=42, device=cuda, backbone paths, demographic anchors
- [x] `configs/cvae_training.yaml`: optimizer, scheduler, loss hyperparameters
- [x] `configs/data_extraction.yaml`: VoxCeleb2 and FaceMesh extraction settings
- [x] `voix/utils/seed.py`: `set_global_seed()`, `get_generator()`
- [x] `voix/utils/config.py`: `load_config()` with environment variable overrides
- [x] Package setup: `pyproject.toml` (`voix-f2s 0.1.0`)

### T1.3: Visual Backbone Integration & Verification
- [x] `voix/data/face_extractor.py`: `ArcFaceExtractor` (512-D, lazy-load, frozen `buffalo_l`)
- [x] `voix/data/face_extractor.py`: `FaceMeshExtractor` (468 landmarks, MediaPipe Tasks API)
- [x] `voix/data/morphology.py`: `compute_craniofacial_ratios()` (32-D canonical anatomical ratios)
- [x] `voix/data/demographics.py`: `SoftDemographicEstimator` (16-D continuous age/sex distributions)
- [x] `tests/test_visual_backbones.py`: passed

### T1.4: Acoustic Backbone Integration & Verification
- [x] `voix/data/speaker_extractor.py`: `ECAPAExtractor` (192-D, SpeechBrain, frozen)
- [x] `tests/test_acoustic_backbones.py`: passed
- [x] Determinism verified: 5 successive identical runs confirmed

### T1.5: Auxiliary Backbone Integration & Verification
- [x] `voix/models/watermarking.py`: `AudioSealWatermarker` (16-bit, lazy-load)
- [x] `tests/test_auxiliary_backbones.py`: passed
- [x] Config loader test suite passed

### T1.6: VRAM Footprint & CUDA Memory Profiling
- [x] VRAM profiling script executed across all backbones
- [x] Confirmed model fits within 4.0 GB VRAM constraint on consumer hardware

---

## Phase 2 — Multimodal Feature Extraction & Data Pipelines (COMPLETE - 2026-09-25)

### T2.1: Head-Pose Estimation & Frontal Frame Selection
- [x] Pitch, yaw, roll estimation via solvePnP on FaceMesh landmarks
- [x] Rejection filters for non-frontal frames ($|\text{pitch}| > 15^\circ, |\text{yaw}| > 15^\circ$)

### T2.2: VoxCeleb2 Curation Index
- [x] Curated multimodal index verifying paired audio-visual integrity across 3,029 identities

### T2.3: Batch Extraction Pipeline
- [x] High-throughput batch feature extractor extracting 560-D face features and 192-D ECAPA targets

### T2.4: FaceFusionLayer (560-D)
- [x] `voix/models/face_fusion.py`: Linear projection + LayerNorm + GELU
- [x] Shape tests verified: $[B, 560] \to [B, 560]$

### T2.5: HDF5 Feature Store
- [x] Indexed `data/processed/voxceleb2_features.h5` containing 75,720 utterance samples across 3,029 identities
- [x] Stored datasets: `face_features`, `speaker_embeddings`, `speaker_centroids`, `speaker_ids`, `gender_labels`

### T2.6: Dataset & DataLoader
- [x] `voix/data/dataset.py`: Memory-efficient chunked HDF5 reader
- [x] Strict identity-disjoint split: 2,629 training speakers (65,720 samples), 400 zero-shot validation speakers (10,000 samples)
- [x] Identity batch sampler for contrastive multi-positive sampling

### T2.7: Test Suite
- [x] All Phase 1 & 2 unit tests passing (35 passed, 2 skipped)

---

## Phase 3 — Probabilistic & Metric Contrastive Mapper (COMPLETE - 2026-10-09)

### T3.1: CVAE Architecture Implementation
- [x] Recognition encoder $q_\phi(z \mid \mathbf{e}_s, \mathbf{e}_f)$, conditional prior $p_\theta(z \mid \mathbf{e}_f)$, generative decoder $p_\psi(\mathbf{e}_s \mid z, \mathbf{e}_f)$ in `voix/models/cvae.py`
- [x] Free-bits KL divergence loss ($\lambda_{\text{fb}} = 0.5$ nats/dim) and linear $\beta(t)$ annealing

### T3.2: Acoustic Mean Collapse Discovery & Diagnosis
- [x] Identified that single-utterance targets with high intra-speaker variance cause CVAE decoder to predict the population mean
- [x] Synthesized voices from population mean exhibited muffled, robotic artifacts

### T3.3: Metric Contrastive Mapper Architecture
- [x] Designed `MetricContrastiveMapper` (`560 -> 512 -> 384 -> 256 -> 192`) with L2 normalization head onto $\mathbb{S}^{191}$
- [x] Added auxiliary gender classification head: `Linear(256, 1) -> Sigmoid`

### T3.4: Supervised InfoNCE Loss over Speaker Centroids
- [x] Implemented `SupervisedInfoNCELoss` with temperature $\tau = 0.07$
- [x] Cleaned training targets: replaced noisy single-utterance vectors with speaker centroids $\mathbf{c}_y$

### T3.5: Training Execution & Optimization
- [x] Trained 40 epochs on GPU using AdamW, cosine learning rate schedule, AMP FP16
- [x] Best checkpoint saved: `results/checkpoints/D_joint_infonce_best.pt`
- [x] Achieved validation cosine similarity of **0.1288** on 400 zero-shot held-out speakers

---

## Phase 4 — Acoustic Synthesis & Vocoder Style Bridge (COMPLETE - 2026-09-28)

### T4.1: Frozen StyleTTS2 Backend Integration
- [x] Integrated frozen StyleTTS2 synthesis backend in `voix/models/tts_backend.py`

### T4.2: VocoderStyleBridge Implementation
- [x] Implemented `VocoderStyleBridge` resolving the acoustic mismatch between ECAPA verification space and StyleTTS2 style latent space

### T4.3: Reference Demographic Anchor Curation
- [x] Integrated curated male and female acoustic anchors in `data/raw/librispeech_ref/`
- [x] Registered fallback and reference anchors in `configs/environment.yaml`

### T4.4: Dual-Tier Style Blending Parameter Tuning
- [x] Conducted perceptual parameter sweep across blending weights ($\alpha$, $\beta$, `blend_weight`)
- [x] Calibrated optimal parameters: $\alpha = 0.70$, $\beta = 0.30$, `blend_weight = 0.75`
- [x] Verified crisp, natural synthesized audio without robotic buzzing or distortion

### T4.5: AudioSeal Watermarking Integration
- [x] Integrated proactive imperceptible watermarking on all generated speech waveforms

---

## Phase 5 — Indian F2S Benchmark Curation (PENDING)

### T5.1: Benchmark Specification & Calibration
- [x] Target specification defined: ~20 hours, 40–60 speakers, 5 Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi)
- [x] Active speaker threshold calibrated (SyncNet confidence $> 4.5$)

### T5.2: Stream Curation & Quality Filtering
- [ ] Ingest CC-licensed educational and speech video streams
- [ ] Execute SNR filtering ($> 20$ dB) and multi-speaker segment removal

### T5.3: ASR & IndicLID Code-Switching Pipeline
- [ ] Generate transcripts via Whisper large-v3
- [ ] Tag code-switching segments (e.g., Hinglish, Tanglish) using IndicLID

### T5.4: Embedding Extraction & Benchmark Release
- [ ] Extract ArcFace 512-D, FaceMesh 32-D, and ECAPA 192-D embeddings
- [ ] Validate schema and publish privacy-preserving embeddings-only benchmark

---

## Phase 6 — Baselines, Ablations & Empirical Evaluation (COMPLETE - 2026-10-12)

### T6.1: Baseline Suite Implementation
- [x] Implemented 6 distinct architectural and retrieval paradigms:
  - **Model A:** Baseline Cosine Regression
  - **Model B:** Speaker-Averaged + Gender Loss
  - **Model C:** Metric Contrastive (Utterances)
  - **Model D:** Joint InfoNCE Centroids (`results/checkpoints/D_joint_infonce_best.pt`)
  - **Model E:** Exemplar kNN Retrieval ($k=5$, `data/processed/retrieval_gallery.pt`)
  - **Model F:** Hybrid Retrieval-Contrastive ($k=10$)

### T6.2: 400 Zero-Shot Unseen Speaker Evaluation Pipeline
- [x] Built evaluation pipeline assessing models on 10,000 utterances across 400 strictly unseen identities

### T6.3: Objective Multi-Metric Computation
- [x] Computed 7 objective metrics: Val Cosine, Mean SECS, Peak SECS, Recall@1, Recall@5, Recall@10, Gender Accuracy, and Inference Latency

### T6.4: Summary Benchmark Table Generation
- [x] Stored empirical results in `results/benchmarks/summary.csv`
- [x] Confirmed Model F superiority: **0.1299** val cosine, **0.0324** mean SECS, **100.0%** gender accuracy

### T6.5: Audio Quality & Perceptual Audit
- [x] Audited synthesized speech across models: confirmed elimination of raspiness and acoustic collapse

---

## Phase 7 — Production Hardening & Reproducibility (IN-PROGRESS - Active)

### T7.1: Legacy Code & Checkpoint Purge
- [x] Purged 25 obsolete training scripts, deprecated checkpoints, duplicate `_final.pt` files, and temporary audio
- [x] Cleaned root repository and documentation directory

### T7.2: Unified `VoixPipeline` Inference Engine
- [x] Verified `VoixPipeline` in `voix/inference/pipeline.py` defaults to production checkpoints:
  - Mapper: `results/checkpoints/D_joint_infonce_best.pt`
  - Scaler: `data/processed/face_scaler.pt`
  - Gallery: `data/processed/retrieval_gallery.pt`

### T7.3: Complete Documentation Synchronization
- [x] Update `project-context/` (`context.md`, `architecture.md`, `research.md`, `tasks.md`, `mvp.md`)
- [x] Update `plans/` (`overview.md`, Phase 3, Phase 4, Phase 6, Phase 7)
- [x] Full `README.md` overhaul — evidence-based 13-section publication-ready documentation (2026-10-13)

### T7.5: Dynamic Hardware Auto-Configuration
- [x] Implemented `configs/hardware.yaml` — universal cross-platform hardware profiles (Windows, Linux, macOS)
- [x] Profiles: `tier_1_budget_gpu` (<=4.5GB CUDA), `tier_2_mid_gpu` (6–12GB), `tier_3_high_gpu` (>=16GB), `tier_apple_silicon` (MPS), `tier_cpu_only`
- [x] `voix/utils/hardware.py` — auto-detection engine querying OS, CUDA, MPS, ROCm, and XPU at runtime
- [x] Tier 2 `batch_voices: true` enabled (parallel synthesis, ~3.8GB peak VRAM); Tier 1 sequential to prevent OOM
- [x] `tests/test_hardware.py` — 11 assertions covering all hardware tiers (89 total tests now passing)

### T7.6: Whitebox End-to-End Inference Audit (`inde/` test case)
- [x] Full pipeline run on unknown test image (`D:/voix/inde.jpeg`) — 3 voices synthesized on CUDA
- [x] `inde/metadata.json` — 78.5 KB whitebox dossier with all 560 input features, 32 craniofacial ratios, gallery top-10 scores, synthesis audit, and WER=0.0% ASR verification
- [x] `inde/` and `inde.jpeg` added to `.gitignore` — local test assets excluded from VCS

### T7.7: Version Control & Remote Sync
- [ ] Stage all cleaned files and updated documentation
- [ ] Push clean commit to GitHub main repository

---

*Read next: [mvp.md](file:///D:/voix/project-context/mvp.md)*

