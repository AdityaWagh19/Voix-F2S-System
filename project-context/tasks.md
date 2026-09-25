# tasks.md
> Implementation checklist - updated as work progresses. Add dates and notes inline.

Status key: [ ] todo | [~] in-progress | [x] done | [!] blocked

---

## Phase 1 - Environment & Pretrained Backbones (COMPLETE - 2026-09-25)

### T1.1: Environment Isolation & Core Dependencies
- [x] Create Python virtual environment (Python 3.12.1, D:\voix\.venv)
- [x] Install PyTorch 2.5.1+cu121 with CUDA 12.1 support
- [x] Install all backbone dependencies (mediapipe 1.0.1, speechbrain 1.1.1, insightface 2.0, transformers 5.17.0, librosa 1.0.0, soundfile 0.14.0)
- [x] Generate requirements.txt with 109 locked packages
- [x] Verify: torch.cuda.is_available()=True, GPU=GTX 1650, VRAM=4.0 GB

### T1.2: Repository Scaffolding & Configuration Schema
- [x] Directory structure: voix/, tests/, configs/, scripts/, checkpoints/, data/processed/, artifacts/
- [x] configs/environment.yaml: seed=42, device=cuda, all backbone model paths
- [x] configs/cvae_training.yaml: optimizer, scheduler, loss hyperparameters
- [x] configs/data_extraction.yaml: VoxCeleb2 and FaceMesh extraction settings
- [x] voix/utils/seed.py: set_global_seed(), get_generator()
- [x] voix/utils/config.py: load_config() with VOIX_* environment variable overrides
- [x] All package __init__.py files created
- [x] pyproject.toml: installable package (voix-f2s 0.1.0) with setuptools.build_meta

### T1.3: Visual Backbone Integration & Verification
- [x] voix/data/face_extractor.py: ArcFaceExtractor (512-D, lazy-load, frozen)
- [x] voix/data/face_extractor.py: FaceMeshExtractor (468 landmarks, mediapipe tasks API)
- [x] voix/data/morphology.py: compute_craniofacial_ratios() 32-D geometric ratios
- [x] voix/data/demographics.py: SoftDemographicEstimator (16-D uniform prior)
- [x] tests/test_visual_backbones.py: 11 tests (9 passed, 2 skipped - model download gated)

### T1.4: Acoustic Backbone Integration & Verification
- [x] voix/data/speaker_extractor.py: ECAPAExtractor (192-D, SpeechBrain, frozen)
- [x] tests/test_acoustic_backbones.py: 12 tests all passed
- [x] Seed determinism: 5 successive identical runs confirmed

### T1.5: Auxiliary Backbone Integration & Verification
- [x] voix/models/watermarking.py: AudioSealWatermarker (16-bit, lazy-load)
- [x] tests/test_auxiliary_backbones.py: 12 tests all passed
- [x] Config loader: test_load_environment_yaml, test_missing_config_raises all passed

### T1.6: VRAM Footprint & CUDA Memory Profiling
- [x] scripts/profile_hardware.py: VRAM profiler per backbone implemented
- [x] CVAE projected memory: <4.0 GB at batch_size=32 on GTX 1650
- [x] Recommended training batch size for GTX 1650: 32

**Test results:** 35 passed, 2 skipped, 0 failed (3.08s)
**Git commit:** 2802626 - feat(phase-1): complete environment scaffold, frozen backbones, and test suite

**Deviations from plan:**
- Python 3.12 used (plan specified 3.10) - all packages compatible on 3.12
- mediapipe 1.0.1 uses tasks API, not solutions API - extractor updated accordingly
- CUDA 12.1 used (plan specified 11.8) - fully supported by GTX 1650 drivers

---

## Phase 2 - Multimodal Feature Extraction & Data Pipelines (PENDING)

### T2.1: VoxCeleb Data Acquisition & Storage Strategy
- [ ] Source data: Official VGG links deprecated (GDPR); acquire via Academic Torrents / Kaggle mirror / Google Drive
- [ ] Alternative for local prototyping: VoxCeleb1 subset (1,251 speakers) or Kaggle 50-speaker slice (~2GB)
- [ ] Cloud extraction plan: Run batch extraction on Colab/Kaggle GPU, save final ~280MB HDF5 to Google Drive
- [ ] Validate 2,000 speaker target: (N, 560) face and (N, 192) speaker embeddings

### T2.2: Face Feature Extraction Pipeline
- [ ] Implement batch ArcFace extraction pipeline with frontal frame selection
- [ ] Head-pose filter: yaw < 10 deg, pitch < 10 deg via MediaPipe
- [ ] Compute 32-D craniofacial ratios per speaker (aggregate over frames)
- [ ] Uniform prior demographic estimation per speaker
- [ ] Concatenate: 512 + 32 + 16 = 560-D face representation

### T2.3: Speaker Feature Extraction Pipeline
- [ ] Implement batch ECAPA-TDNN extraction on 16 kHz audio
- [ ] Per-utterance 192-D speaker embedding
- [ ] Speaker-level mean embedding for training pair construction

### T2.4: HDF5 Feature Store Construction
- [ ] Build voxceleb2_train.h5: (N, 560) face embeddings + (N, 192) speaker embeddings
- [ ] Build voxceleb2_val.h5: held-out 200 speaker validation split
- [ ] Validate pair alignment (same speaker face-audio correspondence)
- [ ] Store metadata: speaker_id, video_id, language, gender_prior

### T2.5: Dataset & DataLoader Implementation
- [ ] Implement voix/data/voxceleb2_dataset.py: HDF5Dataset class
- [ ] Implement voix/data/dataloader.py: seeded DataLoader factory
- [ ] Write tests/test_data_pipeline.py

### T2.6: Extraction Validation
- [ ] PCA/UMAP visualization: confirm speaker clusters in embedding space
- [ ] Cosine similarity sanity check: same-speaker > different-speaker
- [ ] Write extraction_report.md with statistics

---

## Phase 3 - Probabilistic CVAE Mapper (PENDING)

### Week 5: CVAE Implementation
- [ ] Implement encoder q_phi(z|e_f): 4-layer MLP with mu + log_sigma^2 heads
- [ ] Implement decoder p_theta(e_s_hat|z): 2-layer MLP reconstruction head
- [ ] Implement reparameterization trick
- [ ] Implement KL loss with free bits (lambda_fb=0.5 nats)
- [ ] Implement beta annealing schedule (linear warmup, 30% of steps)
- [ ] Unit test: KL=0 during warmup, gradient through reparameterization

### Week 6: StyleTTS2 Integration + Adapter
- [ ] Install StyleTTS2 and verify inference pipeline locally or on Colab
- [ ] Implement style adapter: linear projection from 192-D -> StyleTTS2 style dim
- [ ] Test end-to-end: face -> z -> e_s_hat -> StyleTTS2 waveform

---

## Phase 4 - Acoustic Synthesis & Style Adapter (PENDING)
- [ ] Full end-to-end waveform generation pipeline
- [ ] AudioSeal watermark embedding on every output
- [ ] Perceptual quality validation (MOS proxy)

---

## Phase 5 - Indian F2S Benchmark Curation (PENDING)
- [ ] yt-dlp streaming pipeline for NPTEL/AI4Bharat sources
- [ ] SyncNet lip-sync verification
- [ ] IndicLID code-switching detection
- [ ] Build metadata.csv and speakers.json

---

## Phase 6 - Baselines, Ablations & Evaluation (PENDING)
- [ ] B1-B6 baseline implementations
- [ ] A1-A6 architecture ablations
- [ ] B1-B4 training ablations
- [ ] MOS study protocol
- [ ] Statistical significance tests

---

## Phase 7 - Production Hardening & Reproducibility (PENDING)
- [ ] 3-seed reproducibility audit
- [ ] HuggingFace Hub publication
- [ ] Docker/Conda environment packaging
- [ ] Inference CLI (voix-infer)
- [ ] Final paper-ready results tables