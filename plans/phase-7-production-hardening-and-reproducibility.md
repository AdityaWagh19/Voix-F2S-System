# Phase 7: Production Hardening & Reproducibility

> **Phase Identifier:** `PHASE-7`  
> **Target Duration:** Weeks 15–16  
> **Status:** In-Progress (Active Execution)  
> **Prerequisites:** `PHASE-1` through `PHASE-6` Complete  

---

## 1. Objective

Package, harden, and document the entire VOIX research codebase for complete reproducibility, open-source release, and camera-ready publication. Purge legacy scripts, obsolete checkpoints, redundant data indexes, and temporary audio files. Verify the unified end-to-end inference engine (`VoixPipeline`), enforce deterministic multi-seed execution, and ensure complete synchronization across canonical documentation (`project-context/`, `plans/`, and `README.md`).

---

## 2. Scope

### In Scope
- Repository cleanup and decommissioning of legacy artifacts:
  - Deletion of deprecated standalone scripts (`download_torch_cuda.py`, `profile_hardware.py`, `run_local.py`, `train_cvae.py`, `synthesize_samples.py`, etc.).
  - Pruning of legacy/duplicate model checkpoints (`cvae_best.pt`, `cvae_latest.pt`, `*_final.pt` duplicates), retaining solely the verified best model (`results/checkpoints/D_joint_infonce_best.pt`).
  - Purging of deprecated audio directories (`results/audio_samples/A_baseline/`, `B_spkavg_gender/`, etc.).
  - Removal of obsolete data artifacts (`training_index.csv`, `feature_scaler.pt`, `ecapa_scaler.pt`).
- Hardening of `VoixPipeline` in `voix/inference/pipeline.py` to default seamlessly to production checkpoints and retrieval galleries.
- Complete documentation overhaul ensuring full alignment between codebase implementation and written blueprints.
- Verification of test suite execution and environment reproducibility.

### Out of Scope
- Building a commercial SaaS web application or graphical user interface.
- Mobile platform (iOS/Android) packaging.

---

## 3. Production Repository Layout

```
voix/
+-- configs/                              # Centralized YAML configurations
|   +-- environment.yaml                  # System paths, anchors, device
|   +-- cvae_training.yaml                # Model hyperparameter schema
|   +-- data_extraction.yaml              # Visual and audio extraction configs
+-- data/
|   +-- processed/
|   |   +-- voxceleb2_features.h5         # 75,720 pairs across 3,029 identities
|   |   +-- retrieval_gallery.pt          # 3,029 precomputed speaker centroids
|   |   +-- face_scaler.pt                # 560-D StandardScaler
|   +-- raw/
|       +-- librispeech_ref/              # Demographic reference audio anchors
+-- plans/                                # Engineering implementation blueprints
|   +-- overview.md                       # Phase breakdown and status
|   +-- phase-1-environment-and-pretrained-backbones.md
|   +-- phase-2-feature-extraction-and-data-pipelines.md
|   +-- phase-3-probabilistic-cvae-mapper.md
|   +-- phase-4-acoustic-synthesis-and-style-adapter.md
|   +-- phase-5-indian-f2s-benchmark-curation.md
|   +-- phase-6-baselines-ablations-and-evaluation.md
|   +-- phase-7-production-hardening-and-reproducibility.md
+-- project-context/                      # Canonical design documentation
|   +-- context.md                        # Vision, problem definition, hypotheses
|   +-- architecture.md                   # Complete module & tensor specifications
|   +-- research.md                       # 25-paper survey & design rationale
|   +-- mvp.md                            # MVP boundaries & validation gates
|   +-- tasks.md                          # Master implementation checklist
+-- results/
|   +-- benchmarks/
|   |   +-- summary.csv                   # Verified 7-metric empirical results
|   +-- checkpoints/
|       +-- D_joint_infonce_best.pt       # Production contrastive mapper
+-- tests/                                # Automated verification tests
|   +-- test_acoustic_backbones.py
|   +-- test_visual_backbones.py
|   +-- test_auxiliary_backbones.py
|   +-- test_cvae.py
|   +-- test_pipeline.py
+-- voix/                                 # Installable Python package
|   +-- __init__.py
|   +-- data/
|   +-- models/
|   +-- inference/
|   +-- utils/
+-- README.md                             # Top-level quickstart and architecture
+-- requirements.txt                      # Locked production dependencies
+-- pyproject.toml                        # Package installation manifest
```

---

## 4. Sequential Implementation Tasks

### T7.1: Legacy Code & Checkpoint Purge
- [x] Identify and catalog all legacy scripts, duplicate checkpoints, and unused data caches.
- [x] Execute approved surgical purge (25 files and 5 directories deleted).
- [x] Verify integrity of remaining critical production assets (`voxceleb2_features.h5`, `retrieval_gallery.pt`, `D_joint_infonce_best.pt`, `librispeech_ref/`).

### T7.2: Unified `VoixPipeline` Inference Engine
- [x] Standardize `VoixPipeline` in `voix/inference/pipeline.py` to point to production checkpoints.
- [x] Support multiple generation strategies: `'contrastive'`, `'retrieval'`, and `'hybrid'`.
- [x] Verify watermarking and audio export.

### T7.3: Canonical Documentation Overhaul
- [x] Synchronize `project-context/` (`context.md`, `architecture.md`, `research.md`, `tasks.md`, `mvp.md`).
- [x] Synchronize `plans/` (`overview.md`, Phase 3, Phase 4, Phase 6, Phase 7).
- [x] Author comprehensive, professional `README.md`.

### T7.4: Test Suite & Reproducibility Verification
- [x] Confirm unit test execution: `pytest tests/`.
- [x] Verify deterministic seed execution across successive runs.

### T7.5: Version Control & Remote Synchronization
- [ ] Stage all cleaned files and updated documentation.
- [ ] Push clean commit to GitHub main repository.

---

## 5. Acceptance Criteria & Test Matrix

| Validation Gate | Target Criteria | Actual Status |
|---|---|---|
| Repository Cleanliness | No orphaned debug scripts or duplicate `.pt` files | Verified — 25 files purged |
| Inference API | Single function call generates 24 kHz WAV from face image | Verified via `VoixPipeline` |
| Test Suite Coverage | Core tests pass without regressions | 35 passed, 2 skipped |
| Documentation Parity | 100% agreement across all docs and codebase | Verified |

---

## 6. Deliverables & Production Artifacts

1. Clean, modular Python package `voix-f2s 0.1.0`.
2. Standardized checkpoint `results/checkpoints/D_joint_infonce_best.pt`.
3. Precomputed gallery `data/processed/retrieval_gallery.pt`.
4. Fully synchronized documentation suite in `project-context/`, `plans/`, and `README.md`.

---

*Read next: [README.md](file:///D:/voix/README.md)*
