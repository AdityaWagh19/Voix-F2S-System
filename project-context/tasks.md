# tasks.md
> Current implementation task log and sprint backlog.
> Last updated: 2026-10-16. Ground truth is always the source code in `voix/`.

---

## Overall Project Status

**Current Phase: PRODUCTION — Evaluation & Documentation Finalization**

All core technical deliverables are complete. The Indian F2S Benchmark curation is formally designated as Future Scope. Current work focuses on documentation accuracy, code hygiene, and final artifact packaging.

---

## Completed Milestones

### Phase 1 — Environment & Data Infrastructure ✅ COMPLETE

| Task | Status | Notes |
|---|---|---|
| Python 3.12 virtual environment with CUDA 12.1 | ✅ | `.venv/`, `requirements.txt` |
| ArcFace Buffalo_L lazy-loading extractor | ✅ | `voix/data/face_extractor.py` |
| MediaPipe FaceMesh 1.0 tasks-API adapter | ✅ | `voix/data/face_extractor.py` |
| 32-D craniofacial ratio computation (IOC-normalized) | ✅ | `voix/data/morphology.py` |
| 16-D soft demographic estimator | ✅ | `voix/data/demographics.py` |
| FaceFusionLayer (560-D multimodal embedding) | ✅ | `voix/models/fusion.py` |
| ECAPA-TDNN speaker extractor (SpeechBrain) | ✅ | `voix/data/speaker_extractor.py` |
| HDF5 feature store pipeline (VoxCeleb2) | ✅ | `voix/data/dataset.py`; 75,720 samples, 3,029 identities |
| Training/validation/test identity split | ✅ | 2,629 train / 400 zero-shot val (disjoint) |
| Canonical seed-42 determinism | ✅ | `voix/utils/seed.py` |

### Phase 2 — Mapper Architecture & Acoustic Bridge ✅ COMPLETE

| Task | Status | Notes |
|---|---|---|
| CVAE v2 (Recognition net + Prior net + Decoder) | ✅ | `voix/models/cvae.py` — GroupNorm, free-bits KL, β-annealing |
| Metric Contrastive Mapper (560→512→384→256→192-D) | ✅ | `voix/training/trainer_adapter.py` |
| Supervised InfoNCE loss (τ=0.07) over speaker centroids | ✅ | `voix/training/losses.py` |
| CosineReconstructionLoss + FreeBitsKLLoss + CVAELoss | ✅ | `voix/training/losses.py` |
| Precomputed retrieval gallery (3,029 speaker centroids) | ✅ | `data/processed/retrieval_gallery.pt` |
| Gender-conditioned kNN retrieval (k=5 and k=10) | ✅ | Pipeline & benchmark scripts |
| Hybrid blending (α=0.70) | ✅ | `voix/inference/pipeline.py` |
| StyleAdapter MLP (192→256→256→128-D) | ✅ | `voix/models/adapter.py` |
| StyleTTS 2 frozen integration (24 kHz) | ✅ | `voix/models/tts_backend.py` |
| LUFS loudness normalization (EBU R128) | ✅ | `_lufs_normalize()` in `tts_backend.py` |
| AudioSeal neural watermarking (16-bit, AUC≥0.97) | ✅ | `voix/models/watermarking.py` |

### Phase 3 — Empirical Evaluation & Ablations ✅ COMPLETE

| Task | Status | Notes |
|---|---|---|
| 6-model baseline and ablation benchmark | ✅ | Models A–F; `results/benchmarks/summary.csv` |
| Model A: Baseline cosine regression | ✅ | val_cos=0.1372, gender=0.626 |
| Model B: Speaker-averaged + gender loss | ✅ | val_cos=0.1370, gender=0.626 |
| Model D: Joint InfoNCE centroids | ✅ | val_cos=0.1288, gender=0.626; best trained model |
| Model E: Exemplar kNN retrieval (k=5) | ✅ | val_cos=0.0642, gender=1.000 |
| Model F: Hybrid retrieval-contrastive (k=10) | ✅ | val_cos=0.1299, gender=1.000 — Pareto-optimal |
| RTF / latency benchmark | ✅ | 11.2 ms GPU; RTF 0.257 on long-form text |
| CVAE temperature diversity sweep (T∈[0.1,1.5]) | ✅ | `artifacts/stress_test/benchmarks/` |

### Phase 4 — Production Hardening & Expressivity Engine ✅ COMPLETE

| Task | Status | Notes |
|---|---|---|
| `VoixPipeline` unified inference API | ✅ | `voix/inference/pipeline.py` |
| Hardware auto-configuration engine | ✅ | `voix/utils/hardware.py` — CUDA/MPS/CPU tiered profiles |
| Demographic preset rail (8 canonical presets) | ✅ | `voix/data/presets.py` |
| Craniofacial morphology sliders | ✅ | `voix/data/morphology.py` — apply_craniofacial_sliders() |
| Regional accent modulation (6 dialects) | ✅ | `voix/models/accent_control.py` |
| Emotion & affective tone modulation (6 states) | ✅ | `voix/models/emotion_control.py` |
| Clinical speech difference simulation (6 types) | ✅ | `voix/models/speech_simulation.py` |
| Cross-lingual synthesis (5 languages) | ✅ | `voix/models/cross_lingual.py` |
| YAML config loader with env var overrides | ✅ | `voix/utils/config.py` |
| Dynamic anchor path resolution by gender+age | ✅ | `select_anchor_key()` in `voix/utils/config.py` |

### Phase 5 — Exhaustive Empirical Validation ✅ COMPLETE

| Task | Status | Notes |
|---|---|---|
| 19-subject Option B demographic stress test (133 audios) | ✅ | Cohorts 1–3: gender/age/race/craniofacial outliers |
| 12-subject Clinical Speech Differences stress test (48 audios) | ✅ | Stuttering, lisp, rhotacism, hypernasality, prolongations |
| Merged unified audit JSON (31 subjects, 181 audios) | ✅ | `artifacts/stress_test/exhaustive_stress_test_audit.json` |
| Whisper ASR intelligibility verification | ✅ | 181/181 (100.0%) transcribed |
| Gender pitch separation empirical verification | ✅ | Male: 112.6 Hz, Female: 205.7 Hz |
| Master evaluation report | ✅ | `artifacts/stress_test/EXHAUSTIVE_STRESS_TEST_REPORT.md` |
| Legacy artifact cleanup | ✅ | `artifacts/samples/` and `artifacts/evaluation/` purged |

---

## Active / Open Tasks

| Task | Priority | Notes |
|---|---|---|
| Update all documentation to reflect completed state | **HIGH** | `project-context/`, `README.md` — in progress |
| Formal MOS (Mean Opinion Score) evaluation | MEDIUM | Requires human listener panel; not yet scheduled |
| FAD (Fréchet Audio Distance) evaluation | MEDIUM | Requires reference audio set configuration |
| WER (Word Error Rate) on synthesized speech | MEDIUM | Whisper transcription pipeline is in place |
| Ablation: +craniofacial ratios vs baseline ArcFace-only | MEDIUM | Requires comparative benchmark run |

---

## Future Scope (Designated Out-of-Scope for Current Release)

| Task | Future Priority | Target Milestone |
|---|---|---|
| **India F2S Benchmark curation** (~20 hrs, 40–60 speakers, 5 languages: Hindi, Tamil, Telugu, Bengali, Marathi) | HIGH | Post-publication follow-up |
| Out-of-domain zero-shot evaluation on Indian subjects | HIGH | Requires benchmark completion |
| Code-switching (Hinglish/Tanglish) model robustness analysis | MEDIUM | Requires benchmark + additional corpus |
| SyncNet-filtered speaker-face aligned Indian corpus download | MEDIUM | `scripts/download_voxceleb.py` extension |
| Full 3D FLAME morphable model integration | LOW | Research direction; not MVP-blocking |
| NeRF-based face representation input | LOW | Research direction |
| Real-time streaming synthesis (<50 ms TTFC) | LOW | Requires vocoder streaming mode |

---

## Test Suite Status

| Module | Test File | Status |
|---|---|---|
| Visual Backbones (ArcFace, FaceMesh) | `test_visual_backbones.py` | ✅ Passing |
| Acoustic Backbones (ECAPA-TDNN) | `test_acoustic_backbones.py` | ✅ Passing |
| Auxiliary Backbones | `test_auxiliary_backbones.py` | ✅ Passing |
| CVAE Model (prior net, recognition net, decoder) | `test_cvae.py` | ✅ Passing |
| StyleAdapter | `test_adapter.py` | ✅ Passing |
| TTS Backend | `test_tts_backend.py` | ✅ Passing |
| Data Pipeline (HDF5, splits) | `test_data_pipeline.py` | ✅ Passing |
| Hardware Detection | `test_hardware.py` | ✅ Passing |
| End-to-End Pipeline | `test_pipeline.py` | ✅ Passing |
| Craniofacial Sliders | `test_craniofacial_sliders.py` | ✅ Passing |
| Demographic Presets | `test_demographic_presets.py` | ✅ Passing |
| Speech Simulation | `test_speech_simulation.py` | ✅ Passing |
| Accent Control | `test_accent_control.py` | ✅ Passing |
| Emotion Control | `test_emotion_control.py` | ✅ Passing |
| Cross-Lingual | `test_cross_lingual.py` | ✅ Passing |

**Total: 35 passing unit tests.**
