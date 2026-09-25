# Phase 1: Environment & Pretrained Backbones

> **Phase Identifier:** `PHASE-1`  
> **Target Duration:** Week 1  
> **Status:** COMPLETE - 2026-09-25  
> **Commit:** 2802626  
> **Test Result:** 35 passed, 2 skipped, 0 failed  
> **Prerequisites:** None

---

## 1. Objective

Establish an isolated, fully reproducible development and execution environment across local hardware (NVIDIA RTX 4050 6GB Laptop GPU on Windows 11) and remote compute (Google Colab T4 15GB GPU). Verify that all seven frozen pretrained backbones (ArcFace, MediaPipe FaceMesh, SpeechBrain ECAPA-TDNN, StyleTTS 2, OpenAI Whisper large-v3, IndicLID, and AudioSeal) instantiate deterministically, execute forward passes without memory leaks, and generate fixed-dimension output representations that adhere to architectural contracts.

---

## 2. Scope

### In Scope
- Setup of Python virtual environment (`venv` or `conda` Python 3.10) with exact pinned dependencies.
- PyTorch 2.x installation with CUDA 11.8/12.1 support and cuDNN verification.
- Scaffolding of the core repository structure (`voix/`, `tests/`, `configs/`, `scripts/`).
- Construction of a centralized configuration loader (`configs/environment.yaml`).
- Pretrained model downloading and caching strategies (local disk + Google Drive mirroring).
- Development of unit smoke tests for all seven pretrained model backbones using synthetic inputs.
- Memory benchmarking of each backbone to establish baseline VRAM footprints under batch sizes 1, 8, and 32.

### Out of Scope
- Full VoxCeleb2 batch dataset processing (deferred to Phase 2).
- CVAE mapper network architecture implementation (deferred to Phase 3).
- YouTube stream scraping or data curation (deferred to Phase 5).

---

## 3. Design Decisions & Rationale

### 3.1 Python 3.10 Selection
- **Decision:** Target Python 3.10 strictly, avoiding Python 3.11+.
- **Rationale:** `insightface` (ArcFace) prebuilt wheels on Windows have established compilation stability on Python 3.10. `mediapipe` 0.10.x and `onnxruntime-gpu` maintain full compatibility with Python 3.10 without requiring Visual Studio C++ build tools.

### 3.2 Pretrained Model Freezing & Isolation
- **Decision:** Every upstream model backbone (ArcFace, ECAPA-TDNN, Whisper, FaceMesh, AudioSeal, StyleTTS 2) will be treated as an immutable feature extractor with all parameters set to `requires_grad = False` and evaluated in `eval()` mode.
- **Rationale:** VOIX does not perform end-to-end backpropagation through the face detector or vocoder. Freezing backbones prevents catastrophic gradient explosion, reduces VRAM consumption by 70%, and isolates scientific variables strictly to the generative mapper.

### 3.3 Offline Model Caching Architecture
- **Decision:** All downloaded checkpoints must reside in a standardized local cache directory `checkpoints/` managed via environment variable `VOIX_CHECKPOINT_DIR`.
- **Rationale:** Google Colab sessions are ephemeral. Centralizing checkpoint storage enables identical script execution locally and on Colab by mounting Google Drive directly to `checkpoints/`.

---

## 4. Sequential Implementation Tasks

```
[T1.1] Environment Isolation & Core Dependencies
       +-- Create Python 3.10 environment
       +-- Install PyTorch with CUDA support
       +-- Freeze requirements.txt

[T1.2] Repository Scaffolding & Configuration Schema
       +-- Generate directory tree (voix/, tests/, configs/, scripts/)
       +-- Implement voix/__init__.py and configs/environment.yaml
       +-- Setup deterministic seed utilities (torch, numpy, cuda)

[T1.3] Visual Backbone Integration & Verification
       +-- Implement InsightFace ArcFace ResNet-50 wrapper (512-D output)
       +-- Implement MediaPipe FaceMesh 468 landmark wrapper
       +-- Write test_visual_backbones.py verifying tensor dimensions

[T1.4] Acoustic Backbone Integration & Verification
       +-- Implement SpeechBrain ECAPA-TDNN wrapper (192-D output)
       +-- Implement StyleTTS 2 environment & checkpoint loader
       +-- Write test_acoustic_backbones.py verifying audio processing

[T1.5] Auxiliary Backbone Integration & Verification
       +-- Implement OpenAI Whisper large-v3 loader
       +-- Implement IndicLID classifier wrapper
       +-- Implement AudioSeal neural watermarking wrapper
       +-- Write test_auxiliary_backbones.py

[T1.6] VRAM Footprint & CUDA Memory Profiling
       +-- Profile peak memory consumption per backbone on RTX 4050
       +-- Document batch size limits for subsequent phases
```

### Detailed Task Specifications

#### Task 1.1: Environment Isolation & Core Dependencies
- Create a virtual environment using `python -m venv .venv`.
- Install PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.
- Install backbone dependencies:
  - `insightface==0.7.3`, `onnxruntime-gpu==1.17.1`
  - `speechbrain==1.0.0`
  - `mediapipe==0.10.14`
  - `transformers==4.40.0`, `accelerate==0.30.0`
  - `librosa==0.10.2`, `soundfile==0.12.1`, `scipy==1.13.0`
  - `audioseal==0.1.2`
  - `pytest==8.2.0`, `wandb==0.17.0`
- Generate `requirements.txt` with locked versions.

#### Task 1.2: Repository Scaffolding & Configuration Schema
- Populate the project directory structure as specified in `plans/overview.md`.
- Create `configs/environment.yaml`:
  ```yaml
  seed: 42
  device: "cuda"
  checkpoint_dir: "./checkpoints"
  paths:
    arcface_model: "buffalo_l"
    ecapa_model: "speechbrain/spkrec-ecapa-voxceleb"
    whisper_model: "openai/whisper-large-v3"
    styletts2_checkpoint: "./checkpoints/styletts2"
  ```
- Implement `voix/utils/seed.py`:
  ```python
  def set_global_seed(seed: int = 42):
      import random, os, numpy as np, torch
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```

#### Task 1.3: Visual Backbone Verification
- Create `voix/data/face_extractor.py`:
  - `ArcFaceExtractor`: loads `buffalo_l`, accepts $(B, 3, 112, 112)$ image tensor, returns $(B, 512)$ L2-normalized float32 tensor.
  - `FaceMeshExtractor`: loads MediaPipe FaceLandmarker, accepts $(H, W, 3)$ uint8 RGB, returns $(468, 3)$ coordinate array.
- Create `tests/test_visual_backbones.py` to assert:
  - ArcFace output shape is strictly `(B, 512)` with norm equal to $1.0 \pm 10^{-5}$.
  - FaceMesh returns exactly 468 landmarks with non-zero coordinates on sample portrait images.

#### Task 1.4: Acoustic Backbone Verification
- Create `voix/data/speaker_extractor.py`:
  - `ECAPAExtractor`: loads `speechbrain/spkrec-ecapa-voxceleb`, accepts $(B, T)$ audio waveform tensor at 16 kHz, returns $(B, 192)$ float32 embedding tensor.
- Create `tests/test_acoustic_backbones.py` to assert:
  - ECAPA output shape is strictly `(B, 192)`.
  - Deterministic audio produces identical embeddings across 5 successive runs.

#### Task 1.5: Auxiliary Backbone Verification
- Integrate Whisper large-v3 via HuggingFace `transformers`.
- Integrate `audioseal` watermarking model:
  - Assert that embedding a watermark into 3 seconds of 16 kHz audio alters SNR by less than 0.5 dB while detection detector returns confidence $>0.95$.
- Create `tests/test_auxiliary_backbones.py`.

#### Task 1.6: VRAM Profiling
- Write `scripts/profile_hardware.py` using `torch.cuda.max_memory_allocated()`.
- Record maximum VRAM consumption for each model independently and all models loaded concurrently.

---

## 5. Validation Strategy

Automated test execution using `pytest`:
```bash
pytest tests/test_visual_backbones.py -v
pytest tests/test_acoustic_backbones.py -v
pytest tests/test_auxiliary_backbones.py -v
```

### Manual Spot Checks
- Run `scripts/profile_hardware.py` on the local machine and verify that combined resident memory for inference backbones does not exceed 4.5 GB VRAM on the RTX 4050.

---

## 6. Acceptance Criteria

1. **Environment Reproducibility:** `pip install -r requirements.txt` executes cleanly with zero compilation errors on Windows 11 and Linux (Colab).
2. **CUDA Verification:** `torch.cuda.is_available()` returns `True`, and tensors allocate on `cuda:0` without warnings.
3. **Backbone Determinism:**
   - ArcFace produces identical 512-D embeddings ($L_\infty < 10^{-6}$) on repeated runs for the same input image.
   - ECAPA-TDNN produces identical 192-D embeddings ($L_\infty < 10^{-6}$) for the same input audio.
4. **Watermarking Integrity:** AudioSeal embeds watermarks into 16 kHz audio without audible artifacts and detection accuracy is 100% on 20 test clips.
5. **Memory Constraint:** All core extraction backbones fit concurrently in $\le 4.5\text{ GB}$ VRAM.
6. **Test Coverage:** All unit tests in `tests/test_*.py` pass with zero failures.

---

## 7. Risks & Trade-offs

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| InsightFace on Windows requires MSVC build tools | Setup failure | Use precompiled onnxruntime-gpu wheels or WSL2 environment fallback |
| StyleTTS 2 dependency conflicts with transformers | Broken environment | Isolate StyleTTS 2 inference in a modular subprocess or container if pin conflict occurs |
| 6 GB VRAM overflow during simultaneous model loading | Out-of-memory crash | Enforce pipeline memory cleanup (`del model; torch.cuda.empty_cache()`) between extraction steps |

---

## 8. Deliverables

- `requirements.txt`: Locked dependency manifest.
- `configs/environment.yaml`: System configuration parameters.
- `voix/`: Core package skeleton with verified backbones.
- `tests/`: Backbone smoke test suite (`test_visual_backbones.py`, `test_acoustic_backbones.py`, `test_auxiliary_backbones.py`).
- `scripts/profile_hardware.py`: Hardware memory and throughput audit script.

---

## 9. Documentation Updates

- Update `README.md` Setup Instructions to reflect exact Python 3.10 and CUDA installation commands.
- Mark Phase 1 tasks complete in `project-context/tasks.md`.

---

## 10. Dependencies

- **Predecessor:** None.
- **Successor:** Phase 2 (Multimodal Feature Extraction & Data Pipelines).
