# Phase 7: Production Hardening & Reproducibility

> **Phase Identifier:** `PHASE-7`  
> **Target Duration:** Weeks 15–16  
> **Status:** Pending Execution  
> **Prerequisites:** `PHASE-6` Complete

---

## 1. Objective

Harden, package, and release the VOIX research system for public distribution and peer review. Execute a multi-seed reproducibility audit across seeds `42`, `43`, and `44` to verify numerical stability. Package and host trained model checkpoints on HuggingFace Hub. Standardize and release the India F2S Benchmark dataset (embeddings and metadata only) on HuggingFace Datasets. Provide a turnkey command-line interface (`voix-infer`) and programmatic Python API for one-command synthesis.

---

## 2. Scope

### In Scope
- Multi-seed verification audit: re-running training and evaluation over fixed seeds `42`, `43`, `44` to measure standard deviations on all reported metrics.
- HuggingFace Hub integration: model cards, safetensors serialization, config exports, and automated download helpers.
- HuggingFace Datasets integration: dataset cards, data loaders, automated checksum verification for the India F2S Benchmark.
- Production CLI: `voix-infer` supporting single-image and batch inference modes with AudioSeal watermarking.
- Programmatic Python API: clean pip-installable packaging (`pip install -e .`) with full type annotations.
- Test automation: complete test suite runner spanning unit, integration, and regression checks with automated GitHub Actions workflow definition.
- Ethical safety and licensing compliance audit: confirming no raw media redistribution, validating watermarking integration, and embedding license declarations.

### Out of Scope
- Building a web GUI or mobile app (command-line and Python API are sufficient for research dissemination).
- Commercial closed-source licensing.

---

## 3. Design Decisions & Rationale

### 3.1 Safetensors vs. PyTorch Pickles (`.pt`)
- **Decision:** Convert all published checkpoints to `.safetensors` format prior to public release.
- **Rationale:** Standard PyTorch `.pt` files use Python's `pickle` module, which presents arbitrary code execution security risks. HuggingFace flags unpickled weights and security-conscious researchers reject them. Safetensors prevents security vulnerabilities and supports zero-copy memory mapping.

### 3.2 CLI Packaging via Setuptools Entry Points
- **Decision:** Expose inference via a console entry point `voix-infer` defined in `pyproject.toml` / `setup.py`.
- **Rationale:** Researchers and downstream practitioners should be able to run `voix-infer --image portrait.jpg --text "Hello world"` without digging through source scripts or configuring manual import paths.

### 3.3 Strict Embedding-Only Benchmark Release
- **Decision:** Host only `arcface_512.safetensors`, `ecapa_192.safetensors`, `facemesh_32.safetensors`, and `metadata.csv`. Never upload raw extracted video frames or WAV files.
- **Rationale:** Guarantees absolute compliance with international copyright law, YouTube terms of service, and ethical privacy principles regarding biographical facial likeness.

---

## 4. Sequential Implementation Tasks

```
[T7.1] Multi-Seed Reproducibility Audit
       ├── Execute training runs for seeds 42, 43, 44
       ├── Compute mean +/- std for SECS, Recall@5, DS, ECE
       └── Export reproducibility manifest to artifacts/reproducibility.json

[T7.2] HuggingFace Hub Packaging & Model Card
       ├── Convert CVAE and Adapter checkpoints to .safetensors
       ├── Generate model card with hardware requirements and performance metrics
       └── Build automated downloader (voix/models/hub.py)

[T7.3] India F2S Benchmark Dataset Publication
       ├── Write dataset loading script (india_f2s.py) for datasets library
       ├── Write comprehensive Dataset Card with demographic breakdown
       └── Generate SHA-256 checksum manifests for all embedding archives

[T7.4] Standalone CLI & Programmatic API Packaging
       ├── Implement scripts/voix_cli.py with click / argparse
       ├── Configure pyproject.toml with console entry points
       └── Write test_cli_execution.py

[T7.5] End-to-End Regression Test Suite & CI Automation
       ├── Create .github/workflows/ci.yaml
       ├── Integrate pre-commit hooks (black, isort, flake8, mypy)
       └── Assert 100% test pass rate on clean virtual environment

[T7.6] Final Repository Cleanup & Documentation Sync
       ├── Purge temporary caches and debug artifacts
       ├── Update README.md with HuggingFace Hub and Dataset links
       └── Tag final git release v1.0.0
```

### Detailed Task Specifications

#### Task 7.1: Multi-Seed Audit
- File: `scripts/run_reproducibility_audit.py`
- Iterate over `seeds = [42, 43, 44]`.
- For each seed:
  1. Train CVAE on VoxCeleb2 subset.
  2. Evaluate SECS, Recall@5, Diversity Score, and ECE.
- Compute metric stability: assert $\sigma(\text{SECS}) \le 0.015$ and $\sigma(\text{DS}) \le 0.020$.
- Write output to `artifacts/reproducibility_audit.json`.

#### Task 7.2: HuggingFace Model Card & Weights
- File: `voix/models/hub.py`
- Provide `from_pretrained(repo_id)` method:
  ```python
  from huggingface_hub import hf_hub_download
  from safetensors.torch import load_file

  def load_voix_pretrained(repo_id="AdityaWagh19/voix-f2s-model", device="cuda"):
      cvae_path = hf_hub_download(repo_id, "cvae.safetensors")
      adapter_path = hf_hub_download(repo_id, "adapter.safetensors")
      config_path = hf_hub_download(repo_id, "config.yaml")
      # Load models into memory
  ```

#### Task 7.4: Production CLI
- File: `scripts/voix_cli.py`
- Usage:
  ```bash
  voix-infer \
    --image examples/input_face.jpg \
    --text "Welcome to the demonstration of the VOIX synthesis system." \
    --k 3 \
    --output-dir outputs/demo/ \
    --device cuda
  ```
- Output behavior:
  - Generates `outputs/demo/voice_1.wav`, `outputs/demo/voice_2.wav`, `outputs/demo/voice_3.wav`.
  - Automatically verifies AudioSeal watermark in memory before writing.
  - Prints summary table with predicted fundamental frequency ($F_0$) estimate and pairwise diversity distance.

#### Task 7.5: CI & Quality Hooks
- File: `.github/workflows/ci.yaml`
  - Installs dependencies on `ubuntu-latest`.
  - Runs linting: `flake8 voix/ tests/`, `black --check voix/`.
  - Runs type checking: `mypy voix/`.
  - Executes unit test suite on CPU: `pytest tests/test_backbones.py tests/test_features.py tests/test_cvae.py`.

---

## 5. Validation Strategy

### CLI Verification
```bash
# Verify installation
pip install -e .
voix-infer --help

# Verify end-to-end inference
voix-infer --image tests/fixtures/sample_face.jpg --text "Test synthesis." --output-dir artifacts/test_cli/
```

### Checksum & Integrity Audit
```bash
python scripts/verify_checksums.py --dir data/india_benchmark/
```

---

## 6. Acceptance Criteria

1. **Numerical Reproducibility:** Metrics across seeds `42`, `43`, `44` demonstrate standard deviation $\sigma < 0.015$ on SECS and $\sigma < 0.020$ on Diversity Score.
2. **Security & Weight Safety:** All model weights released strictly as `.safetensors`; zero `.pt` or `.bin` pickle files published to HuggingFace.
3. **Turnkey CLI Operation:** `voix-infer` runs out-of-the-box from a clean shell, creating 3 watermarked 24 kHz WAV files from an input JPEG in $<3.0$ seconds without manual configuration.
4. **Dataset Compliance:** India F2S Benchmark loads successfully via `datasets.load_dataset("AdityaWagh19/india-f2s-benchmark")` in Python.
5. **Code Quality & Typing:** Zero flake8 errors, 100% type hint coverage on public APIs passing `mypy` strict mode.
6. **Watermark Enforcement:** 100% of CLI-synthesized audio passes automated AudioSeal detection verification.

---

## 7. Risks & Trade-offs

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| User machine lacks CUDA / GPU | CLI crash on CPU | Implement automatic CPU fallback with warning that generation latency will increase to ~12s |
| HuggingFace Hub network timeout | Download failure | Cache downloaded weights locally in `~/.cache/voix/` and support offline local path flag |
| PyPI package name collision | Installation confusion | Use namespaced package identifier `voix-f2s` |

---

## 8. Deliverables

- `pyproject.toml` & `setup.py`: Pip-installable package definition.
- `scripts/voix_cli.py`: Production CLI entry point (`voix-infer`).
- `voix/models/hub.py`: HuggingFace Hub download and loading utilities.
- `.github/workflows/ci.yaml`: Automated GitHub Actions continuous integration pipeline.
- `artifacts/reproducibility_audit.json`: 3-seed reproducibility audit report.
- HuggingFace model repo: `AdityaWagh19/voix-f2s-model`.
- HuggingFace dataset repo: `AdityaWagh19/india-f2s-benchmark`.

---

## 9. Documentation Updates

- Update `README.md` with:
  - Quickstart CLI usage guide.
  - HuggingFace Model & Dataset badges and links.
  - Exact citation bibtex entry for peer-reviewed paper.
- Tag git release `v1.0.0`.

---

## 10. Dependencies

- **Predecessor:** `PHASE-6` (Baselines, Ablations & Empirical Evaluation).
- **Successor:** Project complete; paper camera-ready and public release.
