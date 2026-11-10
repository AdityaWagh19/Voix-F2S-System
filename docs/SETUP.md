# VOIX Setup and Operations

This guide covers the P0 demo environment: an isolated Python install, exact
artifact validation, model-cache prefetch, offline use, and the CLI safety
behaviour around CUDA out-of-memory.

VOIX remains a research/demo system. A passing setup check does **not** mean the
generated identity, accent, or emotion is accurate. See
[`docs/MASTER.md`](MASTER.md) and the limitations at the end of this page.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10–3.12 | Validated on 3.12.1. The lock file is named for Python 3.12. |
| Git | The checkout is the project root. |
| `ffmpeg` on `PATH` | Required for video/audio corpus work; optional for a pure synthesis demo. |
| NVIDIA driver + CUDA 12.1 | Only for the `cu121` wheel path. CPU works without a GPU. |
| Disk space | The artifact set is roughly 420 MB; model caches add several hundred MB. |

The audited workstation has a 4.29 GB VRAM GPU. The instructions below are
deliberately conservative for that profile.

---

## 2. Clean environment

The old `.venv` had `include-system-site-packages = true`, which made `pip
check` report 60+ conflicts from the user site. Do not reuse that environment
for a release or demo.

### Recommended: bootstrap script

```powershell
# From the repository root. --recreate clears an existing .venv.
python scripts/setup/bootstrap_env.py --recreate
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

The script:

1. creates a normal isolated venv (`include-system-site-packages = false`);
2. upgrades `pip`, `setuptools`, and `wheel`;
3. installs `torch==2.4.1` and `torchaudio==2.4.1` from the PyTorch CUDA 12.1
   index;
4. installs `-e ".[serving]"` under
   `constraints/serving-py312-cu121.txt`;
5. runs `pip check` and a non-fatal `voix preflight`.

Useful variants:

```powershell
# CPU-only machine
python scripts/setup/bootstrap_env.py --recreate --device cpu --no-lock

# Add evaluation and test dependencies
python scripts/setup/bootstrap_env.py --recreate --with-evaluation --with-dev

# Reuse a separate environment name instead of touching .venv
python scripts/setup/bootstrap_env.py --venv .venv-voix --recreate

# Inspect what would run
python scripts/setup/bootstrap_env.py --dry-run --recreate
```

### Manual equivalent

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchaudio==2.4.1 `
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[serving]" -c constraints/serving-py312-cu121.txt
python -m pip check
```

For a CPU-only or non-Windows machine, omit the CUDA index and the CUDA
constraints file:

```bash
python -m pip install -e ".[serving]"
```

`torchvision` is intentionally not required: no serving-path module imports it.
See [`constraints/README.md`](../constraints/README.md) for the lock's scope and
regeneration policy.

---

## 3. Artifact validation and restore

Model weights and processed data are ignored by git. `configs/artifacts.yaml`
is the single source of truth for what a checkout needs. Each entry records the
project-relative path, required/optional status, byte size, SHA-256 digest,
license, and source.

### Check the checkout

```powershell
python scripts/setup/verify_artifacts.py --verbose
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | All required artifacts are present and valid. |
| 1 | A required artifact is missing or invalid. |
| 2 | The manifest itself is missing or malformed. |

`--no-hash` skips SHA-256 and checks only existence and size. `--strict` also
fails on missing optional artifacts. JSON output is available with `--json`.

### Restore from a mirror

A mirror is a directory containing the same project-relative layout:

```text
E:\voix-artifacts\
  results\checkpoints\cfvn_v2_best.pt
  checkpoints\cvae_cfvn_face192_latent128_diversity.pt
  checkpoints\style_adapter_cfvn_best.pt
  checkpoints\models\buffalo_l\*.onnx
  checkpoints\mediapipe\face_landmarker.task
  data\processed\voxceleb2_v2\face_scaler_v2.pt
```

Then run:

```powershell
python scripts/setup/verify_artifacts.py --restore-from E:\voix-artifacts
```

Each source file is checked against the manifest before it is copied. A partial
copy is never treated as valid. Optional artifacts can be skipped with
`--skip-optional`.

### Required versus optional artifacts

| Artifact | Role | Required |
|---|---|---|
| CFVN, CVAE, StyleAdapter | Identity path | Yes |
| FaceFeatureV2 scaler | Face features | Yes |
| MediaPipe FaceLandmarker | Face features | Yes |
| All five `buffalo_l` ONNX files | Face features | Yes |
| E1 emotion residual | Emotion control | No |
| A3 accent residual | Accent control | No |
| ECAPA-TDNN | Evaluation only | No |

The ArcFace pack is checked as five separate files because InsightFace
`FaceAnalysis` loads detection, landmarks, gender/age, and recognition together.
Checking only `w600k_r50.onnx` allowed a broken offline checkout to pass
preflight.

Artifact licenses are recorded per entry. The repository's MIT code license
does not relicense VoxCeleb2-derived weights, InsightFace models, MSP-Podcast
residuals, or GLOBE-derived residuals. See the limitations below.

---

## 4. Model-cache prefetch

StyleTTS2 and AudioSeal download their own weights on first use. Run the
prefetcher once while online, then use the cache for demos:

```powershell
# Download anything missing (CPU only; no GPU allocation)
python scripts/setup/prefetch_models.py

# Inspect only; no imports or downloads
python scripts/setup/prefetch_models.py --check

# JSON for CI or a dashboard
python scripts/setup/prefetch_models.py --check --json

# Refresh one family
python scripts/setup/prefetch_models.py --only styletts2 --force
```

Cache locations:

| Family | Default | Override |
|---|---|---|
| StyleTTS2 | `~/.cache/cached_path` | `CACHED_PATH_CACHE_ROOT` |
| AudioSeal | `~/.cache/audioseal` | `AUDIOSEAL_CACHE_DIR` (or `XDG_CACHE_HOME`) |
| NLTK `punkt` | `~/AppData/Roaming/nltk_data` (Windows) or `~/nltk_data` | `NLTK_DATA` |

`voix preflight` reports both caches. A missing cache is a warning while
online; with offline mode enabled it is an error.

---

## 5. Preflight and a first synthesis

```powershell
# Artifact, cache, device, and output check; no synthesis
python -m voix preflight --json

# Validate a specific request before spending GPU time
python -m voix preflight -f face.jpg -o output.wav -v 1 --json

# Short CPU smoke test (about 80 s for ~3 s of audio on the audited machine)
python -m voix synth -f face.jpg -t "Hello from VOIX." -o output.wav --device cpu --seed 42

# Default is the configured accelerator; watermarking is opt-in
python -m voix synth -f face.jpg -t "Hello." -o output.wav
python -m voix synth -f face.jpg -t "Hello." -o output.wav --watermark
```

The default prompt is short. `-v/--voices` is bounded to 1–10, temperature to
0.1–3.0, and accent/emotion values are validated by argparse before any model
loads. Progress goes to stderr so `--json` keeps stdout machine-readable.

Output rules are validated **before** synthesis:

- one candidate may target `out.wav` or a directory;
- multiple candidates require a directory;
- an existing file where a directory is expected is rejected.

Accent and emotion flags exist, but the v3 external panel is at chance for both.
They are research controls, not demo-worthy perceptual controls.

---

## 6. Device selection and CUDA OOM fallback

`--device` is accepted globally or after `synth`, `batch`, `interactive`, or
`preflight`:

```powershell
python -m voix --device cpu preflight
python -m voix synth -f face.jpg --device cuda -o output.wav
```

On a CUDA out-of-memory error, the service:

1. clears the failed CUDA attempt;
2. resets the process-wide StyleTTS2 model so it cannot be reused on the wrong
   device;
3. retries the request on CPU;
4. reports `execution_device: "cpu"` and a `fallback_reason` in JSON.

This is a demo safety path, not a performance guarantee. On the 4.29 GB GPU,
prefer one voice and no watermark for a live demo. Multi-voice and watermarking
both increase peak memory.

If CPU fallback also fails, run the request with `--device cpu` explicitly and
inspect the error rather than assuming the model is missing.

---

## 7. Offline mode

After artifacts and caches are present:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:VOIX_OFFLINE = "1"
python -m voix preflight --json
```

`VOIX_OFFLINE=1` makes missing StyleTTS2/AudioSeal caches a preflight error.
Leave it unset for the first online run.

The ViT age estimator (`nateraw/vit-age-classifier`) is a Hugging Face download.
Its failure is caught and InsightFace's gender/age output is used instead, but
an offline first run can still pause while the Hub reports the cache miss. The
age path is optional; it is not part of identity synthesis.

---

## 8. Batch, capabilities, and diagnostics

```powershell
python -m voix batch -d faces\ -o batch_output\ -v 1
python -m voix capabilities --json
python -m voix info --json
```

`info` now reports the manifest-backed checkpoint list, both model caches, the
resolved device, and the requested device. `--json` is suitable for support
logs; do not include private face images or generated audio in a public report.

---

## 9. Verification commands

```powershell
# Fast contracts
python -m pytest tests/test_cli.py tests/test_artifacts.py -q

# Full suite (artifact-dependent tests skip when the V2 files are absent)
python -m pytest -q

# Release-style artifact gate
python scripts/setup/verify_artifacts.py --strict --json

# Cache readiness
python scripts/setup/prefetch_models.py --check --json
```

A green CI gate means the identity path is structurally non-collapsed; it is not
an accuracy or perceptual-quality guarantee.

---

## 10. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `pip check` reports dozens of conflicts | venv includes the polluted user site | Recreate the venv with `bootstrap_env.py --recreate` or a new `--venv` name. |
| Preflight says a required artifact is missing | Weights are git-ignored | Restore from the artifact mirror, then rerun `verify_artifacts.py`. |
| Preflight passes but synthesis downloads models | Runtime cache is empty | Run `prefetch_models.py`, then `prefetch_models.py --check`. |
| `Invalid or missing model checkpoint path. Loading default model...` | Upstream StyleTTS2 uses `None` to select its cached defaults | Expected; it is not a VOIX checkpoint failure. |
| `Logging before InitGoogle()` / XNNPACK message | MediaPipe native logging | Informational. |
| CUDA OOM during multi-voice or watermarking | 4.29 GB VRAM budget | Retry with `-v 1`, omit `--watermark`, or use `--device cpu`; automatic fallback should already report the reason. |
| `nltk` LookupError for `punkt` | Tokenizer data not cached | Run `prefetch_models.py --only nltk`. |
| `voix synth ... --device cpu` rejected | Very old CLI invocation | Current CLI accepts `--device` globally or on the subcommand; verify with `python -m voix synth --help`. |
| Windows console Unicode errors | Legacy code page | The CLI reconfigures stdout/stderr to UTF-8; alternatively set `PYTHONIOENCODING=utf-8`. |

---

## 11. What this setup does not establish

- The v3 external panel scores generated emotion and accent at chance
  (`3/12 = 0.250`, condition-invariant). No MOS, CMOS, emotion-identification,
  or accent-identification study exists.
- The v3 identity distinctness gate fails: the maximum neutral between-voice
  ECAPA cosine is 0.5696 versus 0.4379 for real same-gender speakers.
- CFVN face-to-voice retrieval is weak: R@1 0.0932 against 0.0085 chance.
- The identity adapter is selected on validation but remains below the
  target-mean baseline on the test split, and its face-specific residual is
  weak.
- Evaluation recognizers are evaluation-only and are never part of synthesis.
- VoxCeleb2, InsightFace, MSP-Podcast, GLOBE, and the pretrained TTS/watermark
  weights carry their own terms. Do not present generated audio as a real
  person's voice or use it for impersonation.
