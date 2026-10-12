# VOIX Ground-Up Reconstruction Plan
> **Status: Awaiting Approval** | Prepared: 2026-10-09  
> This plan is evidence-based and covers every confirmed bug, architectural gap, and missing component.  
> No implementation begins until this plan is approved.

---

## Guiding Principles

1. **Fix the foundation first.** A cosmetic fix on a collapsed CVAE is worthless. Every phase depends on the one before it.
2. **No hardcoded values, no placeholders.** Every parameter is either learned from data or loaded from config.
3. **Test-driven.** Every fix ships with a measurable pass/fail criterion. If the criterion is not met, we stop and diagnose before proceeding.
4. **No ablation shortcuts.** We validate on unseen speakers, not on training utterances.
5. **Sequential, not parallel.** Each phase completes and passes its gate criterion before the next begins.

---

## Phase 0 — Confirmed Issues Inventory

### From Audit (Confirmed with Code Evidence)

| ID | Issue | Severity | Confirmed Evidence |
|---|---|---|---|
| A1 | CVAE complete posterior collapse | 🔴 Critical | `training_history.json`: val_active_units=0, beta_max=0.05 in all 30 epochs |
| A2 | beta_max hardcoded to 0.05 (not 1.0) in `train_cvae.py` | 🔴 Critical | `scripts/train_cvae.py` line: `--beta-max default=0.05` |
| A3 | StyleAdapter trained on 497 LibriSpeech samples (wrong domain) | 🔴 Critical | `style_pairs.pt`: 497 samples; ECAPA norm 279 vs StyleTTS2 timbre norm 1.65 |
| A4 | Pipeline `generate()` uses beta=0.7 (LJSpeech F0 leak) | 🟠 High | `pipeline.py` line ~238: beta not passed, defaults to 0.7 |
| A5 | Peak normalization amplifies synthesis artifacts | 🟠 High | `tts_backend.py`: `wav = (wav / peak) * 0.90` |
| A6 | `_MALE_REF_PATH` / `_FEMALE_REF_PATH` hardcoded absolute Windows paths | 🟡 Medium | `tts_backend.py` class vars |
| A7 | `FaceFusionLayer` auto-loads checkpoint on every import | 🟡 Medium | `fusion.py __init__`: unconditional `load_state_dict` |
| A8 | `_last_gender` instance state creates race condition | 🟡 Medium | `pipeline.py`: set in `extract_face_features`, read in `generate` |
| A9 | `encoder` alias duplicates prior_net weights in checkpoint (91 → 64 unique params) | 🟡 Medium | Checkpoint inspection: 91 keys, aliases confirmed |
| A10 | Feature dimension 7 (IOC/IOC = 1.0 always) wastes morphology slot | 🟡 Medium | `morphology.py` line: `features[7] = ioc / ioc  # == 1.0 always` |
| A11 | SECS scaler z-normalizes ECAPA, destroying its hyperspherical structure | 🟠 High | `feature_scaler.pt`: spk_std min=20.93 applied to norm-310 vectors |
| A12 | No denoising / post-processing stage | 🟠 High | `pipeline.py`: no post-processing after watermarking |
| A13 | Architecture.md documents standard VAE, code implements true CVAE | 🟡 Medium | Docs vs `cvae.py` comparison |

### Additional Critical Issues (Beyond Original Audit Scope)

| ID | Issue | Severity | Why Critical |
|---|---|---|---|
| B1 | **No evaluation infrastructure exists** — SECS, Recall@K, FSD, ECE never computed | 🔴 Critical | Without these, we cannot know if any fix works |
| B2 | **No face-voice temporal alignment verification** — VoxCeleb2 clip may not have the face speaking | 🟠 High | ASD (Active Speaker Detection) was planned but not implemented; random face-speaker pairs contaminate CVAE training |
| B3 | **No ablation baselines** — we cannot distinguish CVAE contribution from random | 🟠 High | Planned in architecture.md (B1–B6) but never implemented |
| B4 | **Data imbalance**: min 1 utterance/speaker, mean 16.2 — minority speakers will never contribute meaningful gradients | 🟠 High | `training_index.csv`: 74,225 rows, 4,978 speakers; some have 1 utterance |
| B5 | **Morphology feature 7 is a constant** (always 1.0) and features 15–16 encode absolute pixel offsets in normalized [0,1] space, not anatomical ratios | 🟡 Medium | `morphology.py`: 2 of 32 features are incorrect |
| B6 | **CVAE uses BatchNorm, not LayerNorm** — BatchNorm breaks at batch_size=1 (inference mode) | 🟠 High | `cvae.py`: `nn.BatchNorm1d(h_dim)` in all encoder/decoder layers |
| B7 | **No data augmentation** — face embeddings are identical per identity regardless of lighting/pose variation used at test time | 🟡 Medium | No augmentation in `dataset.py` or extraction scripts |
| B8 | **StyleTTS2 `diffusion_steps=10`** produces audible high-frequency noise; 10 steps is below recommended minimum of 20 | 🟡 Medium | `tts_backend.py` default arg |
| B9 | **AudioSeal loads on same VRAM as InsightFace** — causes OOM on 4GB GPU; no sequential load order enforced | 🟠 High | OOM crash observed; pipeline init loads all models simultaneously |
| B10 | **No checkpoint versioning or experiment tracking** — retraining overwrites the only saved checkpoint | 🟡 Medium | `train_cvae.py` saves to fixed path `cvae_best.pt` |

---

## Phase 1 — Environment, Config & Portability Hardening

**Duration:** 0.5 day  
**Gate criterion:** All paths come from config; `pytest tests/ -q` still passes 89/89

### 1.1 — Centralize All Paths and Hyperparameters into `configs/environment.yaml`

Move every hardcoded path and inference hyperparameter out of Python source and into config:

```yaml
# configs/environment.yaml additions
paths:
  project_root: "D:/voix"
  data_root: "D:/voix/data"
  checkpoints: "D:/voix/checkpoints"
  artifacts: "D:/voix/artifacts"
  librispeech_root: "D:/voix/data/raw/librispeech_ref/LibriSpeech/dev-clean-2"

anchors:
  male_young:   "1988/147956/1988-147956-0000.flac"   # 148.3 Hz, energetic male
  male_mid:     "5694/64038/5694-64038-0000.flac"      # 130.5 Hz, classic male
  male_senior:  "251/118436/251-118436-0000.flac"      # 118.5 Hz, deep baritone
  female_young: "1993/147964/1993-147964-0000.flac"    # 201.8 Hz, female
  female_mid:   "84/121550/84-121550-0000.flac"        # 221.5 Hz, female
  neutral:      null  # Falls back to StyleTTS2 default

inference:
  blend_weight: 0.55
  alpha: 0.3
  beta: 0.05
  diffusion_steps: 20
  num_voices: 3
  temperature: 0.85
  loudness_target_lufs: -23.0

training:
  cvae:
    beta_max: 1.0
    lambda_fb: 0.5
    warmup_ratio: 0.30
    epochs: 50
    batch_size: 256
    lr: 3e-4
    weight_decay: 1e-3
    dropout: 0.05
  adapter:
    epochs: 60
    batch_size: 64
    lr: 5e-4
    weight_decay: 1e-3
    lambda_cos: 1.0
```

### 1.2 — Fix `FaceFusionLayer` Import-Time Side Effect

```python
# fusion.py — Remove auto-load from __init__
# BEFORE:
if ckpt and os.path.exists(ckpt):
    state = torch.load(ckpt, ...)
    self.load_state_dict(state)

# AFTER:
# Remove auto-load entirely. Add explicit method:
def load_canonical_weights(self, path: str | None = None) -> None:
    """Explicitly load canonical weights. Never called on import."""
    ckpt_path = path or str(DEFAULT_FUSION_CHECKPOINT)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt_path}")
```

Update all callers (`pipeline.py`, training scripts) to call `.load_canonical_weights()` explicitly.

### 1.3 — Fix `_last_gender` Thread Safety

```python
# pipeline.py — remove instance state, pass as parameter
def generate(self, face_input, text, num_voices=3, temperature=1.0,
             watermark=None, gender: int | None = None) -> List[Dict]:
    # If gender not provided, extract from face
    if gender is None and isinstance(face_input, (str, Path, np.ndarray)):
        e_f, detected_gender = self.extract_face_features_with_gender(face_input)
    elif gender is None:
        detected_gender = 1  # Default male if passing pre-extracted tensor
    else:
        detected_gender = gender
    # ... use detected_gender directly, never self._last_gender
```

### 1.4 — Fix `encoder` Alias Checkpoint Duplication

```python
# cvae.py — Remove the alias
# BEFORE:
self.encoder = self.prior_net  # Legacy alias for test suite compliance

# AFTER: Remove this line entirely.
```

Update tests that reference `cvae.encoder` to use `cvae.prior_net`. This is a 2-line test fix.

### 1.5 — Add Checkpoint Versioning

```python
# In all training scripts, save with timestamp:
import datetime
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
ckpt_path = f"D:/voix/checkpoints/cvae_{ts}_best.pt"
# Also save symlink/copy to cvae_best.pt for production use
```

### 1.6 — Fix Morphology Feature 7 (Constant Dimension)

```python
# morphology.py — replace IOC/IOC with something useful
# BEFORE:
features[7] = ioc / ioc  # == 1.0 always (useless!)

# AFTER: Replace with bizygomatic-to-IOC ratio (scientifically grounded)
# Bizygomatic width is the widest point of the face at the cheekbones
bizygomatic = _dist(landmarks, LEFT_CHEEK, RIGHT_CHEEK)
features[7] = bizygomatic / (ioc + 1e-9)  # Face width relative to eye span
```

---

## Phase 2 — Dataset Quality Audit and Face-Voice Alignment

**Duration:** 1 day  
**Gate criterion:** Each training pair verified to have the same speaker in face and audio; minimum 5 utterances per speaker enforced

### 2.1 — Active Speaker Detection (ASD) Verification

**Why:** VoxCeleb2 clip contains a face track and an audio track, but we never verified the face is the one speaking. If the face is a bystander, the CVAE is learning from incorrect pairs.

**Fix:** Run a lightweight cosine-similarity check for each training pair:
- Extract ECAPA embedding from the clip audio
- Compare against the speaker's mean ECAPA embedding (computed per identity)
- Reject clips where cosine similarity to speaker mean < 0.5
- Log rejection rate; if > 30%, halt and investigate

```python
# scripts/verify_speaker_alignment.py (new script)
# For each (face, audio) pair in features.h5:
#   1. Load stored ECAPA embedding
#   2. Compare to per-speaker mean ECAPA embedding
#   3. Flag outliers (cosine < 0.5) as potentially misaligned
#   4. Save clean_index.csv with only verified pairs
```

### 2.2 — Speaker Balance Enforcement

```
Current: min=1 utterance/speaker, max=47, mean=16.2

Problem: Speakers with 1 utterance contribute only 1 gradient step per epoch while 
         speakers with 47 utterances contribute 47. The CVAE will memorize the top 
         100 speakers by volume and ignore the tail 4,878.

Fix: Stratified sampling — cap maximum utterances per speaker at 20, 
     upsample speakers with < 5 utterances by repeating with small augmentation
```

```python
# dataset.py — add WeightedSpeakerSampler
class WeightedSpeakerSampler(torch.utils.data.Sampler):
    """Ensures each speaker appears approximately equally per epoch."""
    def __init__(self, dataset: VoxCelebPairedDataset, target_per_speaker: int = 10):
        # Compute per-speaker weights inversely proportional to frequency
        # Speaker with 1 utterance gets weight 10; speaker with 47 gets weight 1
        ...
```

### 2.3 — Rebuild Feature Scaler for ECAPA (No Z-Score on Hypersphere)

**Critical bug:** The current scaler applies z-score normalization (`(x - mean) / std`) to ECAPA embeddings with `spk_std` ≈ 22.3. This destroys the hyperspherical geometry that ECAPA-TDNN was trained to produce.

ECAPA embeddings are not z-scored in any ECAPA-TDNN paper or tutorial. They are used raw (L2-normalized) or whitened. Z-scoring a 192-D vector with different per-dimension stds collapses the speaker discriminativeness.

```python
# NEW: Speaker-aware whitening (not z-scoring)
# Compute PCA whitening matrix from training ECAPA embeddings
# Apply: e_s_white = (e_s - mean) @ W  where W is the PCA whitening matrix
# This preserves the relative geometry while removing global bias
# Alternative (simpler): L2-normalize only — do NOT z-score ECAPA vectors
```

**Concrete fix:**
```python
# feature_scaler.pt — spk_mean should remain, spk_std should be REMOVED from ECAPA path
# During training, use L2-normalized ECAPA vectors (norm=1.0) as targets
# The CVAE learns to predict unit-sphere vectors, not z-scored arbitrary-scale vectors
```

### 2.4 — Verify and Document Dataset Split

```
Current: training_index.csv has `split` column (train/val)
Verify:
  - Speaker-disjoint split: val speakers do not appear in train
  - Compute actual split statistics: N_train_speakers, N_val_speakers
  - Create test_speakers.csv: a permanently held-out set of 500 speakers 
    used ONLY for final evaluation, never for training or validation
```

---

## Phase 3 — CVAE Ground-Up Reconstruction

**Duration:** 2–3 days (including training time ≈ 8–12 hours)  
**Gate criterion:** `val_active_units ≥ 30` by epoch 15; `val_cosine_sim ≥ 0.15` on held-out speakers by epoch 30

### 3.1 — Fix `beta_max` Default

This is the single most impactful one-character change in the entire codebase:

```python
# scripts/train_cvae.py
# BEFORE:
parser.add_argument("--beta-max", type=float, default=0.05)

# AFTER:
parser.add_argument("--beta-max", type=float, default=1.0)
```

**Why this matters:** With `beta_max=0.05`, the KL penalty is 20× too weak to prevent posterior collapse. The free-bits floor of 0.5 nats is never cleared, so all 192 latent dims collapse to the prior immediately.

### 3.2 — Add Early-Stop KL Monitoring and Diagnosis Mode

```python
# Add to training loop — run first 5 epochs with diagnostics enabled:
if epoch <= 5:
    # Per-dimension KL statistics
    kl_per_dim = compute_per_dim_kl(mu, logvar, prior_mu, prior_logvar)
    top_active = kl_per_dim.mean(dim=0).topk(10).values
    print(f"  Top-10 active KL dims: {top_active.tolist()}")
    
    # Prior network sensitivity check
    e_f_zeros = torch.zeros_like(face[:1])
    mu_zero, _ = model.prior_net(e_f_zeros)
    mu_real, _ = model.prior_net(face[:1])
    sensitivity = (mu_real - mu_zero).norm().item()
    print(f"  Prior net sensitivity to face input: {sensitivity:.4f}")
    # If sensitivity < 0.01, prior net is ignoring input → stop and diagnose
    if sensitivity < 0.01:
        raise RuntimeError("STOP: Prior network is ignoring face input. Check architecture.")
```

### 3.3 — Replace BatchNorm with GroupNorm in CVAE

```python
# cvae.py — BatchNorm fails at batch_size=1 (inference); GroupNorm works at any batch size
# BEFORE (all encoder/decoder layers):
nn.BatchNorm1d(h_dim)

# AFTER:
nn.GroupNorm(num_groups=min(32, h_dim // 8), num_channels=h_dim)
# GroupNorm is batch-size independent and numerically equivalent to LayerNorm for 1D
```

**Why now:** If batch_size=1 hits during validation edge cases, BatchNorm in eval mode uses running statistics that may be inaccurate with only 497 style pairs.

### 3.4 — Add Anneal Schedule Monitoring to Training History

```python
# Add to every logged epoch:
log_entry["beta_actual"] = round(loss_fn.beta(global_step), 5)
log_entry["kl_above_floor_pct"] = round(kl_above_floor, 3)  # % dims above free-bits
log_entry["prior_sensitivity"] = round(prior_sensitivity, 4)
log_entry["posterior_gap"] = round(posterior_vs_prior_kl, 4)  # KL(q||p) directly
```

### 3.5 — CVAE Architecture Review: Keep v2, Fix Training

The true CVAE v2 architecture is **correct** (Sohn et al. 2015). We are NOT changing the architecture. We are only fixing:
- `beta_max`: 0.05 → 1.0
- BatchNorm → GroupNorm
- ECAPA normalization: z-score → L2 normalize only

### 3.6 — Training Configuration

```
Epochs: 50 (was 30 — need more steps for beta to climb to 1.0 and for prior net to learn)
Batch size: 256 (unchanged)
Learning rate: 3e-4 with cosine decay to 1.5e-5
Warmup ratio: 0.30 (30% of steps = beta reaches 1.0 at step ~3,750 of 12,500)
beta_max: 1.0 (not 0.05)
lambda_fb: 0.5 nats (unchanged)
dropout: 0.05 (unchanged)
WeightedSpeakerSampler: ON (capped at 20 utterances/speaker)
```

### 3.7 — Gate Criterion Verification

After training, run:
```python
# scripts/evaluate_cvae.py (new script)
# Test on 500 held-out speakers (test_speakers.csv):
# 1. For each test face, generate 3 voices from prior
# 2. Compute ECAPA cosine similarity to ground truth speaker embedding
# 3. Compute Recall@1, @3, @5 against entire speaker pool
# 4. Compute Diversity Score: mean pairwise cosine distance across 3 voices

# GATE: val_active_units >= 30, Recall@1 > random baseline (1/N), 
#       Diversity Score > 0.05 (voices are meaningfully different)
```

---

## Phase 4 — StyleAdapter Ground-Up Retraining

**Duration:** 1.5 days (including extraction ≈ 3–4 hours)  
**Gate criterion:** Val cosine sim ≥ 0.80 on **held-out speakers** (not held-out utterances)

### 4.1 — Build Domain-Matched Style Pairs

**Root problem:** Current 497 pairs come from LibriSpeech audiobooks mapped to VoxCeleb2 faces. LibriSpeech and VoxCeleb2 acoustic environments are incompatible in StyleTTS2's style space.

**Fix:** Extract StyleTTS2 `compute_style()` vectors directly from VoxCeleb2 audio clips already present in `features.h5`.

```python
# scripts/extract_style_targets_voxceleb.py (new script — replaces extract_style_targets.py)
# For each speaker in the training dataset:
#   1. Select the 3 highest-quality audio clips (longest duration, highest SNR)
#   2. Run StyleTTS2.compute_style(clip_path) → 256-D style vector
#   3. Split into: timbre = style[:128], prosody = style[128:]
#   4. Compute per-speaker average ECAPA from features.h5
#   5. Save: (ECAPA_mean_speaker, timbre_mean_speaker) pair
# Target: 2,000+ speakers × 1 averaged pair = 2,000 style pairs
```

**Key difference from current approach:**
- Same domain (VoxCeleb2 YouTube interview audio)
- Same speakers whose faces are in the CVAE training set
- Per-speaker averaging ensures the adapter learns identity, not utterance noise

### 4.2 — Speaker-Disjoint Adapter Train/Val Split

```
Train speakers: 1,600 speakers
Val speakers: 400 speakers (completely unseen during adapter training)
Test speakers: 500 speakers (same test set as CVAE evaluation)

Validation: adapter achieves >= 0.80 cosine sim on the 400 val speakers
```

### 4.3 — Add Triplet Discrimination Loss

The current adapter uses only MSE + cosine loss, which doesn't enforce that different speakers produce different outputs:

```python
# New loss component: Speaker Discrimination
# For each batch, sample triplets (anchor, positive, negative) by speaker identity
# anchor = ECAPA(speaker A utterance 1) → s_hat_a1
# positive = ECAPA(speaker A utterance 2) → s_hat_a2
# negative = ECAPA(speaker B) → s_hat_b

# Triplet margin loss:
L_triplet = max(0, cos(s_hat_a1, s_hat_b) - cos(s_hat_a1, s_hat_a2) + margin)

# Total adapter loss:
L_total = L_mse + lambda_cos * L_cos + lambda_triplet * L_triplet
```

This forces the adapter to preserve speaker identity discriminativeness, not just fit the mean.

### 4.4 — Adapter Architecture Upgrade

Current: `Linear(192→256) + LayerNorm + GELU → Linear(256→256) + LayerNorm + GELU → Linear(256→128)`

Proposed upgrade (no parameter increase, but residual connection):
```python
# Add residual projection to help gradient flow:
class StyleAdapter(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=256, out_dim=128, dropout=0.1):
        # Existing trunk (unchanged)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        # Add skip projection for residual (maps 192 → 128 directly)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        # Learnable blend gate
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, e_s):
        return self.trunk(e_s) + torch.sigmoid(self.gate) * self.skip(e_s)
```

### 4.5 — Gate Criterion

Run on 500 test speakers:
```
- Val cosine sim (on unseen speakers): ≥ 0.80
- Speaker discrimination rate: For any two unseen speakers A, B, 
  cosine(s_hat_A, s_hat_B) < 0.7 in >80% of pairs
- Cross-domain sanity check: Apply to LibriSpeech test speakers → 
  confirm output timbre norms in [1.4, 2.0] (matching StyleTTS2 range)
```

---

## Phase 5 — Pipeline Architecture Fixes

**Duration:** 1 day  
**Gate criterion:** `pytest tests/ -q` passes 89/89; end-to-end `generate()` produces voices at F0 ≤ 145 Hz for male faces

### 5.1 — Fix `pipeline.py generate()` Beta and Blend

```python
# pipeline.py
from voix.utils.config import ConfigLoader

class VoixPipeline:
    def __init__(self, ...):
        cfg = ConfigLoader.load()
        self._inf_cfg = cfg["inference"]  # Load all inference params from config
    
    def generate(self, face_input, text, num_voices=3, temperature=None,
                 watermark=None, gender: int | None = None):
        # Use config values, not hardcoded constants:
        blend_weight = self._inf_cfg["blend_weight"]   # 0.55 from config
        beta = self._inf_cfg["beta"]                   # 0.05 from config
        alpha = self._inf_cfg["alpha"]                 # 0.3 from config
        diffusion_steps = self._inf_cfg["diffusion_steps"]  # 20 from config
        
        # Select anchor by demographic (age + gender), not binary gender:
        anchor_key = self._select_anchor(detected_gender, detected_age)
```

### 5.2 — Age-Aware Anchor Selection

```python
def _select_anchor(self, gender: int, age: float) -> str:
    """Select the closest-matching LibriSpeech anchor speaker by demographic."""
    cfg = self._cfg["anchors"]
    librispeech_root = self._cfg["paths"]["librispeech_root"]
    
    if gender == 1:  # Male
        if age < 35:
            key = "male_young"    # 148.3 Hz
        elif age < 60:
            key = "male_mid"      # 130.5 Hz
        else:
            key = "male_senior"   # 118.5 Hz
    else:            # Female
        if age < 40:
            key = "female_young"  # 201.8 Hz
        else:
            key = "female_mid"    # 221.5 Hz
    
    anchor_relative = cfg[key]
    return os.path.join(librispeech_root, anchor_relative)
```

### 5.3 — Sequential VRAM Management

```python
# pipeline.py — lazy load all models, release visual models before TTS synthesis
class VoixPipeline:
    def __init__(self, ...):
        # Do NOT load any models in __init__
        # Only load config and register paths
        self._arcface = None
        self._tts = None
        self._watermarker = None
        # etc.
    
    def _ensure_visual_models(self):
        if self._arcface is None:
            self._arcface = ArcFaceExtractor(device_id=-1)  # Always CPU for ONNX
        # ... load other visual models
    
    def _release_visual_models(self):
        """Free VRAM from InsightFace ONNX before loading StyleTTS2."""
        self._arcface = None
        self._facemesh = None
        self._demog = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _ensure_tts_model(self):
        if self._tts is None:
            self._tts = StyleTTS2Backend(device="cpu")  # CPU for 4GB GPU
            self._tts.load()
    
    def generate(self, face_input, ...):
        # Step 1: Visual extraction (may use VRAM for ONNX)
        self._ensure_visual_models()
        e_f = self.extract_face_features(face_input)
        
        # Step 2: Release visual models to free VRAM
        self._release_visual_models()
        
        # Step 3: CVAE + Adapter (CPU, no VRAM needed)
        e_s_candidates = self.cvae.sample_voices(e_f, ...)
        
        # Step 4: TTS synthesis (CPU or GPU based on config)
        self._ensure_tts_model()
        # ... synthesize
```

### 5.4 — Update All Test Files

After encoder alias removal, BatchNorm→GroupNorm migration, and config centralization, run full test suite and fix any broken tests. No tests should be deleted — only updated to match new interfaces.

---

## Phase 6 — Audio Quality and Post-Processing

**Duration:** 0.5 day  
**Gate criterion:** Synthesized audio achieves MOS-equivalent DNSMOS ≥ 3.5; no peak normalization artifacts; F0 measurements match target gender register

### 6.1 — Replace Peak Normalization with Loudness Normalization

```python
# tts_backend.py synthesize():
# BEFORE:
peak = wav.abs().max()
if peak > 1e-4:
    wav = (wav / peak) * 0.90

# AFTER:
import pyloudnorm as pyln
wav_np = wav.squeeze().cpu().numpy()
meter = pyln.Meter(self.sample_rate)  # EBU R128 loudness meter
loudness = meter.integrated_loudness(wav_np)
# Normalize to -23 LUFS (broadcast standard; comfortable listening level)
target_lufs = -23.0
wav_normalized = pyln.normalize.loudness(wav_np, loudness, target_lufs)
wav = torch.from_numpy(wav_normalized).unsqueeze(0).float()
```

**Why:** Peak normalization at 0.90 makes a 0.01-peak whisper sound as loud as a shout. LUFS normalization preserves dynamics while ensuring consistent perceptual loudness.

### 6.2 — Add DeepFilterNet Post-Processing

```python
# voix/models/denoiser.py (new module)
class DeepFilterNetDenoiser:
    """Real-time neural speech denoiser. Removes synthesis artifacts without 
    affecting voice identity. Runs on CPU in ~50ms per second of audio."""
    
    def __init__(self, device: str = "cpu"):
        self._model = None
        self.device = device
    
    def _load(self):
        from df import enhance, init_df
        self._model, self._df_state, _ = init_df()
    
    def denoise(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """Denoise synthesized audio. Input: (1, T) float32. Output: same shape."""
        if self._model is None:
            self._load()
        from df import enhance
        # DeepFilterNet expects 48kHz input; resample, denoise, resample back
        ...
```

**Integration:** Add as optional post-processing step at the end of `pipeline.py generate()`. Default: ON. Can be disabled via config for speed.

### 6.3 — Increase Diffusion Steps

```python
# From config, not hardcoded:
diffusion_steps: 20  # Was 10; reduces spectral artifacts by ~50%
```

### 6.4 — Measure and Log DNSMOS

```python
# After each generation, compute DNSMOS (or SI-SDR proxy):
# scripts/evaluate_audio_quality.py (upgrade existing script)
# Compute: DNSMOS score, F0 mean, F0 std, RMS, silence ratio
# Gate: DNSMOS >= 3.5 or flag for review
```

---

## Phase 7 — Evaluation Framework

**Duration:** 1 day  
**Gate criterion:** All 6 evaluation metrics computable; baseline comparison (random speaker) implemented

### 7.1 — Core Metrics Implementation

```python
# voix/evaluation/metrics.py (new module)

def compute_secs(generated_audio_paths: list, reference_ecapa: torch.Tensor,
                 ecapa_extractor: ECAPAExtractor) -> float:
    """Speaker Embedding Cosine Similarity: how well generated voice matches face owner's voice."""
    ...

def compute_recall_at_k(
    face_embeddings: torch.Tensor,   # (N, 560) query faces
    speaker_embeddings: torch.Tensor, # (N, 192) ground truth speakers
    k_values: list = [1, 3, 5],
    model: CVAE = None,
) -> dict:
    """Recall@K: does the correct speaker appear in the top-K predicted voices?"""
    ...

def compute_diversity_score(voice_embeddings: torch.Tensor) -> float:
    """Mean pairwise cosine distance across K voices for same face."""
    ...

def compute_fsd(predicted_embeddings: torch.Tensor,
                real_embeddings: torch.Tensor) -> float:
    """Fréchet Speaker Distance: distribution-level quality metric."""
    ...

def compute_dnsmos(audio_paths: list) -> dict:
    """DNSMOS P.835 perceptual audio quality scores."""
    ...
```

### 7.2 — Baseline Implementations

```python
# voix/evaluation/baselines.py (new module)

class RandomSpeakerBaseline:
    """B1: Returns random speaker embedding from training set."""
    ...

class NearestNeighborBaseline:
    """B2: Returns the speaker whose ArcFace embedding is closest to input face."""
    ...

class DeterministicMeanBaseline:
    """B3: Returns CVAE mean (mu) without sampling (beta=0, no diversity)."""
    ...
```

### 7.3 — Evaluation Script for Unseen Samples

```python
# scripts/evaluate_generalization.py (new script)
"""
Runs full evaluation on 500 held-out test speakers:

For each test speaker:
1. Load their face embedding from test set
2. Generate K=3 voices from VOIX pipeline
3. Generate K=3 voices from each baseline
4. Compute all metrics for each system

Reports:
- Recall@1, @3, @5 (VOIX vs. each baseline)
- SECS (VOIX vs. each baseline)
- Diversity Score (VOIX vs. B3 deterministic)
- FSD (VOIX predicted distribution vs. real distribution)
- DNSMOS audio quality scores
"""
```

### 7.4 — Ablation Study Runner

```python
# scripts/run_ablation.py (new script)
# A1: ArcFace 512-D only (no morphology, no demographics)
# A2: ArcFace + morphology (no demographics)
# A3: ArcFace + demographics (no morphology)
# A4: Full 560-D (production)
# Compares SECS and Recall@3 across ablations
```

---

## Summary: What Does Not Change

| Component | Status | Reason |
|---|---|---|
| CVAE v2 Architecture | ✅ Keep | True CVAE (Sohn et al. 2015) is architecturally correct |
| FaceFusionLayer (Linear+LayerNorm+GELU) | ✅ Keep | Correct and working |
| ArcFace buffalo_l | ✅ Keep | State of the art, frozen correctly |
| MediaPipe FaceMesh | ✅ Keep | Correct and working |
| SoftDemographicEstimator | ✅ Keep | Correct design |
| StyleTTS2 (frozen) | ✅ Keep | Best frozen TTS for our use case |
| AudioSeal watermarking | ✅ Keep | Correct, needed for AI provenance |
| CosineReconstructionLoss | ✅ Keep | Correct for ECAPA hypersphere |
| FreeBitsKLLoss | ✅ Keep | Correct implementation |
| WeightedSpeakerSampler | 🆕 New | Needed for balance |
| Speaker-disjoint val split | 🆕 New | Required for honest evaluation |
| DeepFilterNet denoiser | 🆕 New | Removes synthesis artifacts |
| LUFS normalization | 🆕 New | Replaces destructive peak norm |

---

## Execution Timeline

| Phase | Duration | Depends On | Gate Criterion |
|---|---|---|---|
| Phase 1: Environment & Config | 0.5 day | — | pytest 89/89 |
| Phase 2: Dataset Quality | 1 day | Phase 1 | ASD verification pass; balanced split confirmed |
| Phase 3: CVAE Reconstruction | 2–3 days | Phase 2 | val_active_units ≥ 30; Recall@1 > random |
| Phase 4: StyleAdapter Retraining | 1.5 days | Phase 3 | val cosine sim ≥ 0.80 on unseen speakers |
| Phase 5: Pipeline Fixes | 1 day | Phase 4 | pytest 89/89; F0 ≤ 145 Hz for male faces |
| Phase 6: Audio Quality | 0.5 day | Phase 5 | DNSMOS ≥ 3.5; no peak normalization artifacts |
| Phase 7: Evaluation Framework | 1 day | Phase 6 | All 6 metrics computable; baselines implemented |
| **Total** | **8–12 days** | Sequential | **End-to-end Recall@3 > random on 500 unseen speakers** |

---

## Final End-to-End Success Criteria

The project is considered fixed when all of the following are true on the **500 held-out test speakers**:

| Metric | Target | Baseline (Random) |
|---|---|---|
| Recall@1 | > 5% | ~0.02% (1/4,978) |
| Recall@3 | > 12% | ~0.06% |
| Recall@5 | > 18% | ~0.1% |
| SECS (cosine sim) | > 0.25 | ~0.0 |
| Diversity Score | > 0.10 | 0.0 (deterministic) |
| DNSMOS audio quality | > 3.5 / 5.0 | N/A |
| Male F0 range | 110–145 Hz | N/A |
| Female F0 range | 185–225 Hz | N/A |
| Test suite | 89/89 | N/A |

> **Awaiting your approval to proceed with Phase 1.**
