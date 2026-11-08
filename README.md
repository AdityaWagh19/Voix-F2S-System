# VOIX — Face-to-Voice Synthesis

VOIX is a **research/demo system** that maps a face image (or a precomputed
face representation) to a distribution of plausible speaker identities, then
renders speech with a neural TTS backend. It is built for controlled
experiments, not for identity verification, impersonation, production
deployment, or commercial use.

> **Status:** the V1 face corpus and its checkpoints were degenerate. The V2
> stack is rebuilt from raw VoxCeleb2 video with a speaker-disjoint protocol.
> Canonical identity results live in `results/ci_gate_v2.json`; the current
> research demo is the 24-clip v3 set at `results/demo/research_v3/`; external
> evaluator results are in `results/evaluation/panel_v2/` and
> `results/evaluation/panel_v3/`. The v3 external panel is at chance on
> generated emotion and accent conditions, and no human perceptual study exists. This
> README does not assert unverified numbers. The full audit is in
> [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md); the consolidated research record is in
> [`docs/MASTER.md`](docs/MASTER.md).

### Current scope

The active scope is defined in [`docs/SCOPE.md`](docs/SCOPE.md):

- **In scope:** candidate generation (core), four emotions (`neutral`, `happy`,
  `sad`, `angry`), and four accents (`us`, `england`, `indian_english`,
  `australian`). Emotion and accent are explicit user selections; there is no
  default conditioning.
- **Out of scope permanently:** cross-lingual synthesis, speech-difference
  simulation, and craniofacial sliders. Legacy code and assets may remain for
  reproducibility, but they are not product features and must not be advertised.

Architecture, data access, measured evidence, rejected options, reproduction,
and non-claims are consolidated in [`docs/MASTER.md`](docs/MASTER.md). The
historical code audit is in [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md).

---

## 1. What the system actually does

```
face image
  → ArcFace 512-D identity
  → MediaPipe craniofacial geometry 32-D
  → demographic posterior 16-D
  → per-modality L2 normalization + deterministic fusion
  → 560-D FaceFeatureV2 raw vector
  → train-split z-score scaler
  → CFVN: 560 → 192-D ECAPA-aligned voice identity
  → CVAE: conditional prior p(z | face) → decoder → K candidate voices
  → StyleAdapter: 192 → 256-D StyleTTS2 style
  → StyleTTS2 (or explicit deterministic fallback)
  → 24 kHz waveform
```

The CVAE is deliberately one-to-many: the face constrains a voice manifold, not
a single “correct” voice. The adapter is trained on sampled CVAE decoder outputs,
which are the distribution serving uses; validation uses the deterministic
prior mean so checkpoint selection is reproducible.

---

## 2. Dataset and splits

The authoritative manifest is
[`data/processed/voxceleb2_v2/index.csv`](data/processed/voxceleb2_v2/index.csv).
It is built from the **VoxCeleb2 dev** shards plus the official **VoxCeleb2
test** shard:

| Split | Clips | Speakers | Source |
|---|---:|---:|---|
| train | 33,205 | 5,393 | dev shards 1–6 |
| val | 3,701 | 601 | dev shards 1–6 |
| test | 628 | 118 | official VoxCeleb2 test |

The split is speaker-disjoint by construction. The manifest keeps every dev
speaker, samples 5–8 clips per speaker from distinct videos, balances gender,
and is deterministic. The official test speakers are held out entirely.

The raw videos are **not redistributed**. `data/raw/` and
`data/processed/voxceleb2_v2/` are local research artifacts and are ignored by
git.

The StyleTTS2 LoRA pilot uses a separate 432-clip / 263-speaker / 0.741 h
speaker-disjoint subset of the same corpus, with Whisper `base` ASR transcripts
rather than human ground truth. It is a research pilot, not a naturalness
benchmark; see §5.1.

VoxCeleb2 is governed by its own terms: **research / non-commercial use only**.
Do not use this project or its artifacts to impersonate a real person.

---

## 3. Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

`ffmpeg` must be on `PATH` for video/audio extraction. The project config is
checkout-relative; `configs/environment.yaml` contains no `D:/voix` paths.

---

## 4. Reproducing the V2 corpus

### 4.1 Build the manifest

```bash
python scripts/data/build_voxceleb2_index.py
```

This reads only ZIP central directories over HTTP range requests. It requires
an authenticated Hugging Face token with access to
`ProgramComputer/voxceleb`.

### 4.2 Download only the selected clips

```bash
python scripts/data/download_selected_clips.py \
  --index data/processed/voxceleb2_v2/index.csv \
  --output-dir data/raw/voxceleb2/clips \
  --shards 1,2,3,4,5,6 --workers 16
```

The downloader fetches individual ZIP members with range requests; it never
stores the full 50 GB shard archives.

### 4.3 Extract FaceFeatureV2

Process one shard at a time into the same resumable HDF5:

```bash
python scripts/data/extract_voxceleb2_v2.py \
  --index data/processed/voxceleb2_v2/index.csv \
  --videos data/raw/voxceleb2/clips \
  --output data/processed/voxceleb2_v2/features_v2.h5 \
  --device cuda --shard 1 \
  --failures data/processed/voxceleb2_v2/extraction_failures_v2.csv
```

The extractor samples frames across the whole clip, tracks every face, and
selects the track whose mouth motion correlates with the audio envelope. It
does **not** use “largest face” or “first frontal frame” heuristics. Each row
stores raw ArcFace (512), geometry (32), demographics (16), fused features
(560), ECAPA voice (192), per-clip quality, and selection metadata.

### 4.4 Corpus health gate

Validate the structural contract first. This catches duplicate clip ids,
non-finite features, split speaker leakage, clips absent from the manifest, and
manifest clips that are neither extracted nor recorded as failures:

```bash
python scripts/audit/validate_v2_corpus.py \
  --features data/processed/voxceleb2_v2/features_v2.h5 \
  --index data/processed/voxceleb2_v2/index.csv \
  --failures data/processed/voxceleb2_v2/extraction_failures_v2.csv \
  --output data/processed/voxceleb2_v2/corpus_validation.json \
  --expect-complete
```

Then measure representation quality:

```bash
python scripts/evaluation/evaluate_v2_corpus.py \
  --features data/processed/voxceleb2_v2/features_v2.h5 \
  --index data/processed/voxceleb2_v2/index.csv \
  --output data/processed/voxceleb2_v2/health.json
```

This reports face→face retrieval, voice→voice retrieval, embedding health,
linear probes, gender probes, audio-visual selection quality, extraction
failures, and shuffled/constant negative controls. Do not start training until
the corpus is healthy.

### 4.5 Accent corpus (GLOBE_V3), metadata first

GLOBE_V3 is 348.7 GB in the Hugging Face tree, but 183.2 GB of that is a
duplicate `train/`/`dev/`/`test/` audio copy. The Parquet tree is 165.5 GB and
already contains the 24 kHz audio. The acquisition path never clones it:

```bash
# 1. Index metadata only: footers plus the ~20 KB metadata span per row group.
python scripts/data/index_globe_metadata.py --workers 10

# 2. Common Voice metadata (validated.tsv + clip_durations.tsv only).
python scripts/data/fetch_common_voice_metadata.py

# 3. Recover client_id (speaker) and self-reported accent.
python scripts/data/join_globe_common_voice.py

# 4. Strict balance gate and speaker-disjoint manifest. 2.5 h pre-trim gives
#    headroom for VAD trimming; Irish is evaluation-only at 1.32 h.
python scripts/data/select_globe_accent_subset.py \
  --target-hours 2.5 --min-speakers 30 --max-ratio 1.5 \
  --selection-mode row_group --evaluation-only irish

# 5. Fetch only the selected row groups' audio; decode 24 kHz FLAC to WAV.
python scripts/data/download_globe_selected.py --workers 8

# 6. Authoritative post-trim audit and exclusion of clipped clips.
python scripts/audit/audit_globe_audio.py --workers 8
python scripts/data/prune_globe_manifest.py
python scripts/audit/audit_globe_audio.py --workers 8
```

`scripts/data/index_globe_metadata.py` writes `source_file`, `row_group`, and
`row_in_group` for every row, so the audio pass reads only selected row groups.
The selector resolves self-reported accents (not classifier predictions),
applies quality filters and a per-speaker cap, and fails a class that misses
2 hours or 30 speakers rather than lowering the threshold. The per-class
result is in `data/processed/globe_accent_selection_summary.json`; the manifest
is `data/processed/globe_accent_manifest.csv`.

The completed metadata pass covered **187 shards / 723,532 rows with 0
failures** in 858 s. The Common Voice join matched **723,532/723,532 rows
(100%)**, recovering **50,951 speakers / 1,102.1 h**; 450,835 rows carry a
self-reported accent. The strict selection picks **9,688 clips / 13.83 h
pre-trim / 889 speakers / 543 row groups** for the six accent classes. The
five trainable classes are balanced at 2.50 h pre-trim each (ratio 1.0) with
94–201 train speakers; Irish has only 1.32 h pre-trim and is evaluation-only.
Because GLOBE audio is VAD-trimmed, 2.5 h pre-trim is used to clear the 2.0 h
post-trim gate. The post-download audit measured **9,534 clips / 12.28 h
actual** with 0 health failures after excluding 154 source-clipped clips
(1.6%); the five trainable classes are 2.185–2.257 h actual, 94–198 train
speakers, and ≥5 val/test speakers. Irish is 1.176 h and evaluation-only.
`scripts/audit/audit_globe_audio.py` measures the authoritative hours and never
lowers the 2.0 h / 30-speaker threshold; `scripts/data/prune_globe_manifest.py`
records the exclusion, keeps the unpruned manifest, and moves the excluded WAVs
to `data/processed/globe_audio_excluded` so a glob over `globe_audio` cannot
pick them up.

### Emotion pilot data (CREMA-D)

CREMA-D is acquired with a blob-filtered sparse checkout that skips the video
payload:

```powershell
git clone --filter=blob:none --no-checkout https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/crema_d
cd data/raw/crema_d
git sparse-checkout init --cone
git sparse-checkout set AudioWAV processedResults
git checkout master
cd D:\voix
python scripts/data/curate_crema_d.py
```

The curation parses all **7,442 clips**, joins actor demographics and the
**voice-only** crowd vote, and keeps **6,892 clips / 91 actors / 4.869 h** after
excluding 550 source-clipped clips (7.4%). The six classes are 0.633–0.963 h
(ratio 1.521) with 90–91 speakers, so every class fails the 2.0 h gate. CREMA-D
is therefore **pilot/evaluation-only** and `training_eligible` is false; the
threshold is not lowered. Its voice-only labels are also weak (neutral 0.976,
happy 0.330, sad 0.236, fear 0.356, disgust 0.343, angry 0.568), which is a
finding, not a defect to hide. `scripts/training/train_emotion_classifier.py` produces a
speaker-disjoint pilot classifier (125-D acoustic features, RBF SVM selected on
validation): on 699 held-out clips from 9 speakers it reaches **57.8% accuracy /
0.578 macro-F1** versus a 17.9% majority baseline, with a speaker-cluster
bootstrap 95% CI of **0.513–0.647** accuracy and **0.511–0.643** macro-F1. The
report keeps `control_training_eligible: false`; this is a pilot evaluation
classifier, not a learned control.

---

## 5. Training

All trainers refuse speaker leakage and select checkpoints on held-out
retrieval, not on mean cosine.

```bash
# 0. Fit the canonical 560-D scaler on the train split. Rebuild this after the
#    corpus changes; never fit it on validation or test rows. The scaler records
#    n_train/n_train_speakers: the CFVN trainer rebuilds it when those counts
#    do not match the corpus, while the CVAE/adapter trainers and CI gate fail
#    loudly instead of training or gating on a stale normalization.
python scripts/data/build_face_scaler_v2.py \
  --features data/processed/voxceleb2_v2/features_v2.h5 \
  --index data/processed/voxceleb2_v2/index.csv \
  --output data/processed/voxceleb2_v2/face_scaler_v2.pt

# 1. Canonical 560-D fused CFVN
python -m voix.training.trainer_cfvn \
  --features data/processed/voxceleb2_v2/features_v2.h5 \
  --index data/processed/voxceleb2_v2/index.csv \
  --face-key face_features \
  --output results/checkpoints/cfvn_v2_best.pt

# 2. Raw-ArcFace diagnostic baseline (separate scaler; never overwrites the
#    560-D serving scaler)
python scripts/data/build_face_scaler_v2.py \
  --features data/processed/voxceleb2_v2/features_v2.h5 \
  --index data/processed/voxceleb2_v2/index.csv \
  --face-key arcface \
  --output data/processed/voxceleb2_v2/face_scaler_arcface_v2.pt
python -m voix.training.trainer_cfvn \
  --face-key arcface \
  --scaler data/processed/voxceleb2_v2/face_scaler_arcface_v2.pt \
  --output results/checkpoints/cfvn_v2_arcface.pt

# 3. Compare both representations on the official test split
python scripts/evaluation/compare_cfvn.py \
  --fused results/checkpoints/cfvn_v2_best.pt \
  --arcface results/checkpoints/cfvn_v2_arcface.pt \
  --output results/cfvn_representation_comparison.json

# 4. CVAE on CFVN embeddings
python -m voix.training.trainer_cvae_v3_diversity \
  --cfvn results/checkpoints/cfvn_v2_best.pt \
  --output checkpoints/cvae_cfvn_face192_latent128_diversity.pt

# 5. Adapter on the CVAE serving distribution
python -m voix.training.trainer_adapter_cfvn \
  --cfvn results/checkpoints/cfvn_v2_best.pt \
  --cvae checkpoints/cvae_cfvn_face192_latent128_diversity.pt \
  --style-targets data/processed/style_targets_voxceleb2_5clip.pt \
  --lambda-div 1.0 --lambda-norm 1.0 \
  --output checkpoints/style_adapter_cfvn_best.pt
```

The CFVN defaults (`--reg-target centroid`, `--lambda-reg 5.0`, `--lr 1e-3`)
were selected on shard-1 validation. Speaker-centroid regression targets were
necessary: with the old `lambda_reg=0.5` and per-clip targets, face identity was
preserved but face→voice retrieval stayed at chance.

Before scaling to the full corpus, measure the training-speaker saturation
curve:

```bash
python scripts/audit/cfvn_subset_saturation.py \
  --fractions 0.25,0.5,1.0 --epochs 12 \
  --output results/cfvn_subset_saturation.json
```

Choose the CFVN (`face_features` vs `arcface`) by held-out face→voice
retrieval. Raw ArcFace is the identity-preservation baseline; it is not
automatically the better cross-modal model. Both trainers also report a
gender-controlled retrieval score (gallery restricted to the query's gender):
the fused representation contains gender/age channels, so the unrestricted
score can be inflated by demographic matching. The controlled score is the
identity check; chance rises to roughly `1 / n_same_gender`.

### 5.1 StyleTTS2 prosody LoRA pilot (research only)

The polished, audiobook-like delivery is a checkpoint/prompt mismatch, not a
post-processing artifact. The first naturalness pilot therefore adapts only the
StyleTTS2 **prosody diffusion U-Net** and leaves the text encoder, BERT, style
encoder, predictor, decoder, CFVN, CVAE, and style adapter frozen.

The corpus is a speaker-disjoint VoxCeleb2 subset with real Whisper `base`
pseudo-transcripts, extracted at 24 kHz:

| Split | Clips | Speakers | Hours |
|---|---:|---:|---:|
| train | 369 | 225 | 0.628 |
| val | 63 | 38 | 0.113 |
| total | 432 | 263 | 0.741 |

It is deliberately small: 268 candidate clips were dropped (202 low Whisper
log-probability, 47 too long, 16 too few words, 3 text too long). The
transcripts are ASR output, not human ground truth, and VoxCeleb2 remains
research/non-commercial. This is a pilot corpus, not a naturalness benchmark.

`scripts/training/train_styletts2_lora.py` injects rank-8 / alpha-16 LoRA into 25
`nn.Linear` layers of `model.diffusion.unet` (434,184 trainable parameters),
uses AdamW at `2e-4`, gradient accumulation 4, warmup-cosine, 10 epochs, and
selects the best checkpoint on held-out diffusion loss. The run completed all
10 epochs with 0 failed clips; validation loss fell from 0.5980 to 0.5226 and
the adapter's `lora_B` norms moved from zero to 0.424–0.866.

The paired held-out generation evaluation (25 clips / 21 speakers, same noise
for both conditions, 10 diffusion steps) does **not** show a naturalness win:

| Metric | Base | LoRA | Paired delta (LoRA − base) | 95% paired t CI |
|---|---:|---:|---:|---:|
| WER | 0.0331 | 0.0354 | +0.0022 | −0.0122…+0.0166 |
| ECAPA target similarity | 0.1222 | 0.1267 | +0.0045 | −0.0085…+0.0174 |
| ECAPA reference similarity | 0.2400 | 0.2362 | −0.0039 | −0.0160…+0.0082 |
| F0 mean (Hz) | 162.76 | 157.95 | −4.80 | −9.62…+0.01 |
| Duration (s) | 7.303 | 7.289 | −0.014 | −0.046…+0.018 |
| Energy mean | 0.05336 | 0.05377 | +0.00041 | −0.00029…+0.00110 |

The only paired change that survives Holm correction across the ten metrics is
a small decrease in energy standard deviation (0.04572 → 0.04457), which is a
signal-statistics change, not evidence of better naturalness. The full
ten-metric table is in `docs/AUDIT_REPORT.md` §10.10; raw rows and training history
are under `results/checkpoints/styletts2_lora_prosody*`. Human MOS is
unavailable. The adapter is experimental and is **not** wired into
`StyleTTS2Backend`; the canonical CFVN and style adapter are unchanged. A
credible next step is a larger conversational corpus such as AMI plus real
listener ratings, not shipping this checkpoint.

A second, AMI-based LoRA manifest was built for the naturalness track:
2,424 rows (train 2,320 clips / 8 speakers / 2.79 h; validation 104 / 4 /
0.13 h) with **human** AMI transcripts and deterministic same-speaker reference
clips. A training-only smoke passed (4 optimizer steps, 0 failures, 25 LoRA
layers, 434,184 trainable parameters). The N1 400-step pilot was then scored on
104 held-out pairs: it passed the pre-registered proxy gate narrowly (WER
+0.0078, ECAPA improved, only F0-median error improved by 0.93 Hz) but the
comparison has only four speaker clusters. There is no MOS, and N2 was not
entered. That is a weak objective-proxy result, not evidence of better
naturalness.

The exploratory, confirmatory, and deployment evidence levels, the per-track
status, and every measured claim are consolidated in
[`docs/MASTER.md`](docs/MASTER.md).

---

## 6. Evaluation and CI gate

```bash
python scripts/evaluation/ci_gate_v2.py \
  --features data/processed/voxceleb2_v2/features_v2.h5 \
  --index data/processed/voxceleb2_v2/index.csv \
  --output results/ci_gate_v2.json
```

The gate fails on missing artifacts or missing metrics. It checks:

- held-out face→voice retrieval vs chance on the full official test split,
- shuffled and constant negative controls near chance,
- CFVN embedding health (within/between cosine, collapse ratio),
- CVAE decoder diversity and face margin,
- active latent units,
- adapter output diversity, per-dimension spread, and target-style fidelity.

The gate uses the full test split by default. `--max-faces 400` is a faster
smoke subset, but it is underpowered for the strict 10× chance-multiple check
and can fail even when the full split passes; report the full-split result.

Measured on the final V2 artifacts (`results/ci_gate_v2.json`,
`data/processed/voxceleb2_v2/eval_v2_full.json`,
`results/cfvn_representation_comparison.json`,
`results/adapter_v2_evaluation.json`):

| Metric | Value |
|---|---:|
| Corpus | 35,892 clips / 6,111 speakers |
| Fused CFVN test face-to-voice R@1 | 0.0932 (11.0× chance) |
| Raw-ArcFace CFVN test face-to-voice R@1 | 0.1017 (12.0× chance) |
| Raw fused test face-to-face R@1 | 0.9153 |
| Adapter test raw / centered across cosine | 0.8446 / -0.0022 |
| Adapter test target cosine | 0.8742 |
| Adapter test per-dimension std | 0.0207 |
| Full official-test CI gate | PASS |

The fused representation remains the serving contract because the
ArcFace/fused test R@1 ratio is 1.0909, below the pre-registered 1.5× switch
threshold.

A structure-preservation experiment (`--lambda-struct`, raw-face similarity
distillation) was run to close the CFVN face-face gap. It raises test face-face
R@1 from 0.8475 to 0.8814–0.9153 across the runs, but only the seed-42 s3 run
matches raw 0.9153; the same configuration at seeds 43/44 passes validation and
fails test (0.8814/0.8983 vs 0.9153). The validation-selected winner is seed 44,
whose `test_identity_gate_winner` is **false**. This is a negative result, not a
fix. `results/checkpoints/cfvn_v2_best.pt` remains canonical, and its downstream
CVAE/adapter were not retrained. Details are in
`results/cfvn_structure_comparison.json` and `docs/AUDIT_REPORT.md`.

Measure the HDF5-to-serving gap on the official test split:

```bash
python scripts/evaluation/measure_serving_parity.py \
  --features data/processed/voxceleb2_v2/features_v2.h5 \
  --split test --limit 240 \
  --output data/processed/voxceleb2_v2/serving_parity_test.json
```

This re-reads the stored frame from each raw clip, re-detects and re-extracts a
single frame, and reports bbox/ArcFace/geometry/demographic/fused cosine
agreement plus the resulting CFVN face→face and face→voice retrieval. The
corpus stores a four-frame averaged ArcFace embedding while serving sees one
frame, so this gap is measured rather than assumed away.

Synthesis metrics are in `scripts/evaluation/evaluate_synthesis_v2.py`:

- **WER:** real Whisper transcription plus word-level Levenshtein distance,
  using the exact prompt recorded for each asset.
- **FAD proxy:** Fréchet distance over mel-spectrogram statistics against real
  held-out speech. This is explicitly **not** VGGish FAD and is labelled as a
  proxy.
- **Speaker similarity / diversity:** ECAPA embeddings.
- Missing dependencies or references return `status: unavailable`, never a
  fabricated `0` or `-1`.

Human MOS is a separate protocol:
`python scripts/evaluation/run_mos_evaluation.py` prepares blinded stimuli and a rating
sheet; it computes statistics only from a real ratings CSV.

### 6.1 Expression controls (emotion, accent)

The current research demo is `results/demo/research_v3/`: **24 clips**, three
face-derived voices, each with neutral plus `happy`/`sad`/`angry` and
`us`/`england`/`indian_english`/`australian`. Generation uses seed 42, 10
diffusion steps, serving scales (emotion residual 0.28; accent residual 0.30
with timbre weight 0.25), 24 kHz PCM-16, -23 LUFS, and a -1 dBTP limiter.

Measured v3 checks:

| Check | Result |
|---|---|
| Clips / conditions | 24 / 24 present |
| WER (Whisper `base`) | 0.000 on all 24 clips |
| Emotion direction checks | 9/9 passed |
| Emotion/accent application + ECAPA-to-neutral >= 0.90 | PASS; min 0.9453 (Indian accent included at 0.985-0.992) |
| Noise floor / SNR | PASS; median -62.07 dBFS, SNR 44.04 dB |
| Three neutral voices distinct | **FAIL**; max pair ECAPA 0.5696 vs real same-gender maximum 0.438 |

The control evidence behind the demo is exploratory:

- **Emotion E1 (accepted baseline).** The speaker-balanced class-mean
  prosody-only residual passes 6/6 pre-registered direction checks on held-out
  CREMA-D actors and survives a post-hoc Holm correction. The learned E2
  residual fails sad and does not beat E1, so E2 is rejected.
- **Accent A3 (research transform only).** The class-contrast residual at scale
  0.30 gained one correctly classified clip out of 30 (`7/30` vs `6/30`) and
  had zero recall for Indian and Canadian. It is applied as
  `exploratory_application`, not as a validated accent control.
- **Indian G2P (rejected).** A2c's conservative rule subsets stayed inside the
  `+0.10` WER budget but produced zero classifier gain (`0/6` in both
  conditions). Indian G2P is not applied; Indian English uses the same A3
  residual as the other accents.
- **C1 reference-prosody (not promoted).** Replacing the prosody half with the
  mean of real training references preserved the timbre half exactly, but the
  minimum ECAPA-to-neutral fell to 0.773 (happy), below the 0.90 guard. Sad did
  not reproduce the E1 direction: its F0 rose and its duration did not lengthen.
  WER was not computed. In the C1 external panel, SUPERB ER scored 0.500
  generated accuracy but its real-speech happy recall is only 0.033; emotion2vec+
  scored 0.250 with every clip predicted neutral. C1 remains an exploratory
  comparison, not a serving control.

External evaluator panel on v3 (frozen third-party recognizers; evaluation
only): all three evaluators pass their real-speech and generated-domain gates,
but all score the v3 generated clips at chance.

| Evaluator | Task | Real accuracy | Generated accuracy | Generated n |
|---|---|---:|---:|---:|
| CommonAccent | accent | 0.664 | 0.250 | 12 |
| SUPERB ER | emotion | 0.440 | 0.250 | 12 |
| emotion2vec+ | emotion | 0.910 | 0.250 | 12 |

On v3, CommonAccent predicts only US (8 clips) or England (4); it never predicts
Indian or Australian. SUPERB ER predicts happy (8) or neutral (4). emotion2vec+
predicts neutral for all 12 clips. This is a null external-recognition result,
not proof that listeners hear nothing. There is no human MOS, CMOS,
emotion-identification, or accent-identification study.

Historical v1 diagnostic (superseded): the old 12 identity-controlled
heuristic assets had speaker preservation 0.509 mean / 0.236 min (sad) and worst
WER 0.353 (Indian English), so that gate failed. Those assets used fixed
heuristic deltas, not the current residuals, and are retained only as a
negative baseline. The v1 candidate identity numbers (pairwise ECAPA 0.7236
mean / 0.5408 min; face→voice cosine 0.1019) are also historical.

### 6.2 Noise, loudness, and delivery

**V2 demo (2026-11-05, retained).** The research demo at
`results/demo/research_v2/` has **21 clips**: three face-derived voices
(Lupita Nyong'o, Timothee Chalamet, Jackie Chan), each with neutral plus three
emotions (happy, sad, angry) and three accents (US, England, Australian).
Generation uses seed 42, 10 diffusion steps, `alpha=0.18`, `beta=0.22`,
24 kHz PCM-16, -23 LUFS, and a -1 dBTP limiter.

The v1 hiss came from the raw face-derived style prior, not from clipping, the
output writer, or the diffusion step count. In a controlled matrix that held
face, speaker embedding, text, and seed fixed, raw face style measured
**-49.17 dBFS**; the same model with a real AMI style measured **-71.07 dBFS**.
Steps 5/20/30 and the N1 LoRA did not remove the hiss (N1 measured -46.31 dBFS,
worse than the raw baseline). The v2 demo applies a gender-matched mean
translation, `s' = s + alpha * (mean_real_gender - mean_face_gender)`, at
`alpha=1.0`; this cuts noise by **13.36, 17.58, and 17.12 dB** on the three
selected voices. The 21-clip median is **-63.45 dBFS** (min -64.78, max
-61.75), with median SNR proxy **44.97 dB** (min 42.74, max 46.40), versus a v1
12-clip median of **-49.17 dBFS** (min -49.73, max -46.31). The calibration
artifact is `results/audit/style_calibration/calibration.json`; it is an
exploratory demo transform, not a serving path.

The human AMI reference median is **-76.11 dBFS** (8 clips), so v2 is still
**12.67 dB above** that reference. Alpha 1.25 was rejected: the selected-trio
median improves by only 0.04 dB (-63.51 vs -63.47 dBFS) while the
Timothee-Jackie ECAPA similarity rises to 0.422. No human MOS/CMOS or
noise-preference rating exists. The full measurements and non-claims are in
`results/audit/AUDIT_REPORT.md`.

The historical diagnostic below holds one face, speaker embedding, text, and
seed fixed, disables watermarking to match the demo, and measures both the raw
backend waveform and the waveform written by the current
`save_audio_sample()`:

```bash
python scripts/audit/diagnose_audio_quality.py \
  --device cuda --steps 5 10 20 \
  --save-audio results/audio_quality \
  --output results/audio_quality_diagnostics.json
```

**Historical diagnostic: one face, raw face style, earlier 15-asset demo.**

| Condition | Steps | Raw noise floor (dBFS) | Raw SNR proxy (dB) | Raw peak | Saved LUFS | Saved noise floor (dBFS) |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 5 | -45.70 | 27.42 | 0.642 | -23.00 | -45.70 |
| whispered | 5 | -43.99 | 22.14 | 0.496 | -25.73 | -43.99 |
| neutral | 10 | -47.43 | 29.00 | 0.625 | -23.00 | -47.43 |
| whispered | 10 | -43.26 | 21.73 | 0.423 | -25.73 | -43.26 |
| neutral | 20 | -46.89 | 28.51 | 0.582 | -23.00 | -46.89 |
| whispered | 20 | -40.58 | 18.88 | 0.439 | -25.73 | -40.58 |
| neutral DSP control | 5 | -48.09 | 29.53 | 0.654 | -23.00 | -48.09 |
| neutral + pitch 1.16 / time 0.92 | 5 | -42.92 | 23.91 | 0.531 | -23.00 | -42.92 |

The current writer changes level by **0.00 dB** in every row. The legacy writer
added **+2.94 dB** to neutral and **+5.17 dB** to whispered at 5 steps, and up
to **+6.55 dB** across the sweep, moving saved loudness to -20.6…-18.4 LUFS.
SNR is unchanged by either scalar gain; the difference is that the legacy
writer made quiet conditions artificially loud and flattened intended energy
differences.

`noise_floor_dbfs` is the 10th percentile of 25 ms frame RMS and `snr_proxy_db`
is the 90th minus the 10th percentile. These are diagnostic proxies, not
calibrated noise measurements or MOS.

The historical diagnostic separates four causes:

- **No clipping, but the raw style prior produced hiss.** Peak stays at or below
  0.654 in the diagnostic (the earlier demo's loudest saved file was 0.732),
  clipping fraction is 0.0, energy above 8 kHz is only 4e-05–0.00097 of total
  power, and
  spectral flatness is 0.0034–0.0141. The residual includes low-level
  vocoder/diffusion noise and DSP smearing; the later controlled matrix isolated
  the raw face-style prior as the dominant hiss source, which the v2 calibration
  reduces. This is not clipping, quantization, or a noisy channel.
- **The double normalization is fixed.** The backend targets -23 LUFS and now
  applies only a -1 dBTP true-peak limiter. `save_audio_sample()` no longer
  peak-normalizes; it only applies the same true-peak safety. The controlled
  A/B shows the legacy writer added up to +6.55 dB, while the current writer
  adds 0.00 dB. Neutral files are saved at -23.00 LUFS; expressive offsets are
  intentional (happy -22.11, sad -24.26, angry -21.70, whispered -25.73,
  surprised -22.25).
- **Within that diagnostic, the librosa DSP was a larger noise source than
  diffusion steps.** The neutral DSP control measures -48.09 dBFS / 29.53 dB;
  the same style with pitch 1.16 and time 0.92 measures -42.92 dBFS / 23.91 dB.
  In that diagnostic the emotion/accent path applied this kind of global
  pitch/time stretch; the later matrix found the raw style prior dominates the
  demo.
- **Step count is not a monotonic quality knob here.** The 3-seed × 2-prompt
  ablation (`results/step_ablation.json`) gives neutral SNR 27.06 ± 1.16 dB at
  5 steps, 27.71 ± 1.62 dB at 10, and 26.99 ± 2.13 dB at 20. 10 steps beat 5 in
  5/6 paired samples (+0.65 dB mean), whispered was a wash (21.81 vs 21.79 dB),
  and 20 steps was clearly worse for whispered (19.81 dB, −2.0 dB). The tier-1
  default is therefore 10, not 20; 20 must not be described as "high
  naturalness."

**Polished delivery (unchanged in v2).** The backend logs
`Invalid or missing model checkpoint path. Loading default model...` and loads
StyleTTS2's default LibriTTS checkpoint, which is trained on audiobook
narration. The v2 demo renders one formal 12-word prompt in about 7.1–8.1 s
(about 1.5–1.7 words/s) with no disfluencies. That is why it sounds like a
careful narrator rather than conversational speech; it is a checkpoint and
prompt-style mismatch, not a post-processing artifact or an emotion/accent
effect. The N1 LoRA did not fix it, and the controlled matrix found worse noise
under N1, so N1 is omitted from v2; the negative result is documented in
`results/audit/AUDIT_REPORT.md` §10.10.

**Hardware-profile bug and demo versions.** `VoixPipeline` used to call
`detect_hardware()` with no config path, so it ignored `configs/hardware.yaml`
and used the fallback 5-step profile. The pipeline and service now resolve the
YAML profile. The v2 research demo was regenerated with an explicit
`--diffusion-steps 10`, seed 42, and the calibrated styles; its metadata records
10 steps and the checked-in audio is reproducible. The multi-seed ablation
still supports 10 as the tier-1 default; 20 steps was worse for whispered
speech. The older 15-asset demo remains at `results/demo/research/` as a fixed
v1 artifact and is not the current research demo.

The pre-phase-0 snapshot was not reproducible from the current checkpoints
despite matching seed/config: the regenerated waveforms differ materially, not
just in gain. It was moved to
`D:\voix_archive\20261105_cleanup\files\results\demo_pre_phase0` during the
approved 2026-11-05 cleanup. The v2 research demo remains the fixed v2 artifact
and v3 is the current research demo; v1 remains at `results/demo/research/` for
comparison. The cleanup preserved per-file SHA-256 records at
`results/audit/cleanup/`.

---

## 7. Inference

```python
from voix.inference.pipeline import VoixPipeline
import soundfile as sf

pipeline = VoixPipeline(device="cuda", enable_watermarking=False)

results = pipeline.generate(
    face_input="face.jpg",
    text="The human voice reflects anatomy, culture, and lived experience.",
    num_voices=3,
    temperature=0.8,
)

for result in results:
    sf.write(
        f"voice_{result['voice_index']}.wav",
        result["waveform"].squeeze().numpy(),
        result["sample_rate"],
    )
```

A 560-D raw fused vector can be passed instead of an image. If it is already
z-scored with the canonical scaler, set `face_input_is_normalized=True`.

Demographic presets are versioned FaceFeatureV2 caches:

```python
results = pipeline.generate_from_preset(
    "young_female_resonant",
    text="A preset-driven voice.",
    num_voices=3,
)
```

Preset caches from the old V1 fusion contract (pairwise cosine ~0.9997) are
rejected and rebuilt automatically.

Supported synthesis controls:

| Control | Argument | Effect |
|---|---|---|
| Duration | `duration_scale` | librosa time-stretch (pitch preserved) |
| Pitch | `f0_scale` / `target_f0` | librosa pitch-shift |
| Loudness | `loudness_lufs` | EBU R128 integrated loudness |
| Diffusion steps | `diffusion_steps` | `None` → hardware profile |
| Accent | `accent`, `accent_strength` | A3 class-contrast style residual (exploratory; external panel at chance) |
| Emotion | `emotion`, `emotion_intensity` | E1 prosody-only class-mean residual (exploratory; external panel at chance) |

**Out of scope:** `speech_difference`/`speech_severity` and `language` remain in
legacy code paths for reproducibility, but they are not supported product
controls. Craniofacial sliders are also out of scope. See [`docs/SCOPE.md`](docs/SCOPE.md).

`blend_weight`, `anchor_path`, `f0_variance_scale`, and LibriSpeech anchor
blending were removed because they were not implemented on the neural path.

---

## 8. Repository map

```
configs/
  environment.yaml        # checkout-relative paths, V2 artifacts, inference defaults
  hardware.yaml           # hardware auto-detection profiles
checkpoints/
  cvae_cfvn_face192_latent128_diversity.pt
  style_adapter_cfvn_best.pt
  emotion_residuals/      # E1 class mean (exploratory)
  accent_residuals/       # A3 class mean (research transform)
data/processed/voxceleb2_v2/
  index.csv               # speaker-disjoint manifest
  features_v2.h5          # FaceFeatureV2 corpus (local)
  face_scaler_v2.pt       # 560-D train-split scaler
data/processed/styletts2_lora/
  manifest.csv            # 432-clip speaker-disjoint LoRA pilot
docs/
  MASTER.md               # consolidated research record
  SCOPE.md                # active scope contract
  AUDIT_REPORT.md         # historical code audit
  voix_banner.jpg
results/
  checkpoints/            # CFVN and experimental checkpoints
  ci_gate_v2.json         # canonical identity gate
  demo/research/          # v1 15-asset demo (historical)
  demo/research_v2/       # v2 21-clip demo (historical)
  demo/research_v3/       # v3 24-clip demo (current)
  emotion/                # E0/E1/E2 findings and C1
  accent/                 # A0/A2/A2c/A3 findings
  evaluation/panel_v2/    # external panel on v2
  evaluation/panel_v3/    # external panel on v3
  audit/                  # audio, style, ECAPA, and final manifests
scripts/
  data/                   # corpus builders and feature extraction
  training/               # identity and expression trainers
  evaluation/             # gates, panels, metrics
  demo/                   # research demos and C1 generator
  audit/                  # audio, style, and data-firewall audits
  maintenance/            # cleanup and manifest utilities
voix/
  data/                   # face features, demographics, presets
  models/                 # CFVN, CVAE, adapter, TTS backend, controls
  training/               # CFVN, CVAE, adapter, and LoRA trainers
  evaluation/             # retrieval, gates, and expression metrics
  inference/              # pipeline, service, CLI, IO
tests/
Initial Docs/             # original literature survey and outline
```

---

## 9. Tests

```bash
python -m pytest -q
```

Artifact-dependent tests skip cleanly when the V2 checkpoints are absent; they
do not silently fall back to random weights. Once the artifacts exist, run the
full suite and the CI gate.

---

## 10. Limitations

- Cross-modal face→voice identity is inherently weak. A face constrains a
  distribution; it does not determine a unique voice. Report retrieval against
  chance, not only raw cosine: CFVN R@1 on the official test split is 0.0932
  against 0.0085 chance. On the v3 demo, pairwise neutral ECAPA cosines are
  0.299, 0.251, and 0.570; the 0.570 pair (C1_02/C2_09) exceeds the maximum
  real same-gender between-speaker cosine (0.438), so the three neutral voices
  fail the demo's distinctness check. The v1 candidate numbers (face→voice
  cosine 0.1019) are historical.
- The adapter's style targets are per-speaker StyleTTS2 `[timbre | prosody]`
  vectors averaged over up to five VoxCeleb2 clips
  (`data/processed/style_targets_voxceleb2_5clip.pt`). The targets are
  speaker-disjoint and aligned with the voice that produced them, but style is
  only partly predictable from voice: the target's centered across-speaker
  cosine is ~0.001, so most residual style variation is orthogonal across
  speakers. The adapter is selected on validation, never test, and its output
  distribution is checked against the target's centered mean and residual
  magnitude. On the official test split its target cosine is 0.874, slightly
  below the 0.898 target-mean baseline: the adapter learns speaker-consistent
  residual variation, but that variation is not fully predictable from voice.
- The FAD implementation is a mel-spectrogram proxy, not the official VGGish
  FAD. Do not compare it to published FAD numbers.
- The StyleTTS2 prosody LoRA pilot passed its pre-registered 104-pair proxy
  gate narrowly: WER +0.0078 stayed inside the +0.05 budget, ECAPA improved,
  and only F0-median absolute error improved (-0.93 Hz) across four speaker
  clusters. There is no MOS, and speech-rate error worsened. The corpus is only
  0.741 h of VoxCeleb2 with Whisper pseudo-transcripts. The adapter is not wired
  into serving; the canonical identity checkpoints remain unchanged.
- Emotion and accent controls are exploratory style residuals, not validated
  controls. E1 (prosody-only class mean, serving scale 0.28) passed 6/6
  pre-registered direction checks on held-out CREMA-D actors. A3 (class
  contrast, scale 0.30, timbre weight 0.25) gained one correctly classified clip
  out of 30 and has zero recall for Indian and Canadian. The external panel
  scores generated emotion and accent at chance (0.250) for all three
  evaluators. Indian G2P is rejected (zero classifier gain), and C1 failed the
  0.90 identity guard. There is no cross-corpus-validated expression model and
  no human MOS/CMOS or identification study. Treat the demo controls as
  research controls, not production features.
- The v1 15-asset showcase is a scoped demonstration, not a benchmark. It has one
  prompt and one intensity per condition, so it cannot establish intensity
  monotonicity, generalization across speakers, or confidence intervals.
- The deterministic fusion layer is a fixed random projection. It preserves the
  representation contract across extraction, training, and serving, but it is
  not a learned identity model.
- VoxCeleb2 is research/non-commercial data. The repository does not grant
  rights to the videos, voices, or derived biometric features.

---

## 11. License

The source code is released under the MIT License. Dataset terms are separate
and take precedence for any use of VoxCeleb2.
