# Identity Phase 3 Pre-registration - Paired Face/Voice Supervision

**Status:** proposed; not approved, not downloaded, not trained.
**Date:** 2026-11-12
**Scope:** the frozen `CFVN -> CVAE -> StyleAdapter -> StyleTTS2` identity path.
Only the adapter candidate and its demo calibration may change. The canonical
adapter, the demo, the fixed gates, and the accent/emotion controls remain
unchanged until a candidate passes.
**Pre-registration rule:** thresholds, faces, splits, objectives, and
hyperparameters in this document are fixed before data extraction. Results may
not be used to move them. A run that changes any of them is a new,
separately pre-registered experiment.

---

## 1. Context and hypothesis

Phase 1 compared eight face representations under a train-only linear map. All
eight failed the pre-registered decision matrix in
`docs/IDENTITY_NEXT_PHASE.md`; the selected `cfvn_es_arcface` input failed the
linear gates and the demo gate. Raw ArcFace improved the centered pair tail but
lost target fidelity. No representation supplied both.

Candidates A and B changed the adapter objective on the same VoxCeleb2
supervision. Candidate B reduced the canonical adapter's validation centered
pair p90 from `0.902` to `0.527` and the eight-face held-out identity maximum to
`0.6415`, but still failed the identity and audio gates. Its worst pairs sit at
the ridge linear ceiling for the same `e_s` input.

Phase 1 therefore separates two questions:

- **target fidelity:** can the adapter reconstruct the real style residual?
- **identity separation:** can distinct faces produce distinct rendered voices?

Phase 2, another objective on the same VoxCeleb2-only targets, is not entered:
the Phase 1 result already shows that the worst pairs sit at the linear ceiling
for the same `e_s`. The remaining in-scope lever is supervision on paired
face/voice identities from a clean, consented or licensed corpus.

**Hypothesis H3:** paired neutral face/voice supervision on an external corpus
teaches the adapter to use identity variation in the frozen serving `e_s` that
VoxCeleb2 style targets did not expose, and the resulting candidate passes the
fixed identity and audio gates on the three selection faces and all eight
held-out faces.

**Falsification:** if the pre-registered candidate trained on the paired corpus
fails either fixed gate, H3 is rejected for this corpus and this fine-tuning
design. The frozen `e_s` path has then not been shown to be fixable by paired
adapter supervision. The next lever is a new encoder/scope decision, not
another adapter objective on the same input.

---

## 2. Fixed constraints

- The serving identity path remains `CFVN -> CVAE -> StyleAdapter ->
  StyleTTS2`. CFVN, CVAE, and StyleTTS2 stay frozen. No new face encoder is
  trained in this phase.
- `checkpoints/style_adapter_cfvn_best.pt` is read-only. It may be used as an
  initialization source, but it is not modified, replaced, or overwritten.
- The four accents (`us`, `england`, `indian_english`, `australian`) and four
  emotions (`neutral`, `happy`, `sad`, `angry`) remain explicit. No default
  conditioning is introduced.
- Evaluation recognizers (ECAPA, ArcFace, and any ASR/identification model) are
  evaluation-only. They are never used to synthesize, select, rank, or filter
  audio.
- The fixed identity gate is unchanged:
  `max neutral pairwise ECAPA cosine <= 0.4378563828374123`.
- The fixed audio floors are unchanged:
  `noise_floor_dbfs <= -58.0` and `snr_proxy_db >= 40.0`.
- The three demo selection faces are `C1_02`, `C2_09`, `C2_12`. The eight
  demo held-out faces are `C1_01`, `C1_03`, `C1_05`, `C2_08`, `C1_04`,
  `C2_07`, `C2_13`, `C2_15`. They are evaluation-only.
- No accent or emotion work starts until a candidate passes both fixed gates on
  all 11 gate faces.
- No MOS, CMOS, identification-accuracy, or perceptual-distinctness claim
  follows from style, ECAPA, or audio proxies.

---

## 3. Data source

### 3.1 Primary corpus: CREMA-D

CREMA-D is the primary paired corpus because it is already partly local, is
distributed under the ODbL 1.0 / DbCL 1.0 terms below, and provides multiple
neutral recordings per actor.

| field | value |
|---|---|
| full name | Crowd-sourced Emotional Multimodal Actors Dataset |
| actors | 91 (48 male, 43 female), ages 20-74 |
| clips | 7,442 original; 6,892 in the local curated manifest |
| neutral clips | 1,087 in the local curated manifest; 11-12 per actor |
| local audio | `data/raw/crema_d/AudioWAV/*.wav` (16 kHz) |
| local metadata | `data/processed/crema_d_manifest.csv`, `data/raw/crema_d/VideoDemographics.csv` |
| local video | **not present**; `VideoFlash/` must be fetched |
| license | ODbL 1.0 (database), DbCL 1.0 (contents) |
| access request | the repository README asks users to complete the CREMA-D access form |
| citation | Cao et al., IEEE Transactions on Affective Computing, 2014 |

The local manifest is authoritative for which clips are usable. It already
records 6,892 clips after the curation gate (`data/processed/crema_d_curation_report.json`).
The neutral subset is 1,087 clips, versus a nominal 1,092 (91 actors x 12
clips). The five-clip difference is whatever the existing curation gate
removed; Phase 3 does not add a new neutral filter.

**Why neutral only.** Phase 3 is about identity, not emotion. All six CREMA-D
emotion classes are acted; mixing them would confound voice identity with
prosody and would introduce an emotion label into the identity path. The 11-12
neutral clips per actor are enough for a stable per-actor style target. No
happy, sad, angry, disgust, or fear clips are used for training or selection.

**Known video caveats.** The CREMA-D README documents neutral video files with
missing or suspect audio (`1076_ITH_NEU_XX.flv`, `1076_TAI_NEU_XX.flv`,
`1076_MTI_NEU_XX.flv`). They are not manually special-cased. The decode and
quality gates below exclude any clip that fails. An excluded video may still
contribute its WAV to the actor-level style target, because the WAV is valid
and the pairing is at actor level, not clip level.

### 3.2 Acquisition

The full CREMA-D repository is about `7.55 GB`. Phase 3 does not need the
non-neutral videos, MP3 files, or duplicate media. Only the neutral video
subset is fetched into the existing clone:

```text
# from data/raw/crema_d, with Git LFS installed
git lfs pull --include="VideoFlash/*_NEU_*.flv"
```

Before fetching:

1. Complete the CREMA-D access form linked from
   `data/raw/crema_d/README.md` and record the submission date in the Phase 3
   provenance JSON.
2. Verify the local `AudioWAV/` and manifest hashes are unchanged.
3. Record the Git LFS object count, byte count, and SHA-256 of every fetched
   `.flv`.
4. Confirm no non-neutral video, MP3, or full-repository checkout was pulled.

If the neutral subset cannot be fetched without the full repository, or if the
access request is refused, Phase 3 uses the GRID backup in section 9 rather
than scraping an alternative source.

### 3.3 Licensing and ethics

- The CREMA-D license is ODbL 1.0 for the database and DbCL 1.0 for its
  contents. The access form is a community record request, not a substitute
  for the license. Complete it anyway.
- Phase 3 is research/demo use. This document is not commercial-use clearance.
  Third-party data remains under its own terms after processing.
- Do not redistribute raw CREMA-D media, derived face crops, or derived voice
  embeddings in the repository. Processed artifacts stay in ignored local
  paths and are described by hashes and provenance.
- The candidate is a derivative of the VoxCeleb2-trained canonical adapter,
  so VoxCeleb2's research/non-commercial terms continue to apply. CREMA-D's
  terms apply to the paired data. Record both in `docs/MASTER.md` and the Phase
  3 `provenance.json`; if a processed artifact is added to
  `configs/artifacts.yaml`, record its license there too.
- Do not scrape a real person's face or voice. Do not use the 19 demo faces as
  training identities or adapter supervision; the 11 non-held-out faces are
  used only by the fixed calibration builder. Public figures are not a license
  to their voices.

### 3.4 Demo faces

The 19 demo faces are evaluation/calibration only. The 11 gate faces are
listed in section 2. They are never used as training identities. The eight
non-gate demo faces are calibration-only. The fixed calibration builder fits on
the 11 non-held-out faces, which are those eight plus the three selection
faces.

A data-hygiene check runs before training: compute the ArcFace centroid over
each CREMA-D actor's neutral-clip face embeddings and the cosine to each of the
19 demo faces. If any actor has a cosine `>= 0.6` to a demo face, exclude that
actor and record the exclusion.
This is a face-overlap diagnostic, not an audio-selection or gate metric. It
cannot prove disjointness, but it prevents an obvious identity collision.

---

## 4. Pre-registered identity split

### 4.1 Official labels

CREMA-D's official split is by actor, not by clip:

| official split | actors | male | female | role in Phase 3 |
|---|---:|---:|---:|---|
| train | 73 | 41 | 32 | source of the 57 training and 16 selection identities |
| val | 9 | 2 | 7 | final paired reporting only |
| test | 9 | 5 | 4 | final paired reporting only |

The official validation split is gender-imbalanced (`2M/7F`). Using it for
checkpoint selection would couple selection to gender. Phase 3 therefore uses
a separate, gender-stratified split inside the official train actors and
reserves both official val and official test for final paired reporting.

### 4.2 Primary split: 57 train / 16 selection / 18 report

All 73 official-train actors are eligible before the section 3.4 face-overlap
screen and the section 5.4 quality gate; actors that fail either screen are
handled by the section 4.3 fallback. The split below is gender-stratified and
deterministic:

```text
For each gender stratum in the official train actors:
  rank actors by SHA-256("voix-phase3-v1|<actor_id>") hex ascending
  take the first 32 male and the first 25 female as training
  assign the remaining 9 male and 7 female to selection
```

This gives:

| role | actors | male | female | neutral clips |
|---|---:|---:|---:|---:|
| training | 57 | 32 | 25 | 680 |
| selection (validation) | 16 | 9 | 7 | 192 |
| official val + test, final report | 18 | 7 | 11 | 215 |
| total | 91 | 48 | 43 | 1,087 |

The 57 training identities exceed the Phase 3 minimum of 19 paired training
identities. They are disjoint by actor from the 16 selection identities and
from the 18 official val/test actors.

The official val/test rows keep their official labels in the reporting
manifest, but are stored as `test` in the adapter corpus so the trainer cannot
select on them. The selection identities are stored as `val`.

The exact actor lists and a split hash are in Appendix A. The hash is
`ff0326e5689b3c82618f1f89ce96a8c40fa6a046196547b5578400bba9af7e9d`, computed
as `sha256("|".join(train_ids) + "||" + "|".join(selection_ids))` where the
IDs are decimal strings and each list is sorted ascending. The split manifest
must be written before feature extraction and must match this hash for the
primary split. If section 4.3 changes the split after extraction, the final
manifest and its new hash are recorded alongside the original; the original is
not overwritten silently, and the readiness check uses the final manifest.

### 4.3 Fallback if actors fail the data-quality gate

The 57/16 split assumes all 73 official-train actors pass. If some fail
section 5.4, the following deterministic fallback applies, in order:

1. Remove failed actors. Re-rank the remaining official-train actors by the
   same hash within each gender.
2. For each gender stratum `g` with `n_eligible_g` actors, set
   `n_train_g = floor(0.78 * n_eligible_g)`. The first `n_train_g` actors in
   hash order are training; the remaining `n_eligible_g - n_train_g` are
   selection. This is applied independently to the male and female strata, so
   the training/selection proportion is preserved within one actor per
   stratum.
3. Require at least 44 training and at least 12 selection identities in total.
   If either minimum fails, stop the CREMA-D attempt and move to GRID.
4. Record the final lists and a new split hash before training. Do not restore
   the original 57/16 counts by borrowing from official val/test.

The minimum of 44 training identities is above the 19-identity Phase 3
requirement and keeps the experiment in the "dozens of identities" regime.
The 18 official val/test actors are never used to backfill.

### 4.4 Leakage controls

- No actor appears in more than one role. The split is by `actor_id`, not by
  clip.
- The style target for an actor is computed only from that actor's neutral
  clips. No target uses another actor's audio.
- The adapter selection metric uses only the selection-role identities (16 in
  the primary split). Test, official val/test, and demo results are reported
  after selection.
- The demo calibration uses only the 11 non-held-out demo faces; the eight
  demo held-out faces are excluded from the fit.
- The canonical adapter, CFVN, CVAE, StyleTTS2, face scaler, and real AMI
  style bank are frozen and hash-checked before extraction.
- The face-overlap check in section 3.4 runs before training. An excluded actor
  is not silently replaced by a demo face.

---

## 5. Paired corpus construction

### 5.1 Face features

Build a FaceFeatureV2-compatible corpus with one row per usable neutral clip:

1. Input: `VideoFlash/<clip_id>.flv` for the neutral clips of actors in the
   frozen split and neutral manifest.
2. Use the frozen `buffalo_l` detector and the existing FaceFeatureV2
   extraction defaults. Do not retune detection size, face-area threshold,
   track length, AV-sync threshold, or frame count for CREMA-D.
3. Sample frames from the clip, select the speaker face track, and average the
   raw ArcFace embedding over the selected recognition frames.
4. Compute craniofacial geometry with the frozen FaceMesh model and demographics
   with the frozen soft estimator. Use the same serving contract as the demo;
   do not substitute CREMA-D ground-truth age/sex into the model input.
5. Fuse to the 560-D FaceFeatureV2 vector and store it **raw**. The canonical
   fused scaler `data/processed/voxceleb2_v2/face_scaler_v2.pt` is applied by
   the trainer and by the section 6 sanity check, exactly as for VoxCeleb2; do
   not store pre-scaled features.
6. Store `face_features` (raw 560-D), `speaker_features` (192-D ECAPA from the
   video audio, required by the corpus contract), `clip_ids`, `speaker_ids`,
   `split`, and `gender`. Set `speaker_id` to the CREMA-D actor ID string and
   map `Male`/`Female` to the VoxCeleb2-compatible `m`/`f` gender strings in
   both the index and HDF5. The reporting manifest keeps `official_split`.
7. Reject clips with no detected face, no decodable video, no audio for the
   ECAPA contract, non-finite features, or inconsistent actor/clip metadata.

The face feature is per clip, not per frame. A clip row is the extracted
FaceFeatureV2 row, not an average over arbitrary frames. The actor centroid is
used only for diagnostics and the face-overlap check. The serving `e_s` is not
stored in the extraction HDF5; section 6 recomputes it from the raw features
through the frozen `CFVN -> CVAE prior mean -> decoder -> L2 normalize` path.

The extraction index is `data/processed/crema_d_phase3/index.csv`. It must
contain at least `clip_id`, `speaker_id`, `split`, `gender`, and `path`, where
`path` is relative to `data/raw/crema_d` (for example
`VideoFlash/1001_DFA_NEU_XX.flv`). `split` is the adapter corpus role
(`train`/`val`/`test`, with both official val and official test stored as
`test`); `official_split` stays in `split_manifest.json` and `provenance.json`.

**Domain shift is explicit.** CREMA-D is studio video with frontal faces; the
demo faces are in-the-wild public-figure images. The canonical scaler, CFVN,
and CVAE were fit on VoxCeleb2. This mismatch is a limitation, not something
Phase 3 hides by retuning the frozen components.

### 5.2 Style targets

Build a `StyleTargetV2` artifact with one row per actor:

1. Input: the curated `AudioWAV/<clip_id>.wav` for the actor's neutral clips.
   Use the WAV, not the compressed video audio track. A clip whose video failed
   may still contribute its WAV to the actor-level target; this is the same
   actor-level pairing rule as section 3.1.
2. Run the frozen StyleTTS2 `compute_style` on each usable WAV and average the
   256-D style vectors over up to 12 clips per actor.
3. Record `speaker_ids`, `clip_ids`, `source_clip_ids`, `n_clips`, `split`
   (the adapter corpus role, `train`/`val`/`test`), `gender` (`m`/`f`, matching
   the corpus), `clips_per_speaker`, and the source hash. Use the existing
   `StyleTargetV2` schema so the adapter trainer can load it unchanged; keep
   `official_split` only in the reporting manifest.
4. Require at least 8 successful style clips per actor and a split-half
   reliability of at least `0.70`. Split-half reliability is the cosine
   between the mean style of the odd-indexed and even-indexed clips after
   sorting by `clip_id`.
5. Compute the training target center and centered norm from the training-role
   clip rows only, matching the trainer's clip-weighted aggregation (each
   actor weighted by its usable clip count). Do not compute them from
   selection, official val/test, or demo data.

The adapter-training center in step 5 is clip-weighted because the trainer
repeats each actor target once per face row. The section 6 diagnostic center is
the unique-actor mean of the training-role targets, matching the Phase 1 audit.
These are two different definitions and must not be interchanged.

The target is actor-level. This is intentional: the goal is a stable identity
style, not a clip-specific prosody label. A clip-level target would fit
sentence and recording noise.

### 5.3 Artifact contract

Proposed local paths (ignored by git):

```text
data/processed/crema_d_phase3/
  index.csv
  features_v2.h5
  style_targets_neutral.pt
  split_manifest.json
  provenance.json
  extraction_failures.csv
```

`provenance.json` records: CREMA-D access-form date, license text hashes,
fetched LFS object count/bytes, SHA-256 of each source artifact, the canonical
CFVN/CVAE/scaler/adapter hashes, the split hash, extraction parameters, and
the software commit. No raw media, face crops, or derived biometric vectors
are committed.

### 5.4 Data-quality gates

An actor is eligible only if all of the following hold:

| gate | requirement |
|---|---|
| face clips | at least 8 neutral clips with a valid FaceFeatureV2 row |
| face stability | at least 6 clips with `frames_used >= 4`; median `frames_used >= 4` |
| style clips | at least 8 neutral WAVs with a successful style extraction |
| style reliability | odd/even split-half style cosine `>= 0.70` |
| finite features | no NaN/Inf in `face_features`, `e_s`, or the style target |
| identity hygiene | ArcFace cosine to every demo face `< 0.6` |

A clip may fail the video/face path while its WAV still contributes to the
actor's style target. That is allowed because the pairing is actor-level and
the target is a voice property. The reverse is not allowed: a clip cannot
contribute a face row without a valid style target for its actor.

If fewer than 44 training or 12 selection identities survive, the CREMA-D
attempt is aborted before training. The failure report lists every excluded
actor and the reason.

---

## 6. Pre-training encoder sanity check

This check is a go/no-go for spending training compute. It is **not** a
promotion gate and does not change the fixed gates. It uses train and selection
identities only; official val/test and demo faces are not touched.

Fit a train-only ridge map from the serving `e_s` to the CREMA-D actor style
targets. Use the same preprocessing and aggregation as Phase 1:

- Recompute serving `e_s` from the raw `face_features` with the frozen scaler,
  CFVN, CVAE prior mean, decoder, and L2 normalization. Verify parity against
  the Phase 1 reference path with the same check.
- Aggregate to one row per actor: average the per-clip `e_s` within the actor
  and L2-normalize, matching Phase 1's speaker centroids. Associate the actor
  style target.
- Fit the ridge on the training-actor rows only, with standardization computed
  on those rows. The target is the actor style target.
- Select the ridge lambda on the selection-actor rows by residual cosine over
  the fixed Phase 1 grid `[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`. Do not extend
  the grid after seeing selection results.
- Evaluate at actor level: compare each predicted selection-actor style to its
  actor target, then report residual cosine, centered pair p90, and
  `frac_gt_0.8` for the selection actors.
- Compute the canonical adapter's predictions on the same selection actors and
  targets as the baseline. The canonical adapter is frozen and is not refit.

Definitions used by the checks:

- **residual cosine:** the mean cosine between `prediction - center` and
  `target - center` over the selection actors, where `center` is the
  unique-actor mean of the training-role targets. Higher is better.
- **centered pair p90:** for a set of one vector per actor, subtract the
  unique-actor training target mean from each vector, then take the 90th
  percentile of cosine similarity over all distinct actor pairs. S1 applies it
  to training-role targets; S4/S5 apply it to the ridge predictions on
  selection actors.
- **`frac_gt_0.8`:** the fraction of those same pairs above `0.8`.

All five checks must pass:

| check | threshold | meaning |
|---|---|---|
| S1 target separability | real training-role target centered pair p90 `<= 0.60` | the corpus has enough identity style variation |
| S2 absolute linear signal | selection residual cosine `>= 0.35` | `e_s` carries usable paired signal |
| S3 improvement over canonical | ridge residual cosine `>= canonical + 0.05` | the linear map finds signal the canonical adapter misses |
| S4 tail improvement | ridge centered p90 `<= canonical p90 - 0.05` | the improvement is in the collapsed tail, not only the mean |
| S5 absolute tail | ridge centered p90 `<= 0.65` | the target geometry is not hopelessly collapsed |

If S1 fails, the neutral CREMA-D targets are too similar for a voice-identity
experiment; do not train. If S2-S5 fail, the frozen `e_s` does not expose the
paired signal under a linear probe; a nonlinear adapter is not justified by
this screen. In either case, record the report and move to GRID (section 9) or
stop. No threshold in this table may be changed after seeing the features.

The ridge is a diagnostic, not a serving model. Lambda is selected and the
residual is reported on the same selection actors, so these checks are
optimistic screens, not unbiased estimates. No audio is synthesized in this
check.

---

## 7. Candidate training

### 7.1 Candidate

**Name:** `style_adapter_crema_paired_v1`

One candidate only. No objective variants, scalar sweeps, seed sweeps, or
post-hoc checkpoint averaging.

### 7.2 Initialization

Initialize from `checkpoints/style_adapter_cfvn_best.pt` (read-only). Add an
`--init-adapter` option to the existing trainer if it is not present. The load
order is fixed:

1. Verify the canonical checkpoint SHA-256 against `configs/artifacts.yaml`.
2. Construct the `StyleAdapter` with the candidate dimensions and
   `whiten_input=True`.
3. Load the canonical `model_state_dict` with `strict=False`. The missing keys
   must be exactly `input_mean` and `input_whiten`; there must be no unexpected
   keys. If the candidate were built without whitening, the load must be
   strict, because no buffer may be missing.
4. Fit the train-only whitening transform and install `input_mean` and
   `input_whiten`.
5. Overwrite `timbre_head.bias` and `prosody_head.bias` with the CREMA-D
   training-role target center, then freeze both biases when
   `--freeze-output-bias` is set.
6. Build the optimizer only after steps 1-5.

The canonical output biases are therefore loaded and then deliberately
replaced; the canonical checkpoint itself is never written. Re-verify its
SHA-256 after the run completes. Any missing key other than the two whitening
buffers, any unexpected key, or any shape mismatch is a hard error.

Fine-tuning is deliberate: a few dozen training identities are too few to
learn the whole style manifold from scratch, and the canonical adapter already
carries target-fidelity knowledge. The experiment tests **paired-data
fine-tuning**, not from-scratch paired training. A from-scratch paired model is
a separate hypothesis and is not authorized here.

### 7.3 Input and targets

- Training input: CVAE prior samples decoded from the frozen CFVN embedding,
  L2-normalized to the serving `e_s` distribution, with `--adapter-input cvae`,
  `--samples-per-clip 4`, `--cvae-temperature 1.0`. This matches the canonical
  trainer and the serving distribution.
- Validation input: deterministic CVAE prior-mean decoder output, L2-normalized.
- Face scaling: use the canonical VoxCeleb2 scaler
  `data/processed/voxceleb2_v2/face_scaler_v2.pt` unchanged. The existing
  trainer rejects a scaler whose `n_train` or `n_train_speakers` does not match
  the training corpus. Add an explicit `--allow-external-scaler` flag that
  bypasses only those two same-corpus count checks; schema, `face_key`,
  dimension, and the `configs/artifacts.yaml` SHA-256 check still apply. The
  checkpoint records `scaler_mode: "external_frozen"` and the scaler hash. No
  scaler is refit on CREMA-D.
- Input whitening: `--whiten-input --whiten-shrink 0.05`, fit on the training
  actors only. The whitening source is the prior-mean `e_s` of the training
  clips, as in the existing trainer.
- Targets: the CREMA-D actor-level `StyleTargetV2` artifact.
- The style target center and centered norm come from the training role only.

### 7.4 Objective

Use Candidate B's objective unchanged:

```text
--residual-objective
--center-output-bias
--freeze-output-bias
--norm-loss-mode symmetric
--lambda-div 1.0
--lambda-norm 1.0
--lambda-mean 1.0
--lambda-pair 1.0
--selection residual
```

The residual objective removes the common style mean from the MSE/cosine
target. The output bias is initialized to the CREMA-D training center and
frozen. The symmetric norm term prevents residual shrinkage. The pairwise
structure term targets the high-similarity tail that the mean diversity term
leaves unconstrained.

No new loss is added. All unlisted objective weights use the trainer defaults:
`--lambda-cos 1.0`, `--lambda-var 0.01`, and `--target-norm-scale 1.0`. This
keeps the comparison to Candidate B clean: the only changes are the paired
corpus, the initialization, and the lower learning rate.

### 7.5 Hyperparameters

| parameter | value |
|---|---|
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `1e-3` |
| batch size | `128` |
| epochs | `30` |
| patience | `10` |
| scheduler | cosine annealing to `1e-6` |
| seed | `42` |
| dropout | `0.0` |
| hidden dimension | `256` |
| device | CUDA if available, otherwise CPU with the reason logged; no silent OOM retry |

The lower learning rate is for fine-tuning a pretrained adapter. If the run
produces NaN/Inf, stop and report it. Do not silently lower the rate, change
the batch size, or restart with another seed.

### 7.6 Selection

The selection metric is the existing residual selection loss:

```text
-val_residual_cosine
+ 1.0 * val_diversity_loss
+ 1.0 * val_centered_norm_loss
+ 1.0 * val_pairwise_structure_loss
```

Selection uses the selection-role identities only. The best epoch is the lowest
selection loss. Official val/test and demo faces are not evaluated during
training. There is no early stop on demo audio, ECAPA, or style-cosine gates.

The candidate artifacts record:

- candidate name and epoch in the checkpoint;
- canonical source SHA-256;
- training/selection split hash;
- whitening mean/matrix;
- target center and centered norm;
- objective flags and all hyperparameters;
- the full selection history in the sibling `.history.json`;
- the code commit.

---

## 8. Evaluation and promotion

### 8.1 Calibration

Build a candidate-specific demo calibration with
`scripts/training/build_adapter_style_calibration.py`:

- adapter: the candidate checkpoint;
- face source: `results/audit/face_voice_diversity/face_voice_diversity.json`
  (the frozen 19-face cached `e_s` report);
- exclude the eight demo held-out faces;
- fit on the 11 non-held-out demo faces;
- `--alpha 1.0`; no deviation-gain sweep;
- output: `results/audit/style_calibration/calibration_crema_paired_v1.json`.

The calibration method is unchanged: gender-matched mean translation to the
real AMI style bank. Only the adapter weights change. The eight held-out faces
are excluded from the fit, so the held-out evaluation is not calibrated on
itself.

### 8.2 Fixed gate

Evaluate the candidate on the three demo selection faces and the eight demo
held-out faces with the existing gate evaluator, unchanged, in two separate
runs so each report's max pair is the relevant subset max:

- neutral-only synthesis with no accent conditioning; the canonical serving
  scales (emotion `0.28`, accent `0.30`, timbre weight `0.25`) remain unchanged
  in the demo but are not exercised by this neutral gate;
- watermark off for measurement; the demo watermark remains opt-in;
- the same TTS renderer, diffusion steps, and audio metrics as the Candidate
  A/B gate;
- the same face images as the canonical baseline; the three-face run uses the
  selection order and the eight-face run uses the held-out order in section
  8.2.1.

The identity evaluator and the audio-quality audit run on the same generated
WAV files. Neither tool selects, ranks, or filters audio; they measure the
already-generated candidate output. The four gate rows below are the promotion
decision.

Required for promotion:

| gate | requirement |
|---|---|
| identity, selection faces | max neutral pairwise ECAPA cosine `<= 0.4378563828374123` on `C1_02`, `C2_09`, `C2_12` |
| identity, held-out faces | max neutral pairwise ECAPA cosine `<= 0.4378563828374123` on the eight held-out faces |
| audio, selection faces | every clip `noise_floor_dbfs <= -58.0` and `snr_proxy_db >= 40.0` |
| audio, held-out faces | every clip `noise_floor_dbfs <= -58.0` and `snr_proxy_db >= 40.0` |

All four rows must pass. A candidate that passes only the selection faces or
only the held-out faces is not promoted. The thresholds are not moved.

#### 8.2.1 Gate face labels (fixed order)

The first three labels are the selection faces for step 8a; the last eight are
the held-out faces for step 8b. These are the exact `--faces` values:

```text
C1_02_young_male_Timothee_Chalamet
C2_09_east_asian_male_Jackie_Chan
C2_12_black_female_Lupita_Nyongo
C1_01_young_female_Zendaya
C1_03_adult_female_Sandra_Bullock
C1_05_senior_female_Helen_Mirren
C2_08_south_asian_female_Priyanka_Chopra
C1_04_adult_male_George_Clooney
C2_07_south_asian_male_Shah_Rukh_Khan
C2_13_caucasian_male_Brad_Pitt
C2_15_latino_male_Pedro_Pascal
```

### 8.3 Paired reporting

After checkpoint selection, report the following on the 18 official val/test
actors. These are descriptive and do not select the checkpoint or promote the
candidate:

- style-target residual cosine and centered pair p90;
- fraction of centered pairs above `0.5` and `0.8`;
- generated-audio ECAPA similarity to the same actor's real neutral audio and
  to other actors, clearly labeled as an evaluation proxy;
- the same metrics for the canonical adapter baseline.

The paired reporting is useful because CREMA-D has real voices for the same
faces. It is not a substitute for the demo gate, and it is not a perceptual
study.

### 8.4 Promotion

Promotion is a separate step after all four gate rows pass:

1. Keep a versioned copy of the current canonical checkpoint and its hash.
2. Save the candidate as `checkpoints/style_adapter_crema_paired_v1.pt`.
3. Update `configs/artifacts.yaml`, the demo calibration reference, and the
   documentation in one reviewed change.
4. Re-run the demo smoke test and the artifact verifier.
5. Only then update the canonical path and the demo.

If the candidate fails, no checkpoint, calibration, demo, or documentation
claim changes. The failure is recorded in this document's result ledger.

---

## 9. Backup corpus: GRID

GRID is activated only if CREMA-D access fails, the section 4.3/5.4
data-quality gates leave too few identities, or the pre-training gates in
section 6 fail. It is not downloaded or trained in parallel.

| field | value |
|---|---|
| corpus | GRID audiovisual sentence corpus |
| talkers | 34 (18 male, 16 female); video missing for talker 21, so at most 33 paired talkers |
| utterances | 1,000 per talker, fixed six-word command grammar |
| audio | 25 kHz endpointed WAV |
| video | MPG, normal or high quality |
| license | CC BY 4.0 (Zenodo DOI `10.5281/zenodo.3625687`); attribution required |
| local status | not downloaded |

GRID's fixed grammar and read speech may produce less style diversity than
CREMA-D. The same section 5 quality gates and section 6 sanity gates apply; no
threshold is relaxed for GRID.

Proposed split, frozen before download once the 33 video talkers' genders are
resolved from the corpus metadata:

- 19 training talkers: the first 10 male and first 9 female in hash order;
- 6 selection talkers: the next 3 male and next 3 female in hash order;
- 8 final-reporting talkers: the remaining talkers, at least 3 male and 3
  female.

Rank within each gender stratum by
`sha256("voix-phase3-grid-v1|<talker_id>")` hex ascending before assigning the
slots. Record the exact lists and a split hash before download. The 11 demo
gate faces remain evaluation-only and are never GRID training talkers.

If GRID also fails the pre-training gates, stop. The next step is a scope
decision on a consented internal collection or a synthetic paired corpus with
explicit limitations. Do not run another adapter objective on the same `e_s`.

---

## 10. Stop conditions and result ledger

The run stops, without a gate change, if any of the following occurs:

| condition | action |
|---|---|
| access form refused or neutral videos unavailable | use GRID |
| license review finds a use restriction incompatible with the research/demo scope | use GRID or stop |
| fewer than 44 training or 12 selection CREMA-D identities survive | use GRID |
| any section 6 check fails | do not train; use GRID or stop |
| NaN/Inf or a training crash | stop; report; no silent retry |
| identity or audio gate fails | no promotion; record; stop the `e_s` adapter line |
| any gate, face, split, or calibration change after seeing results | invalidates the run |

The result ledger is appended here after each stage:

| stage | status | artifact |
|---|---|---|
| data acquisition | not started | - |
| split freeze | not started | - |
| corpus extraction | not started | - |
| data-quality gate | not started | - |
| encoder sanity gate | not started | - |
| candidate training | not started | - |
| demo calibration | not started | - |
| fixed gate | not started | - |
| decision | not started | - |

---

## 11. Limitations

- CREMA-D actors are not the demo public-figure distribution. Studio, frontal,
  acted neutral video differs from in-the-wild demo images. Paired supervision
  on CREMA-D may not transfer.
- CREMA-D actor identities are anonymized. The ArcFace overlap screen is a
  biometric heuristic, not proof that no CREMA-D actor is one of the 19 demo
  public figures. The demo faces are never used as training identities, and an
  actor above the screen threshold is excluded.
- The canonical adapter was trained on VoxCeleb2, and `C2_07` has a nearest
  ArcFace cosine of `0.7466` to a VoxCeleb2 train face. Its held-out status is
  therefore not proven. Phase 3 does not retrain on VoxCeleb2, but the
  canonical initialization may carry that overlap; the `C2_07` gate result must
  be read with this caveat.
- The frozen `e_s` is still the input bottleneck. Paired supervision can teach
  the adapter to use existing variation; it cannot add identity information
  that `e_s` does not contain.
- A few dozen training identities is small. Fine-tuning reduces overfitting
  risk but does not eliminate it. The selection-role identities (16 in the
  primary split) are the only checkpoint-selection signal.
- CREMA-D neutral read speech may have lower style diversity than in-the-wild
  speech. The S1 target-separability gate can fail before training; that is a
  valid negative result, not a reason to lower the threshold.
- The canonical fused scaler, CFVN, and CVAE were fit on VoxCeleb2. Applying
  them to CREMA-D is a domain shift. The phase does not retune them.
- The soft demographic estimator is applied to CREMA-D faces even though
  ground-truth age/sex exist, because serving uses the estimator. This may add
  noise.
- ECAPA, style cosine, noise floor, and SNR are proxies. A gate pass is not a
  perceptual guarantee. No MOS, CMOS, identification-accuracy, or
  perceptual-distinctness claim is made.
- The candidate inherits VoxCeleb2 research/non-commercial terms and CREMA-D
  ODbL/DbCL terms. No commercial clearance is implied.
- The 11 demo gate faces are a small evaluation set. A pass on them is not a
  speaker-generalization benchmark.

---

## 12. Approval checklist

- [ ] User approves the Phase 3 plan and the CREMA-D data acquisition.
- [ ] CREMA-D access form completed; date recorded.
- [ ] License review recorded in `docs/MASTER.md` and `provenance.json`; any
      new artifact entry carries its license.
- [ ] Only neutral `VideoFlash/*_NEU_*.flv` fetched; LFS object hashes recorded.
- [ ] Split manifest written and hash matches Appendix A (or the documented
      fallback).
- [ ] Canonical checkpoint, CFVN, CVAE, scaler, and style-bank hashes verified.
- [ ] New extraction/style/split scripts and the `--init-adapter` /
      `--allow-external-scaler` trainer paths have focused tests.
- [ ] Data-quality gate passes; exclusions recorded.
- [ ] Encoder sanity gate passes; report saved.
- [ ] Candidate training completes; selection history saved.
- [ ] Candidate calibration fits on the 11 non-held-out demo faces only.
- [ ] Fixed gate runs on the three selection and eight held-out faces.
- [ ] Decision and all reports recorded; no gate or face changes.

---

## Appendix A - Frozen CREMA-D split

Training actors (57):

```text
1002 1003 1005 1006 1007 1008 1009 1011 1012 1013 1015 1018 1021 1022
1024 1027 1029 1031 1033 1034 1035 1036 1038 1039 1041 1042 1043 1044
1046 1048 1049 1050 1052 1053 1057 1058 1059 1060 1061 1062 1063 1064
1065 1066 1069 1073 1076 1077 1079 1080 1081 1085 1086 1087 1088 1090
1091
```

Selection actors (16):

```text
1001 1004 1016 1017 1023 1025 1030 1032 1054 1055 1056 1067 1068 1070
1071 1082
```

Official val actors (9; final reporting only):

```text
1010 1026 1028 1037 1074 1075 1078 1083 1084
```

Official test actors (9; final reporting only):

```text
1014 1019 1020 1040 1045 1047 1051 1072 1089
```

Split hash:
`ff0326e5689b3c82618f1f89ce96a8c40fa6a046196547b5578400bba9af7e9d`.

The hash input is
`"|".join(sorted(training_ids)) + "||" + "|".join(sorted(selection_ids))`
with decimal-string actor IDs and no trailing separator.

The gender-stratified hash rule is:

```python
sha256(f"voix-phase3-v1|{actor_id}").hexdigest()
```

with male actors ranked first for the 32 training slots and female actors
ranked for the 25 training slots; the remainder of each official-train gender
stratum is selection.

---

## Appendix B - Planned commands

These are the commands the implementation must reproduce. They are not run
until the approval checklist is complete.

```text
# 1. Fetch only neutral video (from data/raw/crema_d)
git lfs pull --include="VideoFlash/*_NEU_*.flv"

# 2. Build the frozen split manifest and the extraction index; verify the hash
python scripts/data/build_crema_phase3_split.py \
    --manifest data/processed/crema_d_manifest.csv \
    --output data/processed/crema_d_phase3/split_manifest.json \
    --index-output data/processed/crema_d_phase3/index.csv

# 3. Extract face features (existing FaceFeatureV2 path; frozen defaults)
python scripts/data/extract_voxceleb2_v2.py \
    --index data/processed/crema_d_phase3/index.csv \
    --videos data/raw/crema_d \
    --output data/processed/crema_d_phase3/features_v2.h5 \
    --failures data/processed/crema_d_phase3/extraction_failures.csv

# 4. Extract actor-level style targets from the neutral WAV files
python scripts/data/extract_style_targets_crema.py \
    --manifest data/processed/crema_d_manifest.csv \
    --split-manifest data/processed/crema_d_phase3/split_manifest.json \
    --output data/processed/crema_d_phase3/style_targets_neutral.pt \
    --clips-per-speaker 12

# 5. Data-quality and encoder-sanity gates
python scripts/audit/check_crema_phase3_readiness.py \
    --features data/processed/crema_d_phase3/features_v2.h5 \
    --index data/processed/crema_d_phase3/index.csv \
    --split-manifest data/processed/crema_d_phase3/split_manifest.json \
    --style-targets data/processed/crema_d_phase3/style_targets_neutral.pt \
    --scaler data/processed/voxceleb2_v2/face_scaler_v2.pt \
    --cfvn results/checkpoints/cfvn_v2_best.pt \
    --cvae checkpoints/cvae_cfvn_face192_latent128_diversity.pt \
    --demo-face-report results/audit/face_voice_diversity/face_voice_diversity.json \
    --output results/audit/crema_phase3/readiness.json

# 6. Train the single candidate (after gates pass)
python -m voix.training.trainer_adapter_cfvn \
    --features data/processed/crema_d_phase3/features_v2.h5 \
    --index data/processed/crema_d_phase3/index.csv \
    --face-key face_features \
    --scaler data/processed/voxceleb2_v2/face_scaler_v2.pt \
    --allow-external-scaler \
    --style-targets data/processed/crema_d_phase3/style_targets_neutral.pt \
    --cfvn results/checkpoints/cfvn_v2_best.pt \
    --cvae checkpoints/cvae_cfvn_face192_latent128_diversity.pt \
    --adapter-input cvae --samples-per-clip 4 --cvae-temperature 1.0 \
    --init-adapter checkpoints/style_adapter_cfvn_best.pt \
    --whiten-input --whiten-shrink 0.05 \
    --residual-objective --center-output-bias --freeze-output-bias \
    --norm-loss-mode symmetric \
    --lambda-div 1.0 --lambda-norm 1.0 --lambda-mean 1.0 --lambda-pair 1.0 \
    --selection residual \
    --lr 1e-4 --weight-decay 1e-3 --batch-size 128 \
    --epochs 30 --patience 10 --seed 42 \
    --output checkpoints/experiments/style_adapter_crema_paired_v1.pt

# 7. Candidate-specific calibration (exclude the eight held-out faces)
python scripts/training/build_adapter_style_calibration.py \
    --adapter checkpoints/experiments/style_adapter_crema_paired_v1.pt \
    --exclude-faces C1_01 C1_03 C1_05 C2_08 C1_04 C2_07 C2_13 C2_15 \
    --alpha 1.0 \
    --output results/audit/style_calibration/calibration_crema_paired_v1.json

# 8a. Fixed identity gate, three selection faces only
python scripts/demo/synthesize_identity_candidate.py \
    --adapter checkpoints/experiments/style_adapter_crema_paired_v1.pt \
    --output-dir results/demo/identity_crema_paired_v1_selection \
    --face-report results/audit/face_voice_diversity/face_voice_diversity.json \
    --faces C1_02_young_male_Timothee_Chalamet C2_09_east_asian_male_Jackie_Chan C2_12_black_female_Lupita_Nyongo \
    --calibration results/audit/style_calibration/calibration_crema_paired_v1.json \
    --calibration-alpha 1.0 --calibration-deviation-gain 1.0 \
    --ecapa-reference results/audit/ecapa_reference/ecapa_reference.json \
    --diffusion-steps 10 --seed 42 \
    --alpha 0.18 --beta 0.22 --loudness-lufs -23.0

# 8b. Fixed identity gate, eight held-out faces only
python scripts/demo/synthesize_identity_candidate.py \
    --adapter checkpoints/experiments/style_adapter_crema_paired_v1.pt \
    --output-dir results/demo/identity_crema_paired_v1_holdout8 \
    --face-report results/audit/face_voice_diversity/face_voice_diversity.json \
    --faces C1_01_young_female_Zendaya C1_03_adult_female_Sandra_Bullock C1_05_senior_female_Helen_Mirren C2_08_south_asian_female_Priyanka_Chopra C1_04_adult_male_George_Clooney C2_07_south_asian_male_Shah_Rukh_Khan C2_13_caucasian_male_Brad_Pitt C2_15_latino_male_Pedro_Pascal \
    --calibration results/audit/style_calibration/calibration_crema_paired_v1.json \
    --calibration-alpha 1.0 --calibration-deviation-gain 1.0 \
    --ecapa-reference results/audit/ecapa_reference/ecapa_reference.json \
    --diffusion-steps 10 --seed 42 \
    --alpha 0.18 --beta 0.22 --loudness-lufs -23.0

# 9. Audio floors on the same gate WAVs (same measure_clip path as Candidate A/B)
python scripts/audit/audit_audio_quality.py \
    --audio-dir results/demo/identity_crema_paired_v1_selection \
    --output-dir results/audit/crema_phase3/audio_selection \
    --label crema_paired_v1_selection --no-spectrograms
python scripts/audit/audit_audio_quality.py \
    --audio-dir results/demo/identity_crema_paired_v1_holdout8 \
    --output-dir results/audit/crema_phase3/audio_holdout8 \
    --label crema_paired_v1_holdout8 --no-spectrograms
```

The step 2 script must verify its generated lists against the Appendix A hash
and fail closed on a mismatch. The step 5 readiness report contains both the
section 5.4 data-quality result and the section 6 encoder-sanity result; a
failed check stops the run before step 6.

The exact CLI flags in steps 3, 4, and 6 depend on the current scripts and the
new `--init-adapter` and `--allow-external-scaler` options; verify them against
`--help` before execution. The pre-registration fixes the data, split,
objective, hyperparameters, faces, and thresholds; it does not authorize
changing the evaluator defaults. Steps 8a/8b use the same
`synthesize_identity_candidate.py` gate path as the Candidate A/B runs. Step 9
uses the same `audit_audio_quality.py::measure_clip` function as the Candidate
A/B audio checks. The 11 full face labels in section 8.2.1 must match the
frozen `results/audit/face_voice_diversity/face_voice_diversity.json` report
and must not be substituted.
