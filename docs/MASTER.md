# VOIX Master Research Record

**Status:** research/demo record, last updated 2026-11-07.
**Scope:** face-to-voice identity plus four explicit emotion and four explicit
accent controls. This is not a production system, an identity-verification
system, or an impersonation tool.

This file is the consolidated record for architecture, data access, measured
evidence, rejected options, reproduction, and non-claims. The top-level
[`README.md`](../README.md) is the shorter entry point. A short status index is
in [`results/RESEARCH_STATUS.md`](../results/RESEARCH_STATUS.md). Older planning
files (`BASELINE_REPORT_PHASE_0.md`, `DATASET_ACCESS.md`,
`EMOTION_ACCENT_PROD_PLAN.md`, `PARALLEL_RESEARCH_PLAN.md`) were merged here so
there is one authoritative research record.

**Evidence levels**

| Level | Meaning | Allowed claim |
|---|---|---|
| Exploratory | Pilot corpus, proxy metric, one seed, or small speaker set | Directional research result only |
| Confirmatory | Speaker-disjoint corpus above the size gate, pre-registered gates, multiple seeds, held-out test untouched during selection | Generalization claim inside the measured domain |
| Deployment | Confirmatory plus human MOS/CMOS and identification, license review, serving parity | Product claim for the validated population |

No VOIX emotion or accent condition has reached confirmatory or deployment
level. No human MOS, CMOS, emotion-identification, accent-identification, or
voice-triad study exists.

---

## 1. Scope and invariants

### 1.1 In scope

- Candidate generation: face image or precomputed 560-D FaceFeatureV2 vector to
  a distribution of plausible voices.
- Emotions: `neutral`, `happy`, `sad`, `angry`.
- Accents: `us`, `england`, `indian_english`, `australian`.
- Evaluation-only research artifacts may contain additional GLOBE classes
  (Canadian, Irish) inherited from the accent corpus. They are not exposed as
  serving controls.
- Objective evaluation, external evaluator panels, and reproducible artifacts.

### 1.2 Out of scope

- Cross-lingual synthesis.
- Speech-difference or speech-severity simulation.
- Craniofacial style modulation and slider controls. The deterministic
  craniofacial ratio computation remains part of the face-feature contract; it
  is not a synthesis control.
- Face-inferred emotion or accent. Both are explicit user selections or
  explicit demo conditions.
- Recognition models as controllers. All recognizers are evaluation-only.
- Commercial use of VoxCeleb2 or other research-only data.

### 1.3 Serving rules

- No default accent, no default emotion, and no implicit conditioning in the
  CLI or Python API. Omit the argument to get the neutral path.
- The canonical face-to-voice path is frozen:
  `CFVN -> CVAE -> StyleAdapter -> StyleTTS2`.
- Expression controls are style residuals or text transforms applied around that
  path. They do not modify canonical checkpoints.
- The validated serving scales are emotion residual `0.28`, accent residual
  `0.30`, and accent timbre weight `0.25`. The v2 demo used exploratory
  `0.56/1.0`; that scale is not serving guidance.
- A demo or research artifact must label every expression condition
  `exploratory_application` unless a confirmatory study exists.

---

## 2. Architecture and canonical artifacts

### 2.1 Identity path

```
face image
  -> ArcFace 512-D identity
  -> MediaPipe craniofacial geometry 32-D
  -> demographic posterior 16-D
  -> per-modality L2 normalization + deterministic fusion
  -> 560-D FaceFeatureV2
  -> train-split z-score scaler
  -> CFVN 560 -> 192-D ECAPA-aligned voice identity
  -> CVAE p(z | face) -> decoder -> K candidate voices
  -> StyleAdapter 192 -> 256-D StyleTTS2 style
  -> StyleTTS2 (frozen) -> 24 kHz waveform
```

The first 128 style dimensions are timbre; the second 128 are prosody.
The CVAE is intentionally one-to-many. The adapter is trained on sampled CVAE
decoder outputs, which are the distribution serving uses; checkpoint selection
uses the deterministic prior mean. Style calibration for the demo is a
gender-matched mean translation; it is not part of serving.

### 2.2 Canonical checkpoints

| Artifact | Path | SHA-256 |
|---|---|---|
| CFVN | `results/checkpoints/cfvn_v2_best.pt` | `0253bbbf96def21ca9a3a9df3a4466c1e1538ef36986d9eb9fe86bfabd7b2f6b` |
| CVAE | `checkpoints/cvae_cfvn_face192_latent128_diversity.pt` | `bb7ad29b7b5cebb2141073892b9d5736d2e996c0bd3e81acc4c671c1ea66656b` |
| Style adapter | `checkpoints/style_adapter_cfvn_best.pt` | `97750ee84e50f9878e3d98de6b2e3dc0ca6829f448f2a5e4a31185adafc0a69c` |
| Face scaler | `data/processed/voxceleb2_v2/face_scaler_v2.pt` | recorded by the corpus gate |

The demo writer verifies these hashes before and after generation. Experimental
adapters live under `results/checkpoints/experimental/` and are never loaded by
`VoixPipeline` by default.

### 2.3 Expression artifacts

| Artifact | Path | Role |
|---|---|---|
| Emotion class mean | `checkpoints/emotion_residuals/class_mean.pt` | E1 prosody-only residual; accepted exploratory baseline |
| E2 learned adapters | `checkpoints/emotion_adapter/` | Rejected; retained for reproducibility |
| Accent class mean | `checkpoints/accent_residuals/class_mean.pt` | A3 class-contrast residual; research transform |
| Accent classifier | `checkpoints/accent_classifier_globe.joblib` | A0 proxy evaluator only |
| Style calibration | `results/audit/style_calibration/calibration.json` | Demo-only gender-matched translation |

The accent artifact uses GLOBE names (`us`, `england`, `australian`,
`canadian`, `india_south_asia`). `accent_residual_key()` maps the serving label
`indian_english` to `india_south_asia`. `canadian` remains in the artifact for
research provenance but is not exposed by the CLI.

---

## 3. Data access, licensing, and consent

**This section is not commercial-use clearance.** Research-only datasets remain
research-only after any processing VOIX performs.

### 3.1 Datasets

| Dataset | Role | License | Size | Access status |
|---|---|---|---|---|
| VoxCeleb2 | Identity training/evaluation; StyleTTS2 LoRA pilot | Research/non-commercial | 35,892 clips / 6,111 speakers locally; LoRA pilot 432 clips / 263 speakers / 0.741 h | Present; original terms apply |
| CREMA-D | Emotion pilot/evaluation | ODbL 1.0 | 7,442 clips / 91 actors / 5.26 h raw; 6,892 kept / 4.869 h | Downloaded; below 2 h/class gate |
| GLOBE_V3 | Accent training/evaluation | CC0 1.0 | 723,532 rows; 9,534 clips / 12.281 h after audit | Selected subset downloaded |
| Common Voice 21 metadata | Recover `client_id` for speaker-disjoint GLOBE splits | CC0 1.0 | `validated.tsv` + `clip_durations.tsv` | Metadata from public mirror; replace with official MDC copy when credentials exist |
| AMI | Naturalness pilot | CC BY 4.0 | 3,315 clips / 16 speakers / 4.25 h local pilot | Present |
| MSP-Podcast | Confirmatory emotion corpus | Academic license, institutional signature | 264,705 turns / 409 h | **Blocked**: agreement not executed |
| Expresso | Optional expressive corpus | CC BY-NC 4.0 | ~40 h / 4 speakers | Not used |
| RAVDESS | Optional evaluation | CC BY-NC-SA default | 7,356 files / 24 actors | Not used |

### 3.2 Accent corpus acquisition

GLOBE_V3 does not expose `speaker_id`; it exposes `common_voice_path`. The
pipeline joins that path to Common Voice 21 metadata to recover `client_id`, then
builds speaker-disjoint splits. Without the join, speaker leakage cannot be
ruled out.

The metadata-first pipeline avoids cloning the 348.7 GB repository tree:

1. Index Parquet footers and small metadata spans over HTTP ranges
   (`scripts/data/index_globe_metadata.py`).
2. Fetch Common Voice metadata only (`scripts/data/fetch_common_voice_metadata.py`).
3. Join on `common_voice_path` (`scripts/data/join_globe_common_voice.py`).
4. Select a balanced, speaker-disjoint subset
   (`scripts/data/select_globe_accent_subset.py`).
5. Fetch only the selected row groups (`scripts/data/download_globe_selected.py`).
6. Audit and prune source-clipped clips
   (`scripts/audit/audit_globe_audio.py`,
   `scripts/data/prune_globe_manifest.py`).

Measured results:

- Metadata pass: 723,532 rows across 187 shards, 0 failures, 858 s.
- Common Voice join: 100% of rows, 50,951 speakers, 1,102.1 h; 450,835 rows have
  a self-reported accent.
- Selection: 9,688 clips / 13.83 h pre-trim / 889 speakers / 543 row groups.
- Post-audit: 9,534 clips / 12.281 h; 154 source-clipped clips (1.6%) excluded.
- Five trainable classes: 2.185-2.257 h each, 94-198 train speakers.
- Irish: 1.176 h actual, evaluation-only.

The serving scope exposes four accents: US, England, Indian English, and
Australian. Canadian remains in the five-class research classifier and residual
artifact; Irish is evaluation-only. No face-derived accent labels are used.

### 3.3 Emotion corpus acquisition

CREMA-D is downloaded directly from the Cheyney Computer Science GitHub
repository (ODbL 1.0, no access form). `scripts/data/curate_crema_d.py` parses
`ActorID_Sentence_Emotion_Level.wav`, joins actor demographics, uses the
**voice-only** crowd vote block from `tabulatedVotes.csv`, checks 16 kHz mono
audio, and assigns deterministic speaker-disjoint splits with salt
`crema-d-v1`.

Measured results:

- 7,442 clips parsed; 6,892 kept / 91 actors / 4.869 h after excluding 550
  source-clipped clips (7.4%).
- Class hours 0.633-0.963 h (ratio 1.521), 90-91 speakers per class.
- Every class is below the 2 h confirmatory gate. `training_eligible=false`.
- Voice-only recognition is uneven: neutral 0.976, happy 0.330, sad 0.236,
  fear 0.356, disgust 0.343, angry 0.568. Acted labels are not clean perceptual
  ground truth.

Do not lower the 2 h gate to make CREMA-D confirmatory. MSP-Podcast or another
licensed corpus is required first.

### 3.4 Human ratings and consent

- Use real listeners; never invent ratings or placeholder values.
- Obtain informed consent for participation and for any audio/text collected.
- Obtain IRB/ethics review when the institution requires it.
- Store ratings pseudonymously; do not publish listener identities.
- MOS remains `unavailable` until a real ratings CSV exists.
- `scripts/evaluation/run_mos_evaluation.py` prepares blinded stimuli and
  computes statistics only from a real ratings file.

### 3.5 Demo provenance

Use a synthetic or consented face. Do not scrape a real person's face or voice.
Confirm the provenance of `demo/subject_face.png` before public presentation.
VoxCeleb2-derived voices and embeddings remain research/non-commercial. A
generated voice is not a claim about the person in the image.

---

## 4. Evidence ledger

### 4.1 Identity path

Measured on the official VoxCeleb2 test split (118 speakers, 628 clips) by
`scripts/evaluation/ci_gate_v2.py`:

| Metric | Value |
|---|---:|
| CFVN R@1 | 0.0932 |
| CFVN R@5 | 0.2627 |
| CFVN R@10 | 0.4068 |
| Chance R@1 | 0.0085 |
| Shuffled control R@1 | 0.0000 |
| Raw ArcFace R@1 | 0.9322 |
| Fused 560-D R@1 | 0.9153 |
| CFVN within-speaker cosine | 0.6541 |
| CFVN between-speaker cosine | 0.2528 |
| CVAE face margin | 0.2229 |
| Adapter output/target cosine | 0.8736 |
| Adapter target-mean baseline | 0.8980 |
| Style-target centered across-speaker cosine | 0.0013 |

The canonical gate passes: R@1 is 11x chance, the shuffled and constant controls
are at chance, and the CVAE does not collapse. The adapter is slightly below the
target-mean baseline, so it captures speaker-consistent residual variation but
not all predictable style. A face-face structure-preservation experiment
reached 0.8475 vs 0.9153 raw R@1 and was not stable across seeds; the canonical
CFVN remains unchanged.

The frozen-diversity report stores both `style_vector` (the retired
post-craniofacial-modulation style) and `style_without_geo_vector` (the canonical
pre-modulation style). Reproduction checks compare against
`style_without_geo_vector` / `style_without_geo_sha256`.

### 4.2 Naturalness and audio quality

**V1 hiss diagnosis.** Holding face, speaker embedding, text, and seed fixed, the
raw face-derived style measured **-49.17 dBFS**; the same model with a real AMI
style measured **-71.07 dBFS**. Steps 5/20/30 and the N1 LoRA did not remove the
hiss (N1 measured -46.31 dBFS, worse than raw). The web was not clipping:
clipping fraction was 0.0, energy above 8 kHz was 4e-05-0.00097 of total power,
and spectral flatness was 0.0034-0.0141.

**V2 calibration.** The demo applies a gender-matched mean translation,
`s' = s + alpha * (mean_real_gender - mean_face_gender)`, at `alpha=1.0`. On the
three selected voices this cut noise by **13.36, 17.58, and 17.12 dB**. The
21-clip v2 median was **-63.45 dBFS** (min -64.78, max -61.75) with median SNR
proxy **44.97 dB** (min 42.74). Alpha 1.25 was rejected: the median improved by
only 0.04 dB while the Timothee-Jackie ECAPA similarity rose to 0.422. The human
AMI reference median is **-76.11 dBFS**, so v2 remained 12.67 dB above it.

**V3 demo.** 24 clips: median noise floor **-62.07 dBFS** (range -63.03 to
-60.80), median SNR proxy **44.04 dB** (min 42.34). The 14.0 dB gap to the AMI
reference remains. Values are diagnostic proxies, not calibrated noise
measurements or MOS.

**Step count and writer.** Neutral SNR across 3 seeds x 2 prompts was 27.06 +
1.16 dB at 5 steps, 27.71 + 1.62 at 10, and 26.99 + 2.13 at 20. Whispered was
21.81 + 0.76, 21.79 + 0.61, and 19.81 + 1.47. Ten steps is the tier-1 default;
20 must not be called "high naturalness." The current writer changes level by
0.00 dB; the legacy writer added up to +6.55 dB and flattened intended energy
differences. The backend targets -23 LUFS with a -1 dBTP limiter.

**Naturalness pilots.** N0b prosody transfer moved pause/tilt slightly toward AMI
without worsening WER, but ECAPA was 0.716-0.883, below the 0.95 guard, so it was
rejected. N1 LoRA ran one 400-step epoch (validation diffusion loss 0.5616 on
104 clips). Its pre-registered 104-pair gate (4 speakers; 208 base/LoRA rows)
passed narrowly: WER
+0.0078 (CI [-0.0008, +0.0264]) stayed inside the +0.05 budget, target ECAPA
improved by +0.0181 and reference ECAPA by +0.0274, and the only prosody metric
whose CI excluded zero was F0-median absolute error, -0.9268 Hz (CI [-2.7009,
-0.0765]). Speech-rate error worsened by +0.0355 (CI [+0.0039, +0.0547]). This
is a four-speaker, one-epoch, diffusion-only proxy result with no MOS, and N2
was not entered. N1 is omitted from the demo because its controlled matrix had
a worse noise floor than the calibrated baseline.

### 4.3 Emotion

**Protocol.** CREMA-D, 73 train / 9 validation / 9 test actors. The identity base
is the frozen StyleTTS2 style encoder on one real neutral clip per actor, not the
canonical face adapter. Paired grid: 108 neutral/emotion pairs per emotion over
9 validation actors and 12 sentences. Generation: seed 0, 10 steps, watermark
off, neutral scalar modifiers. Test actors were untouched.

The class mean uses 4,031 real neutral/emotion pairs from 73 train actors.
E1 applies only the prosody half (`dims 128:256`). E2 is a 199,168-parameter
bounded MLP (256-D style, 6-emotion one-hot, intensity -> 256-D residual,
`max_norm=0.5`, timbre half weighted 0.25), trained on 4,031 examples with
speaker-disjoint internal early stopping.

**Direction gate (paired emotion minus neutral; speaker-cluster bootstrap CIs).**

| Condition | Happy F0 (Hz) | Happy duration (s) | Sad F0 (Hz) | Sad duration (s) | Angry F0 (Hz) | Angry duration (s) | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 class-mean prosody | **+16.19** [+9.68, +22.00] | **-0.055** [-0.063, -0.047] | **-11.59** [-21.75, -2.99] | **+0.048** [+0.040, +0.055] | **+18.13** [+12.79, +23.57] | **-0.054** [-0.065, -0.044] | **6/6** |
| E2 learned residual | **+18.63** [+10.05, +25.72] | **-0.034** [-0.052, -0.016] | +4.21 [-1.13, +9.70] | +0.016 [-0.011, +0.043] | **+20.32** [+14.54, +26.79] | **-0.027** [-0.044, -0.013] | **4/6** |
| E2 corrected label null | **+12.94** [+4.42, +21.39] | -0.005 [-0.034, +0.023] | +9.81 [+3.11, +17.18] | -0.005 [-0.034, +0.024] | +7.94 [-1.53, +17.57] | -0.003 [-0.033, +0.025] | **1/6** |

Bold values pass the pre-registered direction check. E1 passes all six and
survives a post-hoc Holm correction (weakest sad F0, Holm p=0.0013). E2 fails sad
and does not beat E1. The corrected label null passes only happy F0, which is a
generic average-residual effect, not label-specific control.

**Identity and intelligibility.**

| Condition | ECAPA to generated neutral | ECAPA to real speaker | WER |
|---|---:|---:|---:|
| E1 happy | 0.893 | 0.185 | 0.0310 |
| E1 sad | 0.928 | 0.194 | 0.0387 |
| E1 angry | 0.901 | 0.189 | 0.0328 |
| E2 happy | 0.881 | 0.194 | 0.0367 |
| E2 sad | 0.900 | 0.181 | 0.0341 |
| E2 angry | 0.889 | 0.206 | 0.0301 |

Generated neutral ECAPA to the real speaker is 0.204, below the 0.50 sanity
threshold; only relative comparisons are interpretable. All relative ECAPA
values are above 0.80, but absolute identity is not established. WER is 2.8-3.9%
across conditions with no confidence interval; it supports "intelligible," not a
condition difference.

**Emotion classifier proxy.** Generated audio is far outside the real CREMA-D
feature distribution: neutral standardized-centroid distance 67.35, 8.36x the
real validation spread (invalid threshold 3.0x). The classifier collapses to one
class, so generated accuracy (0.0 E1, 0.167 E2) is invalid as an emotion result.
The real-speech calibration is 0.535 accuracy / 0.532 macro-F1.

**Verdict.** Keep E1 prosody-only class mean as the supported emotion baseline
and serving scale 0.28. Do not promote E2. Do not start E3 from CREMA-D. Fear and
disgust remain unavailable as validated controls.

### 4.4 Accent

**A0 classifier (real speech).** 222-D ECAPA + prosody features; GLOBE
speaker-disjoint train/validation/test. Five classes: US, England, Australian,
Canadian, India/South Asia. Irish is evaluation-only.

| Feature set | Dim | Val accuracy | Val macro-F1 |
|---|---:|---:|---:|
| Base | 222 | 0.5102 | **0.5422** |
| Frozen SSL | 768 | 0.5332 | 0.4968 |
| Combined | 990 | 0.5255 | 0.5424 |

Base was selected because neither fallback condition (>=0.55 macro-F1 and
+0.05 over base) held. One-time test: **0.5504 accuracy / 0.5630 macro-F1** on
605 clips / 44 speakers, majority baseline 0.2579, speaker-cluster 95% CI
[0.4240, 0.6749]. This is a real-speech classifier result and is
confirmatory-eligible; the TTS accent track is not.

**A2/A2b G2P (rejected).** The hand-written IPA rules raised WER for every
accent and failed the +0.10 budget for Australian, Indian, and Irish English.
Indian G2P was catastrophic at WER 0.6866. A post-hoc rule ablation brought
Australia inside the budget but left India at 0.2332, still failing.

**A2c Indian subset ablation (rejected).** Six prompts, one frozen US donor,
seed 0, 10 steps, 12 clips per subset. Gate: WER may rise by at most +0.10 and
the classifier must gain at least 0.05 accuracy. Domain-shift checks passed
(ratio 0.878-0.885 against threshold 3.0).

| Subset | Regex WER | G2P WER | Delta [95% CI] | Classifier regex | Classifier G2P | Gain |
|---|---:|---:|---|---:|---:|---:|
| Full | 0.01389 | 0.06423 | +0.05034 [+0.01515, +0.08889] | 0/6 | 0/6 | 0.000 |
| Stop-only | 0.01389 | 0.08274 | +0.06886 [+0.01515, +0.14141] | 0/6 | 0/6 | 0.000 |
| Vowel-only | 0.01389 | 0.03241 | +0.01852 [0.00000, +0.05556] | 0/6 | 0/6 | 0.000 |

All WER budgets pass, but no subset produces a classifier gain. Indian G2P is
therefore **not applied** in the demo or serving. The rule implementation and
`--disable-rules` ablation switch remain for future frontend research.

**A3 class-contrast residual (research transform).** Style subset: 1,765 clips /
607 speakers, up to three clips per speaker, train split only. Residual:
speaker-balanced class mean per accent minus the unweighted mean of the five
class means; timbre half scaled 0.25, prosody half unscaled. Residual norms
0.19-0.46; between/within separation ratio 0.034, so accent differences are
small relative to speaker variation. GLOBE has no within-speaker cross-accent
pairs, so accent and speaker identity remain confounded. This is not a
speaker-disentangled direction.

Evaluation: one frozen validation US donor (`common_voice_en_17993335`), five
accents x six prompts x four conditions, seed 0, 10 steps, 120 clips. Selection
rule declared before the sweep: valid generated-domain classifier proxy; WER
increase <= +0.05 over regex; ECAPA donor drop <= 0.05; then highest classifier
accuracy, ties to smaller scale.

| Condition | Classifier accuracy | Gain vs regex | WER delta | ECAPA donor delta | Eligible |
|---|---:|---:|---:|---:|---|
| regex | 0.200 (6/30) | - | - | - | baseline |
| style_s0.1 | 0.200 (6/30) | 0.000 | +0.0033 | -0.0001 | no |
| style_s0.2 | 0.200 (6/30) | 0.000 | +0.0067 | -0.0003 | no |
| style_s0.3 | **0.2333 (7/30)** | **+0.0333** | +0.0067 | +0.0041 | **yes** |

Domain-shift ratio 0.9048. Scale 0.3 is selected mechanically, but the gain is
one clip. Per-class recall at 0.3: US 1/6, England 5/6, Australian 1/6,
Canadian 0/6, India 0/6; the classifier predicts England for 22/30 clips and
never predicts Canada or India. WER and identity budgets were not the limiting
factor. **This is not evidence that A3 produces recognizable accents.** It is
used only as `exploratory_application` in the demo, with timbre weight 0.25.

### 4.5 External evaluator panel

The panel is evaluation-only. Frozen third-party recognizers never feed back into
synthesis. Each evaluator must pass a real-speech gate (accuracy above chance and
above 0.35, no collapsed predictions) and a generated-domain check
(standardized-centroid distance / real-speech spread <= 3.0). If the domain gate
fails, no category claim is made.

Panel data: GLOBE accent train/test (320/449 clips; 223/36 speakers), CREMA-D
emotion train/test (320/455 clips; 73/9 speakers), 80 train clips per class.
Speaker-cluster bootstrap uses 10,000 resamples.

**Panel v2 (21-clip demo; three accents, three emotions).**

| Evaluator | Task | Real accuracy [95% CI] | Real macro-F1 | Generated accuracy | Generated n | Shift ratio |
|---|---|---:|---:|---:|---:|---:|
| CommonAccent | accent | 0.6637 [0.5548, 0.7671] | 0.6851 | 0.333 (6 US, 3 England) | 9 | 0.680 |
| SUPERB ER | emotion | 0.4396 [0.3528, 0.5165] | 0.3775 | 0.250 (8 neutral, 4 happy) | 12 | 1.635 |
| emotion2vec+ | emotion | 0.9099 [0.8612, 0.9475] | 0.9318 | 0.333 (11 neutral, 1 sad) | 12 | 0.847 |

**Panel v3 (24-clip demo; four accents, four emotions).**

| Evaluator | Task | Real accuracy [95% CI] | Real macro-F1 | Generated accuracy | Generated n | Shift ratio |
|---|---|---:|---:|---:|---:|---:|
| CommonAccent | accent | 0.6637 [0.5548, 0.7671] | 0.6851 | 0.250 | 12 | 0.737 |
| SUPERB ER | emotion | 0.4396 [0.3528, 0.5165] | 0.3775 | 0.250 | 12 | 1.651 |
| emotion2vec+ | emotion | 0.9099 [0.8612, 0.9475] | 0.9318 | 0.250 | 12 | 0.809 |

V3 per-class generated recall:

- CommonAccent: US 2/3, England 1/3, Indian 0/3, Australian 0/3. All 12 clips
  are predicted US (8) or England (4); the evaluator never predicts Indian or
  Australian.
- SUPERB ER: neutral 1/3, happy 2/3, sad 0/3, angry 0/3. Predictions are 8 happy,
  4 neutral.
- emotion2vec+: neutral 3/3, happy 0/3, sad 0/3, angry 0/3. Every generated clip
  is predicted neutral.

**Conclusion.** All three evaluators are valid on real speech and pass the
generated-domain gate, but none recognizes the generated emotion or accent
conditions above the four-class chance rate of 0.25. Adding the Indian accent to
v3 does not improve accent recognition (accuracy 0.250 vs 0.333 on the easier
three-class v2 set; both are at chance for their class counts). The objective
direction checks and identity-preservation checks do not imply perceptual
control. No human study exists, so this is a null external-recognition result,
not proof that listeners hear nothing.

### 4.6 Demo v3

`results/demo/research_v3/` is the current research demo. Design: 3 face-derived
voices (Lupita Nyong'o, Timothee Chalamet, Jackie Chan), 24 clips = 3 voices x
(neutral + happy/sad/angry + US/England/Indian/Australian), one prompt, seed 42,
10 diffusion steps, 24 kHz PCM-16, -23 LUFS, -1 dBTP limiter, serving scales
(emotion 0.28; accent 0.30 with timbre weight 0.25), gender-matched style
calibration at alpha 1.00.

Verification:

| Check | Result |
|---|---|
| Canonical checkpoint hashes unchanged | PASS |
| Face-style recomputation | PASS |
| Audio hashes after write | PASS |
| Metadata completeness | PASS |
| Noise floor / SNR thresholds | PASS (median -62.07 dBFS; SNR 44.04 dB) |
| Three neutral voice identities distinct | **FAIL**: max pair C1_02/C2_09 ECAPA 0.5696; real same-gender between-speaker max 0.438; 100th percentile |
| Identity preserved across conditions | PASS: min ECAPA-to-neutral 0.9453 (emotion 0.9453, accent 0.9613) |
| Emotion direction checks | PASS 9/9 |
| Emotion application | PASS: 9/9 conditions present, distinct styles, ECAPA >= 0.90 |
| Accent application | PASS: 12/12 conditions present, distinct styles, ECAPA >= 0.90, Indian included |
| WER (Whisper base) | 0.000 on all 24 clips |

The identity-distinctness failure is real under the demo's pre-declared criterion
and is not hidden: two of the three face-derived neutral voices are closer than
the maximum real AMI same-gender between-speaker pair. It is a limitation of the
selected face/voice trio, not evidence that the conditions are correct. The
README writer now reports the failure explicitly.

The v3 demo uses the A3 residual only for accents. Indian G2P is not applied,
because A2c found no classifier gain. Sad is carried by longer duration only;
the E1 sad F0 lowering did not transfer to calibrated face styles, and a
post-synthesis pitch shift was rejected because it collapsed ECAPA identity to
about 0.57.

### 4.7 C1: reference-prosody transfer

C1 keeps the calibrated face-derived timbre half (`style[:128]`) and replaces the
prosody half (`style[128:]`) with the mean prosody of real CREMA-D **training**
clips for the requested emotion. References are sex-matched, one clip per actor,
eight actors per sex and emotion, selected only from `train`; no validation/test
actor is used. No residual is trained and no recognizer is used as a controller.
This is a research transform, not a serving path.

C1 is evaluated as a comparison point for E1/A3, not as a promoted control.
Measured on 12 clips (3 voices x neutral + happy/sad/angry), seed 42, 10
diffusion steps, serving loudness settings.

**Identity and directions.** The timbre half was preserved exactly by
construction, and the four styles per voice are distinct. The cost appears in
the measured speaker embedding: the minimum ECAPA-to-neutral is **0.773**
(C2_09 happy), below the 0.90 demo preservation guard. Per-voice minima are
0.849 (C2_12 angry), 0.779 (C1_02 happy), and 0.773 (C2_09 happy).

Happy and angry raise F0 strongly in all three voices (happy +37 to +49 Hz,
angry +35 to +60 Hz) and shorten the clip. Sad does **not** reproduce the E1
direction: its F0 rises by +5.5 to +14.3 Hz and its duration does not lengthen
(three deltas -0.20, 0.00, and -0.63 s). WER was not computed for this run, so
intelligibility is not established for C1.

**External panel (evaluation only).** Both evaluators pass their real-speech and
generated-domain gates (shift ratios 1.591 and 0.842 against threshold 3.0).

| Evaluator | Real accuracy | Generated accuracy | Generated n | Generated per-class recall |
|---|---:|---:|---:|---|
| SUPERB ER | 0.4396 | **0.500** | 12 | neutral 1/3, happy 3/3, sad 2/3, angry 0/3 |
| emotion2vec+ | 0.9099 | 0.250 | 12 | neutral 3/3, happy 0/3, sad 0/3, angry 0/3 |

SUPERB ER predicts happy 8, neutral 2, and sad 2; it never predicts angry.
emotion2vec+ predicts neutral for all 12 clips. The 0.500 SUPERB ER result is
**not** a strong positive: its real-speech per-class recall is only 0.033 for
happy and 0.140 for neutral (sad 0.656, angry 0.980), so it is not a calibrated
happy/neutral evaluator, and the generated set is 12 clips from one seed. The
disagreement between the two evaluators means C1 does not have consistent
external recognition.

**Verdict.** C1 is **not promoted**. It preserves the timbre half but fails the
0.90 ECAPA-to-neutral identity guard (0.773), does not produce the sad
direction, has no WER measurement, and has inconsistent external recognition.
It remains in `results/emotion/c1/` as an exploratory comparison and is not
wired into serving.

---

## 5. Evaluation protocol and gates

### 5.1 Pre-declared gates

| Gate | Emotion | Accent |
|---|---|---|
| Class balance | >=30 speakers and >=2 h per class; max/min <=1.5 | >=30 speakers and >=2 h per class; max/min <=1.5 |
| Classifier accuracy | >=2x chance and >=50% per class | >=2x chance and >=50% per class |
| WER | <= neutral + 0.10 absolute | <= neutral + 0.10 absolute |
| Speaker similarity | >=0.80 to generated neutral | >=0.80 to generated neutral |
| Intensity monotonicity | rho >=0.8 for intended dimension | rho >=0.8 for intended dimension |
| MOS | >=4.0/5 with real ratings | >=4.0/5 with real ratings |
| Human identification | >=2x chance | >=2x chance |

Report every class separately. A class that cannot meet the balance or
classifier gate is evaluation-only; do not hide it in a macro average. Never
lower a gate silently to turn a failure into a pass.

For served expression conditions, the demo additionally requires
ECAPA-to-neutral >=0.90 for every clip. That is an identity-preservation guard,
not a human similarity judgment.

### 5.2 External panel rules

- Train the evaluator on real speech only; never train it on generated clips.
- Use a speaker-disjoint real test split.
- Require real accuracy above chance and above 0.35, with no single-class
  collapse.
- Require generated-domain standardized-centroid ratio <=3.0 before interpreting
  generated predictions.
- Use speaker-cluster bootstrap for confidence intervals.
- Report every class and the prediction distribution, not only pooled accuracy.
- A panel result is a model output, not a perceptual result.
- Recognizers are evaluation-only and must not control synthesis.

### 5.3 Statistics and provenance

- Paired comparisons where possible; report paired mean and 95% CI.
- Bootstrap resamples cluster by speaker, not by clip.
- Pre-register the direction and the gate before generation.
- Keep seed, diffusion steps, alpha, beta, text, model hashes, audio hashes, and
  style hashes in every report.
- One seed is exploratory. Confirmatory work needs multiple seeds and a
  human-identification test.

---

## 6. Research status, rejected options, and blockers

### 6.1 Track status

| Track | Status | Evidence |
|---|---|---|
| Naturalness | N1 narrow proxy pass; no MOS | N0b rejected on identity; N1 104-pair gate passed narrowly (F0-median only); v3 noise improved but still 14 dB above AMI |
| Emotion | E1 narrow exploratory positive; E2 rejected | E1 6/6 directions; E2 4/6; v3 panel at chance |
| Accent | A0 real-speech classifier confirmatory-eligible; A2/A2c rejected; A3 research transform only | A0 test macro-F1 0.5630; A2c zero classifier gain; A3 one-clip gain; v3 panel at chance |
| Identity | Frozen canonical path | CFVN R@1 0.0932 (11x chance); gate passes |
| C1 | Reference-prosody research transform; not promoted | Timbre half exact, but min ECAPA-to-neutral 0.773 (guard 0.90); sad direction absent; evaluators disagree (0.500 vs 0.250) |

### 6.2 Rejected or blocked options

| Option | Decision | Reason |
|---|---|---|
| E2 learned emotion residual | Rejected | Fails sad (4/6); does not beat E1 |
| Indian G2P in demo/serving | Rejected | A2c WER budget passes but zero classifier gain |
| A3 as a serving accent adapter | Not promoted | One-clip gain, zero Indian/Canadian recall, no human identification |
| AccentCL | Rejected | No declared license, coarse regional classes, ~7.7 GB |
| AccentScore | Rejected | No verifiable model/checkpoint |
| Accent conversion | Rejected | Wrong task, no public checkpoints |
| OpenVoice V2 and other full TTS models | External baselines only | No face input or categorical emotion/accent API |
| Recognition models as controllers | Prohibited | Evaluation-only; would leak evaluator bias into synthesis |
| Cross-lingual synthesis | Out of scope permanently | Not part of the research question |
| Speech-difference simulation | Out of scope permanently | Not part of the research question |
| Craniofacial style modulation | Removed | Not a valid expression control; core ratios remain in face features |
| MSP-Podcast confirmatory emotion | Blocked | Institutional license agreement not executed |
| N1 LoRA in demo | Omitted | Worse noise floor than calibrated baseline |
| CFVN structure-preservation replacement | Not adopted | 0.8475 vs 0.9153 raw R@1; unstable across seeds |

### 6.3 Open blockers

- No human perceptual studies: MOS/CMOS, emotion identification, accent
  identification, and voice-triad confirmation are all unavailable.
- MSP-Podcast licensing blocks confirmatory emotion work.
- GLOBE labels are self-reported and accent/speaker identity are confounded;
  there are no within-speaker cross-accent pairs.
- The 4 GB GPU limits batch size and serializes TTS/SSL/evaluation runs.
- The three-voice demo does not establish speaker generalization.
- C1 has not been validated perceptually or in serving; it fails the 0.90
  ECAPA-to-neutral guard and is not a promoted emotion control.

---

## 7. Reproduction

### 7.1 Environment

The P0 setup path is documented in [`SETUP.md`](SETUP.md). The short version:

```powershell
python scripts/setup/bootstrap_env.py --recreate
.venv\Scripts\Activate.ps1
python scripts/setup/verify_artifacts.py --restore-from <artifact-mirror>
python scripts/setup/prefetch_models.py
python -m voix preflight --json
```

The bootstrap creates an isolated venv (`include-system-site-packages = false`),
installs the pinned PyTorch CUDA 12.1 wheels, installs `-e ".[serving]"` under
[`constraints/serving-py312-cu121.txt`](../constraints/serving-py312-cu121.txt),
and runs `pip check`. `configs/artifacts.yaml` records the serving/evaluation
files, sizes, SHA-256 digests, licenses, and sources; `voix preflight` and
`scripts/setup/verify_artifacts.py` read that manifest.

`ffmpeg` must be on `PATH`. The repository config is checkout-relative;
`configs/environment.yaml` contains no absolute `D:/voix` paths.

Measured environment: Python 3.12.1, `torch==2.4.1+cu121`, `numpy==1.26.4`,
`scipy==1.12.0`, `librosa==0.10.2.post1`, `transformers==4.40.2`,
`styletts2==0.1.6`, `audioseal==0.2.0`, `openai-whisper==20250625`,
`speechbrain==1.1.1`, `huggingface_hub==0.36.2`, `modelscope==1.38.1`,
`funasr==1.0.27`. GPU: NVIDIA GTX 1650, 4.29 GB.

The local TensorFlow install is broken (`module 'tensorflow' has no attribute
'Tensor'`). Set `USE_TF=0` and `TRANSFORMERS_NO_TF=1`; the TTS backend also
primes einops to the torch backend before TensorFlow can be selected. A clean
venv does not install TensorFlow, so this is mainly a dirty-workstation note.
This workaround is covered by `tests/test_tts_backend.py`.

### 7.2 Corpus and training

```powershell
# Identity corpus and scaler
python scripts/data/build_voxceleb2_index.py
python scripts/data/download_selected_clips.py
python scripts/data/extract_voxceleb2_v2.py
python scripts/data/build_face_scaler_v2.py
python scripts/audit/validate_v2_corpus.py

# Identity training and gate
python scripts/training/train_emotion_classifier.py   # emotion proxy, not a control
python scripts/evaluation/ci_gate_v2.py

# Accent corpus (metadata-first; see section 3.2)
python scripts/data/index_globe_metadata.py --workers 10
python scripts/data/fetch_common_voice_metadata.py
python scripts/data/join_globe_common_voice.py
python scripts/data/select_globe_accent_subset.py --target-hours 2.5 `
  --min-speakers 30 --selection-mode row_group --evaluation-only irish
python scripts/data/download_globe_selected.py --workers 8
python scripts/audit/audit_globe_audio.py --workers 8
python scripts/data/prune_globe_manifest.py
python scripts/audit/audit_globe_audio.py --workers 8

# Emotion corpus
git clone --filter=blob:none --no-checkout `
  https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/crema_d
# sparse-checkout AudioWAV + processedResults, then:
python scripts/data/curate_crema_d.py
python scripts/audit/audit_expression_data.py
```

The CFVN/CVAE/adapter trainers and configs remain the original V2 pipeline.
Expression work never trains the CFVN, CVAE, or style adapter.

### 7.3 Expression research

```powershell
# E1 class-mean emotion (train actors only)
python scripts/training/build_emotion_class_mean.py --reuse-cache
python scripts/demo/generate_crema_e0.py --condition class_mean `
  --residual-variant prosody_only --emotions happy,sad,angry `
  --scalar-modifiers neutral --condition-label class_mean_prosody `
  --output-dir results/emotion/e1c/audio `
  --index results/emotion/e1c/generation_index_prosody.csv `
  --report results/emotion/e1c/generation_report_prosody.json
python scripts/evaluation/evaluate_crema_e0.py `
  --index results/emotion/e1c/generation_index_prosody.csv `
  --generation-report results/emotion/e1c/generation_report_prosody.json `
  --output results/emotion/e1c/report_prosody.json `
  --run-id e1_class_mean_prosody_neutral_scalars --device cuda --workers 4

# A0 accent classifier and A3 residual
python scripts/training/train_accent_classifier.py
python scripts/data/build_accent_style_cache.py
python scripts/training/build_accent_class_mean.py
python scripts/demo/generate_accent_a3.py
python scripts/evaluation/evaluate_accent_a3.py
```

### 7.4 Demo v3

```powershell
$env:USE_TF='0'; $env:TRANSFORMERS_NO_TF='1'
python scripts/demo/generate_research_demo.py --device cuda --compute-wer
```

Outputs: `results/demo/research_v3/{audio,report.json,index.csv,README.md}`.
The generator targets exactly 24 clips and enforces the serving scales.
Demo v1 remains at `results/demo/research/`; demo v2 remains at
`results/demo/research_v2/`.

### 7.5 External panel

Reuse the real-speech embedding caches by copying `cache/` from a prior panel
run. Fingerprints validate the cached paths and model artifacts; generated
features are recomputed when the demo set changes.

```powershell
$env:USE_TF='0'; $env:TRANSFORMERS_NO_TF='1'
python scripts/evaluation/evaluate_panel.py `
  --demo results/demo/research_v3 `
  --output-dir results/evaluation/panel_v3 `
  --evaluators commonaccent superb_er emotion2vec `
  --n-resamples 10000 --batch-size 8 --device cuda
```

Outputs: `report.json`, `predictions.csv`, `README.md`. The panel never writes
back into synthesis artifacts.

### 7.6 C1

```powershell
python scripts/demo/generate_reference_prosody_c1.py --device cuda
python scripts/evaluation/evaluate_panel.py `
  --demo results/emotion/c1 `
  --output-dir results/emotion/c1/panel `
  --evaluators superb_er emotion2vec `
  --n-resamples 10000 --batch-size 8 --device cuda
```

C1 references are selected from CREMA-D `train` only, sex-matched, one clip per
actor, eight actors per sex and emotion.

### 7.7 Tests

```powershell
python -m pytest -q
```

Artifact-dependent tests skip cleanly when checkpoints are absent; they do not
silently use random weights.

---

## 8. Repository and artifact map

```
README.md                     entry point and quickstart
docs/MASTER.md                this research record
docs/SCOPE.md                 scope contract
docs/SETUP.md                 clean env, artifacts, caches, offline/OOM
docs/AUDIT_REPORT.md          historical code audit
docs/voix_banner.jpg          banner asset
configs/                      environment, hardware, artifact manifest, training configs
constraints/                  validated dependency pins
data/processed/               local corpora, manifests, caches, scalers
checkpoints/                  canonical and experimental weights
results/
  RESEARCH_STATUS.md          short status index
  checkpoints/                CFVN and experimental checkpoints
  audit/                      audio, style, ECAPA, and cleanup audits
  demo/research/              v1 15-asset demo
  demo/research_v2/           v2 21-clip demo
  demo/research_v3/           v3 24-clip demo
  emotion/                    E0/E1/E2 findings, C1
  accent/                     A0/A2/A2c/A3 findings
  evaluation/panel_v2/        external panel on v2
  evaluation/panel_v3/        external panel on v3
scripts/
  data/                       corpus builders and feature extraction
  training/                   identity and expression trainers
  evaluation/                 gates, panels, metrics
  demo/                       research demo and C1 generators
  audit/                      audio, style, and data-firewall audits
  maintenance/                cleanup and manifest utilities
  setup/                      bootstrap, artifact verification, cache prefetch
voix/
  data/                       face features, demographics, presets
  models/                     CFVN, CVAE, adapter, TTS backend, controls
  training/                   trainers
  evaluation/                 metrics, gates, expression metrics
  inference/                  pipeline, service, CLI, IO
tests/                        unit, artifact, and neural integration tests
Initial Docs/                 original literature survey and outline
```

---

## 9. Known issues and dependency notes

- P0 resolved the old dependency record: `requirements.txt` was a UTF-16
  `pip freeze` of the polluted user site (it pinned `numpy==2.5.2`,
  `scipy==1.18.1`, omitted `styletts2`, `audioseal`, `pyloudnorm`, and `funasr`,
  and contained a self-referential VCS install). It is now a UTF-8 serving
  installer backed by `constraints/serving-py312-cu121.txt`,
  `configs/artifacts.yaml`, and `scripts/setup/`. The validated environment
  still uses `huggingface-hub==0.36.2` with `styletts2==0.1.6`; that combination
  passed the full suite and the CPU synthesis smoke test.
- P0 also removed unused `torchvision` from core dependencies, moved
  `soundfile` into core, and made preflight check all five `buffalo_l` ONNX
  files instead of only the recognition model.
- TensorFlow is broken. Use the torch/einops priming described in section 7.1.
- emotion2vec+ uses the FunASR model license (`other`, not OSI-approved).
  Review redistribution terms before shipping model files.
- emotion2vec labels are bilingual (`生气/angry`); the panel normalizes them.
- `funasr` exits with code 1 after successful inference. Output is correct; the
  teardown status is unexplained.
- `torch.load` on some SpeechBrain artifacts warns about future
  `weights_only=True` defaults; this is a warning, not a failure.
- The 8 GB host RAM limit has caused silent process kills when ECAPA workers and
  StyleTTS2 run together. Serialize model runs and cap workers at 2-3.
- The panel cache is under `<output-dir>/cache`; fingerprints do not include the
  cache path, so caches can be copied between output directories.

---

## 10. Limitations and non-claims

- No human MOS, CMOS, emotion-identification, accent-identification, or
  voice-triad study exists. Every perceptual claim is unavailable.
- The external panel is at chance on generated emotion and accent conditions.
  This does not prove listeners cannot hear differences; it proves the frozen
  recognizers do not identify the intended categories.
- Generated emotion/accent direction checks are objective proxies. They do not
  establish perceived emotion, perceived accent, appropriateness, or
  naturalness.
- The accent residual is a class contrast confounded with speaker identity.
  GLOBE has no within-speaker cross-accent pairs.
- The accent classifier's generated-audio proxy is weak. It never predicts the
  Indian class for generated clips.
- The emotion classifier proxy is invalid for generated audio (domain-shift
  ratio 8.36x in E1/E2; the external panel is the better evaluator).
- The demo has three voices. It does not establish speaker generalization. Two
  of the three neutral voices fail the demo's identity-distinctness criterion.
- The generated neutral ECAPA-to-real-speaker similarity is 0.204, below the
  0.50 sanity threshold. Absolute identity is domain-shifted; only relative
  comparisons are interpretable.
- Face-to-voice identity is inherently weak. A face constrains a distribution;
  it does not determine a unique voice. CFVN R@1 is 0.0932, not 0.93.
- The style adapter's output/target cosine 0.8736 is slightly below the
  target-mean baseline 0.8980. It does not predict all residual style.
- The FAD implementation is a mel-spectrogram proxy, not official VGGish FAD.
- The StyleTTS2 LoRA pilot is 0.741 h of VoxCeleb2 with Whisper
  pseudo-transcripts. Its 104-pair proxy gate passed narrowly (WER +0.0078;
  only F0-median absolute error improved, -0.93 Hz) with four speaker clusters;
  there is no MOS and it is not serving.
- The backend loads StyleTTS2's default LibriTTS checkpoint. The formal prompt
  and audiobook checkpoint explain the careful narrator delivery; it is not a
  naturalness or emotion result.
- VoxCeleb2 is research/non-commercial. This repository grants no rights to the
  videos, voices, or derived biometric features. Do not use generated audio to
  impersonate a real person.

---

## Appendix A. Historical Phase 0 baseline

The Phase 0 report recorded the pre-rebuild environment (Python 3.12.1, PyTorch
2.4.1+cu121, GTX 1650), the then-current checkpoint hashes, and the P0-P3 fixes:
fusion projection execution, diffusion/tier defaults, true-peak limiting,
normalization-contract repairs, CVAE retraining with FaceSepLoss
(face_separation 0.9915 -> 0.0842), and neural integration tests. Those hashes
and paths are historical and were superseded by the V2 rebuild.

The Phase 0 report's closing claim that the system was "ready for production use"
is **withdrawn**. It predates the V2 corpus, the expression-control evidence, the
external panel, and the current non-claims. The authoritative current state is
this document plus the measured artifacts it references.
