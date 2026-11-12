# Identity Calibration Audit — Face-Specific Voice Collapse

**Date:** 2026-11-11
**Scope:** canonical `CFVN -> CVAE -> StyleAdapter -> StyleTTS2` path, the demo
style calibration, and exploratory calibration/adapter candidates. No canonical
checkpoint was modified.
**Verdict:** Two distinct failures are present. The demo's mean-only style
translation collapses what separation remains, **and** the canonical adapter
already maps distinct same-gender faces to nearly identical raw styles (cosine
`0.95-0.99`) before calibration. Refitting the calibration on disjoint faces
rules out evaluation leakage: canonical gain 1 still reaches max ECAPA `0.9486`
on held-out faces. Candidate A improves style-space residual fidelity but does
not pass the identity or audio gates. Candidate B, trained with a pairwise
structure objective, cuts the validation centered pair p90 from `0.902` to
`0.527` and the held-out identity max to `0.6415`, but still fails both gates;
its two worst pairs sit at the ridge linear ceiling for the same `e_s` input.
The demo is unchanged and no candidate is promoted.

---

## 1. Pre-registered gates

Identity gate, fixed before the investigation:

```
max neutral pairwise ECAPA cosine <= 0.4378563828374123
```

That value is the maximum real same-gender between-speaker cosine in
`results/audit/ecapa_reference/ecapa_reference.json` (`n=324`).  The real
cross-gender maximum is `0.1931`, i.e. stricter, so the gate is not unfairly
lenient for the male/female pairs.

Naturalness thresholds used by the demo verification:

```
noise_floor_dbfs <= -58.0
snr_proxy_db     >=  40.0
```

The gate was not moved or retuned after seeing results.

---

## 2. Root cause

### 2.1 Identity information is present

A ridge map from centered style residual to normalized `e_s`, fitted on the
adapter's train-split outputs, reconstructs the serving identity on val at
cosine `0.9237`.  On the three demo faces, the **uncalibrated** style residuals
reconstruct the true `e_s` at `0.920`, `0.934`, and `0.886`.

So the adapter has not thrown the identity away.

Linear decodability is necessary but not sufficient. A decoder can recover a
direction that is present in every residual while the residuals themselves are
nearly parallel. On validation, the canonical adapter's centered
different-speaker style cosine has p90 `0.902` against `0.423` for the real
targets; every same-gender demo pair has raw style cosine `0.948-0.988` even
when `e_s` cosine is near zero. This is quantified in section 6.

### 2.2 The constant translation adds a common identity offset

The demo calibration is

```
s' = s + delta_gender,   delta = mean_real - mean_face
```

with `||delta|| = 2.584` (female) and `2.809` (male), while the identity-bearing
style residual is about `1.0` in norm.  After the translation the same linear
decoder reconstructs the true identities at only `-0.002`, `0.203`, and `0.108`,
and the decoded demo identities become nearly parallel (`0.963`, `0.963`,
`1.000`).  The common offset dominates the face-specific residual.  This is the
collapse mechanism.

### 2.3 The demo faces are in-distribution for the adapter

Nearest-neighbour cosine in normalized `e_s` space:

| set | mean | median | min | max |
|---|---:|---:|---:|---:|
| val prior-mean inputs (`n=3539`) | 0.7648 | 0.7621 | 0.5697 | 0.9456 |
| demo faces (`n=19`) | 0.7595 | 0.7742 | 0.6451 | 0.8512 |

The demo inputs are not a simple out-of-distribution extrapolation.

### 2.4 The adapter's face-style spread is about half the real spread

From `results/audit/style_calibration/calibration.json`:

| gender | face centered RMS | real centered RMS | ratio |
|---|---:|---:|---:|
| female | 0.0653 | 0.1299 | 1.989 |
| male | 0.0535 | 0.1067 | 1.996 |

The mean-only translation moves the cluster but leaves it under-dispersed.
Variance matching (`gain ~= 2`) was therefore tested as a principled variant.

---

## 3. Candidate results on the three demo faces

`C2_12` (female), `C1_02` (male), `C2_09` (male).  Identity is the maximum
neutral pair cosine; the gate is `<= 0.4379`.  Audio values are noise floor
dBFS / SNR proxy dB.  The alpha `1.0` row is the v3 demo neutral clips
(`results/demo/research_v3`), which used the same calibration alpha.

| setting | max ECAPA | identity | noise (C2_12 / C1_02 / C2_09) | SNR | naturalness |
|---|---:|---|---:|---:|---|
| canonical, alpha 0.0 | **0.2408** | pass | -45.5 / -41.4 / **-25.1** | 26.4 / 22.4 / **3.3** | fail |
| alpha 0.0, 25 diffusion steps | 0.6380 | fail | -45.7 / -40.0 / -43.2 | 26.9 / 20.8 / 24.4 | fail |
| canonical, alpha 0.5 | 0.5773 | fail | -56.4 / -60.3 / -59.0 | 38.0 / 41.7 / 40.4 | fail (one clip) |
| canonical, alpha 1.0 | 0.5696 | fail | -60.9 / -62.6 / -62.1 | 42.7 / 44.0 / 44.3 | pass |
| timbre-only (timbre 1.0) | 0.6195 | fail | -50.7 / -50.3 / -50.4 | 32.4 / 32.1 / 32.6 | fail |
| prosody-only (prosody 1.0) | 0.0627 | pass | -53.1 / -52.0 / **-25.7** | 34.3 / 33.1 / **0.4** | fail |
| alpha 1.0, deviation gain 2.0 | 0.4762 | fail | -57.9 / -58.7 / -58.1 | 38.7 / 40.7 / 38.2 | near |
| identity-null timbre, gain 1.0 | **0.4679** | fail | -58.1 / -57.8 / -58.7 | 40.1 / 39.1 / 40.3 | near |
| identity-null timbre, gain 2.0 | 0.6175 | fail | -57.7 / -55.5 / -57.4 | 39.3 / 37.2 / 36.3 | fail |

The `-25 dBFS` / `3.3 dB` and `-25.7 dBFS` / `0.4 dB` rows are degenerate
outputs (flat energy, no pauses, uninformative F0), not speech.  A low ECAPA
cosine on such a clip is not evidence of identity preservation.

The prosody-only row shows the same trap: it passes the ECAPA gate while
`C2_09` is noise.

---

## 4. Held-out validation of the closest candidate

The identity-null calibration was the closest setting on the three selection
faces, but it still failed the gate (`0.4679 > 0.4379`).  It was then run on
eight held-out faces (four female, four male):

```
C1_01, C1_03, C1_05, C2_08, C1_04, C2_07, C2_13, C2_15
```

| check | value |
|---|---:|
| same-gender max ECAPA | **0.9473** |
| cross-gender max ECAPA | 0.5861 |
| all-pairs max ECAPA | **0.9473** |
| same-gender median | 0.6533 |
| cross-gender median | 0.3818 |
| noise floor pass | 6 / 8 |
| SNR proxy pass | 5 / 8 |

The worst pair was `C1_03` / `C1_05` (two female faces, cosine `0.9473`).  The
calibration fails the pre-registered identity criterion on both the selection
faces and the held-out faces.

**Reproducibility note:** an earlier unseeded CVAE draw produced a three-face
max of `0.3876` and appeared to pass.  The builder now fixes the seed; the
reproducible artifact gives `0.4679`.  The unseeded pass is not evidence and is
superseded by the seeded result.

Artifacts:

- `results/demo/identity_validation_idnull_8faces_final/report.json`
- `results/demo/identity_candidate_canonical_idnull_final/report.json`
- `results/audit/style_calibration/calibration_identity_null.json`

---

## 5. Disjoint-calibration leakage control

The original demo calibration was fit on all 19 report faces, including the
faces later used for evaluation. That could make held-out numbers optimistic.
To rule this out, both calibrations were rebuilt with `--exclude-faces` on the
eight held-out faces, leaving 11 fit faces (4 female, 7 male). The builder is
`scripts/training/build_adapter_style_calibration.py`; the exclusion list is
recorded in each artifact's provenance.

| calibration | female delta norm | male delta norm |
|---|---:|---:|
| canonical, 11-face fit | 2.1341 | 2.5657 |
| Candidate A, 11-face fit | 2.3747 | 2.0876 |
| Candidate B, 11-face fit | 2.5081 | 2.1461 |

Three selection faces (`C1_02`, `C2_09`, `C2_12`), identity gate `0.4379`:

| setting | max ECAPA | identity | audio |
|---|---:|---|---|
| canonical, gain 1.0 | 0.3920 | pass | 3 / 3 |
| canonical, gain 2.0 | 0.3205 | pass | 3 / 3 |
| Candidate A, gain 1.0 | not available | crash before report | - |
| Candidate B, gain 1.0 | 0.3179 | pass | 2 / 3 (`C2_12` noise `-55.66`, SNR `37.19`) |

Candidate A's disjoint-calibration three-face rerun crashed with a Windows
access violation before writing a report. Its earlier three-face pass (`0.405`)
used the all-19-face calibration and is not clean evidence.

Eight held-out faces (`C1_01`, `C1_03`, `C1_05`, `C2_08`, `C1_04`, `C2_07`,
`C2_13`, `C2_15`):

| setting | max ECAPA | worst pair | audio | failure detail |
|---|---:|---|---:|---|
| canonical, gain 1.0 | **0.9486** | `C1_03` / `C1_05` | 8 / 8 | identity fail |
| canonical, gain 2.0 | **0.8939** | `C1_03` / `C1_05` | 7 / 8 | `C2_08` degenerates: 36.8 s, noise `-24.75`, SNR `0.01` |
| Candidate A, gain 1.0 | **0.7914** | `C1_04` / `C2_15` | 7 / 8 | `C1_01` SNR `39.57` |
| Candidate B, gain 1.0 | **0.6415** | `C1_03` / `C1_05` | noise 6 / 8, SNR 5 / 8 | `C1_01` `-56.53/37.96`; `C2_08` `-58.07/38.88`; `C2_13` `-56.48/37.01` |

Candidate A also has `C1_01` / `C2_08` at `0.7782`; Candidate B has
`C1_04` / `C2_15` at `0.5773`. The canonical collapse reproduces with disjoint
calibration, so leakage was not the cause.

Artifacts:

- `results/audit/style_calibration/calibration_canonical_rebuilt_holdout8.json`
- `results/audit/style_calibration/calibration_whiten_norm_v1_holdout8.json`
- `results/audit/style_calibration/calibration_pairwise_v1_holdout8.json`
- `results/demo/identity_canonical_holdout8cal_gain1_8faces/report.json`
- `results/demo/identity_canonical_holdout8cal_gain2_8faces/report.json`
- `results/demo/identity_whiten_norm_holdout8cal_gain1_8faces/report.json`
- `results/demo/identity_pairwise_v1_holdout8cal_gain1_8faces/report.json`

---

## 6. Why calibration cannot separate these faces

The adapter output is nearly directionless across same-gender faces. The table
shows held-out same-gender pairs: `e_s` cosine (input), raw adapter style
cosine, centered style cosine, calibrated style cosine at gain 1 and 2, and the
resulting audio ECAPA cosine.

| pair | `e_s` | raw style | centered | cal. gain 1 | cal. gain 2 | audio gain 1 | audio gain 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C1_03/C1_05` | 0.581 | **0.988** | 0.984 | 0.995 | 0.983 | **0.949** | 0.894 |
| `C1_01/C1_03` | 0.316 | 0.959 | 0.761 | 0.898 | 0.577 | 0.813 | 0.688 |
| `C1_01/C1_05` | 0.364 | 0.948 | 0.773 | 0.899 | 0.606 | 0.793 | 0.602 |
| `C1_04/C2_15` | 0.069 | 0.985 | 0.524 | 0.980 | 0.930 | 0.751 | 0.543 |
| `C2_13/C2_15` | 0.076 | 0.986 | 0.260 | 0.982 | 0.934 | 0.709 | 0.396 |
| `C2_07/C2_15` | -0.103 | 0.973 | 0.225 | 0.964 | 0.856 | 0.633 | 0.357 |

Raw style cosine is `0.948-0.988` for every canonical same-gender pair,
including pairs whose identity inputs are nearly orthogonal (`C1_04/C2_15` at
`0.069`). The adapter is discarding input distinctions before calibration.
Candidate B reduces the raw range to `0.783-0.935`, and the remaining high pairs
are the ones that still fail:

| pair | `e_s` | raw style | centered | cal. gain 1 | audio |
|---|---:|---:|---:|---:|---:|
| `C1_03/C1_05` | 0.581 | 0.883 | 0.676 | 0.907 | **0.641** |
| `C1_04/C2_15` | 0.069 | 0.923 | 0.523 | 0.930 | **0.577** |
| `C1_04/C2_07` | -0.024 | 0.909 | 0.485 | 0.920 | 0.150 |
| `C1_01/C2_08` | 0.570 | 0.935 | 0.211 | 0.914 | 0.200 |
| `C1_01/C1_05` | 0.364 | 0.809 | 0.108 | 0.816 | 0.150 |
| `C2_07/C2_13` | 0.192 | 0.783 | -0.431 | 0.816 | 0.078 |

Audio ECAPA now tracks centered style cosine rather than the calibrated cosine:
the six pairs above span centered `-0.431` to `0.676` and audio `0.078` to
`0.641`, while most other held-out pairs fall to `0.08-0.26`. The two failures
are the only pairs with centered cosine above `0.5`. Gain 2 can reduce some
audio cosines, but it does not rescue the worst pair and it degenerates
`C2_08`; gain is not a substitute for style separation.

The same defect is systemic on validation. Centered different-speaker cosine
over `12.5M` pairs:

| set | mean | p90 | p99 | max | frac > 0.5 | frac > 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| real targets | 0.0021 | 0.4231 | 0.6434 | 0.9237 | 5.4% | 0.03% |
| canonical adapter | 0.0028 | **0.9023** | 0.9846 | 0.9996 | **34.0%** | 19.2% |
| Candidate A | 0.0025 | **0.8375** | 0.9435 | 0.9987 | **38.3%** | 14.0% |
| Candidate B | 0.0152 | **0.5273** | 0.7840 | 0.9972 | **11.6%** | 0.8% |

On the speaker-disjoint test split (`361,308` different-speaker pairs) the same
ordering holds:

| set | mean | p90 | p99 | max | frac > 0.5 | frac > 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| real targets | 0.0440 | 0.5051 | 0.6840 | 0.8157 | 10.2% | 0.02% |
| canonical adapter | 0.0124 | **0.9200** | 0.9858 | 0.9990 | **37.7%** | 22.5% |
| Candidate A | 0.0245 | **0.8722** | 0.9582 | 0.9923 | **42.2%** | 18.8% |
| Candidate B | 0.0458 | **0.5925** | 0.8197 | 0.9670 | **15.8%** | 1.4% |

Candidate B moves the tail much closer to the target distribution. It does not
match it, and the residual max remains high (`0.9670` test). The mean diversity
loss matches only the average (`~0.002-0.015`) and never constrained this tail;
the pairwise structure term is what reduced it.

A train-fitted ridge map is the linear predictability ceiling from the same
CVAE inputs; scaling its residual to the target norm gives `C1_03/C1_05` style
cosine `0.931`, better than canonical `0.988` but still well above `0.5`, and
Candidate A reaches `0.874`. Candidate B's centered styles for the two residual
failures are `0.676` (`C1_03/C1_05`, ridge `0.656`) and `0.523`
(`C1_04/C2_15`, ridge `0.585`): it is at or beyond the linear ceiling, so the
remaining separation is not recoverable from the same `e_s` by a better adapter
objective alone.

---

## 7. TTS render probe: styles do separate voices

To isolate the adapter from the renderer, eight real AMI styles (4 female,
4 male) were synthesized through the frozen StyleTTS2 backend with fixed text,
seed, and diffusion settings, and their ECAPA embeddings compared. Style pairs
span cosine `0.175-0.949`. Selection used style cosine only; ECAPA was measured
after synthesis and never used to choose styles or audio.

| pairwise style cosine | pairs | ECAPA mean | ECAPA min | ECAPA max |
|---|---:|---:|---:|---:|
| `< 0.5` | 6 | **0.301** | 0.144 | 0.475 |
| `0.5 - 0.8` | 1 | 0.504 | 0.504 | 0.504 |
| `>= 0.8` | 6 | **0.601** | 0.421 | 0.722 |

Style-to-ECAPA Pearson is `0.899`. Distinct styles are rendered as distinct
voices: at style cosine below `0.5`, output ECAPA falls to `0.14-0.48`. At
style cosine above `0.8`, output ECAPA is `0.42-0.72`, which fails the neutral
gate for most pairs. The canonical adapter's `0.95-0.99` output is therefore the
primary bottleneck, not an inability of StyleTTS2 to render different voices.

Two caveats: even low-cosine real styles can fail the audio thresholds
(`FIE081` noise `-52.3`, SNR `34.3`; `MIO082` noise `-53.5`, SNR `35.8`), and
the probe uses real AMI styles, not styles paired with the demo faces. The gate
is strict and style separation must be well below `0.5` to leave margin.

Artifact: `results/audit/tts_style_identity_probe/report.json`.

---

## 8. Why the calibration path cannot fix this

- The dominant failure is upstream of calibration: for the canonical adapter,
  every same-gender held-out pair has raw style cosine `0.948-0.988`, and
  validation centered different-speaker p90 is `0.902` against `0.423` for real
  targets. A shared post-hoc transform cannot restore distinctions the adapter
  did not produce.
- The pairwise structure objective does reduce that tail: Candidate B reaches
  val p90 `0.5273` / test `0.5925`, and most held-out pairs fall to audio ECAPA
  `0.08-0.26`. But it still fails both gates: held-out identity max `0.6415`
  (`C1_03/C1_05`, `C1_04/C2_15` `0.5773`) and audio misses on `C1_01`, `C2_08`,
  and `C2_13`.
- The two remaining identity pairs are at the ridge linear ceiling from the same
  `e_s` (`C1_03/C1_05` centered `0.676` vs ridge `0.656`; `C1_04/C2_15` `0.523`
  vs `0.585`). More objective engineering on the same face embeddings is
  unlikely to close the gap; the next lever is paired face/voice supervision or
  a better face encoder.
- A mean translation that puts the style cloud on the natural TTS manifold
  adds a constant vector larger than the face-specific residual.  The TTS then
  renders a common natural voice.  This is the observed collapse.
- Variance matching (`gain ~= 2`) restores the real style spread but does not
  remove the common offset; on the three faces it still fails
  (`0.4762 > 0.4379`) and it degrades audio. With disjoint calibration it still
  fails the eight held-out faces (`0.8939`) and degenerates `C2_08`.
- Removing the offset's projection onto a linear identity decoder
  (`identity-null`) preserves the decoder's identity ordering by construction,
  but the TTS/ECAPA relationship is nonlinear.  It still fails the three-face
  gate (`0.4679`) and the held-out voices converge (`0.9473`).
- More diffusion steps clean the degenerate `C2_09` clip but leave the SNR at
  `20-27 dB` and collapse the other male pair to `0.638`.  This is not a fix.
- Selecting CVAE samples by ECAPA would violate the rule that evaluation
  recognizers are evaluation-only, and style-space selection alone was already
  shown to leave the calibrated worst pair above the gate.

The remaining failure is not a calibration scalar.  It is the adapter/input
contract for these faces: the adapter preserves identity in a linear decode but
collapses pairwise style directions, and even the best objective tested here
stops at the linear ceiling of the face embedding. The TTS can render distinct
voices from distinct styles; the missing distinctions must come from better
face/voice supervision or a stronger face representation.

---

## 9. Decision and next step

**Decision:** do not change the demo calibration, do not promote Candidate A or
Candidate B, and keep `checkpoints/style_adapter_cfvn_best.pt` unchanged.  The
P0 CLI/environment work remains as committed.

Candidate B was the pre-registered training fix. It improved the pairwise tail
substantially but failed both promotion gates, and its worst pairs are at the
ridge ceiling for the same input. Further scalar calibration sweeps and further
objectives on the same `e_s` are not the next step.

The next credible fix is training/data work:

1. Collect paired face/voice data for the demo face distribution, so the
   adapter is supervised on the input domain it is served on.
2. Or replace/augment the face representation with one that separates the demo
   faces more strongly; the current `e_s` is the limiting factor for
   `C1_03/C1_05` and `C1_04/C2_15`.
3. Keep the neutral-pair ECAPA gate and the audio thresholds unchanged; any
   candidate must pass both on the three selection faces and at least eight
   held-out faces before the demo or a checkpoint changes.

Do not claim perceptual distinctness, MOS, or identity accuracy from these
numbers.  ECAPA is a speaker-embedding proxy and the audio metrics are signal
proxies.

---

## 10. Reproduction

```powershell
# canonical, no calibration
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/style_adapter_cfvn_best.pt `
    --calibration-alpha 0.0 `
    --output-dir results/demo/identity_candidate_canonical_alpha0

# canonical, full demo calibration
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/style_adapter_cfvn_best.pt `
    --calibration-alpha 1.0 `
    --output-dir results/demo/identity_candidate_canonical_alpha1

# variance matching
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/style_adapter_cfvn_best.pt `
    --calibration-alpha 1.0 --calibration-deviation-gain 2.0 `
    --output-dir results/demo/identity_candidate_canonical_spread2

# identity-null calibration artifact (deterministic; default seed 42)
python scripts/evaluation/build_identity_null_calibration.py --seed 42

# held-out validation
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/style_adapter_cfvn_best.pt `
    --calibration results/audit/style_calibration/calibration_identity_null.json `
    --calibration-alpha 1.0 `
    --faces C1_01_young_female_Zendaya C1_03_adult_female_Sandra_Bullock `
        C1_05_senior_female_Helen_Mirren C2_08_south_asian_female_Priyanka_Chopra `
        C1_04_adult_male_George_Clooney C2_07_south_asian_male_Shah_Rukh_Khan `
        C2_13_caucasian_male_Brad_Pitt C2_15_latino_male_Pedro_Pascal `
    --output-dir results/demo/identity_validation_idnull_8faces_final

# disjoint calibration fit (exclude the eight held-out faces)
$exclude = 'C1_01 C1_03 C1_05 C2_08 C1_04 C2_07 C2_13 C2_15'.Split(' ')
python scripts/training/build_adapter_style_calibration.py `
    --adapter checkpoints/style_adapter_cfvn_best.pt `
    --exclude-faces $exclude `
    --output results/audit/style_calibration/calibration_canonical_rebuilt_holdout8.json
python scripts/training/build_adapter_style_calibration.py `
    --adapter checkpoints/experiments/style_adapter_residual_whiten_norm_v1.pt `
    --exclude-faces $exclude `
    --output results/audit/style_calibration/calibration_whiten_norm_v1_holdout8.json

# disjoint-calibration held-out identities and audio
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/style_adapter_cfvn_best.pt `
    --calibration results/audit/style_calibration/calibration_canonical_rebuilt_holdout8.json `
    --calibration-alpha 1.0 --faces $exclude `
    --output-dir results/demo/identity_canonical_holdout8cal_gain1_8faces
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/style_adapter_cfvn_best.pt `
    --calibration results/audit/style_calibration/calibration_canonical_rebuilt_holdout8.json `
    --calibration-alpha 1.0 --calibration-deviation-gain 2.0 --faces $exclude `
    --output-dir results/demo/identity_canonical_holdout8cal_gain2_8faces
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/experiments/style_adapter_residual_whiten_norm_v1.pt `
    --calibration results/audit/style_calibration/calibration_whiten_norm_v1_holdout8.json `
    --calibration-alpha 1.0 --faces $exclude `
    --output-dir results/demo/identity_whiten_norm_holdout8cal_gain1_8faces

# Candidate B: pairwise structure objective
python -m voix.training.trainer_adapter_cfvn `
    --cfvn results/checkpoints/cfvn_v2_best.pt `
    --cvae checkpoints/cvae_cfvn_face192_latent128_diversity.pt `
    --style-targets data/processed/style_targets_voxceleb2_5clip.pt `
    --output checkpoints/experiments/style_adapter_pairwise_v1.pt `
    --whiten-input --residual-objective `
    --center-output-bias --freeze-output-bias `
    --norm-loss-mode symmetric `
    --lambda-div 1.0 --lambda-norm 1.0 --lambda-mean 1.0 --lambda-pair 1.0 `
    --selection residual --epochs 40 --patience 12 --seed 42
python scripts/training/build_adapter_style_calibration.py `
    --adapter checkpoints/experiments/style_adapter_pairwise_v1.pt `
    --exclude-faces $exclude `
    --output results/audit/style_calibration/calibration_pairwise_v1_holdout8.json
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/experiments/style_adapter_pairwise_v1.pt `
    --calibration results/audit/style_calibration/calibration_pairwise_v1_holdout8.json `
    --calibration-alpha 1.0 `
    --output-dir results/demo/identity_pairwise_v1_holdout8cal_gain1
python scripts/demo/synthesize_identity_candidate.py `
    --adapter checkpoints/experiments/style_adapter_pairwise_v1.pt `
    --calibration results/audit/style_calibration/calibration_pairwise_v1_holdout8.json `
    --calibration-alpha 1.0 --faces $exclude `
    --output-dir results/demo/identity_pairwise_v1_holdout8cal_gain1_8faces
python scripts/evaluation/evaluate_adapter_v2.py `
    --adapter checkpoints/experiments/style_adapter_pairwise_v1.pt `
    --output results/adapter_v2_evaluation_pairwise_v1.json
python scripts/evaluation/diagnose_adapter_residual.py `
    --adapter checkpoints/experiments/style_adapter_pairwise_v1.pt `
    --output results/adapter_residual_diagnosis_pairwise_v1.json

# real-style TTS render probe (selection by style cosine only)
python scripts/audit/probe_tts_style_identity.py
```

Audio quality is measured with
`scripts/audit/audit_audio_quality.py::measure_clip`, the same function used by
the demo verification.
