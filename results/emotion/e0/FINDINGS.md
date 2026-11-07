# E0 fixed-delta harness result (seed 0, validation actors)

**Status: exploratory, gate not passed.** This is a real run on the 9 held-out
CREMA-D validation actors. It is not a MOS study and does not claim perceptual
emotion quality.

## Run

| Field | Value |
|---|---|
| Manifest | `data/processed/crema_d_paired_eval_val.csv` |
| Actors | 9 validation actors (train/test actors untouched) |
| Sentences | 12 per actor |
| Conditions | neutral, fixed delta happy/sad/angry |
| Seeds | 0 |
| Diffusion steps | 10 |
| Identity base | frozen StyleTTS2 style/predictor encoder on one real neutral clip per actor |
| Clips | 432 generated, 0 failures |
| Neutral-zero | exact, max abs error `0.0` |
| Report | `results/emotion/e0/report.json` |
| Generation | `results/emotion/e0/generation_report.json` |

Fear and disgust are unavailable for E0 because this checkout has no fixed-delta
artifact for them. They are reported as unavailable, not rendered as neutral.

## Paired effects (emotion minus generated neutral)

Speaker-cluster 95% percentile bootstrap, 10,000 resamples.

| Condition | Metric | Mean delta | 95% CI | Direction correct? |
|---|---|---:|---|---|
| happy | F0 mean (Hz) | +8.89 | [-15.05, +30.10] | yes, CI includes zero |
| happy | duration (s) | -0.265 | [-0.283, -0.249] | yes, CI excludes zero |
| happy | energy mean | +0.0076 | [+0.0068, +0.0084] | yes, CI excludes zero |
| sad | F0 mean (Hz) | -32.79 | [-43.34, -24.59] | yes, CI excludes zero |
| sad | duration (s) | +0.457 | [+0.442, +0.473] | yes, CI excludes zero |
| sad | energy mean | -0.0067 | [-0.0071, -0.0062] | yes, CI excludes zero |
| angry | F0 mean (Hz) | +7.81 | [-13.84, +27.35] | yes, CI includes zero |
| angry | duration (s) | -0.309 | [-0.322, -0.296] | yes, CI excludes zero |
| angry | energy mean | +0.0101 | [+0.0096, +0.0106] | yes, CI excludes zero |

The pre-registered E0 direction gate requires the expected direction **and** a
speaker-cluster CI that excludes zero. Duration and energy pass for all three
emotions. Sad F0 passes. Happy and angry F0 point in the expected direction but
their intervals include zero, so the gate fails.

Interpretation: the fixed deltas reliably change timing and energy, but their
F0 effect for high-arousal emotions is not stable across the 9 validation
speakers. The demo's perceived emotion cannot be attributed to a robust F0
shift for happy/angry on this evidence.

## Identity guardrail

| Condition | ECAPA to generated neutral | ECAPA to real actor reference | Reference drop |
|---|---:|---:|---:|
| neutral | 1.000 | 0.204 | baseline |
| happy | 0.272 | 0.110 | +0.094 |
| sad | 0.208 | 0.089 | +0.115 |
| angry | 0.328 | 0.115 | +0.089 |

- Generated neutral similarity to the real speaker is **0.204**, below the
  neutral-sanity threshold of 0.50.
- All emotion conditions fall below the deployment-style
  `min_ecapa_neutral=0.80` budget.
- The guardrail therefore reports identity drift, and condition comparisons are
  **relative only** because neutral sanity fails.

The fixed-delta path itself drifts identity even before emotion is applied; the
emotion deltas add further drift. This is consistent with the earlier demo
diagnostic (emotion-to-primary similarity 0.26-0.37).

## Emotion classifier proxy is invalid on generated audio

The train-only real-validation calibration model reaches 0.535 accuracy
(0.532 macro-F1), so the evaluator itself is not broken. On generated audio it
predicts `fear` for all 432 clips, including every neutral clip.

A feature-domain diagnostic explains why:

| Quantity | Value |
|---|---:|
| Standardized generated-neutral centroid distance from real validation neutral | 67.35 |
| Real validation neutral spread | 8.05 |
| Ratio | 8.36x |
| Invalid threshold | 3.0x |

Generated RMS is about 0.065 versus 0.014 for real neutral CREMA-D, and the MFCC
distribution is far outside the real-speech training range. The classifier is a
real-speech evaluator; generated audio is a severe domain shift. Its 0% accuracy
is reported as **unavailable for decision-making**, not as evidence that the
emotions are absent.

## Decision

- E0 does **not** pass its pre-registered gate. The harness is usable for
  prosody and identity measurement, but the fixed-delta F0 effect is not stable
  for happy/angry on held-out speakers.
- E1 class-mean comparisons remain exploratory. They can ask whether a
  data-derived residual changes prosody/identity relative to the fixed deltas;
  they cannot establish perceptual emotion quality.
- Do not use the emotion classifier as a primary endpoint until a
  generated-domain calibration or a human identification study exists.
- No human MOS or emotion-identification study was run. `mos` is `unavailable`.

## Artifacts

- `results/emotion/e0/generation_index.csv`
- `results/emotion/e0/generation_report.json`
- `results/emotion/e0/report.json`
- `results/emotion/e0/audio/`
- `results/emotion/e0/evaluate_seed0.log`
