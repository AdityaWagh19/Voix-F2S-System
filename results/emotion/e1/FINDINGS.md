# E1 - Class-mean emotion residual (exploratory)

**Status: exploratory positive under controlled scalar modifiers; not a
deployment result.** The controlled rerun (`e1c`) holds every post-synthesis
scalar modifier at neutral. It passes all six pre-registered F0/duration
direction checks and the identity guardrail's relative check. Absolute identity
is still not established: generated neutral ECAPA similarity to the real speaker
is 0.204, below the 0.50 sanity threshold. The emotion-classifier proxy is
invalid for generated audio. No human MOS, CMOS, or emotion-identification
study was run.

## Correction to the original E1 report

The original E1 run (`results/emotion/e1/`) did **not** hold scalar modifiers
constant. It applied each emotion's post-synthesis scalar profile, so its
direction effects mixed the class-mean residual with F0/energy/duration
modifiers. Those numbers are preserved in
`results/emotion/CONFOUNDED_COMPARISON.md` as the confounded condition. They are
not residual evidence.

The authoritative controlled run is `results/emotion/e1c/`, generated with
`--scalar-modifiers neutral`. The original E1 report's identity failure was also
driven by the scalar profiles; the controlled run's relative identity check
passes. The absolute identity warning remains.

## What was run

- Identity base: frozen StyleTTS2 style/predictor encoder on one real neutral
  clip per actor. No face, CFVN, CVAE, or canonical face-to-voice adapter is
  used. Canonical checkpoint hashes were verified unchanged.
- Residual: speaker-balanced class mean estimated from the 73 CREMA-D training
  actors only; the 9 validation actors were never used to estimate it.
- Primary condition: class-mean **prosody-only** residual (dims 128:256).
- Controlled generation: neutral, happy, sad, angry; 108 pairs per emotion,
  432 rows, 9 validation actors, seed 0, 10 diffusion steps, watermark
  disabled, `duration_scale=1.0`, `alpha=0.18`, `beta=0.22`,
  `scalar_modifiers=neutral`.
- Neutral-zero: exact (`max_abs_error=0.0`); neutral audio is byte-identical to
  the E0 neutral audio, so comparisons are paired.
- 0 generation failures.

Artifacts:

| Item | Path / SHA256 |
|---|---|
| Paired manifest | `data/processed/crema_d_paired_eval_val.csv` / `b782e6a5a9d3d5b88f669e5836803182573fac4df42bddc892fb8e42da6b6912` |
| Class-mean artifact | `checkpoints/emotion_residuals/class_mean.pt` / `6bb4f17ff4bfaf4a9fe4b69514ba3b2e83f4dfe4c8a80f8706c9172cf4bb4408` |
| Controlled index | `results/emotion/e1c/generation_index_prosody.csv` |
| Controlled generation report | `results/emotion/e1c/generation_report_prosody.json` |
| Controlled evaluation report | `results/emotion/e1c/report_prosody.json` |

## Controlled direction gate

Paired emotion-minus-neutral effects, speaker-cluster bootstrap CIs over the 9
validation actors (108 pairs per emotion). Expected directions are
pre-registered: happy/angry F0 up, sad F0 down; happy/angry duration down, sad
duration up.

| Condition | F0 delta (Hz) [95% CI] | Duration delta (s) [95% CI] | Gate |
|---|---|---|---|
| class-mean prosody, happy | **+16.19 [+9.68, +22.00]** | **-0.055 [-0.063, -0.047]** | pass |
| class-mean prosody, sad | **-11.59 [-21.75, -2.99]** | **+0.048 [+0.040, +0.055]** | pass |
| class-mean prosody, angry | **+18.13 [+12.79, +23.57]** | **-0.054 [-0.065, -0.044]** | pass |

All six checks pass. The effects are smaller than the confounded run because
the post-synthesis F0/energy/duration profiles are no longer adding to the
residual. The direction is consistent across the nine validation actors; the
speaker-cluster CIs quantify actor sampling only.

The earlier full-residual ablation is not part of the controlled rerun. In the
confounded condition it failed happy F0, so prosody-only remains the primary
class-mean baseline.

## Identity guardrail

The guardrail compares each condition's generated audio to the same actor's
generated neutral. Because generated neutral itself is far from the real
speaker, only relative comparisons are interpretable.

| Condition | ECAPA to generated neutral | ECAPA to real speaker | Reference drop [95% CI] |
|---|---|---|---|
| generated neutral | 1.000 | 0.204 | - |
| class-mean prosody, happy | 0.893 | 0.185 | +0.019 [+0.002, +0.033] |
| class-mean prosody, sad | 0.928 | 0.194 | +0.009 [+0.003, +0.016] |
| class-mean prosody, angry | 0.901 | 0.189 | +0.015 [-0.001, +0.030] |

`reference_drop` is `neutral_reference - condition_reference`; positive means
the condition moved away from the real speaker relative to that actor's
generated neutral. All three drops are below the pre-registered 0.05 budget,
and all three ECAPA-to-generated-neutral values are above 0.80, so the identity
gate passes its relative checks. The neutral-sanity check still fails:
generated neutral ECAPA to the real speaker is 0.204 < 0.50. This is a
domain-shift warning, not evidence that the residual preserves real identity.
Do not claim absolute identity preservation.

## Emotion classifier proxy

The generated audio is far outside the real-speech classifier feature
distribution (standardized centroid distance 67.4, 8.4x the real validation
spread; invalid threshold 3.0x). The classifier collapses to a single predicted
class, so generated accuracy and macro-F1 are not reported as emotion results.
The real-speech calibration remains a modest evaluator only; it does not
validate generated emotion.

No human MOS, CMOS, or emotion-identification study was run. MOS is
`unavailable`.

## Confounded original E1 run

For the record, the original run applied each emotion's post-synthesis scalar
profile. Its direction effects were larger and its identity check failed:

| Condition | F0 delta (Hz) [95% CI] | Duration delta (s) [95% CI] | ECAPA to generated neutral |
|---|---|---|---|
| happy | +23.37 [+1.45, +42.51] | -0.255 [-0.270, -0.241] | 0.279 |
| sad | -37.71 [-49.08, -27.57] | +0.399 [+0.386, +0.413] | 0.211 |
| angry | +25.14 [+4.37, +45.52] | -0.324 [-0.339, -0.310] | 0.326 |

These are confounded by the scalar modifiers and are **not** evidence for the
class-mean residual. They are retained only to document the correction.

## Interpretation

- E1 supports one narrow claim: a speaker-balanced, prosody-only class-mean
  residual estimated from real paired CREMA-D training speech moves F0 and
  duration in the expected directions on held-out actors when post-synthesis
  scalar modifiers are held neutral.
- E1 does **not** support claims of better perceived emotion, better identity,
  or better naturalness. The classifier cannot adjudicate generated emotion,
  absolute identity is below the sanity threshold, and no listeners were run.
- The effect is exploratory: CREMA-D is acted speech, every emotion class is
  below the 2 h confirmatory threshold, and only 9 validation actors are used.
- Fear and disgust are not part of the controlled E1 gate. The original
  fear/disgust rows used a neutral scalar fallback and are not comparable.

## Decision

Keep the prosody-only class mean as the E2 baseline. E2 must be judged on the
same controlled grid. The E2 main adapter's controlled result fails sad, and
the corrected label-shuffled control is reported separately in
`results/emotion/FINDINGS.md`. No learned-emotion claim is made from E2.

## Reproduction

```powershell
python scripts/build_emotion_class_mean.py --reuse-cache
python scripts/generate_crema_e0.py `
  --condition class_mean --residual-variant prosody_only `
  --emotions happy,sad,angry --scalar-modifiers neutral `
  --condition-label class_mean_prosody `
  --output-dir results/emotion/e1c/audio `
  --index results/emotion/e1c/generation_index_prosody.csv `
  --report results/emotion/e1c/generation_report_prosody.json
python scripts/evaluate_crema_e0.py `
  --index results/emotion/e1c/generation_index_prosody.csv `
  --generation-report results/emotion/e1c/generation_report_prosody.json `
  --output results/emotion/e1c/report_prosody.json `
  --run-id e1_class_mean_prosody_neutral_scalars `
  --device cuda --workers 4
```

## Limitations

- CREMA-D is acted speech; every emotion class is below the 2 h confirmatory
  threshold. This is exploratory.
- One generation seed. The speaker-cluster CIs quantify actor sampling, not
  diffusion-sampling or training-seed variance.
- The identity base is the StyleTTS2 style encoder, not the canonical face
  adapter. Results do not transfer automatically to the face-to-voice path.
- Generated neutral ECAPA is 0.204, below the 0.50 sanity threshold. Absolute
  identity numbers are domain-shifted; only paired comparisons are
  interpretable, and even those are relative to a shifted neutral.
- The classifier proxy is invalid for generated audio; no generated emotion
  accuracy is claimed.
- No human listening test; perceptual naturalness and emotion remain
  unmeasured.
