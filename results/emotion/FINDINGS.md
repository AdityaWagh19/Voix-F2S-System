# Emotion track findings (exploratory)

**Status: E1 is a narrow positive result; E2 is a negative result.** Under
controlled post-synthesis scalars (`--scalar-modifiers neutral`), the
speaker-balanced class-mean prosody residual passes all six pre-registered
F0/duration direction checks on held-out CREMA-D actors. The learned E2 adapter
passes four of six and fails both sad checks, so it does not beat the simple
class-mean baseline and is not promoted. The corrected label-shuffled null
passes one of six, which is the expected null behaviour. No human MOS,
CMOS, or emotion-identification study was run; the emotion-classifier proxy is
invalid for generated audio. This is not a deployment result.

## Data and protocol

- Corpus: CREMA-D, 6,892 clips / 91 actors. Every emotion class is below the 2 h
  confirmatory threshold, so this track is exploratory.
- Identity base: frozen StyleTTS2 style/predictor encoder on one real neutral
  clip per actor. No face, CFVN, CVAE, or canonical face-to-voice adapter is
  used. Canonical checkpoint hashes were verified unchanged.
- Split: 73 train actors, 9 validation actors, 9 test actors held out. The
  class mean and both adapters use train actors only. Validation actors are used
  for every number in this report; test actors are untouched.
- Paired grid: 108 neutral/emotion pairs per emotion over 9 validation actors
  and 12 sentences (`data/processed/crema_d_paired_eval_val.csv`).
- Generation: seed 0, 10 diffusion steps, watermark disabled,
  `duration_scale=1.0`, `alpha=0.18`, `beta=0.22`, and neutral scalar modifiers
  for every emotion.
- Neutral-zero is exact (`max_abs_error=0.0`) in every controlled run. Neutral
  audio is byte-identical across E1c/E2c, so condition comparisons are paired.

## Correction history

Two methodological corrections were made during the track and are part of the
record:

1. **Scalar confound.** The first E1/E2 pass applied each emotion's
   post-synthesis F0/energy/duration profile. Those runs are preserved in
   `results/emotion/CONTROLLED_COMPARISON.md` as the `emotion_scalars`
   conditions, but they are confounded and are not residual evidence. Only the
   `neutral_scalars` reruns are authoritative.
2. **Control mislabel.** The first "label-shuffled" control permuted emotion
   labels and target residuals together while leaving each neutral style fixed.
   That preserves the label-to-target association, so it was actually a
   style-shuffled diagnostic. It is preserved as
   `checkpoints/emotion_adapter/e2_adapter_style_shuffled_seed0.pt`. The
   corrected control (`shuffle_labels`) permutes emotion labels only and keeps
   each neutral style's real target residual fixed. That is the null reported
   below.

## Controlled direction results

Paired emotion-minus-neutral effects; speaker-cluster bootstrap CIs over the 9
validation actors, 108 pairs per emotion. Expected directions were
pre-registered before the controlled rerun.

| Condition | Happy F0 (Hz) | Happy dur. (s) | Sad F0 (Hz) | Sad dur. (s) | Angry F0 (Hz) | Angry dur. (s) | Gate |
|---|---|---|---|---|---|---|---|
| E1 class-mean prosody | **+16.19** [+9.68, +22.00] | **-0.055** [-0.063, -0.047] | **-11.59** [-21.75, -2.99] | **+0.048** [+0.040, +0.055] | **+18.13** [+12.79, +23.57] | **-0.054** [-0.065, -0.044] | **6/6** |
| E2 learned residual | **+18.63** [+10.05, +25.72] | **-0.034** [-0.052, -0.016] | +4.21 [-1.13, +9.70] | +0.016 [-0.011, +0.043] | **+20.32** [+14.54, +26.79] | **-0.027** [-0.044, -0.013] | **4/6** |
| E2 corrected label-shuffled null | **+12.94** [+4.42, +21.39] | -0.005 [-0.034, +0.023] | +9.81 [+3.11, +17.18] | -0.005 [-0.034, +0.024] | +7.94 [-1.53, +17.57] | -0.003 [-0.033, +0.025] | **1/6** |

Bold values are checks that passed: the CI excludes zero in the expected
direction. The null's single pass is happy F0. Its predictions are nearly
label-invariant (pairwise cosine about 0.99), so that pass reflects a generic
average residual that raises F0, not a learned happy-specific direction. All
three duration checks fail in the null, and sad/angry F0 fail. This is the
expected null and it strengthens the specificity of the E2 duration and angry
F0 effects, but it does not rescue E2's sad failure.

### Holm sensitivity (post-hoc)

The pre-registered gate uses speaker-cluster bootstrap CIs. As a post-hoc
robustness check, `scripts/emotion_holm_sensitivity.py` reconstructs a
one-sample t test from the reported paired mean, SD, and `n_pairs` (108 per
emotion) and applies Holm correction across the six direction checks. This is a
summary-level sensitivity analysis, not a replacement for the pre-registered
gate, and it assumes approximate normality of the paired differences.

| Condition | Two-sided Holm passes | Expected-direction Holm passes | Note |
|---|---:|---:|---|
| E1 class-mean | 6/6 | 6/6 | weakest is sad F0, Holm p = 0.0013 |
| E2 learned residual | 5/6 | 5/6 | sad F0 fails, Holm p = 0.147 |
| E2 corrected label null | 2/6 | 1/6 | happy F0 passes; sad F0 is significant in the wrong direction (+9.81 Hz) |

E1's direction result survives a simple multiple-comparison correction. E2 still
fails: sad F0 is null after correction. One discrepancy needs to be stated
plainly: E2's sad duration has a significant t test (Holm p = 0.0071) but its
pre-registered bootstrap CI [-0.011, +0.043] includes zero. The mean shift is
only +0.016 s, and the paired distribution is evidently skewed or heavy-tailed.
The pre-registered bootstrap gate remains authoritative, so E2 stays at 4/6 and
is not promoted. The null's happy F0 pass is a generic F0 shift; its sad F0
shift is significant but in the wrong direction, so it is not a directional
pass. The sensitivity analysis therefore supports the existing verdicts rather
than changing them.

### E1 class-mean prosody

The speaker-balanced class mean is estimated from 4,031 real pairs over 73
train actors. The primary condition uses only the prosody half (dims 128:256).
It passes all six checks. Effects are smaller than the confounded run because
the post-synthesis F0/energy/duration profiles are no longer adding to the
residual. The effect is consistent across the nine validation actors; the
speaker-cluster CIs quantify actor sampling only.

E1 is the only emotion condition that supports a positive claim: a simple
prosody-only class-mean residual from real paired speech moves F0 and duration
in the expected directions on held-out actors. It does **not** establish
perceived emotion, identity preservation, or naturalness.

### E2 learned residual

The adapter is a 199,168-parameter bounded MLP (256-D frozen style input,
6-emotion one-hot, intensity; 256-D residual, `max_norm=0.5`; timbre half
weighted 0.25). It was trained on 4,031 examples from 73 actors with
speaker-disjoint internal early stopping (3,430 fit / 601 internal validation).
Best epoch 6, internal validation cosine 0.539.

Controlled generation produced all five emotions (108 pairs each, 648 clips,
0 failures). Happy and angry pass both checks. Sad fails both: F0 +4.21 Hz with
a CI crossing zero and duration +0.016 s with a CI crossing zero. The learned
adapter therefore does not beat the class-mean baseline on the pre-registered
gate (4/6 vs 6/6). It produces a larger F0 shift for happy and angry but a
smaller duration shift for all three, and it reverses the sad direction. No
learned-emotion claim is made.

Fear and disgust have no fixed-delta artifacts, so they are outside the
direction gate. Their E2 effects are descriptive only: fear F0 +24.51 Hz
[+15.99, +33.70], duration -0.024 s [-0.047, -0.001]; disgust F0 +7.52 Hz
[+3.36, +11.87], duration +0.039 s [+0.007, +0.069]. They are not evidence of
correct fear or disgust control.

## Why the style-shuffled control is not a null

The historical style-shuffled diagnostic passes 6/6, which looks stronger than
the main adapter. That is expected: it reassigns label/target pairs to neutral
styles but keeps each label's real target distribution, so it learns the class
mean independently of the input style. Its predicted residuals align with the
class-mean targets at cosine 0.94-0.98, versus 0.75-0.94 for the main adapter
and 0.41-0.66 for the corrected label null. Passing the direction gate is
therefore not evidence of learned, style-conditioned emotion; it is evidence
that the gate measures the class direction. The corrected label null is the
condition that tests label specificity, and it fails 5/6.

## Identity guardrail

The guardrail compares each condition's generated audio to the same actor's
generated neutral. Relative checks pass, but absolute identity is not
established.

| Condition | ECAPA to generated neutral | ECAPA to real speaker | Reference drop [95% CI] |
|---|---|---|---|
| generated neutral | 1.000 | 0.204 | - |
| E1 happy | 0.893 | 0.185 | +0.019 [+0.002, +0.033] |
| E1 sad | 0.928 | 0.194 | +0.009 [+0.003, +0.016] |
| E1 angry | 0.901 | 0.189 | +0.015 [-0.001, +0.030] |
| E2 happy | 0.881 | 0.194 | +0.010 [-0.004, +0.023] |
| E2 sad | 0.900 | 0.181 | +0.023 [+0.007, +0.034] |
| E2 angry | 0.889 | 0.206 | -0.002 [-0.013, +0.009] |
| E2 label null, all | 0.888-0.890 | 0.191-0.196 | +0.008 to +0.013 |

All ECAPA-to-generated-neutral values are above 0.80 and all reference drops
are below 0.05, so the relative gate passes. The neutral-sanity check still
fails: generated neutral ECAPA to the real speaker is 0.204 < 0.50. This is a
domain-shift warning, not evidence that the residual preserves real identity.
Do not claim absolute identity preservation.

## Intelligibility (WER)

WER was measured with Whisper `base` on the generated audio against the known
CREMA-D transcript: 108 clips per emotion, plus the shared neutral condition.
It measures intelligibility only; it is not a naturalness or emotion metric.

| Run | Neutral | Happy | Sad | Angry | Fear | Disgust | Mean | Worst |
|---|---|---|---|---|---|---|---|---|
| E1 class-mean prosody | 0.0291 | 0.0310 | 0.0387 | 0.0328 | - | - | 0.0329 | 0.0387 |
| E2 learned residual | 0.0291 | 0.0367 | 0.0341 | 0.0301 | 0.0317 | 0.0278 | 0.0316 | 0.0367 |

All WERs are low (2.8-3.9%). The largest condition difference is 0.0096
absolute, and no confidence interval or significance test was computed for WER
differences on these short clips. The data therefore support "intelligible" but
do not support a claim that one condition is more intelligible than another.
The label-shuffled and style-shuffled controls were not scored for WER.

## Emotion classifier proxy

Generated audio is far outside the real-speech classifier feature
distribution. The neutral standardized centroid distance is 67.35, 8.36x the
real validation spread (invalid threshold 3.0x); over all conditions the ratio
is about 6.05x. The classifier collapses to a single predicted class, so
generated accuracy (0.0 for E1, 0.167 for E2) and macro-F1 are not emotion
results. The real-speech calibration is 0.535 accuracy / 0.532 macro-F1 and is
a modest evaluator only; it does not validate generated emotion.

No human MOS, CMOS, or emotion-identification study was run. MOS is
`unavailable`. WER is measured and reported above, but human perceptual
naturalness and emotion remain unmeasured, so this is not a deployment result.

## Decision

- Keep the E1 prosody-only class mean as the supported emotion baseline.
- Do not promote the E2 adapter. It fails sad and does not beat E1 on the
  pre-registered gate. The corrected label null shows the E2 happy/angry
  duration and angry F0 effects are not fully explained by label-invariant
  residual structure, but that is not enough for a learned-emotion claim.
- Do not start E3 from CREMA-D. E3 requires MSP-Podcast or another corpus with
  at least 2 h per emotion class; that agreement is still blocked.
- The showable emotion demo may use E1 class-mean prosody for happy, sad, and
  angry with neutral scalars. It must label the result exploratory and must not
  claim learned emotion, absolute identity, naturalness, or MOS.
- Fear and disgust remain unavailable as controlled conditions: there is no
  fixed-delta artifact for them, and the learned adapter's descriptive effects
  are not validated.

## Artifacts

| Item | Path / SHA256 |
|---|---|
| Paired manifest | `data/processed/crema_d_paired_eval_val.csv` / `b782e6a5a9d3d5b88f669e5836803182573fac4df42bddc892fb8e42da6b6912` |
| Class-mean artifact | `checkpoints/emotion_residuals/class_mean.pt` / `6bb4f17ff4bfaf4a9fe4b69514ba3b2e83f4dfe4c8a80f8706c9172cf4bb4408` |
| E1c report | `results/emotion/e1c/report_prosody.json` |
| E2 main adapter | `checkpoints/emotion_adapter/e2_adapter_seed0.pt` / file `dfc57dfb3debef9400944fdf70963c692cdd150dda36f0c6f2915ad50a0b0545`, state_dict `43970fbc52d79828e8a796e02c5115b35fbbcb3d9721e6e8196d026ec45bf540` |
| E2 label-shuffled control | `checkpoints/emotion_adapter/e2_adapter_label_shuffled_seed0.pt` / file `1091059140e2138a618f91e2db5e34cd135131301b28fdbff0109a118ef175a0`, state_dict `47cf1ffce7dfbbef21870c3c2309809266282fc233ec4dd223a8686fa62dda7c` |
| E2 style-shuffled diagnostic | `checkpoints/emotion_adapter/e2_adapter_style_shuffled_seed0.pt` / file `0b8e94d3203f3b91c0248f6559f53ba54bf3ecf1c90d4ff5b16929f36bcc823b`, state_dict `c2e0fa9252b305660929d122c6013896815d30adc138679c604674514db2af60` |
| E2 main controlled report | `results/emotion/e2c/report_main.json` |
| E2 label-shuffled report | `results/emotion/e2c/report_label_shuffled.json` |
| E2 style-shuffled report | `results/emotion/e2c/report_style_shuffled.json` |
| Holm sensitivity | `results/emotion/holm_sensitivity.json` / `scripts/emotion_holm_sensitivity.py` |
| Comparison table | `results/emotion/CONTROLLED_COMPARISON.md` / `controlled_comparison.json` |

The E2 main adapter was retrained after generation with an identical
`state_dict` (verified `43970fbc...`); the file hash changed only because
metadata/timestamps were rewritten. The generation report records the
generation-time file hash and the state_dict hash.

## Reproduction

```powershell
# Rebuild the class mean from train actors
python scripts/build_emotion_class_mean.py --reuse-cache

# E1 controlled generation + evaluation
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

# E2 main + corrected label-shuffled control
python scripts/train_emotion_adapter.py --device cpu --seed 0
python scripts/generate_crema_e0.py `
  --condition learned_residual `
  --adapter-artifact checkpoints/emotion_adapter/e2_adapter_seed0.pt `
  --scalar-modifiers neutral `
  --emotions happy,sad,angry,fear,disgust `
  --output-dir results/emotion/e2c/audio_main `
  --index results/emotion/e2c/generation_index_main.csv `
  --report results/emotion/e2c/generation_report_main.json `
  --condition-label learned_residual --device cuda --seeds 0 --diffusion-steps 10
python scripts/evaluate_crema_e0.py `
  --index results/emotion/e2c/generation_index_main.csv `
  --generation-report results/emotion/e2c/generation_report_main.json `
  --output results/emotion/e2c/report_main.json `
  --run-id e2_learned_residual_neutral_scalars `
  --device cuda --workers 4
# Repeat generation/evaluation with
#   --adapter-artifact checkpoints/emotion_adapter/e2_adapter_label_shuffled_seed0.pt
#   --condition-label learned_residual_label_shuffled
#   --output-dir results/emotion/e2c/audio_label_shuffled
#   --index results/emotion/e2c/generation_index_label_shuffled.csv
#   --report results/emotion/e2c/generation_report_label_shuffled.json
#   --output results/emotion/e2c/report_label_shuffled.json

# Comparison table
python scripts/summarize_emotion_controls.py `
  --report e1_class_mean_neutral_scalars=results/emotion/e1c/report_prosody.json `
  --report e2_main_neutral_scalars=results/emotion/e2c/report_main.json `
  --report e2_label_shuffled_neutral_scalars=results/emotion/e2c/report_label_shuffled.json `
  --report e2_style_shuffled_neutral_scalars=results/emotion/e2c/report_style_shuffled.json `
  --output-json results/emotion/controlled_comparison.json `
  --output-md results/emotion/CONTROLLED_COMPARISON.md

# Post-hoc Holm sensitivity (does not replace the pre-registered gate)
python scripts/emotion_holm_sensitivity.py
```

## Limitations

- CREMA-D is acted speech; every emotion class is below the 2 h confirmatory
  threshold. This is exploratory.
- One generation seed. The speaker-cluster CIs quantify actor sampling, not
  diffusion-sampling or training-seed variance.
- Only 9 validation actors; speaker-cluster intervals are provisional.
- The identity base is the StyleTTS2 style encoder, not the canonical face
  adapter. Results do not transfer automatically to the face-to-voice path.
- Generated neutral ECAPA is 0.204, below the 0.50 sanity threshold. Absolute
  identity numbers are domain-shifted; only paired comparisons are
  interpretable, and even those are relative to a shifted neutral.
- The classifier proxy is invalid for generated audio; no generated emotion
  accuracy is claimed.
- WER is measured (2.8-3.9% across conditions) but has no CI; it is not a
  naturalness measure and does not establish a condition difference.
- No human listening test; perceptual naturalness and emotion remain
  unmeasured.
- Fear and disgust have no controlled fixed-delta reference and are not part of
  the direction gate.
