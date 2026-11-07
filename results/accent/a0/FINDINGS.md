# Accent A0 findings: real GLOBE classifier and SSL ablation

**Status: base 222-D classifier selected; frozen SSL does not help.** The
ECAPA + prosody classifier passes the A0 stop/go rule (it beats both the
majority baseline and 2x chance on held-out speakers), but its validation
macro-F1 is 0.5422, below the 0.55 threshold that triggers the SSL fallback.
The fallback was run: SSL alone has lower macro-F1 (0.4968), and the combined
990-D feature set is effectively identical to base (+0.0002 macro-F1, far below
the +0.05 A1 rule). Base therefore remains the selected classifier.

The test split was scored once after selection: accuracy 0.5504, macro-F1
0.5630. That test value is above 0.55, but the gate and fallback decision were
made on validation before test, so the classifier is not retroactively promoted.

This is a real-speech classifier result. The GLOBE five-class data pass the
confirmatory size threshold, and the test was scored once after validation-only
selection, so the classifier result is confirmatory-eligible. The TTS accent
track is not: in A2 the same model scored 0.200 on generated audio, exactly
chance, and A2 was rejected on WER and classifier grounds. A3/A4 remain blocked
by A2.

## Data and splits

- GLOBE / Common Voice English, 9,534 clips / 884 speakers.
- Five trainable classes: US, England, Australian, Canadian, India/South Asia.
  Irish (855 clips) is evaluation-only and is not a training class.
- Speaker-disjoint splits for the five trainable classes:
  - train: 7,682 clips / 699 speakers
  - validation: 392 clips / 48 speakers
  - test: 605 clips / 44 speakers
- Labels are self-reported accent classes. `predicted_accent` was not used as a
  label.
- No face, CFVN, CVAE, or canonical style adapter is involved.

## Features

| Block | Dim | Source |
|---|---|---|
| base | 222 | ECAPA embedding + F0/duration/energy/spectral/ZCR statistics |
| ssl | 768 | frozen torchaudio `WAV2VEC2_BASE`, layer 8, mean-pooled, L2-normalized |
| combined | 990 | base followed by ssl |

The SSL cache was extracted for all 9,534 rows. A first detached run crashed on
a checkpoint write after row 8,500; a resumed run re-extracted the remaining
1,270 rows (including 236 rows that had failed transiently) with 0 failures.
The authoritative cache is `data/processed/globe_accent_ssl_features_full.npz`;
the partial checkpoint is preserved as
`data/processed/globe_accent_ssl_features_partial_8500.npz`. Base features,
clip ids, labels, speakers, and splits match the original 222-D cache exactly.

## Validation ablation

Model selection used validation speakers only. The grid is logistic regression
over a small C grid; the selected C and metrics are:

| Feature set | Dim | C | Val accuracy | Val macro-F1 | Per-class F1 (US / Eng / Aus / Can / Ind) |
|---|---|---|---|---|---|
| base | 222 | 0.01 | 0.5102 | **0.5422** | 0.453 / 0.652 / 0.404 / 0.302 / 0.899 |
| ssl | 768 | 0.01 | 0.5332 | 0.4968 | 0.575 / 0.408 / 0.254 / 0.435 / 0.812 |
| combined | 990 | 0.1 | 0.5255 | 0.5424 | 0.514 / 0.636 / 0.366 / 0.309 / 0.887 |

- SSL raises plain accuracy slightly (0.5332 vs 0.5102) but lowers macro-F1
  (0.4968 vs 0.5422). The accuracy gain comes from the large US class; England
  and Australia lose F1. This does not satisfy the A1 rule.
- Combined is 0.0002 macro-F1 above base, with a wider C and a different
  per-class balance. It is not a meaningful improvement and is below the 0.55
  threshold.
- Selection rule: combined only if validation macro-F1 is at least 0.55 **and**
  at least base + 0.05. Neither condition holds, so base is selected.

The speaker probe is a diagnostic, not an accent metric. Top-1 speaker
identification is 0.108 for base, 0.632 for SSL, and 0.108 for combined
(chance 0.00165). SSL features carry far more speaker information; that is a
plausible reason they do not improve accent macro-F1 and is a warning against
reading SSL accuracy as accent-specific.

## One-time test evaluation (base)

The test split was untouched during selection and was scored once.

| Metric | Value |
|---|---|
| Clips / speakers | 605 / 44 |
| Accuracy | 0.5504 |
| Macro-F1 | 0.5630 |
| Speaker-cluster 95% CI (accuracy) | [0.4240, 0.6749] |
| Majority baseline accuracy | 0.2579 |

Per-class test F1: US 0.454, England 0.556, Australian 0.701, Canadian 0.255,
India/South Asia 0.849.

The confusion matrix shows where the errors are:

- Canada is the weakest class: 81 of 156 Canadian clips are predicted US and 30
  Australian; only 33 are correct.
- US is confused with Canada (42 of 120) but is otherwise mostly correct (64).
- Australia is strong on test (110 of 159 correct), despite weak validation F1.
- India is strong (79 of 101 correct), with 16 predicted England.

The test macro-F1 is above 0.55, but this is a descriptive held-out estimate.
The pre-registered selection and fallback decision used validation only, and the
test result is not used to reverse that decision. The wide speaker-cluster CI
reflects only 44 test speakers.

## Irish evaluation-only

Irish has no classifier class. On 69 Irish test clips / 6 speakers, the model
predicted England 50, US 9, Canada 7, Australia 3, and India 0. The report's
Irish "accuracy" is 0.0 by construction (no Irish class), so it is not a
meaningful recognition score. The prediction distribution is the useful result:
Irish speech is mapped mostly onto England/US, which is consistent with the
known limitation that the classifier cannot represent Irish.

## Gates and decision

| Check | Result |
|---|---|
| Beat majority baseline | pass: 0.5102 vs 0.2015 |
| Beat 2x chance (0.40) | pass: 0.5102 vs 0.40 |
| Validation macro-F1 >= 0.55 (A1 trigger) | fail: 0.5422 |
| SSL fallback improves macro-F1 by >= 0.05 | fail: combined +0.0002 |
| Selected classifier | base 222-D, C=0.01 |

The formal A0 stop/go rule is "must beat majority and 2x chance before it can
evaluate TTS"; base passes that. The 0.55 threshold is the A1 fallback trigger,
not a hard A0 gate, and the fallback was tried and rejected. The classifier is
therefore usable for real-speech GLOBE analysis but is weak, especially for
Canada and the US/Canada boundary.

It is **not** yet a valid evaluator of generated accent. A2 measured 0.200
accuracy on generated audio (chance for five classes), even though the
generated-vs-real domain-shift ratio was 0.877. A2 also failed its WER budget,
so A2 is rejected independently. A3/A4 stay blocked until A2 is revised and the
classifier is shown to respond to generated accent.

## Limitations

- GLOBE is read speech from Common Voice, not conversational speech.
- Accent labels are self-reported and may mix nationality, L1, and accent.
- Per-class test speakers are few; the speaker-cluster CI is wide.
- Canada and the US/Canada boundary are the dominant error mode.
- SSL features are strongly speaker-discriminative; accent and speaker remain
  entangled.
- No human accent-identification study was run. Accent accuracy is a proxy, not
  proof of perceived accent quality.
- Irish is evaluation-only and cannot be recognized by this five-class model.

## Artifacts

- `results/accent/a0/report_base.json` - selected base model, validation, test,
  Irish, gates.
- `results/accent/a0/report_ssl.json` - SSL-only ablation.
- `results/accent/a0/report_combined.json` - combined ablation.
- `results/accent/a0/ssl_feature_report.json` - full SSL cache report.
- `checkpoints/accent_classifier_globe_base.joblib` - selected model.
- `checkpoints/accent_classifier_globe_ssl.joblib`,
  `checkpoints/accent_classifier_globe_combined.joblib` - ablation models.
- `data/processed/globe_accent_ssl_features_full.npz` - authoritative 990-D
  cache.
- `data/processed/globe_accent_ssl_features_partial_8500.npz` - preserved
  partial checkpoint from the crashed first run.
