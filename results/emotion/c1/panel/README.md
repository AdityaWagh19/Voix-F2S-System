# External evaluator panel

This directory contains an **evaluation-only** run of frozen external recognizers on real holdout audio and generated VOIX demo audio.
The recognizers are not controllers and no result was fed back into synthesis.

> No MOS, CMOS, human emotion-identification, or human accent-identification study was run. All numbers below are model outputs on this specific audio.

## Run

- Created: 2026-11-07 12:52:36 UTC
- Demo: `results\emotion\c1`
- Generated clips: 12
- Device: `cuda`
- Domain-shift threshold: 3.00

```text
python.exe scripts/evaluation/evaluate_panel.py --demo results/emotion/c1 --output-dir results/emotion/c1/panel --evaluators superb_er emotion2vec --n-resamples 10000 --batch-size 8 --device cuda
```

## Data

| Split | Clips | Speakers | Classes |
| --- | --- | --- | --- |
| accent_train | 320 | 223 | australian=80, england=80, indian_english=80, us=80 |
| accent_test | 449 | 36 | australian=159, england=69, indian_english=101, us=120 |
| emotion_train | 320 | 73 | angry=80, happy=80, neutral=80, sad=80 |
| emotion_test | 455 | 9 | angry=101, happy=122, neutral=107, sad=125 |

## Panel results

| Evaluator | Task | Real accuracy [95% CI] | Real macro-F1 | Shift ratio | Gate | Generated accuracy | Generated n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| superb_er | emotion | 0.440 [0.353, 0.516] | 0.378 | 1.59 | yes | 0.500 | 12 |
| emotion2vec | emotion | 0.910 [0.861, 0.947] | 0.932 | 0.84 | yes | 0.250 | 12 |

## Gates and interpretation

### superb_er

Panel valid: **yes**.

The evaluator passed its real-speech and domain-shift gates. Generated accuracy is a measurable external-classifier result, not a human perceptual result.

| Label | Real support | Real recall | Generated support | Generated recall |
| --- | --- | --- | --- | --- |
| neutral | 107 | 0.140 | 3 | 0.333 |
| happy | 122 | 0.033 | 3 | 1.000 |
| sad | 125 | 0.656 | 3 | 0.667 |
| angry | 101 | 0.980 | 3 | 0.000 |

### emotion2vec

Panel valid: **yes**.

The evaluator passed its real-speech and domain-shift gates. Generated accuracy is a measurable external-classifier result, not a human perceptual result.

| Label | Real support | Real recall | Generated support | Generated recall |
| --- | --- | --- | --- | --- |
| neutral | 107 | 0.953 | 3 | 1.000 |
| happy | 122 | 0.934 | 3 | 0.000 |
| sad | 125 | 0.864 | 3 | 0.000 |
| angry | 101 | 0.891 | 3 | 0.000 |

## Limitations

- CommonAccent is a 16-class model. Its accuracy on the four VOIX classes is not the published 16-class accuracy and is reported only for this subset.
- GLOBE labels are self-reported; the evaluator uses the speaker-disjoint test split.
- CREMA-D is acted speech. Fear and disgust were excluded because VOIX has no validated direction for them.
- The generated set has three voices and a small number of target clips. Generated accuracy is a sanity check, not a speaker-generalization estimate.
- A failed domain-shift gate means the recognizer cannot measure the generated category. It does not prove the generated control is wrong.
- emotion2vec+ uses the FunASR model license (`other`), not an OSI-approved license. Review redistribution terms before shipping model files.
- The panel does not replace a listener study. No perceptual claim is made.

