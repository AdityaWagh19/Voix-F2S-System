# External evaluator panel

This directory contains an **evaluation-only** run of frozen external recognizers on real holdout audio and generated VOIX demo audio.
The recognizers are not controllers and no result was fed back into synthesis.

> No MOS, CMOS, human emotion-identification, or human accent-identification study was run. All numbers below are model outputs on this specific audio.

## Run

- Created: 2026-11-07 06:53:27 UTC
- Demo: `D:\voix\results\demo\research_v2`
- Generated clips: 21
- Device: `cuda`
- Domain-shift threshold: 3.00

```text
python.exe scripts/evaluation/evaluate_panel.py --evaluators commonaccent superb_er --output-dir results/evaluation/panel_v2_smoke --train-per-class 20 --max-test-per-class 20 --n-resamples 200 --batch-size 8
```

## Data

| Split | Clips | Speakers | Classes |
| --- | --- | --- | --- |
| accent_train | 80 | 72 | australian=20, england=20, indian_english=20, us=20 |
| accent_test | 80 | 29 | australian=20, england=20, indian_english=20, us=20 |
| emotion_train | 80 | 44 | angry=20, happy=20, neutral=20, sad=20 |
| emotion_test | 80 | 9 | angry=20, happy=20, neutral=20, sad=20 |

## Panel results

| Evaluator | Task | Real accuracy [95% CI] | Real macro-F1 | Shift ratio | Gate | Generated accuracy | Generated n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| commonaccent | accent | 0.625 [0.430, 0.766] | 0.667 | 0.67 | yes | 0.333 | 9 |
| superb_er | emotion | 0.450 [0.339, 0.539] | 0.385 | 1.59 | yes | 0.250 | 12 |

## Gates and interpretation

### commonaccent

Panel valid: **yes**.

The evaluator passed its real-speech and domain-shift gates. Generated accuracy is a measurable external-classifier result, not a human perceptual result.

| Label | Real support | Real recall | Generated support | Generated recall |
| --- | --- | --- | --- | --- |
| us | 20 | 0.800 | 3 | 0.667 |
| england | 20 | 0.400 | 3 | 0.333 |
| indian_english | 20 | 0.750 | 0 | 0.000 |
| australian | 20 | 0.550 | 3 | 0.000 |

### superb_er

Panel valid: **yes**.

The evaluator passed its real-speech and domain-shift gates. Generated accuracy is a measurable external-classifier result, not a human perceptual result.

| Label | Real support | Real recall | Generated support | Generated recall |
| --- | --- | --- | --- | --- |
| neutral | 20 | 0.100 | 3 | 0.667 |
| happy | 20 | 0.100 | 3 | 0.333 |
| sad | 20 | 0.600 | 3 | 0.000 |
| angry | 20 | 1.000 | 3 | 0.000 |

## Limitations

- CommonAccent is a 16-class model. Its accuracy on the four VOIX classes is not the published 16-class accuracy and is reported only for this subset.
- GLOBE labels are self-reported; the evaluator uses the speaker-disjoint test split.
- CREMA-D is acted speech. Fear and disgust were excluded because VOIX has no validated direction for them.
- The generated set has three voices and a small number of target clips. Generated accuracy is a sanity check, not a speaker-generalization estimate.
- A failed domain-shift gate means the recognizer cannot measure the generated category. It does not prove the generated control is wrong.
- emotion2vec+ uses the FunASR model license (`other`), not an OSI-approved license. Review redistribution terms before shipping model files.
- The panel does not replace a listener study. No perceptual claim is made.

