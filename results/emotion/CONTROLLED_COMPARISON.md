# Emotion condition comparison

Direction effects use paired CREMA-D validation actors. A direction
check is only interpretable when `scalar_modifiers` is `neutral`.
`None` means the run predates the field and used each emotion's
post-synthesis scalar profile; those runs are confounded.

`e2_label_shuffled_neutral_scalars` is the corrected label-shuffled
null (labels permuted, targets fixed). The `e2_style_shuffled_*` rows
are historical diagnostics: they preserve the label-target association
and are not valid nulls.

WER is measured with Whisper on the generated audio against the CREMA-D
transcript. It is paired but noisy on short clips; `unavailable` means it
was not requested or could not be measured.

| Condition | Scalars | Directions passed | Neutral-zero | Identity gate | WER mean |
|---|---|---|---|---|---|
| e1_class_mean_emotion_scalars | pre-field (emotion profile) | 6/6 | True | False | unavailable |
| e2_main_emotion_scalars | pre-field (emotion profile) | 5/6 | True | False | unavailable |
| e2_style_shuffled_emotion_scalars | pre-field (emotion profile) | 6/6 | True | False | unavailable |
| e1_class_mean_neutral_scalars | neutral | 6/6 | True | True | 0.0329 |
| e2_main_neutral_scalars | neutral | 4/6 | True | True | 0.0316 |
| e2_label_shuffled_neutral_scalars | neutral | 1/6 | True | True | unavailable |
| e2_style_shuffled_neutral_scalars | neutral | 6/6 | True | True | unavailable |

| Condition | Emotion | F0 delta (Hz) [95% CI] | Duration delta (s) [95% CI] | ECAPA drop vs gen. neutral | WER |
|---|---|---|---|---|---|
| e1_class_mean_emotion_scalars | happy | +23.37 [+1.45, +42.51] | -0.255 [-0.270, -0.241] | 0.721 | unavailable |
| e1_class_mean_emotion_scalars | sad | -37.71 [-49.08, -27.57] | +0.399 [+0.386, +0.413] | 0.789 | unavailable |
| e1_class_mean_emotion_scalars | angry | +25.14 [+4.37, +45.52] | -0.324 [-0.339, -0.310] | 0.674 | unavailable |
| e2_main_emotion_scalars | happy | +20.36 [-1.27, +39.18] | -0.239 [-0.265, -0.216] | 0.738 | unavailable |
| e2_main_emotion_scalars | sad | -31.37 [-41.79, -21.49] | +0.363 [+0.338, +0.389] | 0.796 | unavailable |
| e2_main_emotion_scalars | angry | +32.98 [+12.31, +51.56] | -0.299 [-0.321, -0.279] | 0.689 | unavailable |
| e2_style_shuffled_emotion_scalars | happy | +30.94 [+9.84, +48.60] | -0.237 [-0.245, -0.228] | 0.726 | unavailable |
| e2_style_shuffled_emotion_scalars | sad | -39.81 [-56.36, -26.26] | +0.404 [+0.370, +0.438] | 0.797 | unavailable |
| e2_style_shuffled_emotion_scalars | angry | +32.44 [+20.37, +42.75] | -0.302 [-0.310, -0.293] | 0.689 | unavailable |
| e1_class_mean_neutral_scalars | happy | +16.19 [+9.68, +22.00] | -0.055 [-0.063, -0.047] | 0.107 | 0.0310 |
| e1_class_mean_neutral_scalars | sad | -11.59 [-21.75, -2.99] | +0.048 [+0.040, +0.055] | 0.072 | 0.0387 |
| e1_class_mean_neutral_scalars | angry | +18.13 [+12.79, +23.57] | -0.054 [-0.065, -0.044] | 0.099 | 0.0328 |
| e2_main_neutral_scalars | happy | +18.63 [+10.05, +25.72] | -0.034 [-0.052, -0.016] | 0.119 | 0.0367 |
| e2_main_neutral_scalars | sad | +4.21 [-1.13, +9.70] | +0.016 [-0.011, +0.043] | 0.100 | 0.0341 |
| e2_main_neutral_scalars | angry | +20.32 [+14.54, +26.79] | -0.027 [-0.044, -0.013] | 0.111 | 0.0301 |
| e2_label_shuffled_neutral_scalars | happy | +12.94 [+4.42, +21.39] | -0.005 [-0.034, +0.023] | 0.112 | unavailable |
| e2_label_shuffled_neutral_scalars | sad | +9.81 [+3.11, +17.18] | -0.005 [-0.034, +0.024] | 0.110 | unavailable |
| e2_label_shuffled_neutral_scalars | angry | +7.94 [-1.53, +17.57] | -0.003 [-0.033, +0.025] | 0.111 | unavailable |
| e2_style_shuffled_neutral_scalars | happy | +15.23 [+10.24, +20.79] | -0.029 [-0.042, -0.018] | 0.121 | unavailable |
| e2_style_shuffled_neutral_scalars | sad | -10.29 [-19.71, -1.56] | +0.049 [+0.032, +0.067] | 0.093 | unavailable |
| e2_style_shuffled_neutral_scalars | angry | +16.44 [+10.43, +23.32] | -0.031 [-0.042, -0.021] | 0.113 | unavailable |
