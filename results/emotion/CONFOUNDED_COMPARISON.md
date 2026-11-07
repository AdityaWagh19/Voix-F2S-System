# Controlled emotion comparison

Direction effects use paired CREMA-D validation actors. A direction
check is only interpretable when `scalar_modifiers` is `neutral`.

| Condition | Scalars | Directions passed | Neutral-zero | Identity gate |
|---|---|---|---|---|
| e1_emotion_scalars | None | 6/6 | True | False |
| e2_main_emotion_scalars | None | 5/6 | True | False |
| e2_shuffled_emotion_scalars | None | 6/6 | True | False |

| Condition | Emotion | F0 delta (Hz) [95% CI] | Duration delta (s) [95% CI] | ECAPA drop |
|---|---|---|---|---|
| e1_emotion_scalars | happy | +23.37 [+1.45, +42.51] | -0.255 [-0.270, -0.241] | 0.721 |
| e1_emotion_scalars | sad | -37.71 [-49.08, -27.57] | +0.399 [+0.386, +0.413] | 0.789 |
| e1_emotion_scalars | angry | +25.14 [+4.37, +45.52] | -0.324 [-0.339, -0.310] | 0.674 |
| e2_main_emotion_scalars | happy | +20.36 [-1.27, +39.18] | -0.239 [-0.265, -0.216] | 0.738 |
| e2_main_emotion_scalars | sad | -31.37 [-41.79, -21.49] | +0.363 [+0.338, +0.389] | 0.796 |
| e2_main_emotion_scalars | angry | +32.98 [+12.31, +51.56] | -0.299 [-0.321, -0.279] | 0.689 |
| e2_shuffled_emotion_scalars | happy | +30.94 [+9.84, +48.60] | -0.237 [-0.245, -0.228] | 0.726 |
| e2_shuffled_emotion_scalars | sad | -39.81 [-56.36, -26.26] | +0.404 [+0.370, +0.438] | 0.797 |
| e2_shuffled_emotion_scalars | angry | +32.44 [+20.37, +42.75] | -0.302 [-0.310, -0.293] | 0.689 |
