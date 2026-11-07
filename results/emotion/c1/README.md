# C1: reference-prosody transfer

C1 keeps the frozen face-derived timbre half of the StyleTTS2 style
vector and replaces the prosody half with the mean prosody of real
CREMA-D **training** clips for the requested emotion.

> This is a research transform. It is not wired into serving, and no
> MOS, CMOS, or human emotion-identification study was run.

## References

- CREMA-D manifest: `D:\voix\data\processed\crema_d_manifest.csv`
- Cached train styles: `D:\voix\data\processed\crema_d_style_cache_train.npz`
- Split: `train` only
- Gender match: yes
- References per class: 8 (one clip per actor)

| Sex | Emotion | References | Mean prosody norm |
| --- | --- | --- | --- |
| Female | happy | 8 | 3.014 |
| Female | sad | 8 | 2.817 |
| Female | angry | 8 | 2.914 |
| Male | happy | 8 | 2.574 |
| Male | sad | 8 | 2.706 |
| Male | angry | 8 | 2.257 |

## Clips

| Voice | Condition | Audio |
| --- | --- | --- |
| C2_12_black_female_Lupita_Nyongo | neutral | `D:\voix\results\emotion\c1\audio\C2_12__neutral.wav` |
| C2_12_black_female_Lupita_Nyongo | happy | `D:\voix\results\emotion\c1\audio\C2_12__emotion_happy.wav` |
| C2_12_black_female_Lupita_Nyongo | sad | `D:\voix\results\emotion\c1\audio\C2_12__emotion_sad.wav` |
| C2_12_black_female_Lupita_Nyongo | angry | `D:\voix\results\emotion\c1\audio\C2_12__emotion_angry.wav` |
| C1_02_young_male_Timothee_Chalamet | neutral | `D:\voix\results\emotion\c1\audio\C1_02__neutral.wav` |
| C1_02_young_male_Timothee_Chalamet | happy | `D:\voix\results\emotion\c1\audio\C1_02__emotion_happy.wav` |
| C1_02_young_male_Timothee_Chalamet | sad | `D:\voix\results\emotion\c1\audio\C1_02__emotion_sad.wav` |
| C1_02_young_male_Timothee_Chalamet | angry | `D:\voix\results\emotion\c1\audio\C1_02__emotion_angry.wav` |
| C2_09_east_asian_male_Jackie_Chan | neutral | `D:\voix\results\emotion\c1\audio\C2_09__neutral.wav` |
| C2_09_east_asian_male_Jackie_Chan | happy | `D:\voix\results\emotion\c1\audio\C2_09__emotion_happy.wav` |
| C2_09_east_asian_male_Jackie_Chan | sad | `D:\voix\results\emotion\c1\audio\C2_09__emotion_sad.wav` |
| C2_09_east_asian_male_Jackie_Chan | angry | `D:\voix\results\emotion\c1\audio\C2_09__emotion_angry.wav` |

## Verification

- Clips generated: 12
- Timbre half preserved per voice: True
- Styles distinct from neutral: True
- Minimum ECAPA-to-neutral (all rows): 0.7732372058296921
- Minimum ECAPA-to-neutral (emotion rows): 0.7732372058296921

## Limitations

- CREMA-D is acted speech; the references are not a naturalistic corpus.
- Only three demo voices and three emotions were tested.
- The timbre and prosody halves of StyleTTS2's style vector may not be
  fully independent; replacing one half is an approximation.
- Panel recognition results, if present, are model outputs, not human
  perception.
