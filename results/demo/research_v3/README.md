# VOIX research demo

Three face-derived voices, three emotions, and four accents from one
canonical face-to-voice pipeline. Every clip uses the same prompt, seed,
diffusion settings, and loudness target so the conditions are comparable.

> This is a research demo, not a serving endpoint. No MOS, CMOS, or human
> emotion/accent-identification study was run. Metrics are objective
> proxies only.

## Design

- Voices: 3 (C2_12_black_female_Lupita_Nyongo, C1_02_young_male_Timothee_Chalamet, C2_09_east_asian_male_Jackie_Chan)
- Emotions: happy, sad, angry
- Accents: us, england, indian_english, australian
- Clips: 24 (neutral + 3 emotions + 4 accents per voice)
- Text: `The human voice carries identity, emotion, and the rhythm of lived experience.`
- Seed: 42; diffusion steps: 10
- Emotion scale: 0.28
- Accent scale: 0.30; timbre weight: 0.25
- Style calibration: `gender_matched_mean_translation` at alpha 1.00

## Verification

- PASS: Canonical checkpoint hashes unchanged
- PASS: Face-style recomputation matches the frozen report
- PASS: Audio file hashes verified after write
- PASS: Metadata completeness
- PASS: Noise floor / SNR thresholds
- FAIL: Three voice identities distinct
- PASS: Identity preserved across conditions
- PASS: Emotion direction checks
- PASS: All emotion conditions applied
- PASS: All accent conditions applied

## Conditions

| Voice | Condition | Track | Level | WER | F0 median (Hz) | Duration (s) | Noise (dBFS) | ECAPA to neutral | Audio |
|---|---|---|---|---|---|---|---|---|---|
| C2_12_black_female_Lupita_Nyongo | neutral | baseline | baseline | 0.000 | 187.7 | 6.42 | -60.9 | 1.000 | [C2_12__neutral.wav](audio/C2_12__neutral.wav) |
| C2_12_black_female_Lupita_Nyongo | happy | emotion | exploratory_application | 0.000 | 194.3 | 6.35 | -61.3 | 0.956 | [C2_12__emotion_happy.wav](audio/C2_12__emotion_happy.wav) |
| C2_12_black_female_Lupita_Nyongo | sad | emotion | exploratory_application | 0.000 | 188.8 | 6.45 | -60.8 | 0.986 | [C2_12__emotion_sad.wav](audio/C2_12__emotion_sad.wav) |
| C2_12_black_female_Lupita_Nyongo | angry | emotion | exploratory_application | 0.000 | 196.6 | 6.12 | -61.7 | 0.945 | [C2_12__emotion_angry.wav](audio/C2_12__emotion_angry.wav) |
| C2_12_black_female_Lupita_Nyongo | us | accent | exploratory_application | 0.000 | 186.6 | 6.45 | -61.1 | 0.967 | [C2_12__accent_us.wav](audio/C2_12__accent_us.wav) |
| C2_12_black_female_Lupita_Nyongo | england | accent | exploratory_application | 0.000 | 189.3 | 6.42 | -61.1 | 0.984 | [C2_12__accent_england.wav](audio/C2_12__accent_england.wav) |
| C2_12_black_female_Lupita_Nyongo | indian_english | accent | exploratory_application | 0.000 | 189.9 | 6.42 | -61.0 | 0.992 | [C2_12__accent_indian_english.wav](audio/C2_12__accent_indian_english.wav) |
| C2_12_black_female_Lupita_Nyongo | australian | accent | exploratory_application | 0.000 | 187.7 | 6.35 | -60.8 | 0.961 | [C2_12__accent_australian.wav](audio/C2_12__accent_australian.wav) |
| C1_02_young_male_Timothee_Chalamet | neutral | baseline | baseline | 0.000 | 122.4 | 6.80 | -62.6 | 1.000 | [C1_02__neutral.wav](audio/C1_02__neutral.wav) |
| C1_02_young_male_Timothee_Chalamet | happy | emotion | exploratory_application | 0.000 | 128.9 | 6.70 | -63.0 | 0.983 | [C1_02__emotion_happy.wav](audio/C1_02__emotion_happy.wav) |
| C1_02_young_male_Timothee_Chalamet | sad | emotion | exploratory_application | 0.000 | 121.0 | 6.90 | -62.6 | 0.988 | [C1_02__emotion_sad.wav](audio/C1_02__emotion_sad.wav) |
| C1_02_young_male_Timothee_Chalamet | angry | emotion | exploratory_application | 0.000 | 128.9 | 6.55 | -62.4 | 0.973 | [C1_02__emotion_angry.wav](audio/C1_02__emotion_angry.wav) |
| C1_02_young_male_Timothee_Chalamet | us | accent | exploratory_application | 0.000 | 118.9 | 6.82 | -62.7 | 0.986 | [C1_02__accent_us.wav](audio/C1_02__accent_us.wav) |
| C1_02_young_male_Timothee_Chalamet | england | accent | exploratory_application | 0.000 | 123.8 | 6.82 | -62.9 | 0.988 | [C1_02__accent_england.wav](audio/C1_02__accent_england.wav) |
| C1_02_young_male_Timothee_Chalamet | indian_english | accent | exploratory_application | 0.000 | 123.8 | 6.87 | -62.3 | 0.989 | [C1_02__accent_indian_english.wav](audio/C1_02__accent_indian_english.wav) |
| C1_02_young_male_Timothee_Chalamet | australian | accent | exploratory_application | 0.000 | 120.7 | 6.75 | -62.9 | 0.989 | [C1_02__accent_australian.wav](audio/C1_02__accent_australian.wav) |
| C2_09_east_asian_male_Jackie_Chan | neutral | baseline | baseline | 0.000 | 116.9 | 7.27 | -62.1 | 1.000 | [C2_09__neutral.wav](audio/C2_09__neutral.wav) |
| C2_09_east_asian_male_Jackie_Chan | happy | emotion | exploratory_application | 0.000 | 121.7 | 7.05 | -62.0 | 0.979 | [C2_09__emotion_happy.wav](audio/C2_09__emotion_happy.wav) |
| C2_09_east_asian_male_Jackie_Chan | sad | emotion | exploratory_application | 0.000 | 116.9 | 7.45 | -62.2 | 0.986 | [C2_09__emotion_sad.wav](audio/C2_09__emotion_sad.wav) |
| C2_09_east_asian_male_Jackie_Chan | angry | emotion | exploratory_application | 0.000 | 123.1 | 6.80 | -61.5 | 0.973 | [C2_09__emotion_angry.wav](audio/C2_09__emotion_angry.wav) |
| C2_09_east_asian_male_Jackie_Chan | us | accent | exploratory_application | 0.000 | 115.5 | 7.25 | -61.9 | 0.989 | [C2_09__accent_us.wav](audio/C2_09__accent_us.wav) |
| C2_09_east_asian_male_Jackie_Chan | england | accent | exploratory_application | 0.000 | 117.6 | 7.22 | -62.1 | 0.990 | [C2_09__accent_england.wav](audio/C2_09__accent_england.wav) |
| C2_09_east_asian_male_Jackie_Chan | indian_english | accent | exploratory_application | 0.000 | 118.9 | 7.37 | -62.2 | 0.990 | [C2_09__accent_indian_english.wav](audio/C2_09__accent_indian_english.wav) |
| C2_09_east_asian_male_Jackie_Chan | australian | accent | exploratory_application | 0.000 | 116.9 | 7.20 | -62.1 | 0.985 | [C2_09__accent_australian.wav](audio/C2_09__accent_australian.wav) |

## Evidence and non-claims

- **Voices**: canonical face -> CFVN -> CVAE -> StyleAdapter styles,
  translated onto the gender-matched real-AMI style mean. The
  translation is a demo calibration; it is not wired into serving and
  does not modify any canonical checkpoint.
- **Noise**: the demo noise floor is median -62.07 dBFS (range -63.03 to -60.80). It passes the demo gate, but the human AMI reference median is -76.11 dBFS, so residual hiss remains (about 14.0 dB gap).
- **Identity distinctness**: the largest neutral pair is C1_02/C2_09 at ECAPA 0.570, the 100.0th percentile of real same-gender between-speaker pairs (median 0.115, p90 0.250, max 0.438). This exceeds the observed real maximum, so the three neutral voices are not distinct under this criterion; human triad confirmation is unavailable.
- **Emotion**: E1 class-mean prosody residual at the serving scale
  0.28. The direction passed 6/6 pre-registered checks on held-out
  CREMA-D actors. Happy and angry raise F0 and shorten the clip; angry
  shortens more. The E1 sad F0 lowering did not transfer to the
  calibrated face styles. A post-synthesis pitch shift was tested and
  rejected because it collapsed ECAPA identity to about 0.57; sad is
  therefore carried by its longer duration only.
- **Accent**: A3 class-contrast residual at the serving scale 0.30 with
  timbre weight 0.25. The A3 selection was based on one clip (7/30 vs
  6/30) and per-class recall was US 1/6, England 5/6, Australian 1/6.
  This is **not** evidence that the accents are recognizable. The
  Indian G2P rules were evaluated separately in A2c and are not
  applied: they increased WER without a classifier gain.
- **No human study**: no MOS, CMOS, emotion-identification, or
  accent-identification result exists. Do not read these clips as a
  validated naturalness or diversity result.
- **Identity**: the generated voice is not a claim about the person in
  the image. ECAPA similarity is an embedding proxy.
- **LoRA**: the N1 adapter is omitted because the controlled matrix
  showed a worse noise floor than the calibrated baseline.

## Artifacts

- face_report: `D:\voix\results\audit\face_voice_diversity\face_voice_diversity.json` (`6b260d95568d85fd3fede4e6929da3dbf90f614edf1cd87eefddf0b23fa6cf32`)
- emotion_residual: `D:\voix\checkpoints\emotion_residuals\class_mean.pt` (`6bb4f17ff4bfaf4a9fe4b69514ba3b2e83f4dfe4c8a80f8706c9172cf4bb4408`)
- accent_residual: `D:\voix\checkpoints\accent_residuals\class_mean.pt` (`4781e800d756242ce82a0058291fbcaf7ba5a46e55aeaaef95a332ed2838de32`)
- cfvn: `D:\voix\results\checkpoints\cfvn_v2_best.pt` (`0253bbbf96def21ca9a3a9df3a4466c1e1538ef36986d9eb9fe86bfabd7b2f6b`)
- cvae: `D:\voix\checkpoints\cvae_cfvn_face192_latent128_diversity.pt` (`bb7ad29b7b5cebb2141073892b9d5736d2e996c0bd3e81acc4c671c1ea66656b`)
- style_adapter: `D:\voix\checkpoints\style_adapter_cfvn_best.pt` (`97750ee84e50f9878e3d98de6b2e3dc0ca6829f448f2a5e4a31185adafc0a69c`)

## Files

- `report.json`: full provenance, per-condition metrics, and verification.
- `index.csv`: flat condition table.
- `audio/`: 24 kHz PCM-16 WAV files.
- `faces/`: input face images used to derive the styles.
