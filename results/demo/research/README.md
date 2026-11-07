# VOIX research demo

One face-derived identity, rendered under the measured research conditions.

> This is a research demo, not a serving endpoint. No MOS, CMOS, or human
> listening study was run. Metrics below are objective proxies only.

## Identity

- Face: `D:\voix\demo\subject_face.png`
- Detected gender: `0`; age: `26.5`
- Base style SHA-256: `7852507d994cfbaca3ab18050c5a994093e3f941c458dc2c40593a50dc747de9`

## Conditions

| Condition | Track | Evidence level | Support | WER | F0 median (Hz) | Duration (s) | Audio |
|---|---|---|---|---|---|---|---|
| `neutral` | baseline | baseline | canonical face-to-voice path | 0.000 | 208.3 | 6.67 | [neutral.wav](audio/neutral.wav) |
| `emotion_happy` | emotion | exploratory_application | results/emotion/FINDINGS.md (E1 direction 6/6, Holm 6/6) | 0.000 | 213.1 | 6.60 | [emotion_happy.wav](audio/emotion_happy.wav) |
| `emotion_sad` | emotion | exploratory_application | results/emotion/FINDINGS.md (E1 direction 6/6, Holm 6/6) | 0.000 | 208.3 | 6.72 | [emotion_sad.wav](audio/emotion_sad.wav) |
| `emotion_angry` | emotion | exploratory_application | results/emotion/FINDINGS.md (E1 direction 6/6, Holm 6/6) | 0.000 | 211.9 | 6.55 | [emotion_angry.wav](audio/emotion_angry.wav) |
| `emotion_fear` | emotion | exploratory_unvalidated_extension | E1 class mean exists, but no confirmatory gate includes fear | 0.000 | 213.1 | 6.67 | [emotion_fear.wav](audio/emotion_fear.wav) |
| `emotion_disgust` | emotion | exploratory_unvalidated_extension | E1 class mean exists, but no confirmatory gate includes disgust | 0.000 | 208.3 | 6.72 | [emotion_disgust.wav](audio/emotion_disgust.wav) |
| `accent_us` | accent | exploratory_application | results/accent/a3/scale_selection.json (scale 0.3 selected; classifier 7/30 vs 6/30 regex, one-clip gain) | 0.000 | 205.9 | 6.70 | [accent_us.wav](audio/accent_us.wav) |
| `accent_england` | accent | exploratory_application | results/accent/a3/scale_selection.json (scale 0.3 selected; classifier 7/30 vs 6/30 regex, one-clip gain) | 0.000 | 209.5 | 6.67 | [accent_england.wav](audio/accent_england.wav) |
| `accent_australian` | accent | exploratory_application | results/accent/a3/scale_selection.json (scale 0.3 selected; classifier 7/30 vs 6/30 regex, one-clip gain) | 0.000 | 207.1 | 6.65 | [accent_australian.wav](audio/accent_australian.wav) |
| `accent_canadian` | accent | exploratory_application | results/accent/a3/scale_selection.json (scale 0.3 selected; classifier 7/30 vs 6/30 regex, one-clip gain) | 0.000 | 207.1 | 6.65 | [accent_canadian.wav](audio/accent_canadian.wav) |
| `accent_india_south_asia` | accent | exploratory_application | results/accent/a3/scale_selection.json (scale 0.3 selected; classifier 7/30 vs 6/30 regex, one-clip gain) | 0.000 | 209.5 | 6.75 | [accent_india_south_asia.wav](audio/accent_india_south_asia.wav) |
| `naturalness_lora` | naturalness | exploratory_proxy_application | results/naturalness/n1/gate_report.json (pre-registered N1 gate) | 0.000 | 210.7 | 6.72 | [naturalness_lora.wav](audio/naturalness_lora.wav) |

## Evidence notes

- `happy`, `sad`, `angry`: E1 direction passed on CREMA-D; this
  face-derived application is exploratory and was not re-measured.
- `fear`, `disgust`: exploratory extensions with no confirmatory gate.
- Accent residuals: A3 class contrasts. Scale 0.3 was selected by the
  pre-declared rule, but the classifier gain is one clip (7/30 vs 6/30)
  and recall is 0/6 for Canada and India. This is not evidence that the
  accent is recognizable. Applying it to a face-derived identity is
  exploratory.
- `naturalness_lora`: included only when the pre-registered N1 proxy
  gate passed on AMI. Applying it to a face-derived identity was not
  measured. It is not a MOS/CMOS result.

## Artifacts

- face: `D:\voix\demo\subject_face.png` (`84567470d6ab4600160c61dd18cb602a8ea73d5f234f11bc333fbc7a4b8b8104`)
- emotion_residual: `D:\voix\checkpoints\emotion_residuals\class_mean.pt` (`6bb4f17ff4bfaf4a9fe4b69514ba3b2e83f4dfe4c8a80f8706c9172cf4bb4408`)
- cfvn: `D:\voix\results\checkpoints\cfvn_v2_best.pt` (`0253bbbf96def21ca9a3a9df3a4466c1e1538ef36986d9eb9fe86bfabd7b2f6b`)
- cvae: `D:\voix\checkpoints\cvae_cfvn_face192_latent128_diversity.pt` (`bb7ad29b7b5cebb2141073892b9d5736d2e996c0bd3e81acc4c671c1ea66656b`)
- style_adapter: `D:\voix\checkpoints\style_adapter_cfvn_best.pt` (`97750ee84e50f9878e3d98de6b2e3dc0ca6829f448f2a5e4a31185adafc0a69c`)
- accent_residual: `D:\voix\checkpoints\accent_residuals\class_mean.pt` (`4781e800d756242ce82a0058291fbcaf7ba5a46e55aeaaef95a332ed2838de32`)
- n1_lora: `D:\voix\results\naturalness\n1\ami_lora_pilot.pt` (`36ce2fe4beba4c181e43d1a84397d3871ee3c0e479f694facf29e3d3a8849357`)

## Non-claims

- No human naturalness, emotion, or accent-identification ratings.
- No claim that the face-derived identity is the person in the image.
- No serving checkpoint was modified; the LoRA is loaded only here.
