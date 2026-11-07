# VOIX research demo audit

**Date:** 2026-11-05
**Repository:** `D:\voix`, branch `main` at `3bbf21d`
**Scope:** the face-to-voice research demo, its checkpoints, configurations,
synthesis outputs, verification code, and audit evidence.
**Serving status:** unchanged. No canonical checkpoint was modified and no
experimental adapter was wired into serving.

## 1. Evidence level

This is an **exploratory** audit. It measures generated audio with objective
proxies and verifies that the intended conditions were applied. It does **not**
contain a human MOS, CMOS, emotion-identification, accent-identification, or
speaker-triad study. No objective proxy in this report is a perceptual quality
score.

| Level | What it means here | Status |
|---|---|---|
| Exploratory | One prompt, one seed per clip, objective proxies, no human ratings | This work |
| Confirmatory | Pre-registered hypothesis, held-out speakers, confidence intervals, human listening where perceptual | Not available for the demo |
| Deployment | Serving-path validation, monitoring, rollback, human acceptance | Not attempted |

The working tree contained 401 status entries at the start of this audit:
59 modified tracked files, 33 deleted tracked files, and 309 untracked files.
Those are treated as user work. Nothing in this audit reverts, deletes, moves,
or renames them.

## 2. Executive findings

### Confirmed by controlled measurement

1. **The residual noise is caused by the face-derived style being off the real
   StyleTTS2 style manifold.** With the same model, text, seed, and synthesis
   settings, a real AMI style gives a noise floor of **-71.07 dBFS**; the raw
   face style gives **-49.17 dBFS**. The 21.9 dB difference is a style-prior
   effect, not a diffusion-step or LoRA effect.
2. **Gender-matched mean translation removes a large part of that noise.** On
   the three selected voices, alpha 1.0 improves the noise floor by
   **13.36, 17.58, and 17.12 dB**. The final 21-clip median is
   **-63.45 dBFS** (range -64.78 to -61.75), up from -49.17 dBFS in v1.
3. **The noise is reduced, not eliminated.** The human AMI reference median is
   **-76.11 dBFS**. The v2 demo still has a **12.7 dB** gap. Its worst clip is
   -61.75 dBFS. A -58 dBFS demo gate passes, but that gate is not a perceptual
   quality target.
4. **The N1 LoRA does not fix the noise.** The same-style control gives
   -46.31 dBFS with LoRA versus -49.17 dBFS without it. Diffusion steps 5, 20,
   and 30 stay between -45.50 and -49.38 dBFS. Neither knob is a remediation.
5. **The three selected voices are objectively separable, but not equally.**
   Neutral ECAPA pairs are Lupita-Timothee **0.119**, Lupita-Jackie **0.114**,
   and Timothee-Jackie **0.316**. The same-gender real reference has median
   0.115, p90 0.250, and max 0.438. Timothee-Jackie is at the **97.8th
   percentile**; the demo passes only because its criterion uses the observed
   real maximum.
6. **Emotion directions are applied 9/9, but perceptual distinctness is not
   established.** Happy and angry both raise F0; angry shortens the clip more.
   Sad does not lower F0 on the calibrated face styles and is carried by
   duration only. A post-synthesis pitch shift was tested and rejected because
   it collapsed ECAPA identity to about 0.57.
7. **Accents are applied 9/9, but recognition is not established.** The A3
   classifier gained one clip (7/30 versus 6/30) and per-class recall was US
   1/6, England 5/6, Australian 1/6. Canada and India were 0/6 and were dropped
   from the demo. The demo uses scale 1.0, above the A3-selected 0.3.
8. **Metadata and provenance are now substantially complete in
   `report.json`.** All 21 clips pass the required-field check, all 21 audio
   hashes verify, the face styles recompute with max absolute difference 0.0,
   and the canonical checkpoint hashes match.
9. **WER is 0.000 on all 21 clips for two ASR models** (`whisper-base` in the
   report, `whisper-small` in an independent cross-check). This is an ASR
   proxy, not an intelligibility or naturalness rating.

### Not established or not fixed

1. **"Different faces produce genuinely distinct voices" is not fully
   confirmed.** Two pairs are in the normal real-speaker range; Timothee-Jackie
   is in the upper tail. No human triad test exists. The correct claim is
   "objectively separable voices", not "perceptually confirmed distinct
   speakers".
2. **Happy versus angry may be confusable.** Both raise F0 by a similar amount;
   angry is separated mainly by greater shortening. No human confusion matrix
   exists.
3. **Sad is a duration-only condition on these voices.** It is not a validated
   sad prosody.
4. **The three accents are applied timbre conditions, not validated accents.**
   No human accent-identification result exists.
5. **The "over-polished" delivery is not fixed.** The backend still uses the
   default LibriTTS StyleTTS2 checkpoint and the same formal prompt. The N1
   LoRA was not a naturalness fix. The v2 pausing proxy is closer to human AMI
   than v1, but there is no human naturalness judgment.
6. **The top-level documentation is stale.** `README.md` section 6.2 and the
   top-level `AUDIT_REPORT.md` section 10.9 still describe the residual as "not
   broadband hiss". That conclusion was incomplete because it lacked a
   real-style control. They also need to be updated for the 3-voice/3-emotion/
   3-accent v2 demo. They were not edited because they contain user changes.
7. **Configuration is inconsistent.** `configs/environment.yaml` sets
   `diffusion_steps: null` and `tts_device: cpu`, while `configs/hardware.yaml`
   selects GPU profiles and the generator defaults to `cuda` with 10 steps. The
   v2 report records what was actually used, but one resolved config is not the
   source of truth.

## 3. Baseline evidence

Three baselines were measured before changing the demo.

| Corpus | Clips | Noise floor median | SNR proxy median | 6-8 kHz share median | Pause ratio median | F0 median |
|---|---:|---:|---:|---:|---:|---:|
| v1 `results/demo/research` | 12 | -49.17 dBFS | 31.15 dB | -28.09 dB | 0.007 | 208.9 Hz |
| Legacy `demo/` | 15 | -43.87 dBFS | 25.56 dB | -28.59 dB | 0.038 | 225.8 Hz |
| Human AMI reference | 8 | -76.11 dBFS | 38.75 dB | -33.45 dB | 0.255 | 106.6 Hz |

The v1 noise range was -49.73 to -46.31 dBFS. The legacy range was -48.01 to
-39.35 dBFS. The AMI range was -80.94 to -73.71 dBFS. AMI is not loudness
normalized, so its absolute F0 and level comparisons are descriptive only.

The v1 demo also had metadata gaps: there was no audio SHA-256, no per-clip
noise/SNR, no per-clip ECAPA, and no per-voice identity block. It included five
accents and five emotions, including fear, disgust, Canada, and India. Those
conditions were not supported by confirmatory evidence and are not in v2. The
v1 face (`demo/subject_face.png`) is not one of the 19 audited faces, so the
v1-to-v2 comparison is distributional, not paired by identity.

The old conclusion that the residual was "not broadband hiss" ruled out
clipping and quantization but did not test a real-style control. The controlled
synthesis matrix below closes that gap.

## 4. Root cause: off-manifold face style

The controlled matrix used the same model, text, and seed while changing only
the style source or a synthesis knob. This is the strongest causal evidence in
the audit.

| Condition | Noise dBFS | SNR dB | 6-8 kHz share dB | F0 Hz |
|---|---:|---:|---:|---:|
| Raw face style, 10 steps | -49.17 | 31.28 | -28.26 | 208.3 |
| 5 diffusion steps | -45.50 | 27.54 | -26.94 | 207.1 |
| 20 diffusion steps | -49.38 | 31.55 | -27.76 | 200.0 |
| 30 diffusion steps | -48.94 | 30.94 | -26.69 | 200.0 |
| N1 LoRA | -46.31 | 28.40 | -27.86 | 210.7 |
| Face posterior style | -47.34 | 29.28 | -29.43 | 211.9 |
| Face style without geometry | -48.20 | 29.64 | -23.64 | 208.9 |
| Real AMI reference style | **-71.07** | **51.91** | **-38.24** | 111.3 |
| Real AMI style, default alpha/beta | -60.46 | 42.38 | -33.65 | 105.9 |

Diffusion steps do not fix the noise. The N1 LoRA makes it worse. The posterior
and no-geometry variants do not recover the real-style floor. The real-style
control does. This establishes the style prior as the dominant cause.

The style geometry explains how a face style can be plausible in norm but wrong
in direction:

- 19 face styles: pairwise cosine median **0.831**, max **0.994**;
  79/171 pairs above 0.90 and 42/171 above 0.95.
- 8 real AMI style references: pairwise cosine median **0.873**, max **0.959**.
- Face-style norms are similar to real-style norms (mean 3.50 versus 3.51), so
  the problem is direction, not scale.
- The selected trio's raw face styles have cosine to the real mean of
  **0.522, 0.624, and 0.697**. After alpha 1.0 they are **0.945, 0.962, and
  0.959**.
- CVAE `e_s` vectors are diverse (median pairwise cosine 0.110, max 0.803),
  while the StyleAdapter `style` outputs collapse. This is consistent with the
  adapter being the collapse bottleneck, but it is an inference from the
  geometry, not a formal layer ablation.

## 5. Style calibration

The calibration is a translation in the 256-D style space:

```
s' = s + alpha * (mean_real_gender - mean_face_gender)
```

Within each gender it preserves pairwise style differences exactly (the
translation is constant) and moves the face cloud's mean onto the gender-matched
real AMI mean. Cross-gender pairwise differences do change, because each gender
gets a different translation vector. Alpha 1.0 was selected. The artifact is
`results/audit/style_calibration/calibration.json`
(SHA-256 `2676efbe1792c4e5a1b885b1b55eb5364a3871e4b097617305d9102d6a803aca`).
It is a demo transform, not a serving path, and changes no checkpoint.

| Face | alpha 0 noise | alpha 1 noise | Improvement | alpha 1 SNR | alpha 1 style cosine | alpha 1.25 noise |
|---|---:|---:|---:|---:|---:|---:|
| C2_12 Lupita | -51.06 | -64.42 | 13.36 dB | 45.54 | 0.945 | -68.14 |
| C1_02 Timothee | -44.63 | -62.21 | 17.58 dB | 43.12 | 0.962 | -63.51 |
| C2_09 Jackie | -46.35 | -63.47 | 17.12 dB | 45.33 | 0.959 | -63.29 |

Alpha 1.25 is **not** a better setting for the selected trio. Its selected-trio
median is -63.51 dBFS, essentially the same as alpha 1.0's -63.47 dBFS. Lupita
improves 3.7 dB, Timothee improves 1.3 dB, and Jackie gets 0.2 dB worse.
Alpha 1.25 also raises the Timothee-Jackie ECAPA pair from 0.316 to **0.422**,
close to the real same-gender maximum of 0.438. The frequently quoted -66.1
dBFS figure for alpha 1.25 is the median across all five swept faces, not the
selected trio. Alpha 1.0 is retained.

The nonlinearity is worth recording: the raw alpha-0 Timothee-Jackie ECAPA pair
is 0.413, alpha 1.0 is 0.316, and alpha 1.25 is 0.422. Translation of the style
mean does not translate monotonically into speaker-embedding distance.

## 6. Voice diversity

The three selected voices are:

| Voice | Face | Style cosine (alpha 0 / alpha 1) | ECAPA neutral pair vs others | Real percentile |
|---|---|---|---:|---:|
| C2_12 Lupita | black female, age 26.4 | 0.522 / 0.945 | vs Timothee 0.119 | 51.5th |
| C2_12 Lupita | black female, age 26.4 | 0.522 / 0.945 | vs Jackie 0.114 | 49.7th |
| C1_02 Timothee | young male, age 24.7 | 0.624 / 0.962 | vs Jackie 0.316 | 97.8th |
| C2_09 Jackie | east Asian male | 0.697 / 0.959 | - | - |

The real AMI reference (32 clips, 16 speakers) gives:

- same speaker: median **0.779**, p10 0.638, p90 0.825, max 0.833
- between speakers, same gender: median **0.115**, p90 **0.250**, max **0.438**
- between speakers, cross gender: median **0.063**, p90 **0.140**, max **0.193**

The identity verification compares every pair to the same-gender distribution.
That is a permissive common scale. For the two cross-gender pairs, the more
relevant cross-gender distribution is lower: Lupita-Timothee 0.119 and
Lupita-Jackie 0.114 are above its median 0.063 but below its p90 0.140 and
maximum 0.193.
The warning pair is the same-gender Timothee-Jackie pair at 0.316, which is
above the same-gender p90 0.250 and below the maximum 0.438.

**Conclusion:** the trio is objectively separable; it is not perceptually
validated as three distinct speakers. Timothee-Jackie is the weak pair. The
style-only farthest trio would have been Lupita, Timothee, and Morgan Freeman;
the final choice of Jackie is a demo decision, not a perceptual optimum. A
human triad test is required before claiming three distinct speakers.

Face-style distance cannot be used as a diversity guarantee: the face styles
are collapsed (79/171 pairs above 0.90 cosine), and Timothee-Jackie has a style
cosine of 0.946 while still producing a 0.316 ECAPA pair. Audio-side ECAPA is
the relevant check, and even that is an embedding proxy.

## 7. Demo design and verification

- 3 voices: C2_12 Lupita, C1_02 Timothee, C2_09 Jackie (1 female, 2 male)
- 3 emotions: happy, sad, angry
- 3 accents: US, England, Australian
- 21 clips: neutral + 3 emotions + 3 accents per voice
- Text: `The human voice carries identity, emotion, and the rhythm of lived experience.`
- Seed 42, 10 diffusion steps, alpha 0.18, beta 0.22, 24 kHz PCM-16
- Loudness target -23 LUFS
- Emotion scale 0.56 (2x the E1-validated 0.28)
- Accent scale 1.0, timbre weight 0.25 (above the A3-selected 0.3)
- Style calibration alpha 1.0

The 21-clip cross product is deliberate: every emotion and every accent is
tested on every voice, rather than showing each factor on only one voice.

| Verification block | Result |
|---|---|
| Canonical checkpoint hashes | PASS; CFVN `0253bbbf...`, CVAE `bb7ad29b...`, StyleAdapter `97750ee8...` |
| Face-style recomputation | PASS; 3/3, max absolute difference 0.0 |
| Audio hashes | PASS; 21/21 |
| Metadata completeness | PASS; 21/21, no missing required fields |
| Noise / SNR | PASS against the demo gate (-58 dBFS); not against the human target |
| Identity | PASS against the observed real maximum 0.438; warning pair above p90 |
| Identity preservation | PASS; minimum ECAPA-to-neutral 0.935, floor 0.90 |
| Emotion direction | PASS; 9/9 |
| Emotion application | PASS; 9/9 present, distinct styles, min ECAPA 0.935 |
| Accent application | PASS; 9/9 present, distinct styles, min ECAPA 0.958 |
| WER | `whisper-base` 0.000 (21/21); `whisper-small` 0.000 (21/21) |

The run took 264.6 s on an NVIDIA GTX 1650 with torch 2.4.1+cu121, Python
3.12.1. The report records the runtime, environment, per-clip synthesis time,
and per-clip metric time.

### Per-clip metrics

| Clip | Noise dBFS | SNR dB | F0 Hz | Duration s | ECAPA to neutral | WER |
|---|---:|---:|---:|---:|---:|---:|
| C2_12__neutral | -64.35 | 45.48 | 181.3 | 7.55 | 1.000 | 0.000 |
| C2_12__emotion_happy | -64.44 | 45.87 | 197.7 | 7.30 | 0.969 | 0.000 |
| C2_12__emotion_sad | -64.35 | 45.70 | 181.3 | 7.70 | 0.988 | 0.000 |
| C2_12__emotion_angry | -64.78 | 46.40 | 196.6 | 7.02 | 0.963 | 0.000 |
| C2_12__accent_us | -64.59 | 45.88 | 174.1 | 7.62 | 0.973 | 0.000 |
| C2_12__accent_england | -64.40 | 45.90 | 186.6 | 7.57 | 0.980 | 0.000 |
| C2_12__accent_australian | -64.00 | 44.93 | 179.2 | 7.45 | 0.988 | 0.000 |
| C1_02__neutral | -62.24 | 43.15 | 117.6 | 7.20 | 1.000 | 0.000 |
| C1_02__emotion_happy | -62.54 | 43.53 | 129.7 | 6.95 | 0.951 | 0.000 |
| C1_02__emotion_sad | -62.25 | 42.82 | 117.6 | 7.45 | 0.983 | 0.000 |
| C1_02__emotion_angry | -62.50 | 43.87 | 133.5 | 6.55 | 0.935 | 0.000 |
| C1_02__accent_us | -62.10 | 43.08 | 112.6 | 7.12 | 0.974 | 0.000 |
| C1_02__accent_england | -62.98 | 43.68 | 122.4 | 7.17 | 0.958 | 0.000 |
| C1_02__accent_australian | -61.75 | 42.74 | 116.9 | 7.02 | 0.981 | 0.000 |
| C2_09__neutral | -63.45 | 45.31 | 114.9 | 7.62 | 1.000 | 0.000 |
| C2_09__emotion_happy | -63.67 | 44.92 | 123.1 | 7.32 | 0.963 | 0.000 |
| C2_09__emotion_sad | -63.19 | 44.97 | 115.2 | 8.10 | 0.979 | 0.000 |
| C2_09__emotion_angry | -63.49 | 44.84 | 126.0 | 6.75 | 0.937 | 0.000 |
| C2_09__accent_us | -63.26 | 44.99 | 109.7 | 7.72 | 0.976 | 0.000 |
| C2_09__accent_england | -63.68 | 45.41 | 117.6 | 7.75 | 0.964 | 0.000 |
| C2_09__accent_australian | -63.36 | 45.07 | 112.9 | 7.40 | 0.980 | 0.000 |

Per-voice high-band behaviour differs. Lupita's 8-10 kHz share is about -34.4
dB, Timothee's about -53.3 dB, and Jackie's about -55.5 dB. The noise floor is
still above the human reference for all three. The female voice has more
high-band speech energy, not a higher measured noise floor; do not read the
band share alone as hiss.

## 8. Emotion evidence

| Voice | Emotion | F0 delta | Duration delta | ECAPA to neutral | Direction |
|---|---|---:|---:|---:|---|
| C2_12 | happy | +16.41 Hz | -0.250 s | 0.969 | F0 up, shorter |
| C2_12 | sad | 0.00 Hz | +0.150 s | 0.988 | duration only |
| C2_12 | angry | +15.27 Hz | -0.525 s | 0.963 | F0 up, shorter |
| C1_02 | happy | +12.13 Hz | -0.250 s | 0.951 | F0 up, shorter |
| C1_02 | sad | 0.00 Hz | +0.250 s | 0.983 | duration only |
| C1_02 | angry | +15.93 Hz | -0.650 s | 0.935 | F0 up, shorter |
| C2_09 | happy | +8.24 Hz | -0.300 s | 0.963 | F0 up, shorter |
| C2_09 | sad | +0.33 Hz | +0.475 s | 0.979 | duration only |
| C2_09 | angry | +11.12 Hz | -0.875 s | 0.937 | F0 up, shorter |

All 9 direction checks pass. The upstream E1 direction evidence was 6/6
pre-registered checks on held-out CREMA-D actors at scale 0.28. The demo is an
exploratory 2x extension. Happy and angry share the same F0 direction; angry is
separated mainly by greater shortening. Sad is not a reproduced E1 sad
prosody. A global pitch shift at 0.93 lowered sad F0 but collapsed ECAPA to
0.649 (Lupita), 0.570 (Timothee), and 0.565 (Jackie); at 0.90 it fell to 0.466,
0.415, and 0.526. That compensation was rejected.

Fear and disgust were dropped because only happy, sad, and angry had
confirmatory E1 direction support. This is a deliberate reduction, not a
missing feature.

## 9. Accent evidence

| Voice | Accent | F0 delta | Duration delta | ECAPA to neutral | WER |
|---|---|---:|---:|---:|---:|
| C2_12 | US | -7.18 Hz | +0.075 s | 0.973 | 0.000 |
| C2_12 | England | +5.31 Hz | +0.025 s | 0.980 | 0.000 |
| C2_12 | Australian | -2.08 Hz | -0.100 s | 0.988 | 0.000 |
| C1_02 | US | -4.98 Hz | -0.075 s | 0.974 | 0.000 |
| C1_02 | England | +4.85 Hz | -0.025 s | 0.958 | 0.000 |
| C1_02 | Australian | -0.68 Hz | -0.175 s | 0.981 | 0.000 |
| C2_09 | US | -5.19 Hz | +0.100 s | 0.976 | 0.000 |
| C2_09 | England | +2.68 Hz | +0.125 s | 0.964 | 0.000 |
| C2_09 | Australian | -1.97 Hz | -0.225 s | 0.980 | 0.000 |

The accent directions are small and not consistently separated: US lowers F0,
England raises it, and Australian is near neutral. The A3 classifier was weak:
7/30 versus 6/30 baseline is a one-clip gain, and per-class recall was US 1/6,
England 5/6, Australian 1/6, Canada 0/6, India 0/6. Canada and India were
dropped. The demo scale 1.0 is an exploratory extension above the selected 0.3.
The correct label is "accent-conditioned timbre", not "validated accent".

## 10. Naturalness and the over-polished delivery

The N1 LoRA was the previous attempt to address over-polished delivery. In the
controlled matrix it made the noise floor worse (-46.31 versus -49.17 dBFS) and
did not produce a conversational style. It was not wired into the demo. Do not
repeat the same fine-tuning run as a noise or naturalness fix.

Objective prosody proxies are mixed:

| Metric | v1 | v2 | Human AMI |
|---|---:|---:|---:|
| Pause ratio median | 0.007 | 0.217 | 0.255 |
| F0 standard deviation median | 3.59 st | 2.35 st | 2.63 st |
| Duration median | 6.1 s speech | 5.1 s speech | corpus-dependent |

The v2 pause ratio is closer to the human reference than v1; its F0 variability
is slightly lower than the AMI median. The prompt and the default LibriTTS
checkpoint are unchanged, so this does not prove the over-polished concern is
resolved. It is a hypothesis for a separate naturalness branch, not a completed
fix. No MOS/CMOS exists to support either direction.

## 11. Metadata and provenance

`results/demo/research_v2/report.json` is the authoritative source. It contains:

- schema/version, creation time, runtime, device, Python/torch/CUDA versions
- design: voices, emotions, accents, clip count, text, seed, diffusion steps,
  alpha/beta, loudness target, emotion/accent scales, calibration block
- per clip: voice id/label, track, condition, level, support, note, text, WAV
  path, file name, audio SHA-256, sample rate, duration, seed, diffusion steps,
  alpha/beta, loudness target, F0 scale, style SHA-256, calibrated style
  SHA-256, residual variant/scale/norm/hash, timing, metrics, ECAPA vector,
  ECAPA-to-neutral, WER, transcript
- artifacts: face report, emotion residual, accent residual, CFVN, CVAE,
  StyleAdapter, each with path and SHA-256
- verification: the ten pass/fail blocks, the exploratory accent block, and the
  WER block, each with per-check evidence
- limitations and MOS status

`index.csv` has 21 rows and 26 flat columns. It is a convenience table, not the
full provenance record. It does not include the face image SHA-256, the
calibration artifact hash, or the git commit. Those are in `report.json` or in
the audit manifest, not in every CSV row.

The face image hashes in the report are the actual input images. The face-style
recomputation used the canonical pipeline and matched all three frozen report
hashes with max absolute difference 0.0.

Remaining provenance gaps, to close before a deployment claim:

- no git commit or dirty-tree flag inside `report.json`
- no exact command line inside `report.json` (the command is recorded in this
  audit)
- no JSON-schema validator in the test suite
- CSV does not repeat the calibration/face hashes per row

## 12. WER cross-check

The generator reports `whisper-base` WER 0.000 on all 21 clips. An independent
cross-check with `whisper-small` also gives mean 0.000, max 0.000, n=21
(`results/audit/wer_crosscheck/wer_small.json`). The cross-check pins the source
report SHA-256 and all 21 audio SHA-256 values; they match the report's
verification block. Transcripts vary only by comma placement, which normalized
WER ignores.

This is an ASR proxy. It does not establish naturalness, emotion recognition,
accent recognition, or speaker distinctness.

## 13. Prioritized remediation plan

The full plan with acceptance criteria is `results/audit/REMEDIATION_PLAN.md`.
The priority order is:

### P0 - freeze and stop overclaiming

- Keep `results/demo/research_v2/` immutable after this run.
- Publish the corrected claims: residual hiss is reduced but still 12.7 dB above
  the AMI median; accents are applied conditions, not validated accents; sad is
  duration-only; happy/angry perceptual separation is unproven;
  Timothee-Jackie is in the upper tail.
- Run one pre-registered human study: naturalness CMOS/MUSHRA, 3-way emotion
  identification, 3-way accent identification, and a three-voice triad test.
- Decide whether v2 replaces the v1 gallery. That requires user approval; see
  `CLEANUP_INVENTORY.md`.

**Acceptance:** all pass/fail verification blocks pass, and the exploratory
accent block is explicitly labelled as unvalidated; every document contains the
non-claims above; the human study reports mean, 95% CI, and confusion matrices;
no MOS/CMOS is claimed unless collected.

### P1 - close measurable gaps

| Area | Current | Target | Status |
|---|---:|---:|---|
| Noise median | -63.45 dBFS | <= -68 dBFS | Not met |
| Noise max | -61.75 dBFS | <= -65 dBFS | Not met |
| SNR median | 44.97 dB | >= 40 dB | Met |
| Min ECAPA-to-neutral | 0.935 | >= 0.90 | Met |
| Max neutral pair | 0.316 | <= 0.250 (real p90) | Not met; below real max 0.438 |
| Emotion directions | 9/9 | 9/9 | Met |
| Emotion 3-way human accuracy | unavailable | >= 60% | Not measured |
| Accent application | 9/9 | 9/9 | Met |
| Accent 3-way human accuracy | unavailable | >= 60% | Not measured |
| WER | 0.000 (two ASR models) | <= 0.05 | Met as an ASR proxy |
| Metadata required fields | 21/21 | 100% | Met |
| Metadata hash chain | 21/21 | 100% | Met |
| Human naturalness CMOS vs v1 | unavailable | >= 0 | Not measured |

For noise, test a style prior/posterior interpolation or a real-style anchor per
gender; do not raise diffusion steps and do not retrain N1. Reject any change
that pushes a pair above the real same-gender maximum 0.438 or below the 0.90
identity floor.

For diversity, keep the trio only with the upper-tail caveat, or replace Jackie
with a third voice whose pair is at or below the real p90 0.250. Do not use raw
face-style distance as the fix.

For emotion, test whether the human 3-way test separates happy from angry. If
sad fails the 50% per-emotion floor, remove it from the headline or add a
validated prosody mechanism; do not use a global pitch shift.

For accent, the current GLOBE class contrast is speaker-confounded. Collect
within-speaker accent pairs or use a corpus/model that separates accent from
speaker identity, and evaluate on speaker-disjoint splits. Until then keep the
"exploratory application" label.

### P2 - reproducibility and hygiene

- Resolve the config inconsistency and record one resolved config in the report.
- Add the verification blocks to CI or a documented pre-release script.
- Add a JSON-schema/provenance test and a "no MOS/CMOS" report guard.
- Add git commit, dirty-tree flag, and exact command line to the report.
- Keep v2 and the audit evidence out of any cleanup deletion set.
- Cleanup was executed on 2026-11-05 after approval: 7,053 files (936.7 MiB)
  archived to `D:\voix_archive\20261105_cleanup`, 170 cache files (159.8 MiB)
  deleted, and every archive destination re-hashed. See
  `results/audit/cleanup/` and `CLEANUP_INVENTORY.md`.

## 14. Configuration and documentation debt

`configs/environment.yaml` has `diffusion_steps: null` and `tts_device: cpu`.
`configs/hardware.yaml` selects GPU profiles. The generator's CLI defaults are
`cuda` and 10 steps, and the v2 report records `cuda` and 10. This is a real
reproducibility hole: the config files do not describe the run. Fix it before
calling the demo one-command reproducible.

The top-level `README.md` and `AUDIT_REPORT.md` predate the controlled matrix and
the v2 gallery. They need:

- the corrected noise statement (style prior, real-style control, residual gap)
- the 3-voice/3-emotion/3-accent design
- the Timothee-Jackie upper-tail caveat
- the sad/accent non-claims
- a pointer to this audit and the v2 demo

They were not edited in this pass because they are among the 59 modified
tracked files and may contain user work. Updating them is a documentation task
in the remediation plan, not a silent overwrite.

## 15. Artifact index

| Artifact | Path |
|---|---|
| Final demo report | `results/demo/research_v2/report.json` |
| Flat demo table | `results/demo/research_v2/index.csv` |
| Demo README | `results/demo/research_v2/README.md` |
| Demo gallery | `results/demo/research_v2/index.html` |
| Demo audio | `results/demo/research_v2/audio/` (21 WAVs) |
| Demo faces | `results/demo/research_v2/faces/` (3 images) |
| Controlled before/after | `results/audit/before_after/index.html`, `comparison.csv`, `comparison.json` |
| WER cross-check | `results/audit/wer_crosscheck/wer_small.json` |
| Calibration artifact | `results/audit/style_calibration/calibration.json` |
| Calibration sweep | `results/audit/style_calibration/style_calibration_audit.json` |
| Face/voice audit | `results/audit/face_voice_diversity/face_voice_diversity.json` |
| Condition response | `results/audit/condition_response/condition_response.json` |
| Demo settings | `results/audit/demo_settings/demo_settings.json` |
| Controlled synthesis matrix | `results/audit/synthesis_matrix/synthesis_matrix.json` |
| ECAPA real reference | `results/audit/ecapa_reference/ecapa_reference.json` |
| v1 baseline metrics | `results/audit/baseline_audio/audio_metrics.json` |
| v1 snapshot | `results/audit/baseline_audio/snapshot_20261105/` |
| Human AMI metrics | `results/audit/reference_audio_metrics/audio_metrics.json` |
| Legacy demo metrics | `results/audit/legacy_demo_audio/audio_metrics.json` |
| Remediation plan | `results/audit/REMEDIATION_PLAN.md` |
| Cleanup inventory | `results/audit/CLEANUP_INVENTORY.md` |
| Final artifact manifest | `results/audit/final_manifest.json` |

## 16. Reproduction commands

The commands below are the ones used in this audit. Run them from `D:\voix`
with temporary directories on `D:`:

```powershell
$env:TMP='D:\voix\.tmp'; $env:TEMP='D:\voix\.tmp'
python scripts\audit_audio_quality.py
python scripts\audit_face_voice_diversity.py
python scripts\audit_synthesis_matrix.py
python scripts\audit_style_calibration.py
python scripts\audit_ecapa_reference.py
python scripts\audit_condition_response.py
python scripts\audit_demo_settings.py
python scripts\generate_research_demo.py --output-dir results\demo\research_v2 --compute-wer
python scripts\build_before_after.py
python scripts\audit_wer_crosscheck.py --model small
python -m pytest tests\test_generate_research_demo.py tests\test_style_calibration.py -q
```

The audit scripts are evidence generators; they do not modify canonical
checkpoints or serving code. The generator verifies the canonical hashes before
synthesis.

## 17. Bottom line

The v2 demo is a real improvement over v1: median noise is 14.3 dB lower, all
21 clips are hash-verified, two ASR models return zero WER, and the three
condition tracks are applied and documented. It is still an exploratory demo.
The remaining hiss is 12.7 dB above the human AMI median, the third voice pair
is in the upper tail of the real speaker distribution, and none of the emotion,
accent, or voice-distinctness claims has human perceptual evidence. The
honest description is:

> Three face-derived voices, three applied emotion conditions, and three applied
> accent conditions, with reduced but still measurable residual noise, objective
> verification, and no human naturalness or recognition study.
