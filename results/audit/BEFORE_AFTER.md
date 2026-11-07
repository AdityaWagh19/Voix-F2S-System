# VOIX before/after audio comparison

**Date:** 2026-11-05
**Evidence level:** exploratory, objective proxies only
**Human listening:** none. No MOS, CMOS, ABX, or preference rating was collected.

The interactive comparison is `results/audit/before_after/index.html`. It has
15 audio cards:

- 6 controlled alpha pairs (same face, prompt, seed, and synthesis settings)
- 4 demo-level neutral clips (one v1, three v2)
- 5 negative controls (raw face, N1 LoRA, 5 steps, 30 steps, real AMI style)

The machine-readable rows are `comparison.csv` and `comparison.json`. Every
metric is copied from the source audit reports; nothing is recomputed or
invented here.

## 1. Controlled before/after: the causal test

These pairs change only the style-manifold translation. Text, seed, diffusion
steps, alpha/beta, and the face are held constant.

| Face | Variant | Noise dBFS | SNR dB | 6-8 kHz share dB | Style cosine to real mean | F0 Hz |
|---|---|---:|---:|---:|---:|---:|
| C2_12 Lupita | alpha 0, raw face | -51.06 | 33.06 | -26.63 | 0.522 | 188.8 |
| C2_12 Lupita | alpha 1, calibrated | -64.42 | 45.54 | -31.42 | 0.945 | 181.3 |
| C1_02 Timothee | alpha 0, raw face | -44.63 | 25.87 | -41.58 | 0.624 | 122.4 |
| C1_02 Timothee | alpha 1, calibrated | -62.21 | 43.12 | -31.63 | 0.962 | 117.6 |
| C2_09 Jackie | alpha 0, raw face | -46.35 | 27.96 | -44.07 | 0.697 | 120.3 |
| C2_09 Jackie | alpha 1, calibrated | -63.47 | 45.33 | -38.44 | 0.959 | 114.9 |

Noise improvement: **13.36 dB** (Lupita), **17.58 dB** (Timothee), **17.12 dB**
(Jackie). SNR improves by 12.5 to 17.3 dB. F0 moves by -7.5 to -4.8 Hz; the
calibration is not a pitch shift.

The 6-8 kHz share needs care. For Timothee and Jackie, the calibrated clips have
a **higher** share than the raw clips (-41.58 to -31.63 and -44.07 to -38.44),
even though the measured noise floor improves by 17 dB. A single band-share
number is not a noise measurement; the noise floor and the controlled
real-style comparison are the relevant evidence.

Alpha 1.25 was tested and rejected. Its selected-trio noise median is
-63.51 dBFS, effectively unchanged from alpha 1.0's -63.47 dBFS. It improves
Lupita by 3.7 dB but makes Jackie 0.2 dB worse and raises the Timothee-Jackie
ECAPA pair from 0.316 to **0.422**, close to the real same-gender maximum 0.438.
The -66.1 dBFS figure often attached to alpha 1.25 is the median across all five
swept faces, not the three selected voices.

## 2. Demo-level before/after

The v1 gallery used a different face, so these rows are distributional, not
paired by identity. Do not treat the v1-to-v2 F0 change as a controlled effect.

| Clip | Noise dBFS | SNR dB | 6-8 kHz share dB | F0 Hz |
|---|---:|---:|---:|---:|
| v1 neutral (single face) | -49.17 | 31.29 | -28.26 | 208.3 |
| v2 neutral C2_12 | -64.35 | 45.48 | -31.42 | 181.3 |
| v2 neutral C1_02 | -62.24 | 43.15 | -31.63 | 117.6 |
| v2 neutral C2_09 | -63.45 | 45.31 | -38.44 | 114.9 |

The v2 neutral median is -63.45 dBFS, **14.28 dB** below v1's -49.17 dBFS.
The worst v2 neutral is -62.24 dBFS; the worst v1 clip was -46.31 dBFS. The
improvement is large and consistent, but the v2 median is still **12.67 dB**
above the human AMI reference median of -76.11 dBFS. The residual is reduced,
not eliminated.

## 3. Negative controls

All rows use the legacy face and the same synthesis settings unless stated.

| Control | Noise dBFS | SNR dB | 6-8 kHz share dB | F0 Hz | Interpretation |
|---|---:|---:|---:|---:|---|
| Raw face style, 10 steps | -49.17 | 31.28 | -28.26 | 208.3 | Baseline |
| N1 LoRA | -46.31 | 28.40 | -27.86 | 210.7 | Worse; do not repeat |
| 5 diffusion steps | -45.50 | 27.54 | -26.94 | 207.1 | Worse |
| 30 diffusion steps | -48.94 | 30.94 | -26.69 | 200.0 | No fix |
| Real AMI style | **-71.07** | **51.91** | **-38.24** | 111.3 | Style prior is the cause |

The real-style control is 21.9 dB better than the raw face style. Diffusion
steps and the N1 LoRA do not approach that. This is why the remediation is a
style-manifold transform, not more steps or more fine-tuning.

## 4. Acceptance status

| Target | Result | Status |
|---|---:|---|
| Median noise <= -68 dBFS | -63.45 dBFS | Not met |
| Maximum noise <= -65 dBFS | -61.75 dBFS | Not met |
| Median SNR >= 40 dB | 44.97 dB | Met |
| Min ECAPA-to-neutral >= 0.90 | 0.935 | Met |
| Max neutral pair <= 0.250 (real p90) | 0.316 | Not met; below real max 0.438 |
| Emotion directions 9/9 | 9/9 | Met |
| Accent application 9/9 | 9/9 | Met |
| WER <= 0.05 | 0.000 (whisper-base and whisper-small) | Met as an ASR proxy |
| Audio hashes 21/21 | 21/21 | Met |
| Human CMOS vs v1 >= 0 | unavailable | Not measured |

The noise gates and the pair-p90 target are **not met**. The demo passes its own
looser runtime checks, not the quality targets in the remediation plan.

## 5. What the comparison supports

Supported by the measurements:

- The face style is the dominant noise source.
- The alpha-1.0 translation removes 13.4 to 17.6 dB of noise on the selected
  voices. In the full 21-clip demo, every condition stays at or above 0.935
  ECAPA-to-neutral.
- The N1 LoRA and extra diffusion steps are not noise fixes.
- The alpha-1.25 setting is not an improvement for the selected trio and costs
  voice distinctness.

Not supported by the measurements:

- That the residual is inaudible.
- That the v2 voices are perceptually three distinct speakers.
- That happy, sad, and angry are perceptually distinguishable.
- That US, England, and Australian are recognizable accents.
- That the delivery is no longer over-polished.
- Any MOS/CMOS or preference claim.

The v1/v2 rows use different faces and one seed per clip. The controlled alpha
pairs are the only rows that support a causal before/after statement.

## 6. How to listen

Open `results/audit/before_after/index.html` in a browser. The cards are grouped:

- `C2_12`, `C1_02`, `C2_09`: alpha 0 versus alpha 1
- `legacy_v1` and the three v2 voices: demo-level neutrals
- `synthesis_matrix`: raw face, N1 LoRA, 5 steps, 30 steps, real AMI style

For a blind check, do not read the labels; compare the alpha-0 and alpha-1 cards
for the same voice first, then compare the three v2 neutrals. A human ABX or
triad test is still required for a perceptual conclusion.

## 7. Reproduce

```powershell
$env:TMP='D:\voix\.tmp'; $env:TEMP='D:\voix\.tmp'
python scripts\build_before_after.py
python scripts\audit_wer_crosscheck.py --model small
```

`build_before_after.py` copies the 15 selected WAVs into
`results/audit/before_after/audio/` and writes the JSON, CSV, and HTML. It does
not resynthesize or alter the source clips.
