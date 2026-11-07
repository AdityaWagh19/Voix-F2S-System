# N0b - AMI prosody prototype (exploratory, inference only)

**Status: completed; gate fails on identity. Reject N0b as a naturalness fix.**

The frozen StyleTTS2 style-space prototype moves two held-out AMI prosody
dimensions slightly toward the AMI distribution (pause ratio and spectral tilt)
and does not worsen WER, but generated-speaker identity falls below the
pre-registered 0.95 cosine threshold at both blend strengths. The effect is
therefore not usable as a cheap naturalness baseline. No human MOS/CMOS was
collected; perceptual naturalness remains **unavailable**.

This run also corrects a measurement bug: the first identity block multiplied
raw ECAPA embeddings, which are not unit-normalized, so the reported values were
dot products rather than cosine similarities. The original report is preserved
as `report_pre_identity_fix.json`; the corrected gate fails. See
"Identity correction" below.

## Design

- AMI pilot corpus: 3,315 clips / 16 speakers / 4.25 h, speaker-disjoint split.
- Evaluation: 16 held-out AMI clips from 8 validation/test speakers, median
  duration 4.68 s, selected only by duration and style-cache availability.
- One fixed real VoxCeleb2 face and one fixed canonical style vector. The
  timbre half (`style[0:128]`) is unchanged; only the prosody half
  (`style[128:256]`) is blended.
- Prototypes from 315 real AMI training clips / 8 training speakers:
  - `mean`: clip-weighted mean `style[128:]`, norm 2.2094.
  - `speaker_balanced`: mean of each training speaker's mean `style[128:]`,
    norm 2.2134.
  - The two prototypes are nearly collinear (cosine 0.999993), so
    `speaker_balanced` is a robustness check, not independent evidence.
  - `speaker_centered_diagnostic`: the retired construction that subtracts each
    speaker's mean and then averages. Its norm is 1.01e-7, i.e. mathematically
    zero; it is not a condition. Its WAVs are quarantined in
    `deprecated_speaker_centered_audio/`.
- Grid: 2 variants x 3 alphas (0, 0.25, 0.5) x 3 seeds x 16 clips = 288
  generations. Alpha 0 is the paired baseline. All 288 rows were measured; 0
  failures.
- Audio generation and acoustic measurement run in separate phases, and the
  process-wide StyleTTS2 model is released before librosa/pyin and ECAPA. This
  was required on the 8 GB host.
- CIs are speaker-cluster bootstraps over the 8 held-out AMI speakers, not
  row-level intervals.

## Toward-AMI results

Negative values mean the absolute error to the matched real AMI clip decreased
(moved toward AMI). Median F0 is a diagnostic only and is excluded from the
gate because the face donor and AMI speakers differ in gender/register.

| Variant | Alpha | F0 range (st) | Speech rate (words/s) | Energy SD (dB) | Pause ratio | Spectral tilt (dB/kHz) |
|---|---:|---|---|---|---|---|
| mean | 0.25 | +0.456 [-0.084, +1.069] | -0.017 [-0.066, +0.040] | +0.181 [-0.006, +0.370] | **-0.0085 [-0.0174, -0.0005]** | **-0.0276 [-0.0558, -0.0009]** |
| mean | 0.50 | +0.445 [-0.295, +1.143] | -0.058 [-0.161, +0.053] | +0.158 [-0.184, +0.478] | **-0.0252 [-0.0450, -0.0036]** | -0.0318 [-0.0899, +0.0135] |
| speaker_balanced | 0.25 | +0.622 [+0.086, +1.215] | -0.017 [-0.066, +0.041] | +0.178 [-0.011, +0.369] | **-0.0086 [-0.0178, -0.0005]** | **-0.0272 [-0.0546, -0.0012]** |
| speaker_balanced | 0.50 | +0.968 [+0.374, +1.610] | -0.057 [-0.160, +0.055] | +0.143 [-0.202, +0.463] | **-0.0254 [-0.0452, -0.0036]** | -0.0302 [-0.0916, +0.0173] |

Bold cells are CIs excluding zero in the improving direction. The gate counts
distinct metrics across alphas, so pause ratio and spectral tilt are the two
improved dimensions. F0 range moves away from AMI for `speaker_balanced` at both
alphas (significantly so), which is a negative result.

The movements are small relative to the remaining gap. Generated alpha-0 pause
ratio is 0.0059 versus 0.2113 for the selected real AMI clips; the alpha-0.25
change closes only about 4% of that gap. Spectral tilt is -4.24 versus -4.59
dB/kHz; the change closes about 8% of the gap. This is not a distribution match.

## WER

Whisper base on the generated audio against the AMI transcript. Lower is better;
the gate allows at most +0.05 absolute over the alpha-0 baseline.

| Variant | Alpha 0 | Alpha 0.25 | Alpha 0.50 |
|---|---:|---:|---:|
| mean | 0.1848 | 0.1608 | 0.1646 |
| speaker_balanced | 0.1848 | 0.1631 | 0.1605 |

WER does not worsen; it improves by about 0.02. This is a paired 48-clip
measurement and is noisy, but it is not a gate failure.

## Identity correction and gate

The first report computed `np.dot(ecapa_a, ecapa_b)` without normalization.
ECAPA embeddings are not unit length, so those values were not cosine
similarities (about 47,275 and 37,985) and the 0.95 identity gate passed
trivially. The corrected values use L2-normalized embeddings:

| Variant | Alpha 0.25 mean [min] | Alpha 0.50 mean [min] |
|---|---:|---:|
| mean | 0.883 [0.663] | 0.716 [0.411] |
| speaker_balanced | 0.883 [0.651] | 0.714 [0.416] |

Both variants and both alphas are below 0.95. The corrected N0b gate is:

| Variant | Improved metrics | WER OK | Identity OK | Passed |
|---|---|---|---|---|
| mean | pause_ratio, spectral_tilt | yes | **no** | **no** |
| speaker_balanced | pause_ratio, spectral_tilt | yes | **no** | **no** |

The original report is kept for audit, not as evidence.

## Decision

- **Reject N0b as a naturalness fix.** Do not use the AMI prosody prototype in
  the research demo or as a serving default.
- Keep it as an honest negative: a frozen style-space prototype can nudge pause
  ratio and spectral tilt and slightly improve WER, but it degrades generated
  identity below the pre-registered threshold at both blend strengths.
- Do not treat `speaker_balanced` as independent confirmation; it is nearly
  identical to `mean` (cosine 0.999993).
- Do not present the retired `speaker_centered` WAVs as a result.
- N1 LoRA remains the next naturalness test. No naturalness claim is made until
  it is trained and evaluated on held-out speakers with identity and WER.

## Artifacts

| Item | Path / SHA256 |
|---|---|
| Clips CSV | `results/naturalness/n0b/clips.csv` / `EFD566BF1A5DD94F59064F4511B78BE66F5D45B0701002CBAC00A823011C9A9E` |
| Corrected report | `results/naturalness/n0b/report.json` / `D5614FA780C1DD7D64200DFAB4F9B34110427F8A177371B4D9196E36541EE7CD` |
| Original identity report | `results/naturalness/n0b/report_pre_identity_fix.json` / `77F06AFED7F1376A687514B064B4222FDD08B1E59E1FFCF5930D81AA40F051EE` |
| Retired `speaker_centered` audio | `results/naturalness/n0b/deprecated_speaker_centered_audio/` |
| AMI style cache | `data/processed/ami_style_cache.npz` |

## Reproduction

```powershell
# Full N0b run (generation + metrics + identity + WER)
python scripts/run_n0b_prosody_prototype.py `
  --device cuda --compute-wer --wer-model base `
  --max-clips-per-split 8 --seeds 0,1,2

# Recompute only the normalized identity block from existing audio
python scripts/recompute_n0b_identity.py --device cuda
```

## Limitations

- Exploratory; no human MOS/CMOS. Perceptual naturalness is unmeasured.
- 16 held-out clips from 8 AMI speakers; speaker-cluster CIs are wide.
- AMI is 16 kHz headset speech; generated audio is 24 kHz. The face donor is not
  an AMI speaker, so absolute F0 comparisons are confounded.
- The effect is a small move toward AMI on two dimensions, not a match to the AMI
  distribution, and it fails identity.
- WER is Whisper-base on disfluent AMI transcripts; it is paired but noisy.
- `speaker_balanced` is not independent of `mean`.
