# Adapter Manifold Retraining — Pre-registration

**Date:** 2026-11-11
**Scope:** `CFVN -> CVAE -> StyleAdapter` training objective. No canonical
checkpoint, demo calibration, or evaluation gate is changed.

## 1. Why retrain

The calibration audit (`docs/IDENTITY_CALIBRATION_AUDIT.md`) showed:

- The adapter's style residual carries identity: a train-fitted linear decoder
  reconstructs the serving `e_s` from uncalibrated demo residuals at cosine
  `0.920 / 0.934 / 0.886`.
- The demo's mean-only calibration adds a common offset (norm `2.58-2.81`) that
  dominates the face-specific residual (norm `~1.36`), which collapses identity.
- Without calibration the identity is preserved, but some styles are off the
  TTS style manifold and synthesize noisy audio.
- The residual fit is weak: canonical val residual cosine is `0.248`, while a
  ridge linear ceiling reaches `0.407` val / `0.433` test.
- A whitened candidate raises val residual cosine to `0.405` and test to
  `0.440`, but its residual norm overshoots the target by `1.43x` because the
  historical norm loss only penalizes under-shoot. Its calibrated styles are
  too large and fail the audio thresholds.

## 2. Hypothesis

The canonical adapter's residuals are both weak and off-manifold. Improving
residual direction while holding residual magnitude at the train target should
move the adapter output distribution onto the real-style distribution. Then the
adapter's own styles should be natural enough that the identity-bearing residual
does not need the collapsing mean translation.

The specific fix tested here is the **one-sided norm loss**: it allows residual
magnitude to grow without bound once directional fitting improves. A symmetric
relative norm penalty should remove that overshoot while leaving the improved
residual direction intact.

## 3. Candidate A (pre-registered)

Training data and frozen upstream artifacts are unchanged: VoxCeleb2 V2,
speaker-disjoint split, frozen CFVN, frozen CVAE, per-speaker 5-clip style
targets. Training inputs are 4 CVAE samples per clip; validation uses the
serving CVAE prior mean.

```powershell
python -m voix.training.trainer_adapter_cfvn `
  --cfvn results/checkpoints/cfvn_v2_best.pt `
  --cvae checkpoints/cvae_cfvn_face192_latent128_diversity.pt `
  --style-targets data/processed/style_targets_voxceleb2_5clip.pt `
  --output checkpoints/experiments/style_adapter_residual_whiten_norm_v1.pt `
  --whiten-input --residual-objective `
  --center-output-bias --freeze-output-bias `
  --norm-loss-mode symmetric `
  --lambda-div 1.0 --lambda-norm 1.0 --lambda-mean 1.0 `
  --selection residual --epochs 40 --patience 12 --seed 42
```

Candidate knobs, fixed before the run:

| knob | value | reason |
|---|---:|---|
| input whitening | on | improved residual fit in the earlier candidate |
| residual objective | on | keeps the common style mean out of the MSE |
| output bias | center, frozen | serving output starts at the real-style center |
| norm loss | symmetric, total | removes the `1.43x` over-shoot |
| target norm scale | 1.0 | no deliberate residual inflation |
| lambda mean | 1.0 | keeps the batch residual mean at zero |
| selection | val residual | `-residual_cosine + div + norm` |

## 4. Selection and reporting

- Select the checkpoint by validation `val_selection_loss`:
  `-val_residual_cosine + lambda_div * val_diversity_loss +`
  `lambda_norm * val_centered_norm_loss`. Candidate B adds
  `+ lambda_pair * val_pairwise_structure_loss` (section 5); the canonical and
  Candidate A runs were selected without the pairwise term.
- Report validation and test target cosine, residual cosine, and residual norm
  ratio only after selection. Test is never used to pick a checkpoint.
- Compare against the canonical adapter (`val residual 0.2481`, test `0.2805`,
  test target `0.8804`, test norm ratio `1.0656`) and the ridge ceiling
  (`test residual 0.4330`, norm ratio `0.5334`).

### Candidate A outcome

Candidate A (`style_adapter_residual_whiten_norm_v1.pt`) reached val residual
`0.405` / val target `0.8877`, and test residual `0.4391` / test target
`0.8938` / norm ratio `1.0248`. It is **not promoted**: with a calibration fit
only on the 11 non-held-out faces, the three selection faces pass, but the eight
held-out faces reach max neutral pairwise ECAPA `0.7914` (`C1_04/C2_15`) and
`C1_01` misses the SNR floor (`39.57`). Style-space residual fidelity did not
transfer to TTS identity separation.

## 5. Candidate B (pre-registered)

**Motivation.** The failure is a high-similarity tail, not the mean. On the V2
validation split, centered different-speaker style cosine is p90 `0.423` for the
real targets, `0.902` for the canonical adapter, and `0.838` for Candidate A;
the fraction above `0.5` is `5.4%`, `34.0%`, and `38.3%` respectively. The mean
diversity term matches only the average and leaves the tail unconstrained. On
the demo faces every same-gender pair has raw adapter style cosine `0.95-0.99`
even when `e_s` cosine is near zero (`C1_03/C1_05`: `e_s 0.581`, raw style
`0.988`; `C1_04/C2_15`: `e_s 0.069`, raw style `0.985`).

A controlled TTS probe (`results/audit/tts_style_identity_probe/report.json`)
shows the frozen renderer is not the primary bottleneck: real AMI styles with
pairwise style cosine `<0.5` synthesize to ECAPA mean `0.301` (min `0.144`, max
`0.475`), while styles at `>=0.8` synthesize to mean `0.601` (min `0.421`, max
`0.722`), Pearson `0.899`. Distinct styles do render distinct voices, so the
adapter must stop mapping distinct inputs to near-identical styles.

**Change.** Add a differentiable pairwise structure loss that matches the
centered different-speaker cosine matrix of the predictions to the target
geometry (squared error over off-diagonal different-speaker pairs, same
centering as the diversity term). Keep Candidate A's whitening, residual
objective, centered/frozen output bias, and symmetric norm loss.

```powershell
python -m voix.training.trainer_adapter_cfvn `
  --cfvn results/checkpoints/cfvn_v2_best.pt `
  --cvae checkpoints/cvae_cfvn_face192_latent128_diversity.pt `
  --style-targets data/processed/style_targets_voxceleb2_5clip.pt `
  --output checkpoints/experiments/style_adapter_pairwise_v1.pt `
  --whiten-input --residual-objective `
  --center-output-bias --freeze-output-bias `
  --norm-loss-mode symmetric `
  --lambda-div 1.0 --lambda-norm 1.0 --lambda-mean 1.0 `
  --lambda-pair 1.0 `
  --selection residual --epochs 40 --patience 12 --seed 42
```

Candidate knobs, fixed before the run:

| knob | value | reason |
|---|---:|---|
| input whitening | on | same as Candidate A |
| residual objective | on | same as Candidate A |
| output bias | center, frozen | same as Candidate A |
| norm loss | symmetric, total | same as Candidate A |
| lambda pair | 1.0 | squared geometry term is O(0.3) initially; the directional and norm terms are O(1) |
| selection | val residual + style terms | `-residual_cosine + div + norm + pair` |

**Selection metric (fixed):**

```
-val_residual_cosine
+ 1.0 * val_diversity_loss
+ 1.0 * val_centered_norm_loss
+ 1.0 * val_pairwise_structure_loss
```

Report validation and test target cosine, residual cosine, norm ratio, and
centered pairwise p90/p99/max/frac>0.5/frac>0.8 after selection. Test is never
used to pick a checkpoint. The promotion gates in section 6 are unchanged.

### Candidate B outcome

Candidate B (`style_adapter_pairwise_v1.pt`) trained to best epoch 12 (7.2 min).

| metric | validation | test |
|---|---:|---:|
| target cosine | 0.8502 | 0.8518 |
| residual cosine | 0.2480 | 0.2661 |
| centered norm ratio | 1.0797 | 1.0923 |
| centered pair p90 | 0.5273 | 0.5925 |
| centered pair p99 | 0.7840 | 0.8197 |
| centered pair max | 0.9972 | 0.9670 |
| frac > 0.5 | 11.6% | 15.8% |
| frac > 0.8 | 0.8% | 1.4% |

The pairwise tail moved substantially toward the real geometry (val target p90
`0.4231`, test `0.5051`; canonical test p90 `0.9200`, Candidate A `0.8722`).
The cost is style fidelity: test target cosine `0.8518` is below the canonical
`0.8804` and the target-mean baseline `0.8980`, and residual cosine `0.2661` is
close to canonical `0.2805` rather than Candidate A's `0.4391`.

Raw held-out demo styles also separated more: same-gender raw cosine min /
median / max is `0.783 / 0.828 / 0.935` for Candidate B, versus
`0.948 / 0.978 / 0.988` canonical and `0.815 / 0.910 / 0.973` Candidate A. The
worst Candidate B pair is `C1_01/C2_08` (`0.935`), followed by `C1_04/C2_15`
(`0.923`) and `C1_04/C2_07` (`0.909`).

Gate results with the disjoint 11-face calibration:

| set | identity max | identity | audio |
|---|---:|---|---|
| three selection faces | 0.3179 | pass | 2 / 3; `C2_12` noise `-55.66`, SNR `37.19` |
| eight held-out faces | **0.6415** `C1_03/C1_05` | **fail** | noise 6 / 8, SNR 5 / 8 |

Held-out audio misses are `C1_01` (noise `-56.53`, SNR `37.96`), `C2_08`
(noise `-58.07`, SNR `38.88`), and `C2_13` (noise `-56.48`, SNR `37.01`).
`C1_04/C2_15` is the second identity failure at `0.5773`.

**Why the remaining pairs cannot be fixed by this adapter alone.** The two
worst centered style pairs are at or beyond the train-fitted ridge ceiling:
`C1_03/C1_05` is `0.676` for Candidate B vs `0.656` for ridge, and
`C1_04/C2_15` is `0.523` vs `0.585`. The input `e_s` does not contain enough
linearly decodable separation for these pairs, so another objective on the same
inputs is unlikely to bring them below the gate without paired supervision or a
different face encoder.

**Decision:** Candidate B is not promoted. It fails both promotion gates
(identity on eight held-out faces, naturalness on the three selection faces and
on the held-out set). The canonical adapter, demo calibration, and demo remain
unchanged, and Candidate A remains not promoted.

## 6. Promotion gates (unchanged)

The candidate is promoted only if both pass on the three selection faces and
on at least eight held-out faces:

```
max neutral pairwise ECAPA cosine <= 0.4378563828374123
noise_floor_dbfs                  <= -58.0
snr_proxy_db                      >=  40.0
```

The three selection faces are `C1_02`, `C2_09`, `C2_12`. The eight held-out
faces are `C1_01`, `C1_03`, `C1_05`, `C2_08`, `C1_04`, `C2_07`, `C2_13`,
`C2_15`.

If a candidate passes identity but misses naturalness, the residual magnitude is
the first diagnostic; if it passes naturalness but misses identity, the
residual scale is the first diagnostic. Any variant is trained and selected on
validation only, and the fixed gates are not moved.

## 7. Limitations

- VoxCeleb2 provides paired face/voice training data, but it does not cover the
  demo face distribution or the TTS rendering conditions.
- A residual cosine near `0.43` is the observed linear ceiling; an adapter
  cannot be expected to match the target style much better from this input.
- ECAPA and the audio metrics are proxies. This protocol cannot establish
  perceptual naturalness, identity accuracy, or MOS.
