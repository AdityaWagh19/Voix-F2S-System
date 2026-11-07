# Naturalness N1 findings: AMI LoRA pilot (exploratory, narrow proxy pass)

**Status: the pre-registered N1 proxy gate passes, but the measured effect is
very small and this is not a MOS/CMOS result.** The LoRA adapter stays within
the WER and identity budgets, and it reduces the absolute error in F0 median
against the real AMI target by about 0.93 Hz. No other pre-registered prosody
proxy improves. The gate uses only four speaker clusters, a single 400-step
training epoch, and a diffusion-only adapter. Treat this as a narrow exploratory
signal that justifies a confirmatory run, not as evidence that the voice sounds
more natural.

## Design

- Data: AMI Meeting Corpus pilot, 3,315 clips / 16 speakers / 4.25 h.
  Speaker-disjoint `train` / `val` / `test`: train 2,993 clips / 8 speakers;
  val 134 clips / 4 speakers; test 188 clips / 4 speakers.
- Training manifest: `data/processed/ami_lora/manifest.csv`, 2,424 rows
  (train 2,320 / 8 speakers / 2.79 h; val 104 / 4 speakers / 0.13 h).
- Adapter: LoRA rank 8, alpha 16, injected into `model.diffusion.unet` only.
  25 layers, 434,184 trainable parameters. The predictor is **not** adapted.
- Training: 400 optimizer steps, one epoch, AdamW, gradient accumulation 4,
  warmup-cosine. Train loss `0.646213`, validation loss `0.561567`, 251.4 s.
- Evaluation: all 104 validation clips, 10 diffusion steps, seed 42. Base and
  LoRA use the same prompts, references, and noise. Generation-only pass:
  208 rows, 0 failures, 169.1 s.

This is narrower than the pre-registered N1 intervention (predictor +
diffusion) and below the N1 speaker target (8 training speakers vs >=10 held-out
speakers). It is a pilot.

## Evaluation method

The full stack (StyleTTS2 + ECAPA + Whisper + librosa) exceeds this machine's
RAM. Metrics were computed in 26 fresh processes of 8 rows each and merged into
`eval_pilot_merged.json` (208 rows, 104 base/LoRA pairs). The gate then required:

- WER within `base + 0.05`;
- target and reference ECAPA similarity within `base - 0.05`;
- at least one pre-registered prosody metric closer to the real AMI target with
  a speaker-cluster bootstrap 95% CI whose upper bound is below zero;
- all 104 base/LoRA pairs present.

Commands and process-boundary details: [`n1/EVAL_METHOD.md`](n1/EVAL_METHOD.md).
The merge and gate scripts are `scripts/merge_n1_eval.py` and
`scripts/evaluate_n1_gate.py`. Gate report: `n1/gate_report.json`.

## Result

Gate: **passed** (`n_paired_clips = 104`, coverage sufficient).

| Metric | Base | LoRA | Delta (LoRA - base) | Speaker-cluster 95% CI | Gate |
|---|---:|---:|---:|---|---|
| WER | 0.1527 | 0.1605 | +0.0078 | [-0.0008, +0.0264] | within +0.05 |
| ECAPA to real target | 0.2948 | 0.3129 | +0.0181 | [+0.0047, +0.0352] | within -0.05 |
| ECAPA to reference | 0.3482 | 0.3757 | +0.0274 | [+0.0122, +0.0523] | within -0.05 |
| F0 median abs. error to AMI | - | - | -0.9268 Hz | [-2.7009, -0.0765] | **improved** |
| F0 median signed shift | - | - | +1.0413 Hz | - | descriptive |
| F0 range abs. error | - | - | +0.2410 st | [-0.4922, +1.5281] | not improved |
| Words/s abs. error | - | - | +0.0355 | [+0.0039, +0.0547] | worse |
| Energy std abs. error | - | - | +0.0040 dB | [-0.0508, +0.0538] | not improved |
| Pause ratio abs. error | - | - | -0.0033 | [-0.0133, +0.0027] | not improved |
| Spectral tilt abs. error | - | - | +0.0723 dB/kHz | [-0.0378, +0.2211] | not improved |

Notes:

- The WER delta is small and its CI includes zero; the point estimate is worse,
  but it stays inside the pre-registered `+0.05` budget.
- The identity deltas are positive, so the LoRA did not degrade ECAPA
  similarity to either the target speaker or the reference clip.
- The only passing prosody metric is F0 median absolute error. The point
  improvement is about `0.93 Hz` on a mean F0 near `117 Hz` (under 1%). A shift
  of this size is not a perceptual naturalness claim.
- F0 metrics use 103 complete target pairs; one validation clip has no usable
  target F0. The gate's coverage check uses all 104 base/LoRA pairs.
- `scripts/evaluate_n1_gate.py` reports `n_pairs` as the number of complete
  rows and `n_clusters` as the number of speakers resampled by the cluster
  bootstrap. With four validation speakers, the CI is fragile even when it
  excludes zero.

## Per-speaker sensitivity (post-hoc, descriptive)

The gate report includes per-speaker means. These do not change the
pre-registered verdict; they show how narrow it is.

| Speaker | Clips | WER delta | ECAPA target delta | F0-median abs-error delta (Hz) |
|---|---:|---:|---:|---:|
| MEE009 | 52 | +0.0020 | +0.0174 | -0.5854 |
| MEE010 | 12 | +0.0164 | +0.0365 | +0.6096 |
| MEE011 | 18 | +0.0331 | +0.0344 | -3.6715 |
| MEE012 | 22 | -0.0040 | -0.0034 | -0.4508 |

- F0-median error improves for 3 of 4 speakers and is concentrated in MEE011
  (18 clips, -3.67 Hz). One speaker (MEE010) gets worse. The pooled CI is
  therefore not evidence of a consistent speaker-level effect.
- WER is worse for 3 of 4 speakers; the largest increase is +0.033 for MEE011,
  still inside the pooled `+0.05` budget. One speaker improves.
- ECAPA to the real target improves for 3 of 4 speakers and is essentially flat
  for MEE012. Reference ECAPA improves for all four.
- This is exactly why the result is labeled exploratory and why a confirmatory
  run needs more speakers and multiple seeds.

## Decision

- The original plan was to include the N1 LoRA condition in the research demo
  **only** as `exploratory_proxy_application`, sourced to this gate report. The
  2026-11-05 update below supersedes that for the v2 demo: N1 is omitted
  because it increases noise. The demo must keep the "no MOS/CMOS" non-claim.
- Do **not** claim that N1 improves perceived naturalness. Do **not** enter N2
  on this result.
- A confirmatory N1 run needs more held-out speakers, multiple seeds, the full
  pre-registered predictor+diffusion adapter, and human listening. The existing
  4-speaker CI is a pilot uncertainty estimate, not a stable effect size.
- AMI remains confined to the naturalness track. No CFVN, CVAE, or adapter
  training used AMI audio.

## Demo decision update (2026-11-05)

The v2 research demo omits the N1 LoRA condition. A controlled matrix held the
face, speaker embedding, text, and seed fixed and measured N1 at **-46.31
dBFS / 28.40 dB SNR**, versus **-49.17 dBFS / 31.28 dB** for the raw face-style
baseline: 2.86 dB worse noise, and about 17 dB worse than the calibrated v2
median (-63.45 dBFS). This changes only the demo decision; it does not alter
the N1 gate result and does not promote N1 to serving.

## Artifacts

- `results/naturalness/n1/ami_lora_pilot.pt` - LoRA adapter.
- `results/naturalness/n1/ami_lora_pilot.history.json` - loss history.
- `results/naturalness/n1/eval_pilot.json` - 208 paired generation rows.
- `results/naturalness/n1/eval_pilot_merged.json` - 208 rows after metrics.
- `results/naturalness/n1/gate_report.json` - pre-registered gate report.
- `results/naturalness/n1/EVAL_METHOD.md` - commands and coverage rule.
