# N0a - Prompt/delivery audit (exploratory, inference only)

**Status: completed.** Conversational phrasing robustly increases speech rate on
the held-out VoxCeleb2 face, but it does not detectably change F0 variability,
F0 range, energy variability, pause ratio, or spectral tilt. Prompt wording
alone is therefore a partial delivery lever, not a fix for audiobook-like
prosody.

No training was run. No checkpoint, adapter, CVAE, CFVN, or canonical
face-to-voice artifact was modified. AMI was not used. No human MOS/CMOS was
collected; perceptual naturalness remains **unavailable**.

## Design

- One real VoxCeleb2 validation face selected deterministically by face-quality
  score: `id06637` / `dev/id06637/OLO2xEIjbC0/00012` (gender 1, age 38.75,
  quality 0.721).
- 10 formal/conversational prompt pairs from
  `configs/research/n0a_prompt_pairs.json`.
- 2 registers x 10 pairs x 3 seeds = 60 clips, 0 failures.
- 10 diffusion steps, temperature 0.85, alpha 0.18, beta 0.22, duration scale
  1.0, watermark disabled.
- The canonical face-to-voice style conditioning is held fixed. The style
  embedding hash is identical for formal and conversational clips in all 30
  pair/seed cells (`style_embedding_consistency=true`, 0 mismatches).
- Canonical checkpoint hashes verified unchanged.
- Prompt pairs, not clips or seeds, are the independent unit. The bootstrap
  resamples the 10 pairs. Seed noise is a descriptive noise floor; it includes
  CVAE identity resampling and is therefore conservative.

Artifacts:

| Item | Path / SHA256 |
|---|---|
| Clips CSV | `results/naturalness/n0a/clips.csv` / `842f843c609d1a31bda62ce01752051a262827025ccf796f2c9f05b7900cf393` |
| Prompt config | `configs/research/n0a_prompt_pairs.json` / `b8abdd0496557a993e6789e372c95d43c70621ece06d67cf773566f3a7c2069f` |
| Face features | `data/processed/voxceleb2_v2/pilot100_features.h5` / `be767576b47402be3b1eca43cd06ff3f168b0b0bf4ab7d409775ec45055e4e28` |
| Report | `results/naturalness/n0a/report.json` |

## Paired prompt effects

Conversational minus formal, averaged over seeds within each prompt pair; 95%
speaker-cluster-style bootstrap over prompt pairs (n=10). `effect_to_noise` is
the absolute mean delta divided by the seed-difference noise floor.

| Metric | Formal mean | Conversational mean | Delta [95% CI] | Effect / noise | Detected |
|---|---:|---:|---|---:|---|
| Speech rate (words/s) | 2.360 | 2.731 | **+0.371 [+0.122, +0.654]** | **3.76** | **yes** |
| F0 SD (semitones) | 2.079 | 2.257 | +0.177 [-0.650, +1.092] | 0.10 | no |
| F0 range (semitones) | 6.511 | 7.152 | +0.641 [-2.036, +3.647] | 0.12 | no |
| Energy SD (dB) | 6.640 | 6.697 | +0.057 [-0.169, +0.280] | 0.05 | no |
| Pause ratio | 0.0102 | 0.0102 | +0.000016 [-0.0092, +0.0093] | 0.001 | no |
| Spectral tilt (dB/kHz) | -3.366 | -3.305 | +0.061 [-0.059, +0.187] | 0.11 | no |

Speech rate is the only detected effect and the only one that also exceeds the
seed-noise floor. It is a moderate-to-large paired effect (Cohen's dz 0.81,
sign consistency 0.70): conversational prompts are spoken faster, but not
uniformly so across all ten pairs.

The F0-variability point estimates are directionally positive but small and
uncertain (F0 SD +0.18 st, CI crosses zero; F0 range +0.64 st, CI crosses
zero). This audit does not support a claim that prompt wording alone makes
delivery more expressive.

## Identity / conditioning check

- Style embedding: identical across registers in 30/30 pair-seed cells.
- ECAPA cosine between formal and conversational audio at the same seed:
  mean 0.733, 95% CI [0.674, 0.788], minimum 0.485 across 30 observations.
- This is a generated-domain prompt-consistency check, not a real-speaker
  verification score and not the CFVN identity guardrail. It says the two
  registers remain broadly similar in generated speaker space; it does not
  establish real identity preservation.

## Decision

The N0a decision is: **prompt register shifts speech rate, but not both F0
variability and speech rate. Do not assume prompt wording alone fixes
audiobook-like delivery.** N0b should test whether a conversational prosody
prototype adds anything beyond prompt phrasing.

N0b update: AMI is now local and validated (3,315 clips / 16 speakers / 4.25 h,
speaker-disjoint pilot split). N0b is testing the AMI prosody prototype; this
does not change N0a's conclusion that prompt wording alone is only a partial
delivery lever.

## Reproduction

```powershell
python scripts/run_n0a_prompt_audit.py --device cuda --diffusion-steps 10 --seeds 0,1,2
```

## Limitations

- Exploratory; no human MOS/CMOS. Perceptual naturalness is unmeasured.
- One face, one voice, one accent. Effects may not generalize.
- Ten prompt pairs; lexical and phonetic content differs between registers, so
  the contrast is register phrasing as realized in these pairs, not a
  word-level causal effect.
- Seed noise includes CVAE identity resampling; the effect-to-noise ratio is
  conservative and descriptive, not a formal power calculation.
- Diffusion sampling noise is only partially averaged by three seeds.
- AMI is not used; N0b/N2 conversational adaptation is untrained and
  data-blocked.
- The ECAPA prompt-consistency check is generated-domain only.
- The pilot100 face-feature store is a 382-clip pilot, not the full VoxCeleb2
  corpus.
- Watermarking is disabled so the measurement is not contaminated by an audio
  watermark.
- No canonical artifact was modified.
