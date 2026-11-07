# A3 method: class-contrast accent residual

A3 tests whether a style-space accent residual can move generated speech toward
a target accent without breaking intelligibility or speaker identity. It does
not change phonemes, so it bypasses the rejected A2 G2P path.

## Why this is a class contrast, not a speaker-centered mean

GLOBE gives each speaker one accent class. Subtracting a speaker's own mean
style and then averaging by accent is exactly zero: every speaker's centered
mean is zero, so every class mean of centered styles is zero. The original A3
note prescribed that construction; it was rejected at implementation time
before any artifact was built.

The implemented baseline instead:

1. averages each speaker's style vectors into a stable per-speaker mean;
2. averages those speaker means into a class mean, so speakers are weighted
   equally;
3. subtracts the unweighted mean of the five class means.

This is a class-contrast direction. Accent and speaker identity remain
confounded because GLOBE has no within-speaker cross-accent pairs. The timbre
half of the residual is down-weighted to `0.25`, and the evaluator measures
ECAPA drift against the real style donor. With only five residual vectors, a
post-hoc speaker probe cannot be trained; the donor ECAPA budget is the identity
check.

## Compute-bounded style subset

The style cache uses `data/processed/globe_accent_style_subset.csv`: up to
three training clips per speaker, speakers with at least two clips, 1,765 clips
across 607 speakers. Every row is `speaker_split=train`; no validation or test
speaker enters the residual. This is a compute bound, not a new split.

## Commands

```powershell
$env:TMP='D:\voix\.tmp'; $env:TEMP='D:\voix\.tmp'
$env:HF_HOME='D:\voix\.cache\huggingface'; $env:TORCH_HOME='D:\voix\.cache\torch'
$env:XDG_CACHE_HOME='D:\voix\.cache'; $env:OMP_NUM_THREADS='1'

python scripts\select_accent_style_subset.py

python scripts\build_accent_style_cache.py `
  --manifest data\processed\globe_accent_style_subset.csv `
  --cache data\processed\globe_accent_style_subset_cache.npz `
  --report results\accent\a3\style_cache_report.json `
  --splits train --device cuda

python scripts\build_accent_class_mean.py `
  --cache data\processed\globe_accent_style_subset_cache.npz `
  --artifact checkpoints\accent_residuals\class_mean.pt `
  --report results\accent\a3\class_mean_report.json

python scripts\generate_accent_a3.py `
  --manifest data\processed\globe_accent_manifest.csv `
  --residual-artifact checkpoints\accent_residuals\class_mean.pt `
  --accents us england australian canadian india_south_asia `
  --scales 0.1 0.2 0.3 --timbre-weight 0.25 --device cuda

python scripts\evaluate_accent_a3.py `
  --index results\accent\a3\generation_index.csv `
  --generation-report results\accent\a3\generation_report.json `
  --report results\accent\a3\scale_selection.json `
  --compute-wer --device cpu
```

## Observed artifacts (2026-11-02)

- Style cache: 1,765 styles, 256-D, 0 failures. Per-class clips: US 479,
  England 400, Australia 280, Canada 260, India/South Asia 346.
- Class-contrast artifact: `checkpoints/accent_residuals/class_mean.pt`,
  schema `voix.accent_class_mean.v2`, 607 speakers. Speaker counts: US 168,
  England 137, Australia 94, Canada 87, India/South Asia 121.
- Residual norms: US 0.251, England 0.367, Australia 0.192, Canada 0.315,
  India 0.463. Class-mean norms are all about 2.33-2.43, so the contrast is a
  small offset on a large shared mean.
- Separation ratio (between-class / within-class variance) is 0.0338. Accent
  class differences are small relative to speaker variation, which is why the
  sweep includes small scales and why ECAPA drift is gated.
- The artifact stores the five residuals and the global mean; it is a research
  baseline, not a serving checkpoint.

## Scale-selection rule (declared before the sweep)

- The A0 classifier proxy must be valid on generated audio (domain-shift
  check). If it is invalid, no scale is selected.
- WER may rise by at most `0.05` over the no-residual baseline.
- ECAPA similarity to the real style donor may fall by at most `0.05`.
- Among eligible scales, select the highest trainable-classifier accuracy.
  Ties go to the smaller scale.
- If no scale improves accuracy while satisfying both budgets, A3 is a null and
  the demo shows no accent condition.

The classifier and ECAPA checks are proxies. No human accent-identification or
MOS study was run, and this rule does not replace the A5 requirement.

## Observed result (2026-11-02)

The rule selects `style_s0.3`, but the classifier gain is one clip: `7/30`
versus `6/30` at `regex`. WER rises by `+0.0067` (two `dance` -> `dunce`
substitutions) and ECAPA donor similarity changes by `+0.0041`; both stay well
inside budget. Per-class recall at `0.3` is US 1/6, England 5/6, Australian 1/6,
Canadian 0/6, India 0/6. This is a mechanical selection, not substantive accent
control. The authoritative result and decision are in
[`FINDINGS.md`](FINDINGS.md).
