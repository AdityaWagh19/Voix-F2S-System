# A3 findings: class-contrast accent residual (exploratory, one-clip selection)

**Status: the pre-declared selection rule picks scale `0.3`, but the measured
gain is one additional correctly classified clip out of 30 and the classifier
has zero recall for Canada and India. This is not evidence that A3 produces
recognizable accents.** The scale is usable only as a clearly labeled
`exploratory_application` in the research demo. No human accent identification
or MOS study was run.

## Design

- Data: GLOBE / Common Voice English, speaker-disjoint `train` split. The
  compute-bounded style subset is 1,765 clips / 607 speakers, up to three clips
  per speaker. No validation or test speaker enters the residual.
- Residual: speaker-balanced class mean per accent minus the unweighted mean of
  the five class means. The first 128 style dimensions (timbre) are scaled by
  `0.25`; the second 128 (prosody) are not. This is a class contrast, not a
  speaker-disentangled direction. GLOBE has no within-speaker cross-accent
  pairs, so accent and speaker identity remain confounded.
- Classes: US, England, Australian, Canadian, India/South Asia. Irish is
  evaluation-only and has no A3 residual.
- Identity donor: one frozen validation US clip, `common_voice_en_17993335`.
  Every condition starts from that donor style, so comparisons are paired.
- Generation grid: five accents x six prompts x four conditions
  (`regex`, `style_s0.1`, `style_s0.2`, `style_s0.3`), seed 0, 10 diffusion
  steps, 120 clips, zero generation failures.
- The A0 base 222-D accent classifier and ECAPA similarity to the real donor
  are the proxy evaluators. They are not human listeners.

## Selection rule (declared before the sweep)

1. The classifier's domain-shift check must accept generated audio.
2. WER may rise by at most `+0.05` over `regex`.
3. ECAPA similarity to the real donor may fall by at most `0.05`.
4. Among eligible scales, pick the highest trainable-classifier accuracy. Ties
   go to the smaller scale.

## Result

The domain-shift check accepts the generated audio: standardized centroid
distance `13.27` versus real validation spread `14.67`, ratio `0.9048` (threshold
`3.0`).

| Condition | Classifier accuracy | Gain vs regex | WER delta [95% CI] | ECAPA donor delta [95% CI] | Eligible |
|---|---:|---:|---|---|---|
| `regex` | 0.200 (6/30) | - | - | - | baseline |
| `style_s0.1` | 0.200 (6/30) | 0.000 | +0.0033 [0.0000, +0.0100] | -0.0001 [-0.0055, +0.0056] | no |
| `style_s0.2` | 0.200 (6/30) | 0.000 | +0.0067 [0.0000, +0.0167] | -0.0003 [-0.0059, +0.0054] | no |
| `style_s0.3` | **0.2333 (7/30)** | **+0.0333** | +0.0067 [0.0000, +0.0167] | +0.0041 [-0.0012, +0.0094] | **yes** |

Only `style_s0.3` clears the rule, because it is the only scale with any
accuracy gain. The entire gain is one clip: `7/30` versus `6/30`. The WER and
identity budgets are not the limiting factor; all three scales stay well inside
them.

### Per-class classifier recall

| Accent class | `regex` | `style_s0.3` | Selected condition predictions |
|---|---:|---:|---:|
| US | 1/6 | 1/6 | 4 |
| England | 4/6 | 5/6 | 22 |
| Australian | 1/6 | 1/6 | 4 |
| Canadian | 0/6 | 0/6 | 0 |
| India/South Asia | 0/6 | 0/6 | 0 |

The classifier predicts England for 22 of 30 clips at scale `0.3`. It never
predicts Canada or India. The pooled accuracy of `0.2333` is barely above the
`0.200` chance rate for five classes and is driven by the England class.

### WER and identity at the selected scale

- WER rises by `+0.0067` overall. The extra errors are two `dance` -> `dunce`
  substitutions, one each for Australia and England; the other three accents
  have the same WER as `regex`.
- Mean ECAPA similarity to the real donor is about `0.467` in every condition.
  At scale `0.3` the paired delta is `+0.0041` with a CI that includes zero.
  The residual does not measurably damage donor identity.
- The donor is a single US speaker, so speaker generalization is not measured.

## Decision

- **Do not treat A3 as a positive accent-control result.** The rule mechanically
  selects `0.3`, but a one-clip classifier gain with zero recall for two classes
  is not substantive evidence.
- The research demo at `results/demo/research/` shows the A3 residual at scale
  `0.3` only as `exploratory_application`, with the one-clip margin and the
  per-class failures stated. It does not claim that listeners hear the intended
  accent.
- Do not promote A3 to confirmatory work or serving. A5 human accent
  identification is still required, and a stronger design needs more than one
  donor and either within-speaker cross-accent data or a genuinely
  speaker-disentangled residual.
- No Irish residual was built. Irish remains evaluation-only.
- No MOS or CMOS was collected.

## Artifacts

- `data/processed/globe_accent_style_subset_cache.npz` - 1,765 styles, 0
  extraction failures.
- `checkpoints/accent_residuals/class_mean.pt` - class-contrast artifact,
  schema `voix.accent_class_mean.v2`.
- `results/accent/a3/generation_index.csv` - 120 generated rows.
- `results/accent/a3/generation_report.json` - generation metadata and donor.
- `results/accent/a3/scale_selection.json` - authoritative A3 selection report.
- `results/accent/a3/METHOD.md` - method, commands, and artifacts.
