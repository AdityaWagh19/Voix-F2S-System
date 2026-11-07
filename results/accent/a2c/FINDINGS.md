# A2c findings: Indian English G2P subset ablation (rejected)

**Status: rejected for the research demo v3.** All three conservative G2P
subsets stay inside the pre-declared `+0.10` WER budget, but none produces any
accent-classifier accuracy gain: the frozen GLOBE classifier predicts the
Indian English clips as US, England, or Australian in both the `regex` and the
`g2p` conditions. The demo therefore uses the A3 accent residual only. No human
accent-identification or MOS study was run.

## Question

A2b applied an aggressive Indian English rule set and produced WER `0.2332`,
which was too costly. A2c asks whether a smaller, phonetically conservative
subset can keep WER acceptable while changing the accent evidence. It is an
ablation over which rules are enabled, not a new residual or a classifier
training run.

## Design

- Data: GLOBE / Common Voice English, evaluation-only Indian English rows.
- Prompts: six shared accent prompts.
- Conditions: `regex` (unmodified: the pipeline's default phonemizer) and
  `g2p` (accent rules enabled).
- Identity donor: frozen validation US clip `common_voice_en_17993335`.
- Generation: seed `0`, 10 diffusion steps, `alpha 0.18`, `beta 0.22`, 12
  clips per subset, zero failures.
- Evaluators: Whisper WER (paired bootstrap) and the frozen A0 GLOBE accent
  classifier. The classifier's generated-domain check is a gate; its
  `india_south_asia` training label is mapped from the current
  `indian_english` serving label for scoring only.
- Gate: `g2p - regex` mean WER may rise by at most `+0.10`, and the classifier
  must gain at least `0.05` accuracy with a valid domain-shift check.

## Subsets

| Subset | Rules enabled | Rules disabled |
| --- | --- | --- |
| `full` | all conservative Indian rules | none |
| `stop_only` | TH/DH stopping, FACE/GOAT monophthongisation | `india_face_monophthong`, `india_goat_monophthong` |
| `vowel_only` | FACE/GOAT monophthongisation | `india_theta_stop`, `india_eth_stop` |

The recorded phoneme provenance now reflects the disabled rules. Earlier A2c
indexes recorded the full rule output even when synthesis honored a subset;
the audio was already correct, only the recorded column was wrong.

## Result

Domain-shift check: generated features are inside the real-speech spread for
all three subsets (standardized-centroid ratio `0.879`, `0.878`, and `0.885`
against threshold `3.0`).

| Subset | Regex WER | G2P WER | Delta [95% CI] | WER budget | Classifier regex | Classifier G2P | Gain |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `full` | 0.01389 | 0.06423 | +0.05034 [+0.01515, +0.08889] | pass | 0/6 | 0/6 | 0.000 |
| `stop_only` | 0.01389 | 0.08274 | +0.06886 [+0.01515, +0.14141] | pass | 0/6 | 0/6 | 0.000 |
| `vowel_only` | 0.01389 | 0.03241 | +0.01852 [0.00000, +0.05556] | pass | 0/6 | 0/6 | 0.000 |

`vowel_only` is the WER-safest subset, but it still gives no classifier gain.
Classifier predictions for `g2p` were US 1 / England 4 / Australian 1
(`full` and `stop_only`) and US 1 / England 2 / Australian 3 (`vowel_only`).
`india_south_asia` was never predicted in any condition. The conditions remain
paired at the donor-style level: mean ECAPA regex-vs-g2p similarity was `0.842`,
`0.849`, and `0.858` for the three subsets.

## Decision

- Do **not** apply Indian G2P rules in `results/demo/research_v3`. The demo
  applies the A3 class-contrast residual to the same phonemized text used for
  the other three accents.
- Do **not** claim an accent improvement from A2c. The WER budget is satisfied,
  but the classifier proxy shows zero gain and the classifier has no Indian
  recall even in the real-speech A3 evaluation.
- Keep the rule implementation and the `--disable-rules` ablation switch; they
  are available for future frontend research, not for serving.

## Limitations

- Six prompts and one voice donor are a diagnostic, not a confirmatory study.
- The WER deltas are small in absolute count: the errors are substitutions in
  a handful of words, so the confidence intervals are wide.
- The A0 classifier is a proxy trained on real read speech; it never predicts
  the Indian class for generated audio, which limits its sensitivity to any
  frontend change.
- No human accent identification, MOS, or CMOS was collected.

## Artifacts

- `results/accent/a2c/full/{generation_report.json,generation_index.csv,report.json}`
- `results/accent/a2c/stop_only/{generation_report.json,generation_index.csv,report.json}`
- `results/accent/a2c/vowel_only/{generation_report.json,generation_index.csv,report.json}`
- Audio: `results/accent/a2c/<subset>/audio/indian_english/{regex,g2p}/`
