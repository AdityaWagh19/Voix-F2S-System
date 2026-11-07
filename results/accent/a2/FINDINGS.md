# Accent A2 findings: IPA G2P versus regex (exploratory, negative)

**Status: A2 fails its pre-registered gate.** The hand-written IPA accent rules
increase WER for every accent, exceed the +0.10 budget for Australian, Indian,
and Irish English, and do not improve accent-classifier accuracy over the
canonical regex phonemizer. Do not add an accent residual on top of this rule
set. Revise the rules and rerun A2 first, as required by the plan's stop/go
criterion.

## Design

- 6 phonetically balanced prompts x 6 accents x 2 conditions (`regex`, `g2p`)
  = 72 generated clips, seed 0, 10 diffusion steps.
- One frozen validation US style donor (`common_voice_en_17993335`) for every
  clip. Gender/age prosody bias is disabled, so phonemes are the only accent
  variable.
- `regex` is StyleTTS2's canonical gruut phonemizer. `g2p` is the rule-based
  IPA path in `voix/models/accent_g2p.py`.
- All six prompts change phonemes under every accent (6/6 each), so the G2P
  condition is not a no-op.
- WER: Whisper `base` against the known prompt, paired by prompt. Classifier:
  the A0 real-speech model, with a generated-vs-real domain-shift check.
- Irish is evaluation-only and has no classifier class.

## WER

| Accent | Regex WER | G2P WER | G2P - regex [95% CI] | Within +0.10 |
|---|---|---|---|---|
| US | 0.0139 | 0.0509 | +0.0370 [0.0000, +0.1111] | yes |
| England | 0.0139 | 0.0815 | +0.0676 [+0.0139, +0.1417] | yes |
| Australian | 0.0139 | 0.1146 | +0.1007 [+0.0333, +0.1680] | **no** |
| Canadian | 0.0139 | 0.1031 | +0.0892 [0.0000, +0.2374] | yes |
| India / South Asia | 0.0139 | 0.6866 | +0.6727 [+0.5592, +0.7640] | **no** |
| Irish (eval only) | 0.0139 | 0.1523 | +0.1384 [+0.0714, +0.1949] | **no** |

The regex baseline is identical across accent rows because the regex condition
does not change the phoneme string; the same six clips are generated once per
accent label. G2P raises WER in every accent. Indian English is catastrophic:
two thirds of words are no longer recognized. Australian and Irish also exceed
the budget.

Likely contributors, not isolated by this run:

- India applies retroflex /ʈ ɖ/ to every /t d/, producing a heavily retroflexed
  string that the acoustic model does not render intelligibly.
- Australian maps /aɪ/ to /ɔɪ/ ("time" -> "tɔɪm"), a stereotyped broad
  realization that merges PRICE with CHOICE and is not a neutral broad
  transcription.
- Irish applies several vowel monophthongizations plus stop substitutions; its
  +0.138 exceeds the budget even though it is evaluation-only.

These are rule-design problems, not evidence that accent-aware G2P is
impossible. A rule ablation is required before any revision is claimed to work.

## Post-hoc rule ablation (A2b)

A2b disables three named rules and reruns the same 72-clip grid:
`india_retroflex_t`, `india_retroflex_d`, and `australian_price`. This is a
post-hoc diagnostic, not a pre-registered condition.

| Accent | A2 G2P WER | A2b G2P WER | A2b delta vs regex | Within +0.10 |
|---|---|---|---|---|
| US | 0.0509 | 0.0509 | +0.0370 | yes |
| England | 0.0815 | 0.0815 | +0.0676 | yes |
| Australian | 0.1146 | 0.0994 | +0.0855 | **yes** |
| Canadian | 0.1031 | 0.1031 | +0.0892 | yes |
| India / South Asia | 0.6866 | 0.2332 | +0.2193 | **no** |
| Irish (eval only) | 0.1523 | 0.1523 | +0.1384 | **no** |

Removing the Australian PRICE rule brings Australia inside the WER budget.
Removing the global Indian retroflex rules cuts Indian WER by two thirds, but
it still fails by a wide margin. The classifier accuracy is unchanged at 0.133
(versus regex 0.200), and predictions still collapse onto England/Australian
with zero India recall. The A2 gate therefore still fails.

The remaining Indian error is not isolated. Candidate contributors include the
global /θ ð/ -> /t d/ stops, the /v w/ -> /ʋ/ approximants, and the vowel
monophthongizations. Further tuning on six prompts would overfit the diagnostic;
revisit A2 after A0 model selection and with a larger prompt set. Do not add an
accent residual on the current rule set.

## Accent classifier

| Condition | Trainable accuracy | Prediction collapse |
|---|---|---|
| Regex | 0.200 | England 24/36, US 6, Australian 6, Canadian 0, India 0 |
| G2P | 0.133 | England 18/36, Australian 15, Canadian 2, US 1, India 0 |

Chance is 0.20 for five classes. Regex is at chance and G2P is below it, so the
gate's "+5 accuracy points" check fails. The generated-vs-real domain-shift
ratio is 0.877 (threshold 3.0), so the classifier is technically in-distribution
enough to score, but its predictions collapse onto England/Australian and never
predict India. This is no evidence of improved accent recognition.

The classifier used here is the A0 base model
(`checkpoints/accent_classifier_globe.joblib`). After A0 selection, the
newly trained `checkpoints/accent_classifier_globe_base.joblib` was verified to
have identical coefficients, C, and classes, so these A2 numbers remain
reproducible with the selected A0 model. The base model is weak on real speech
(validation macro-F1 0.5422) and at chance on generated speech; this is a
separate limitation from the A2 WER failure.

Irish clips are predicted mostly as England (7/12), then Australian (3) and US
(2); there is no Irish class, so this is descriptive only.

## Identity

Paired ECAPA cosine between regex and G2P audio is 0.73-0.87 by accent:
US 0.862, Canada 0.870, England 0.848, Australia 0.842, Ireland 0.791, India
0.732. The lower values for India and Ireland track the larger phoneme changes.
Absolute cosine to the real donor is 0.444 (regex 0.467, G2P 0.420); this is a
generated-vs-real domain-shift number, not a pass/fail identity result, and must
not be read as identity preservation.

## Decision

- **A2 is rejected as an accent control.** WER budget fails for Australian,
  Indian, and Irish; classifier accuracy does not improve. The post-hoc A2b
  ablation fixes the Australian budget but not India or the classifier, so it
  does not reverse the decision.
- Do not train or evaluate A3/A4 on this rule set. The plan says to revise the
  rules when G2P exceeds the WER budget.
- Next step: revisit A2 after A0 SSL model selection and with a larger prompt
  set; do not tune the rules further on six prompts. Any revision is post-hoc
  and must be reported as such.
- No human accent-identification or MOS study was run. All results are
  exploratory and use a six-prompt diagnostic, not a confirmatory corpus.

## Artifacts

- `results/accent/a2/report.json` - WER, classifier, identity, gate.
- `results/accent/a2/generation_report.json` - donor style and phoneme changes.
- `results/accent/a2/generation_index.csv` - 72 generated clips.
- `results/accent/a2/audio/` - generated WAVs.
- `results/accent/a2b/` - post-hoc ablation report, generation report, index,
  and audio for the disabled-rule rerun.
