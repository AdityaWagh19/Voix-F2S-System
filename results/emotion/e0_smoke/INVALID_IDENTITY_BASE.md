# Invalid identity base: adapter-path E0 smoke run

**Status: negative diagnostic, not an E0 result.** Do not cite the audio or
metrics in this directory as evidence about the fixed-delta emotion path.

## What was run

`results/emotion/e0_smoke/` was generated before the identity base was corrected.
For each actor it extracted a frozen ECAPA embedding from one real neutral
CREMA-D clip, passed that embedding through the canonical face-to-voice style
adapter (`checkpoints/style_adapter_cfvn_best.pt`), and applied the fixed emotion
delta to the adapter output.

## Why it is invalid

The canonical adapter was trained to map a **192-D face embedding** to a
StyleTTS2 style vector. An ECAPA speaker embedding is a different modality and
distribution. Feeding it to the adapter is out of distribution.

A direct diagnostic on actor 1010, sentence `IWL`, measured cosine similarity
between the generated neutral and the real speaker's ECAPA embedding:

| identity base | neutral-to-real ECAPA |
| --- | ---: |
| canonical face adapter + ECAPA | 0.067 |
| StyleTTS2 frozen style/predictor encoder | 0.204 |
| official StyleTTS2 cloning path | 0.144 |

The adapter path collapses speaker identity. Its audio therefore cannot support
any claim about emotion effects, identity retention, or classifier behavior.

## What replaced it

`results/emotion/e0_smoke_v2/` and the full E0 run use
`StyleTTS2Backend.compute_style`, i.e. the frozen StyleTTS2 style/predictor
encoder applied to one real neutral clip per actor. The canonical adapter is not
loaded and the face-to-voice mapping remains untouched.

## Keep or delete

Keep this directory until the corrected E0 report is written, so the negative
result is auditable. It is not part of the E0 evidence set.
