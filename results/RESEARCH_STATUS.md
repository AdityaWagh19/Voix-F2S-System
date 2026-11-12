# VOIX research status

**As of 2026-11-12.** This page is a short index, not the authoritative record.
The consolidated methods, evidence tables, gates, rejected options, and
limitations are in [`docs/MASTER.md`](../docs/MASTER.md). Per-track
`FINDINGS.md` files remain authoritative for their own experiments.

## Bottom line

- **Identity:** the canonical frozen path passes its V2 gate. CFVN R@1 is
  `0.0932` against `0.0085` chance; the shuffled and constant controls are at
  chance. The adapter output/target cosine is `0.8736`, slightly below the
  `0.8980` target-mean baseline. The demo's face-specific neutral identity is
  still unresolved: with disjoint calibration, max held-out ECAPA is `0.9486`
  canonical, `0.7914` Candidate A, and `0.6415` Candidate B (pairwise structure
  objective), all above the `0.4379` gate; Candidate B also misses three audio
  floors. The Phase 1 encoder screen
  (`docs/IDENTITY_NEXT_PHASE.md`) is **NO-GO**: no input cleared the
  pre-registered linear gate, and all eight screened inputs are NO-GO in the
  selection-sensitivity block. The selected `cfvn_es_arcface` failed L1-L5 and
  the demo gate; raw ArcFace improved the centered pair tail (test p90
  `0.6597` vs `0.8516`) but its residual cosine fell below baseline. The next
  step is paired face/voice data, not another VoxCeleb2-only objective on the
  same `e_s`. Report:
  `results/audit/face_encoder_ceiling/report.json`.
- **Emotion:** E1 (speaker-balanced class-mean, prosody-only, serving scale
  `0.28`) passes `6/6` pre-registered direction checks on held-out CREMA-D
  actors. E2 (learned residual) is negative at `4/6` (sad fails) and is
  rejected. The v3 external panel scores generated emotion at chance (`0.250`).
  C1 reference-prosody is **not promoted**: timbre is preserved exactly, but
  the minimum ECAPA-to-neutral is `0.773` (guard `0.90`), the sad direction is
  absent, and the two evaluators disagree (`0.500` vs `0.250`).
- **Accent:** the A0 real-speech classifier is confirmatory-eligible on its own
  data (test macro-F1 `0.5630`). The TTS accent track is not supported. A2 and
  A2c G2P are rejected; A2c produced zero classifier gain (`0/6`) in every
  subset. A3 selects scale `0.30` by a pre-declared rule, but the gain is one
  clip (`7/30` vs `6/30`) with zero Indian/Canadian recall. The v3 external
  panel scores generated accent at chance (`0.250`) and never predicts Indian or
  Australian.
- **Naturalness:** N0b is rejected on identity (`0.716-0.883` normalized ECAPA,
  below the `0.95` guard). The N1 AMI LoRA pilot passes a narrow proxy gate
  (WER `+0.0078`, only F0-median error improves by `0.93 Hz`, four speaker
  clusters), but there is no MOS and N2 is not entered.
- **Current demo:** `results/demo/research_v3/` has exactly 24 clips
  (`3 voices x neutral + 3 emotions + 4 accents`) at serving scales. WER is
  `0.000` on all 24; emotion directions pass `9/9`; identity preservation
  passes (min ECAPA-to-neutral `0.9453`); the three neutral voices **fail**
  distinctness (max pair `0.5696` vs real same-gender maximum `0.438`).
- **External panel:** on the v3 demo, all three evaluators are valid on real
  speech and pass the generated-domain gate, but generated accuracy is `0.250`
  for all three. This is a null external-recognition result, not proof that
  listeners cannot hear differences.

## Not claimed

- No human MOS, CMOS, emotion-identification, accent-identification, or
  voice-triad study exists.
- No emotion or accent condition is a validated perceptual control, a
  confirmatory result, or a deployment result.
- Recognition models are evaluation-only and never control synthesis.
- The accent residual is a class contrast confounded with speaker identity;
  GLOBE has no within-speaker cross-accent pairs.
- The three-voice demo does not establish speaker generalization. Its identity
  distinctness check fails.
- VoxCeleb2 remains research/non-commercial; no rights to faces, voices, or
  derived biometric features are granted.

## Blocked

- Confirmatory emotion work needs MSP-Podcast or another licensed corpus with
  at least 2 h and 30 speakers per class; the institutional agreement is not
  executed.
- CREMA-D is below the 2 h/class gate and remains exploratory.
- The 4 GB GPU serializes TTS, SSL, and evaluator runs.
- The recorded environment does not match `requirements.txt` exactly
  (`styletts2 0.1.6` declares `huggingface-hub<0.20`; hub `0.36.2` is
  installed), and the local TensorFlow install is broken. See
  `docs/MASTER.md` §9 for the torch/einops workaround.

## Artifacts

- v1 demo: `results/demo/research/`
- v2 demo: `results/demo/research_v2/`
- v3 demo (current): `results/demo/research_v3/`
- External panel: `results/evaluation/panel_v2/`,
  `results/evaluation/panel_v3/`
- C1: `results/emotion/c1/` and `results/emotion/c1/panel/`
