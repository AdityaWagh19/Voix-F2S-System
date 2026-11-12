# Identity Next Phase — Encoder Ceiling, Adapter Side-Channel, Paired Data

**Date:** 2026-11-12
**Status:** Phase 1 pre-registered; results pending.
**Scope:** face-specific identity in the frozen
`CFVN -> CVAE -> StyleAdapter -> StyleTTS2` path. No accent or emotion work
starts until this phase resolves the identity blocker. No canonical checkpoint,
demo calibration, demo audio, or evaluation gate is changed by Phase 1.

---

## 1. Why this phase exists

The identity audit and the adapter retraining experiments established four
facts:

1. The canonical adapter collapses distinct same-gender faces to nearly
   identical styles. On validation the centered different-speaker style cosine
   p90 is `0.902` against `0.423` for the real style targets; on test it is
   `0.920` against `0.505`.
2. Calibration is not the cause. With a calibration fit on 11 disjoint faces,
   the canonical adapter still reaches max neutral ECAPA `0.9486` on eight
   held-out faces. Candidate A reaches `0.7914`; Candidate B, trained with a
   pairwise structure objective, reaches `0.6415`. All exceed the fixed gate
   `0.4378563828374123`.
3. Candidate B reduced the style tail substantially but still failed both the
   identity and naturalness gates. Its two worst centered style pairs,
   `C1_03/C1_05` at `0.676` and `C1_04/C2_15` at `0.523`, are at or beyond the
   ridge linear ceiling from the same `e_s` (`0.656` and `0.585`). Another
   objective on the same input is unlikely to fix them.
4. The renderer is not the primary bottleneck. In a controlled probe, real AMI
   styles with pairwise style cosine below `0.5` synthesize to ECAPA mean
   `0.301` (min `0.144`, max `0.475`); styles at or above `0.8` synthesize to
   mean `0.601`. Distinct styles do render distinct voices, although even some
   low-cosine real styles miss the strict audio floors.

The next lever is therefore the face representation or the supervision, not
another calibration scalar or another objective on the existing `e_s`.

### 1.1 Constraints that do not move

- The canonical serving path stays
  `CFVN -> CVAE -> StyleAdapter -> StyleTTS2`.
- The four accents (`us`, `england`, `indian_english`, `australian`) and four
  emotions (`neutral`, `happy`, `sad`, `angry`) remain explicit; no default
  conditioning is introduced.
- The identity gate stays
  `max neutral pairwise ECAPA cosine <= 0.4378563828374123`.
- The audio floors stay `noise_floor_dbfs <= -58.0` and
  `snr_proxy_db >= 40.0`.
- The three selection faces are `C1_02`, `C2_09`, `C2_12`; the eight held-out
  faces are `C1_01`, `C1_03`, `C1_05`, `C2_08`, `C1_04`, `C2_07`, `C2_13`,
  `C2_15`.
- Evaluation recognizers and ECAPA are evaluation-only. They never select a
  representation, a checkpoint, a style, or audio.
- `checkpoints/style_adapter_cfvn_best.pt` is not modified until a candidate
  passes the pre-registered validation/test gate.
- No MOS, CMOS, identification, or perceptual claim follows from this phase.

---

## 2. Hypotheses

**H1 — the fused face representation loses identity-relevant information.**
Raw ArcFace preserves face identity better than the fused 560-D vector in
retrieval (`0.9322` vs `0.9153` face-face R@1 on test). If a train-fitted ridge
from raw ArcFace has a higher style-prediction ceiling and separates the demo
worst pairs better than the canonical `e_s`, the fused/CFVN path is the
bottleneck.

**H2 — the CFVN `e_s` bottleneck, not the face input, is limiting.**
If raw ArcFace and fused 560-D do not improve the ceiling over `e_s`, the
information needed by the demo pairs is not linearly present in the face
representations. A different adapter objective on the same data will not fix
it; paired face/voice supervision or a different encoder is required.

**H3 — complementary information exists across representations.**
If a concatenation (`e_s + fused`, `e_s + arcface`, or
`e_s + fused + arcface`) improves both residual prediction and demo
separability beyond every single block, a side-channel adapter that keeps the
frozen CFVN/CVAE path is justified.

**H4 — supervision, not representation, is the remaining gap.**
If no representation clears the screen, the VoxCeleb2-only mapping does not
cover the demo face/TTS domain. The next step is paired face/voice data, not
another VoxCeleb2 adapter run.

A positive ridge result is necessary but not sufficient. A ridge is a linear
predictability ceiling, not a serving model, and it can overstate what a
nonlinear adapter will achieve. A negative result is stronger evidence: if the
linear ceiling cannot separate the pairs, an adapter trained on the same input
is unlikely to do so.

---

## 3. Phase 1 — encoder-ceiling diagnostic

### 3.1 Protocol

- **Data:** VoxCeleb2 FaceFeatureV2, speaker-disjoint train/validation/test
  splits (`5,392 / 601 / 118` speakers; `31,749 / 3,539 / 604` clips).
- **Targets:** per-speaker 256-D StyleTTS2 style targets
  (`data/processed/style_targets_voxceleb2_5clip.pt`), aligned by speaker ID.
- **Unit:** one row per speaker for fitting and evaluation. The target is
  per-speaker, so clip-level fitting would pseudo-replicate speakers with more
  clips. The canonical `e_s` baseline is re-estimated under the same protocol;
  the historical per-clip ridge number (`0.4126` validation, `0.4255` test) is
  not directly comparable.
- **Face blocks:**
  - `cfvn_es`: canonical serving `e_s` (192-D): the frozen
    `CFVN -> CVAE prior-mean -> decoder -> L2-normalize` output that the
    StyleAdapter actually consumes.
  - `cfvn_es_arcface`: the same frozen CVAE applied to the raw-ArcFace CFVN
    output (192-D), using `results/checkpoints/cfvn_v2_arcface.pt`. This is a
    diagnostic composite, not a proposed serving change.
  - `arcface`: raw ArcFace identity embedding (512-D).
  - `fused`: fused FaceFeatureV2 representation (560-D).
- **Screened inputs:**
  - single blocks: `cfvn_es`, `cfvn_es_arcface`, `arcface`, `fused`;
  - concatenations: `cfvn_es+fused` (752-D), `cfvn_es+arcface` (704-D),
    `fused+arcface` (1,072-D), `cfvn_es+fused+arcface` (1,264-D).
- **Standardization:** each input dimension is z-scored with train-split
  speaker-centroid mean and standard deviation. Targets are centered on the
  train-split speaker-level target mean.
- **Ridge:** closed-form ridge from the standardized input to the centered
  target. The regularization grid is fixed before the run:
  `[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`. The lambda is selected by
  validation residual cosine only; test is reported after selection. If the
  selected lambda is at a grid edge, that is reported and the grid is not
  extended after seeing test.
- **Selection rule:** the Phase 1 candidate is the input with the highest
  validation residual cosine. Other variants are descriptive. This prevents
  choosing an input by its test or demo result.
- **No synthesis, no ECAPA, no evaluator in the loop.** Phase 1 predicts
  styles only. Any candidate that advances still has to synthesize audio and
  pass the fixed identity and naturalness gates.

### 3.2 Metrics

For each screened input and split (`val`, `test`):

- speaker-level target cosine, residual cosine, centered residual norm ratio,
  and residual MSE;
- centered different-speaker style-pair mean, p90, p99, max,
  fraction above `0.5`, and fraction above `0.8`;
- the same tail statistics for the real style targets as the reference.

For the demo and gate faces:

- input-space cosine for `e_s`, raw ArcFace, and fused features;
- predicted raw and centered style cosine for every same-gender pair;
- the two pre-declared worst pairs `C1_03/C1_05` and `C1_04/C2_15`;
- the gate-face same-gender maximum and the count above `0.5` and `0.65`;
- the canonical adapter's own raw and centered style cosine for comparison,
  computed from the frozen face-diversity report;
- an identity-overlap diagnostic: nearest VoxCeleb2 train face for each demo
  face under raw ArcFace cosine. This flags possible train/evaluation identity
  overlap; it is a diagnostic, not a gate.

### 3.3 Pre-registered Phase 1 decision

The decision is computed against the `cfvn_es` baseline from the same run.
The thresholds are fixed now and are not moved after seeing results.

**Linear gate (all six must hold):**

| ID | Check |
|---|---|
| L1 | validation residual cosine gain over `cfvn_es` >= **0.05** |
| L2 | test residual cosine gain over `cfvn_es` >= **0.05** |
| L3 | validation centered pair p90 reduction >= **0.05** |
| L4 | test centered pair p90 reduction >= **0.05** |
| L5 | test centered pair `frac_gt_0.8` no higher than `cfvn_es` |
| L6 | test target cosine no lower than `cfvn_es` minus **0.01** |

**Demo gate (all four must hold):**

| ID | Check |
|---|---|
| D1 | `C1_03/C1_05` predicted centered style cosine <= **0.50** |
| D2 | `C1_04/C2_15` predicted centered style cosine <= **0.50** |
| D3 | gate-face same-gender max centered style cosine <= **0.60** |
| D4 | both watch pairs improve by >= **0.10** versus the `cfvn_es` prediction |

**Decision matrix:**

- **GO:** linear gate and demo gate both pass. Pre-register one adapter
  side-channel candidate with the winning input (Phase 2).
- **CONDITIONAL:** linear gate passes, both watch pairs improve by >= `0.10`,
  and the gate-face max centered cosine is <= `0.70`, but D1/D2/D3 are not all
  met. Allow **one** Phase 2 retrain, because the linear ceiling improved and
  the adapter is cheap to test. If that candidate fails both fixed gates, stop
  VoxCeleb2-only objectives and move to paired data.
- **NO-GO:** the linear gate fails, or either watch pair gets worse than the
  `cfvn_es` prediction. Do not retrain the adapter on the same supervised data.
  Move directly to paired face/voice data (Phase 3).

The demo gate is a screen-positive, not a promotion. A style cosine of `0.50`
is already above the audio gate in the probe, and some low-cosine real styles
still fail the audio floors. Phase 2 must still pass the actual three-face and
eight-held-out-face ECAPA and audio checks.

### 3.4 Phase 1 artifacts

- Script: `scripts/audit/compare_face_encoder_ceiling.py`.
- Report: `results/audit/face_encoder_ceiling/report.json`.
- Demo-feature cache:
  `results/audit/face_encoder_ceiling/demo_face_features.json`.

Reproduction:

```powershell
python scripts/audit/compare_face_encoder_ceiling.py `
  --output results/audit/face_encoder_ceiling/report.json
```

The script refuses to reuse demo features whose face SHA-256 has changed and
checks that recomputed canonical `e_s` matches the cached `e_s` in the frozen
face-diversity report.

---

## 4. Phase 2 — one adapter side-channel candidate

Phase 2 runs only after a GO or CONDITIONAL Phase 1 result. The upstream
`CFVN`, `CVAE`, and `StyleTTS2` checkpoints remain frozen. The canonical
adapter remains untouched. The candidate is written under
`checkpoints/experiments/`.

### 4.1 Input

Phase 2 keeps `e_s` in the adapter input and adds the winning raw
representation as a side channel. It does not bypass the frozen
`CFVN -> CVAE` path. The input is exactly the winning Phase 1 variant, for
example:

```
[e_s (192) | fused (560) | arcface (512)]  ->  StyleAdapter  ->  256-D style
```

The adapter checkpoint records the input blocks, input dimension, whitening
statistics, and source artifact hashes. The serving pipeline gets an explicit
candidate-only input mode; the canonical demo and default pipeline are
unchanged until promotion.

### 4.2 Training

- Data: the same speaker-disjoint VoxCeleb2 V2 train/validation splits.
- Frozen: CFVN, CVAE, StyleTTS2, and the scalar calibration.
- Inputs: the winning Phase 1 representation, with train-fitted whitening if
  the winning variant used it. The whitening transform is fit on the train
  split only.
- Objective: Candidate B's pre-registered objective
  (`--whiten-input --residual-objective --center-output-bias
  --freeze-output-bias --norm-loss-mode symmetric --lambda-div 1.0
  --lambda-norm 1.0 --lambda-mean 1.0 --lambda-pair 1.0`), unchanged except for
  the input layer and its whitening dimension.
- Selection: validation only, with the fixed metric
  `-val_residual_cosine + val_diversity_loss + val_centered_norm_loss +
  val_pairwise_structure_loss`.
- Test is reported after selection. No test-driven checkpoint choice.
- One candidate only. No scalar sweeps and no objective variants in this
  phase.

### 4.3 Evaluation and promotion

The candidate is synthesized on the three selection faces and the eight
held-out faces with a calibration fit on the 11 non-held-out faces. The
calibration builder must use the same candidate input features and must record
the excluded faces. Promotion requires both fixed gates on both sets:

```
max neutral pairwise ECAPA cosine <= 0.4378563828374123
noise_floor_dbfs                  <= -58.0
snr_proxy_db                      >=  40.0
```

If identity passes but audio fails, residual magnitude is the first diagnostic.
If audio passes but identity fails, centered style separation is the first
diagnostic. The gates are not moved. If the candidate fails either gate, it is
not promoted and Phase 3 starts.

---

## 5. Phase 3 — paired face/voice data fallback

Phase 3 starts if Phase 1 is NO-GO, or if the single Phase 2 candidate fails
either fixed gate. The goal is supervision on the input domain the demo is
served on, not another objective on VoxCeleb2-only data.

### 5.1 Data requirements

- At least 19 consented or properly licensed identities with paired face
  images and real speech, gender-balanced. At least 11 identities are used for
  training/calibration; at least 8 are held out and disjoint from the 11
  evaluation faces.
- The three selection faces and eight held-out faces remain evaluation-only.
  Training on them invalidates the gate.
- Multiple utterances per identity, ideally five or more, so the per-speaker
  style target is stable.
- The paired speech must be real or explicitly documented as synthetic. If
  synthetic, the face/voice pairing and demographic consistency must be
  documented as a limitation.
- Provenance and licensing review before any training. Do not scrape a real
  person's face or voice. VoxCeleb2 remains research/non-commercial; the demo
  faces are public figures and are not a license to their voices.
- If a licensed paired corpus is unavailable, the documented alternatives are
  (a) a consented internal collection, or (b) a synthetic paired corpus with
  explicit limitations. The choice is recorded before training.

### 5.2 Training and evaluation

- Train a face-to-style mapping on the paired training identities, using the
  same frozen StyleTTS2 renderer and the same style-target extraction.
- Keep the speaker-disjoint split and the evaluation recognizers out of the
  training loop.
- Pre-register the objective, selection metric, and candidate name before the
  run. Use validation identities for selection; test identities are reported
  after selection.
- Evaluate on the three selection faces and at least eight held-out faces with
  the fixed identity and audio gates. A candidate that passes only one set is
  not promoted.
- Do not start accent or emotion work until a candidate passes both gates and
  the demo identity check on all required faces.

---

## 6. Phase 1 result and decision ledger

**Run:** 2026-11-12,
`python scripts/audit/compare_face_encoder_ceiling.py --device cpu --reuse-demo-features`.
**Artifacts:** `results/audit/face_encoder_ceiling/report.json`,
`results/audit/face_encoder_ceiling/run.log`,
`results/audit/face_encoder_ceiling/demo_face_features.json`.
**Integrity:** all 19 demo faces matched the frozen face hashes and reproduced
the cached serving `e_s`; the maximum absolute difference over all faces was
`3.43e-7`. The demo features were extracted once and then reused by hash.

| field | value |
|---|---|
| run timestamp | 2026-11-12 |
| selected input (by validation residual cosine) | `cfvn_es_arcface` |
| selected lambda | `1000.0` (upper grid edge; grid not extended) |
| validation residual cosine | `0.4440` (baseline `0.4263`; gain `+0.0177`, below `0.05`) |
| test residual cosine | `0.4507` (baseline `0.4248`; gain `+0.0259`, below `0.05`) |
| validation centered pair p90 | `0.8370` (baseline `0.8520`; reduction `0.0150`, below `0.05`) |
| test centered pair p90 | `0.8648` (baseline `0.8516`; reduction `-0.0132`) |
| `C1_03/C1_05` centered style cosine | `-0.4619` (baseline `0.7021`; improvement `1.1641`, passes D1) |
| `C1_04/C2_15` centered style cosine | `0.8281` (baseline `0.5546`; regression `-0.2735`, fails D2) |
| gate-face max centered style cosine | `0.8981` (threshold `0.60`, fails D3) |
| linear gate | **FAIL**: L1, L2, L3, L4, L5 fail; L6 passes |
| demo gate | **FAIL**: D1 passes only for `C1_03/C1_05`; D2, D3, D4 fail |
| decision | **NO-GO** |

### 6.1 All screened inputs

Speaker-level metrics. The real target reference for the same splits is
centered pair p90 `0.4116` validation / `0.4781` test, with
`frac_gt_0.8` `0.0003` / `0.0004`.

| input | val residual | test residual | val p90 | test p90 | test frac > 0.8 | test target | demo gate max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cfvn_es` | 0.4263 | 0.4248 | 0.8520 | 0.8516 | 0.1476 | 0.9179 | 0.8276 |
| `cfvn_es_arcface` | 0.4440 | 0.4507 | 0.8370 | 0.8648 | 0.1682 | 0.9200 | 0.8981 |
| `arcface` | 0.3520 | 0.3628 | 0.6041 | 0.6597 | 0.0285 | 0.9125 | 0.7778 |
| `fused` | 0.4261 | 0.4115 | 0.8067 | 0.8105 | 0.1098 | 0.9175 | 0.8809 |
| `cfvn_es+fused` | 0.4176 | 0.4028 | 0.7667 | 0.7702 | 0.0818 | 0.9154 | 0.7986 |
| `cfvn_es+arcface` | 0.4268 | 0.4114 | 0.7726 | 0.7874 | 0.0916 | 0.9159 | 0.7993 |
| `fused+arcface` | 0.4255 | 0.4096 | 0.7452 | 0.7637 | 0.0737 | 0.9170 | 0.8256 |
| `cfvn_es+fused+arcface` | 0.4163 | 0.4014 | 0.7323 | 0.7447 | 0.0694 | 0.9146 | 0.7606 |

Seven of eight inputs selected the upper ridge edge `lambda=1000`; `arcface`
selected the lower edge `0.01`. Per the pre-registration, the grid was not
extended after observing test. The selection-sensitivity block in the report
applies the decision matrix to every input: **all eight are NO-GO**. The NO-GO
is therefore not an artifact of selecting by validation residual cosine.

### 6.2 What the screen shows

- The canonical serving `e_s` baseline is reproduced at residual cosine
  `0.4263` validation / `0.4248` test, consistent with the historical ridge
  result. Its centered pair p90 is `0.8520` / `0.8516`, far above the real
  target (`0.4116` / `0.4781`). The adapter's input space is the bottleneck.
- Raw ArcFace is the only input that substantially improves the centered pair
  tail: p90 `0.6041` validation / `0.6597` test, `frac_gt_0.8` `0.0285` test,
  and target cosine `0.9125` (within `0.01` of the baseline). It passes L3-L6
  but fails L1/L2: its residual cosine is `0.0743` / `0.0620` **below** the
  baseline. Raw ArcFace carries identity structure but not the style-target
  fidelity that L1/L2 require.
- Concatenations improve the pair tail further than `e_s` alone (p90
  `0.732-0.787`) and lower `frac_gt_0.8`, but their residual cosine stays at
  `0.401-0.427`, so they also fail L1/L2. No combination recovers both target
  fidelity and separation under a linear map.
- No input clears the demo gate. The best gate-face maximum is
  `cfvn_es+fused+arcface` at `0.7606`, still above `0.60`; raw ArcFace is
  `0.7778`. Only `cfvn_es_arcface` pushes `C1_03/C1_05` below `0.50`
  (`-0.4619`), but it pushes `C1_04/C2_15` to `0.8281`. Raw ArcFace improves
  three of the four watch pairs but leaves `C1_03/C1_05` at `0.648` and
  `C1_04/C2_15` at `0.537`, both above `0.50`.
- The identity-overlap diagnostic found one possible VoxCeleb2 train overlap:
  `C2_07` has nearest-train ArcFace cosine `0.7466` with two train faces at or
  above `0.6`. `C1_01` (`0.5526`), `C2_13` (`0.5397`), and `C2_12` (`0.4730`)
  are below the `0.6` flag. This is a diagnostic, not a gate; it means the
  held-out face set is not proven disjoint from VoxCeleb2 identities, and
  `C2_07`'s result may be optimistic.

The pre-registered screen therefore separates two questions that were
previously conflated: **target fidelity** (L1/L2) and **identity separation**
(L3/L5, demo gate). Raw ArcFace improves the latter but not the former. The
current decision matrix requires both, and no input supplies both linearly.
A nonlinear side-channel adapter could in principle use `e_s` for target
fidelity and raw ArcFace for separation, but that is a **new hypothesis**, not
a result of this phase and not authorized by the Phase 1 gate. It would need a
separate pre-registration before any training.

### 6.3 Decision and next step

**NO-GO.** Phase 2 is not entered. The canonical adapter, demo, style
calibration, and fixed gates are unchanged. Per section 5, the next step is
Phase 3: paired face/voice data planning and licensing review. No
VoxCeleb2-only adapter objective is run on the same `e_s`.

---

## 7. Limitations

- The ridge is linear. A nonlinear adapter may recover interactions the ridge
  cannot, but the screen cannot prove that. In particular, the screen cannot
  rule out a side-channel adapter that uses `e_s` for target fidelity and raw
  ArcFace for separation; that is a new, separately pre-registered hypothesis.
- The selection rule is validation residual cosine, which measures target
  fidelity. It selected `cfvn_es_arcface`, not the input with the best pair
  separation. The descriptive sensitivity block shows every input is NO-GO, but
  the rule is not a pair-tail selector.
- Seven of eight inputs selected a ridge-grid edge (`1000` upper, `0.01`
  lower). The grid was not extended after observing test. The selected
  candidate's residual gain is far below the gate, so the edge does not change
  the NO-GO, but the absolute ceiling is not fully optimized.
- The official test split has only 118 speakers. Speaker-level pair tails are
  therefore noisier than the historical clip-level tails; the target tail is
  reported alongside every prediction.
- The demo set has 19 faces, of which 11 are the gate set. It is not a
  speaker-generalization benchmark. `C2_07` has a possible VoxCeleb2 train
  identity overlap (`0.7466` nearest-face cosine, two train faces at or above
  `0.6`), so its result may be optimistic.
- The demo predictions apply a speaker-centroid ridge to single-image demo
  features. That distribution is not identical to the per-clip corpus rows and
  is an additional source of mismatch.
- ECAPA and the audio metrics are proxies. A style cosine below `0.5` is
  screen-positive, not a guarantee that the frozen renderer will pass the
  audio floors.
- VoxCeleb2-only supervision may not cover the demo face/TTS domain even if a
  representation clears the screen.
- No human perceptual study exists. No claim of perceptual distinctness, MOS,
  or identity accuracy is made.
