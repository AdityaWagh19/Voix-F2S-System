# VOIX remediation plan

**Status: proposed. No production/serving change has been made.**
The v2 research demo is generated and verified, but it remains exploratory.
This plan separates what is already measured from what still requires a human
study, new data, or a model change.

## Evidence levels used here

- **Exploratory**: one prompt, one seed, objective proxies; no human ratings.
- **Confirmatory**: pre-registered hypothesis, held-out speakers, confidence
  intervals, and a human listening study where the claim is perceptual.
- **Deployment**: validated on the serving path with monitoring, rollback, and
  a human acceptance test.

The current demo is exploratory. No MOS, CMOS, emotion-identification, or
accent-identification result exists. No metric in this plan should be reported
as a perceptual quality score.

## P0 — freeze the evidence and stop overclaiming (now)

### P0.1 Freeze the v2 artifact

- Keep `results/demo/research_v2/` immutable after the final run.
- Record SHA-256 for `report.json`, `index.csv`, `README.md`, `index.html`, all
  21 WAVs, and the three copied face images.
- Keep `results/audit/baseline_audio/snapshot_20261105/` as the v1 before/after
  snapshot.
- Do not wire the style-manifold calibration into serving. It is a demo
  transform, not a serving path.

**Acceptance:** a manifest verifies every file; the canonical checkpoint hashes
remain `cfvn 0253bbbf…`, `cvae bb7ad29b…`, `style_adapter 97750ee8…`; the v2
report's `verification` block passes all ten checks.

### P0.2 Correct the claims

- Replace the top-level claim that residual noise is "not broadband hiss" with
  the narrower, measured statement: the old diagnostic ruled out clipping and
  quantization, but did not test a real-style control. The face style is off the
  StyleTTS2 manifold and is the dominant noise source.
- State that the three accents are applied conditions, not validated perceptual
  accents.
- State that sad is carried by duration only on the calibrated face styles; the
  E1 sad F0 lowering did not transfer, and the pitch-shift compensation was
  rejected because it collapsed ECAPA identity to about 0.57.
- State that happy and angry share an F0 rise; angry is separated mainly by a
  larger duration reduction, and perceptual distinctness is unproven.
- State that the Timothée–Jackie neutral pair is in the upper tail of the real
  same-gender ECAPA distribution (0.316; 97.8th percentile), even though it is
  below the observed real maximum (0.438).

**Acceptance:** `results/demo/research_v2/README.md`, `index.html`, and the new
audit report contain these statements; no document claims MOS/CMOS or validated
accent/emotion recognition.

### P0.3 Human listening study (the missing evidence)

Run one study, not several ad hoc ones. Minimum design:

- 20–30 listeners, balanced by gender and accent background where possible.
- Naturalness: CMOS or MUSHRA comparing v1 raw-face, v2 calibrated, and a real
  AMI reference. Include an anchor and a hidden reference.
- Emotion: three-way identification (happy/sad/angry) on the nine emotion clips,
  balanced order, with "unsure" allowed. Report accuracy and confusion matrix.
- Accent: three-way identification (US/England/Australian) on the nine accent
  clips, same protocol. Report accuracy and confusion matrix.
- Voice distinctness: triads with the three neutral voices; ask whether all
  three are the same speaker, two speakers, or three speakers.

**Acceptance:** pre-registered analysis; report mean, 95% CI, and per-listener
counts. Do not report a single aggregate without the confusion matrix.
Interpretation thresholds are in the table below.

### P0.4 Decide the canonical demo

- **Decision (2026-11-05):** keep both `results/demo/research/` (v1) and
  `results/demo/research_v2/` (v2). v2 is the current research demo; v1 remains
  for comparison. No demo folder was moved or renamed. The top-level README and
  audit report now point at v2; see `CLEANUP_INVENTORY.md` for the executed
  cleanup record.

## P1 — close the measurable quality gaps

### P1.1 Residual hiss

**Confirmed cause:** at alpha 0 the face style is far from the real AMI style
manifold (style cosine to real mean 0.522–0.697); the same model/text/seed with a
real AMI style gives −71.1 dBFS, while the raw face style gives −49.2 dBFS.
Alpha 1.0 calibration gives a median −63.45 dBFS across the 21 clips.

**Target:** median noise floor ≤ −68 dBFS and maximum ≤ −65 dBFS, with no
identity loss (ECAPA-to-neutral ≥ 0.90 for every non-neutral clip) and no
regression in the human ABX/CMOS test. AMI reference median is −76.1 dBFS.

**Approach (in order):**

1. Test an alpha sweep between 1.0 and 1.25 on the selected trio, but stop if any
   pairwise ECAPA exceeds the real same-gender p90 (0.250). Alpha 1.25 improves
   Lupita by 3.7 dB and Timothée by 1.3 dB but makes Jackie 0.2 dB worse; the
   selected-trio median is essentially unchanged (−63.5 dBFS vs −63.5 dBFS),
   and Timothée–Jackie rises to 0.422. The −66.1 dBFS figure often quoted for
   alpha 1.25 is the median across all five swept faces, not the selected trio.
2. Test a style prior/posterior interpolation or a real-style anchor per gender.
   Do not simply increase diffusion steps; 5/10/20/30 steps stayed between
   −45.5 and −49.4 dBFS on the raw face style.
3. Do not retrain N1 LoRA for this. The controlled matrix shows it worsens the
   noise floor (−46.3 vs −49.2 dBFS) and raises 4–8 kHz energy.
4. If the target is not reachable without identity loss, keep alpha 1.0 and
   report the residual gap explicitly. Do not apply a denoiser that has not
   been listening-tested; it may remove breathiness and make the delivery even
   more polished.

**Acceptance:** per-clip noise/SNR table; ECAPA identity; human ABX “hiss
audible?” with ≤ 20% detection on the calibrated set; no MOS claim from the
denoiser alone.

### P1.2 Voice diversity

**Current:** neutral ECAPA pairs Lupita–Timothée 0.119, Lupita–Jackie 0.114,
Timothée–Jackie 0.316. Real same-gender between-speaker median 0.115, p90 0.250,
max 0.438.

**Target:** every selected pair ≤ 0.250 (real p90), while each voice's
ECAPA-to-neutral stays ≥ 0.90 across all conditions.

**Approach:** choose a different third voice if Timothée–Jackie remains above
p90 after calibration; or explicitly accept the upper-tail pair and require the
human triad test to show three distinct speakers. Do not use raw face style as a
diversity fix: face-style pairs collapse (79/171 cosine > 0.90; median 0.831).

**Acceptance:** pairwise ECAPA table plus human triad accuracy; no pair above
the real maximum 0.438 under any circumstance.

### P1.3 Emotion distinctness

**Current:** 9/9 direction checks pass; happy F0 +8.2 to +16.4 Hz, angry +11.1
to +15.3 Hz; angry shortens duration more than happy. Sad F0 is flat; sad is
longer by 0.15–0.475 s. No perceptual result.

**Target:** three-way human identification ≥ 60% overall (chance 33%); each
emotion ≥ 50%; happy/angry confusion ≤ 25%. If sad remains duration-only and
fails the 50% floor, remove sad from the headline or add a validated prosody
mechanism (not a global pitch shift, which already failed identity).

**Approach:** keep scale 0.56 for the current demo; test scale 0.84/1.12 only
for sad duration, since it did not lower F0 and reducing identity at scale 1.12
(Jackie 0.946). For happy/angry separation, test a targeted duration/energy
contrast rather than more F0. Do not claim distinctness from residual cosine
alone.

**Acceptance:** human confusion matrix and pre-registered thresholds; objective
F0/duration directions remain 9/9.

### P1.4 Accent conditioning

**Current:** A3 scale 0.3 was selected by 7/30 vs 6/30 classifier accuracy, one
clip; per-class recall US 1/6, England 5/6, Australian 1/6, Canada 0/6, India
0/6. At demo scale 1.0 the F0 shifts are small (US −7.2 to −5.0 Hz, England +2.7
to +5.3 Hz, Australian −2.1 to −0.7 Hz) and ECAPA-to-neutral is 0.958–0.988.

**Target:** human three-way identification ≥ 60%, or relabel the feature as
“accent-conditioned timbre” and remove it from the headline. Objective target
if the model is improved: held-out per-class recall ≥ 60% for each of the three
accents.

**Approach:** the current GLOBE class contrast is speaker-confounded. Either
collect within-speaker accent pairs or use a corpus/model that separates accent
from speaker identity; train and evaluate on speaker-disjoint splits. Do not
raise the scale above 1.0 without a perceptual test; scale 1.5 lowers identity to
0.945–0.977 and still has no human evidence.

**Acceptance:** human accent-identification matrix; speaker-disjoint classifier
recall; no identity below 0.90. If the target is missed, the demo keeps the
“exploratory application” label and does not advertise “three accents” as a
perceptual capability.

### P1.5 Over-polished delivery

**Current:** the backend loads StyleTTS2's default LibriTTS checkpoint and the
fixed prompt is formal. N1 LoRA improved one proxy (F0 median) but worsened
noise; it did not produce a conversational delivery. Step count and alpha/beta
do not fix the speaking style.

**Target:** a conversational-delivery comparison with CMOS ≥ 0 vs the current
LibriTTS output and WER ≤ base + 0.02, noise ≤ −65 dBFS, ECAPA-to-neutral ≥ 0.90.

**Approach:** test an alternative checkpoint or a prosody/style control that does
not add a diffusion adapter to the vocoder. Do not repeat the N1 LoRA. Treat this
as a research branch, not a demo tweak; it changes the voice prior and therefore
requires a new identity/diversity audit.

**Acceptance:** paired human CMOS and the same objective gate as the demo.

### P1.6 Metadata and provenance

**Current v2:** per clip: text, voice id/label, track/condition, level, support,
note, WAV path, audio SHA-256, sample rate, seed, diffusion steps, alpha/beta,
F0 scale, style SHA-256, calibrated style SHA-256, residual, timing, metrics,
ECAPA-to-neutral, WER/transcript. Report: canonical checkpoint hashes, face
report hash, calibration artifact hash, emotion/accent artifact hashes, ECAPA
reference hash, environment, runtime, verification blocks.

**Gaps to close:** add the face image SHA-256 and the calibration/condition
artifact hashes to every CSV row; add the git commit and dirty-tree flag; add
the exact command line; add a machine-readable schema and validation test. The
v1 report had no audio SHA-256, no per-clip noise/SNR, no per-clip ECAPA, and no
per-voice identity block.

**Acceptance:** every required field non-null for all 21 rows; a JSON-schema or
unit test rejects a missing provenance field; a second run reproduces the same
audio hashes on the same hardware, or the nondeterminism is documented.

## P2 — hygiene and reproducibility

- Resolve the config inconsistency: `configs/environment.yaml` sets
  `diffusion_steps: null` and `tts_device: cpu`, while `configs/hardware.yaml`
  selects GPU profiles and the generator defaults to `cuda`, 10 steps. Make the
  demo read one resolved config and record it.
- Add the generator verification to CI (or a documented pre-release script):
  canonical hashes, face-style recomputation, audio hashes, metadata, noise,
  identity, identity preservation, emotion directions, emotion/accent
  application.
- Keep `results/audit/` and the v2 demo out of the cleanup deletion set.
- Clean up only after approval; see `CLEANUP_INVENTORY.md`.
- Add a “no MOS/CMOS” guard to any report template so a future run cannot
  silently label an objective proxy as a perceptual score.

## Acceptance summary

| Area | Metric | Current (v2) | Target | Status |
|---|---|---:|---:|---|
| Noise | median noise floor | −63.45 dBFS | ≤ −68 dBFS | **Not met** (improved from −49.17) |
| Noise | max noise floor | −61.75 dBFS | ≤ −65 dBFS | **Not met** |
| Noise | median SNR proxy | 44.97 dB | ≥ 40 dB | Met |
| Identity | min ECAPA-to-neutral | 0.935 | ≥ 0.90 | Met |
| Identity | max pair ECAPA | 0.316 | ≤ 0.250 (p90) | **Not met** (below real max 0.438) |
| Emotion | direction checks | 9/9 | 9/9 | Met |
| Emotion | human 3-way accuracy | unavailable | ≥ 60% | Not measured |
| Accent | application checks | 9/9 | 9/9 | Met |
| Accent | human 3-way accuracy | unavailable | ≥ 60% | Not measured |
| WER | mean WER | 0.000 | ≤ 0.05 | Met (one ASR model; verify with a second) |
| Metadata | required fields | all pass | 100% | Met |
| Metadata | hash chain | all pass | 100% | Met |
| Human naturalness | CMOS vs v1 | unavailable | ≥ 0 | Not measured |

## Stop conditions

- If a calibration change improves noise but pushes any pair above the real
  same-gender maximum 0.438, reject it.
- If a human study fails its pre-registered threshold, keep the demo
  exploratory and remove the corresponding headline claim.
- If an accent improvement cannot be demonstrated on speaker-disjoint data,
  keep the current “exploratory application” label.
- Do not delete or rewrite evidence to make a metric look better.
- Do not wire the calibration into serving without a separate deployment audit.

## References

- `results/audit/AUDIT_REPORT.md` — measured findings.
- `results/audit/BEFORE_AFTER.md` — controlled before/after numbers.
- `results/audit/CLEANUP_INVENTORY.md` — executed cleanup and per-file
  manifests.
- `results/demo/research_v2/report.json` — per-clip provenance and verification.
- `results/emotion/FINDINGS.md`, `results/accent/a3/FINDINGS.md`,
  `results/naturalness/n1/gate_report.json` — upstream evidence.
